"""Contracts for converting YOLO encoders to native CE classifier networks.

Only encoder loading and head initialization belong here. Freezing, training
modes, and checkpoint writing are owned by the native trainer and tested there.
"""

from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st
from pytorch_metric_learning.losses import SupConLoss
from ultralytics import YOLO

from smartrodent.supcon_model import YoloSupConModel


@pytest.fixture
def classifier_api():
    """Require the proposed model API without blocking legacy test collection."""
    from smartrodent import classifier_model

    return classifier_model


@pytest.fixture
def tiny_architecture(tmp_path):
    """Create a tiny classifier with backbone dropout to exercise strict freezing."""
    torch.set_num_threads(2)
    path = tmp_path / "tiny-cls.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "nn.Dropout", [0.6]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    return path


@pytest.fixture
def supcon_source(tiny_architecture, tmp_path):
    """Save real SupCon-updated features using the existing standalone checkpoint schema."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        source = YoloSupConModel(
            str(tiny_architecture), num_classes=2, output_features=4, dropout=0.25
        )
        source.requires_grad_(True)
        # Source-stage freezing must not override the later CE freeze choice.
        source.full_model.model.model[0].requires_grad_(False)
        source.label_names = {"mouse": 0, "rat": 1}
        source.full_model.model.names = {0: "mouse", 1: "rat"}
        optimizer = torch.optim.AdamW(
            (parameter for parameter in source.parameters() if parameter.requires_grad),
            lr=0.01,
        )
        source.train()
        _, embeddings = source(torch.rand(4, 3, 16, 16))
        SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
        optimizer.step()
        checkpoint = tmp_path / "supcon-best.pt"
        torch.save(
            {
                "epoch": 1,
                "architecture": deepcopy(source.full_model.model.yaml),
                "model_configuration": {
                    "dropout": 0.25,
                    "output_features": 4,
                    "num_classes": 2,
                    "freeze": [0],
                },
                "model_state_dict": source.state_dict(),
                "label_names": dict(source.label_names),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "configuration": {"model_name": str(tiny_architecture)},
                "history": [{"epoch": 1, "mean": 0.5}],
            },
            checkpoint,
        )
        network = source.finalize().model.eval()
    return SimpleNamespace(
        path=checkpoint,
        network=network,
        source_format="supcon",
        architecture=tiny_architecture,
    )


def _native_checkpoint(architecture, path, *, width=8, names=None):
    """Write a standard Ultralytics checkpoint and account for its FP16 save conversion."""
    document = yaml.safe_load(architecture.read_text())
    document["backbone"][0][3][0] = width
    classes = names if names is not None else {0: "mouse", 1: "rat"}
    document["nc"] = len(classes)
    document["head"][-1][3][0] = len(classes)
    config_path = path.with_suffix(".yaml")
    config_path.write_text(yaml.safe_dump(document))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yolo = YOLO(str(config_path), task="classify")
        yolo.model.names = classes
        head = yolo.model.model[-1]
        with torch.no_grad():
            head.conv.conv.weight.fill_(0.25)
            head.conv.bn.running_mean.fill_(1.25)
            head.conv.bn.running_var.fill_(0.75)
            head.linear.weight.fill_(0.125)
            head.linear.bias.copy_(torch.linspace(-0.5, 0.5, len(classes)))
        head.drop.p = 0.35
        yolo.save(path)
        # Compare to what is actually stored, not to pre-quantization FP32 values.
        network = YOLO(str(path), task="classify").model.eval()
    return SimpleNamespace(path=path, network=network, source_format="ultralytics")


@pytest.fixture
def native_source(tiny_architecture, tmp_path):
    """Provide an ordinary YOLO checkpoint with distinctive pretrained head weights."""
    return _native_checkpoint(tiny_architecture, tmp_path / "native-cls.pt")


@pytest.fixture(params=["supcon", "ultralytics"])
def encoder_source(request):
    """Exercise the same conversion contracts against both existing checkpoint formats."""
    fixture_name = "supcon_source" if request.param == "supcon" else "native_source"
    return request.getfixturevalue(fixture_name)


def _initialize(api, source, **updates):
    """Use the agreed default: retain pooled features and train a fresh linear head."""
    options = {
        "source_format": source.source_format,
        "label_names": {"mouse": 0, "rat": 1},
        "head_replace": "linear",
        "head_source": "random",
        "device": "cpu",
    }
    options.update(updates)
    initialization = {
        "encoder": {"source": options["source_format"], "checkpoint": source.path},
        "head": {
            "replace": options["head_replace"],
            "source": options["head_source"],
            "checkpoint": options.get("head_checkpoint"),
        },
    }
    return api.YoloClassifierModel.initialize(
        initialization,
        label_names=options["label_names"],
        device=options["device"],
    )


@pytest.fixture
def native_initializations(monkeypatch):
    """Observe native reset results so bounded-but-zero or custom reinitialization fails."""
    snapshots = {}
    originals = {
        layer: layer.reset_parameters for layer in (torch.nn.Linear, torch.nn.Conv2d)
    }

    def reset(layer):
        originals[type(layer)](layer)
        snapshots[id(layer)] = deepcopy(layer.state_dict())

    for layer in originals:
        monkeypatch.setattr(layer, "reset_parameters", reset)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        yield snapshots


def _replacement(network, head_replace):
    """Locate the explicitly selected YOLO classifier component."""
    head = network.model[-1]
    return head.linear if head_replace == "linear" else head


def _encoder_state(network, head_replace):
    """Snapshot everything retained as the encoder, including BatchNorm buffers."""
    head_name = list(network.model.named_children())[-1][0]
    prefix = f"model.{head_name}."
    if head_replace == "linear":
        prefix += "linear."
    return {
        name: tensor.detach().clone()
        for name, tensor in network.state_dict().items()
        if not name.startswith(prefix)
    }


def _assert_same_state(actual, expected):
    """Require exact checkpoint-state preservation, not approximate numerical similarity."""
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_linear_random_initialization_keeps_the_encoder_and_returns_eval_mode(
    classifier_api, supcon_source
):
    """A linear/random head is fresh; the encoder is unchanged and nothing starts in training mode."""
    model = classifier_api.YoloClassifierModel.initialize(
        {
            "encoder": {"source": "supcon", "checkpoint": supcon_source.path},
            "head": {"replace": "linear", "source": "random"},
        },
        label_names={"mouse": 0, "rat": 1},
    )
    _assert_same_state(
        _encoder_state(model.network, "linear"),
        _encoder_state(supcon_source.network, "linear"),
    )
    head = model.network.model[-1].linear
    assert not torch.equal(head.weight, supcon_source.network.model[-1].linear.weight)
    assert all(not module.training for module in model.modules())


@pytest.mark.parametrize("class_names", [("rat", "mouse"), ("shrew", "rat", "mouse")])
def test_fresh_linear_head_preserves_all_encoder_state(
    classifier_api, encoder_source, class_names, native_initializations
):
    """Keep Classify.conv/pooling and reset linear even when the class count already matches."""
    expected = _encoder_state(encoder_source.network, "linear")
    labels = {name: index for index, name in enumerate(sorted(class_names))}
    model = _initialize(classifier_api, encoder_source, label_names=labels)
    _assert_same_state(_encoder_state(model.network, "linear"), expected)
    head = model.network.model[-1].linear
    assert id(head) in native_initializations
    _assert_same_state(head.state_dict(), native_initializations[id(head)])
    assert torch.count_nonzero(head.weight)
    assert head.out_features == len(class_names)
    assert head.in_features == encoder_source.network.model[-1].linear.in_features
    original = encoder_source.network.model[-1].linear.weight
    assert head.weight.shape != original.shape or not torch.equal(head.weight, original)
    assert model.label_names == {
        name: index for index, name in enumerate(sorted(class_names))
    }
    assert model.network.names == {
        index: name for name, index in model.label_names.items()
    }
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all("projector" not in name for name in model.state_dict())
    assert all(not isinstance(module, YoloSupConModel) for module in model.modules())


def test_full_random_head_replaces_classify_conv_but_not_backbone(
    classifier_api, encoder_source, native_initializations
):
    """Full replacement deliberately discards the convolution inside Classify."""
    model = _initialize(classifier_api, encoder_source, head_replace="full")
    _assert_same_state(
        _encoder_state(model.network, "full"),
        _encoder_state(encoder_source.network, "full"),
    )
    head = model.network.model[-1]
    assert not torch.equal(
        head.conv.conv.weight, encoder_source.network.model[-1].conv.conv.weight
    )
    assert torch.equal(
        head.conv.bn.running_mean, torch.zeros_like(head.conv.bn.running_mean)
    )
    assert torch.equal(
        head.conv.bn.running_var, torch.ones_like(head.conv.bn.running_var)
    )
    assert head.linear.out_features == 2
    assert torch.equal(head.conv.bn.weight, torch.ones_like(head.conv.bn.weight))
    assert torch.equal(head.conv.bn.bias, torch.zeros_like(head.conv.bn.bias))
    assert head.conv.bn.eps == 0.001 and head.conv.bn.momentum == 0.03
    for layer in (head.conv.conv, head.linear):
        assert id(layer) in native_initializations
        _assert_same_state(layer.state_dict(), native_initializations[id(layer)])
        assert torch.count_nonzero(layer.weight)
        fan_in = layer.weight[0].numel()
        bound = fan_in**-0.5
        assert layer.weight.abs().max() <= bound
        if layer.bias is not None:
            assert layer.bias.abs().max() <= bound


@pytest.mark.parametrize("head_replace", ["linear", "full"])
@pytest.mark.parametrize(
    "source_classes,target_classes", [(2, 2), (1000, 2), (1000, 26)]
)
def test_head_checkpoint_initializes_only_compatible_requested_weights(
    classifier_api,
    supcon_source,
    native_source,
    tiny_architecture,
    tmp_path,
    head_replace,
    source_classes,
    target_classes,
):
    """Native partial transfer adapts output size without overwriting the SupCon encoder."""
    if source_classes == 1000:
        native_source = _native_checkpoint(
            tiny_architecture,
            tmp_path / "imagenet-like.pt",
            names={index: f"imagenet_{index}" for index in range(source_classes)},
        )
    labels = (
        {"mouse": 0, "rat": 1}
        if target_classes == 2
        else {f"species_{index:02}": index for index in range(target_classes)}
    )
    model = _initialize(
        classifier_api,
        supcon_source,
        head_replace=head_replace,
        label_names=labels,
        head_source="yolo_checkpoint",
        head_checkpoint=native_source.path,
    )
    _assert_same_state(
        _encoder_state(model.network, head_replace),
        _encoder_state(supcon_source.network, head_replace),
    )
    target = _replacement(model.network, head_replace).state_dict()
    donor = _replacement(native_source.network, head_replace).state_dict()
    for name, tensor in target.items():
        if donor[name].shape == tensor.shape:
            assert torch.equal(tensor, donor[name]), name
    head = model.network.model[-1]
    assert head.linear.out_features == target_classes
    assert model.network.names == {index: name for name, index in labels.items()}
    if source_classes != target_classes:
        assert head.linear.weight.shape == (
            target_classes,
            native_source.network.model[-1].linear.in_features,
        )
        assert torch.count_nonzero(head.linear.weight)
        original = supcon_source.network.model[-1].linear
        if original.out_features == target_classes:
            assert not torch.equal(
                head.linear.weight, original.weight
            ), "Do not retain the old SupCon output map"
    if head_replace == "full":
        # Native transfer loads tensors, not donor Dropout/activation configuration.
        assert head.drop.p == 0.0


def test_linear_transfer_allows_a_different_backbone_when_features_match(
    classifier_api, supcon_source, tiny_architecture, tmp_path
):
    """Compatibility applies to the selected linear layer, not unrelated donor backbone shapes."""
    donor = _native_checkpoint(tiny_architecture, tmp_path / "wide-donor.pt", width=16)
    model = _initialize(
        classifier_api,
        supcon_source,
        head_source="yolo_checkpoint",
        head_checkpoint=donor.path,
    )
    _assert_same_state(
        model.network.model[-1].linear.state_dict(),
        donor.network.model[-1].linear.state_dict(),
    )
    _assert_same_state(
        _encoder_state(model.network, "linear"),
        _encoder_state(supcon_source.network, "linear"),
    )


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_incompatible_head_tensors_are_skipped_like_native_ultralytics(
    classifier_api, supcon_source, tiny_architecture, tmp_path, head_replace
):
    """Keep the target architecture; transfer compatible tensors rather than rejecting a donor."""
    donor = _native_checkpoint(
        tiny_architecture, tmp_path / "incompatible.pt", width=16
    )
    if head_replace == "linear":
        yolo = YOLO(str(donor.path), task="classify")
        yolo.model.model[-1].linear = torch.nn.Linear(7, 2)
        yolo.save(donor.path)
    model = _initialize(
        classifier_api,
        supcon_source,
        head_replace=head_replace,
        head_source="yolo_checkpoint",
        head_checkpoint=donor.path,
    )
    _assert_same_state(
        _encoder_state(model.network, head_replace),
        _encoder_state(supcon_source.network, head_replace),
    )
    head = model.network.model[-1]
    assert (
        head.conv.conv.in_channels
        == supcon_source.network.model[-1].conv.conv.in_channels
    )
    assert head.linear.in_features == supcon_source.network.model[-1].linear.in_features
    loaded_donor = YOLO(str(donor.path), task="classify").model.model[-1]
    assert torch.equal(head.linear.bias, loaded_donor.linear.bias)
    if head_replace == "full":
        _assert_same_state(head.conv.bn.state_dict(), loaded_donor.conv.bn.state_dict())
    assert model(torch.rand(2, 3, 16, 16)).shape == (2, 2)


@pytest.mark.parametrize(
    "names", [{0: "rat", 1: "mouse"}, {0: "mouse", 1: "shrew"}, {0: "0", 1: "1"}]
)
def test_same_size_head_transfer_does_not_require_matching_class_identity(
    classifier_api, supcon_source, tiny_architecture, tmp_path, names
):
    """Native classification transfers matching rows by shape, then applies target names."""
    donor = _native_checkpoint(
        tiny_architecture, tmp_path / "different-labels.pt", names=names
    )
    model = _initialize(
        classifier_api,
        supcon_source,
        head_source="yolo_checkpoint",
        head_checkpoint=donor.path,
    )
    _assert_same_state(
        model.network.model[-1].linear.state_dict(),
        donor.network.model[-1].linear.state_dict(),
    )
    assert model.network.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_initialized_classifier_has_no_hidden_training_mode_policy(
    classifier_api, supcon_source, head_replace
):
    """train() switches every module, because freezing belongs to the native trainer."""
    model = _initialize(classifier_api, supcon_source, head_replace=head_replace)
    assert model.train() is model
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_forward_returns_raw_logits_in_train_and_eval(
    classifier_api, supcon_source, head_replace
):
    """CE receives raw scores without enabling training-mode BatchNorm during evaluation."""
    model = _initialize(classifier_api, supcon_source, head_replace=head_replace)
    with torch.no_grad():
        model.network.model[-1].linear.weight.zero_()
        model.network.model[-1].linear.bias.copy_(torch.tensor([-2.0, 3.0]))
    images = torch.rand(4, 3, 16, 16)
    for training in (True, False, True):
        model.train(training)
        before = {name: value.clone() for name, value in model.named_buffers()}
        output = model(images)
        assert isinstance(output, torch.Tensor) and output.shape == (4, 2)
        torch.testing.assert_close(output, torch.tensor([[-2.0, 3.0]]).expand(4, -1))
        if not training:
            assert all(not module.training for module in model.modules())
            _assert_same_state(dict(model.named_buffers()), before)


def test_supcon_conversion_does_not_keep_a_feature_capture_wrapper_alive(
    classifier_api, supcon_source, monkeypatch
):
    """If conversion uses the SupCon loader, remove its hook rather than retaining its closure."""
    restored_wrappers = []
    original = YoloSupConModel.from_checkpoint

    def restore(cls, *args, **kwargs):
        """Observe wrapper lifetime without requiring this loading implementation."""
        model = original(*args, **kwargs)
        restored_wrappers.append(weakref.ref(model))
        return model

    monkeypatch.setattr(YoloSupConModel, "from_checkpoint", classmethod(restore))
    model = _initialize(classifier_api, supcon_source)
    gc.collect()
    assert all(reference() is None for reference in restored_wrappers)
    assert model(torch.rand(2, 3, 16, 16)).shape == (2, 2)


@pytest.mark.parametrize("location", ["encoder", "head"])
def test_detection_checkpoint_is_not_a_classification_model(
    classifier_api, supcon_source, tmp_path, location
):
    """A real YOLO detection checkpoint must fail clearly, not much later during CE."""
    architecture = tmp_path / "tiny-detect.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
                "head": [[[-1], 1, "Detect", [2]]],
            }
        )
    )
    path = tmp_path / "detector.pt"
    YOLO(str(architecture), task="detect").save(path)
    with pytest.raises(NotImplementedError, match="YOLO|classif|support"):
        if location == "encoder":
            _initialize(
                classifier_api, SimpleNamespace(path=path, source_format="ultralytics")
            )
        else:
            _initialize(
                classifier_api,
                supcon_source,
                head_source="yolo_checkpoint",
                head_checkpoint=path,
            )


@pytest.mark.parametrize("location", ["encoder", "head"])
def test_non_yolo_torch_model_has_an_explicit_unsupported_branch(
    classifier_api, supcon_source, tmp_path, location
):
    """A readable Torch module is not silently accepted as a supported YOLO model."""
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))
    network.names = {0: "mouse", 1: "rat"}
    path = tmp_path / "non-yolo.pt"
    torch.save({"model": network, "train_args": {"task": "classify"}}, path)
    if location == "encoder":
        source = SimpleNamespace(path=path, source_format="ultralytics")
        options = {}
    else:
        source = supcon_source
        options = {"head_source": "yolo_checkpoint", "head_checkpoint": path}
    with pytest.raises(NotImplementedError, match="YOLO|support"):
        _initialize(classifier_api, source, **options)


def test_explicit_source_format_does_not_fall_back_to_another_loader(
    classifier_api, supcon_source
):
    """A SupCon payload is not silently autodetected after an explicit format mismatch."""
    with pytest.raises(ValueError, match="format|checkpoint"):
        _initialize(classifier_api, supcon_source, source_format="ultralytics")


def test_native_checkpoint_cannot_be_read_as_supcon(classifier_api, native_source):
    """An explicit SupCon source format must not silently unpickle a native model."""
    with pytest.raises(ValueError, match="supcon|format|checkpoint"):
        _initialize(classifier_api, native_source, source_format="supcon")


def test_initialization_is_required_rather_than_guessed(classifier_api):
    """Model construction requires explicit, prepared initialization settings."""
    with pytest.raises(TypeError, match="initialization"):
        classifier_api.YoloClassifierModel.initialize(
            label_names={"mouse": 0, "rat": 1}
        )


def test_missing_source_checkpoint_propagates_file_error(classifier_api, tmp_path):
    """A missing local checkpoint must not trigger downloads or another loader fallback."""
    source = SimpleNamespace(path=tmp_path / "missing.pt", source_format="supcon")
    with pytest.raises(FileNotFoundError):
        _initialize(classifier_api, source)


# The checkpoint fixture is read-only across examples; every example builds a
# fresh classifier. No mutable model or optimizer is shared between examples.
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    class_names=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
def test_fresh_head_preserves_supplied_label_mapping(
    classifier_api, supcon_source, class_names
):
    """The model preserves supplied IDs instead of sorting and reassigning them."""
    expected = {name: index for index, name in enumerate(class_names)}
    model = _initialize(classifier_api, supcon_source, label_names=expected)
    assert model.label_names == expected
    assert model.network.names == {index: name for name, index in expected.items()}
    assert model.network.model[-1].linear.out_features == len(class_names)


@pytest.mark.parametrize("location", ["encoder", "head"])
def test_corrupt_official_named_local_checkpoint_is_never_replaced_or_downloaded(
    classifier_api, supcon_source, tmp_path, monkeypatch, location
):
    """A bare official filename remains a read-only local input, even when corrupt."""
    from ultralytics.utils import downloads

    monkeypatch.chdir(tmp_path)
    path = Path("yolo26n-cls.pt")
    original = b"not a valid checkpoint"
    path.write_bytes(original)

    def forbid_download(*args, **kwargs):
        """Prevent the upstream corrupt-cache recovery path from reaching the network."""
        raise AssertionError("Unexpected checkpoint download")

    monkeypatch.setattr(downloads, "safe_download", forbid_download)
    with pytest.raises(ValueError, match="checkpoint|format"):
        if location == "encoder":
            _initialize(
                classifier_api, SimpleNamespace(path=path, source_format="ultralytics")
            )
        else:
            _initialize(
                classifier_api,
                supcon_source,
                head_source="yolo_checkpoint",
                head_checkpoint=path,
            )
    assert path.read_bytes() == original


def _fused_native_checkpoint(source, path, component):
    """Use the official fusion primitive to create a saved, partially/fully fused model."""
    from ultralytics.nn.modules import Conv
    from ultralytics.utils.torch_utils import fuse_conv_and_bn

    yolo = YOLO(str(source.path), task="classify")
    if component == "backbone":
        modules = [yolo.model.model[0]]
    elif component == "head":
        modules = [yolo.model.model[-1].conv]
    else:
        modules = [
            module for module in yolo.model.modules() if isinstance(module, Conv)
        ]
    for module in modules:
        module.conv = fuse_conv_and_bn(module.conv, module.bn)
        del module.bn
        module.forward = module.forward_fuse
    yolo.save(path)
    return SimpleNamespace(path=path, source_format="ultralytics")


@pytest.mark.parametrize(
    "component,boundary",
    [("backbone", "linear"), ("backbone", "full"), ("head", "linear")],
)
def test_fused_retained_encoder_is_rejected_before_classifier_is_returned(
    classifier_api, native_source, tmp_path, component, boundary
):
    """Retained fused state cannot silently become a CE checkpoint that will not restore."""
    source = _fused_native_checkpoint(
        native_source, tmp_path / "fused-encoder.pt", component
    )
    with pytest.raises(ValueError, match="fused|fusion"):
        _initialize(classifier_api, source, head_replace=boundary)


def test_fused_discarded_head_does_not_prevent_full_replacement(
    classifier_api, native_source, tmp_path
):
    """Only retained components matter; a fused head being discarded is harmless."""
    source = _fused_native_checkpoint(
        native_source, tmp_path / "discard-head.pt", "head"
    )
    model = _initialize(classifier_api, source, head_replace="full")
    _assert_same_state(
        _encoder_state(model.network, "full"),
        _encoder_state(native_source.network, "full"),
    )


def test_fused_full_head_donor_is_rejected(
    classifier_api, supcon_source, native_source, tmp_path
):
    """Copying a fused Classify module would make the final classifier unrestorable."""
    donor = _fused_native_checkpoint(
        native_source, tmp_path / "fused-head-donor.pt", "head"
    )
    with pytest.raises(ValueError, match="fused|fusion"):
        _initialize(
            classifier_api,
            supcon_source,
            head_replace="full",
            head_source="yolo_checkpoint",
            head_checkpoint=donor.path,
        )


@pytest.mark.parametrize(
    "component,boundary", [("all", "linear"), ("backbone", "full")]
)
def test_fused_unused_donor_components_do_not_prevent_head_transfer(
    classifier_api, supcon_source, native_source, tmp_path, component, boundary
):
    """Do not reject fused donor layers outside the actual transferred component."""
    donor = _fused_native_checkpoint(
        native_source, tmp_path / "fused-unused-donor.pt", component
    )
    model = _initialize(
        classifier_api,
        supcon_source,
        head_replace=boundary,
        head_source="yolo_checkpoint",
        head_checkpoint=donor.path,
    )
    _assert_same_state(
        _encoder_state(model.network, boundary),
        _encoder_state(supcon_source.network, boundary),
    )
    _assert_same_state(
        _replacement(model.network, boundary).state_dict(),
        _replacement(native_source.network, boundary).state_dict(),
    )


@pytest.fixture
def activation_defaults(monkeypatch):
    """Isolate Ultralytics' process-global default activation in activation regressions."""
    from ultralytics.nn.modules import Conv

    monkeypatch.setattr(Conv, "default_act", torch.nn.SiLU())


def test_full_weight_transfer_keeps_target_activation(
    activation_defaults,
    classifier_api,
    supcon_source,
    tiny_architecture,
    tmp_path,
    monkeypatch,
):
    """Native state transfer does not import the donor's activation into the target graph."""
    from ultralytics.nn.modules import Conv

    architecture = tmp_path / "relu-donor.yaml"
    document = yaml.safe_load(tiny_architecture.read_text())
    document["activation"] = "nn.ReLU()"
    architecture.write_text(yaml.safe_dump(document))
    donor = _native_checkpoint(architecture, tmp_path / "relu-donor.pt")
    monkeypatch.setattr(Conv, "default_act", torch.nn.SiLU())
    model = _initialize(
        classifier_api,
        supcon_source,
        head_replace="full",
        head_source="yolo_checkpoint",
        head_checkpoint=donor.path,
    ).eval()
    assert isinstance(model.network.model[0].act, torch.nn.SiLU)
    assert isinstance(model.network.model[-1].conv.act, torch.nn.SiLU)


def test_random_full_head_activation_is_independent_of_encoder_yaml(
    activation_defaults, classifier_api, tiny_architecture, tmp_path, monkeypatch
):
    """A fresh SiLU head on a saved ReLU encoder keeps both activations."""
    from ultralytics.nn.modules import Conv

    architecture = tmp_path / "relu-encoder.yaml"
    document = yaml.safe_load(tiny_architecture.read_text())
    document["activation"] = "nn.ReLU()"
    architecture.write_text(yaml.safe_dump(document))
    source = _native_checkpoint(architecture, tmp_path / "relu-encoder.pt")
    monkeypatch.setattr(Conv, "default_act", torch.nn.SiLU())
    model = _initialize(classifier_api, source, head_replace="full").eval()
    assert isinstance(model.network.model[0].act, torch.nn.ReLU)
    assert isinstance(model.network.model[-1].conv.act, torch.nn.SiLU)
