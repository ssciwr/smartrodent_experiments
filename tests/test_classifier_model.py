"""Contracts for converting YOLO encoders to CE classifiers and freezing them.

Only model initialization, portable model state, and training-mode policy belong
here. Optimizer creation, the CE trainer, and pipeline orchestration come later.
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
    path.write_text(yaml.safe_dump({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "nn.Dropout", [0.6]]],
        "head": [[-1, 1, "Classify", [2]]],
    }))
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
            (parameter for parameter in source.parameters() if parameter.requires_grad), lr=0.01
        )
        source.train()
        _, embeddings = source(torch.rand(4, 3, 16, 16))
        SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
        optimizer.step()
        checkpoint = tmp_path / "supcon-best.pt"
        torch.save({
            "epoch": 1,
            "architecture": deepcopy(source.full_model.model.yaml),
            "model_configuration": {
                "dropout": 0.25, "output_features": 4, "num_classes": 2, "freeze": [0],
            },
            "model_state_dict": source.state_dict(),
            "label_names": dict(source.label_names),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "configuration": {"model_name": str(tiny_architecture)},
            "history": [{"epoch": 1, "mean": 0.5}],
        }, checkpoint)
        network = source.finalize().model.eval()
    return SimpleNamespace(
        path=checkpoint, network=network, source_format="supcon", architecture=tiny_architecture
    )


def _native_checkpoint(architecture, path, *, width=8, names=None):
    """Write a standard Ultralytics checkpoint and account for its FP16 save conversion."""
    document = yaml.safe_load(architecture.read_text())
    document["backbone"][0][3][0] = width
    config_path = path.with_suffix(".yaml")
    config_path.write_text(yaml.safe_dump(document))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yolo = YOLO(str(config_path), task="classify")
        yolo.model.names = names if names is not None else {0: "mouse", 1: "rat"}
        head = yolo.model.model[-1]
        with torch.no_grad():
            head.conv.conv.weight.fill_(0.25)
            head.conv.bn.running_mean.fill_(1.25)
            head.conv.bn.running_var.fill_(0.75)
            head.linear.weight.fill_(0.125)
            head.linear.bias.copy_(torch.tensor([-0.5, 0.5]))
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
        "freeze_encoder": True,
        "encoder_batchnorm": "eval",
        "device": "cpu",
    }
    options.update(updates)
    initialization = {
        "encoder": {"source": options["source_format"], "checkpoint": source.path},
        "head": {"replace": options["head_replace"], "source": options["head_source"],
                 "checkpoint": options.get("head_checkpoint")},
    }
    return api.YoloClassifierModel.initialize(
        initialization, label_names=options["label_names"],
        freeze_encoder=options["freeze_encoder"], encoder_batchnorm=options["encoder_batchnorm"],
        device=options["device"],
    )


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


def test_default_initialization_is_a_fresh_linear_head_with_strictly_frozen_encoder(
    classifier_api, supcon_source
):
    """Omitted options mean linear/random replacement and frozen encoder with BN eval."""
    model = classifier_api.YoloClassifierModel.initialize(
        {"encoder": {"source": "supcon", "checkpoint": supcon_source.path},
         "head": {"replace": "linear", "source": "random"}},
        label_names={"mouse": 0, "rat": 1},
    )
    _assert_same_state(_encoder_state(model.network, "linear"), _encoder_state(supcon_source.network, "linear"))
    head = model.network.model[-1].linear
    assert not torch.equal(head.weight, supcon_source.network.model[-1].linear.weight)
    model.train()
    head_ids = {id(parameter) for parameter in head.parameters()}
    assert all(parameter.requires_grad == (id(parameter) in head_ids) for parameter in model.network.parameters())
    assert all(not module.training for module in model.network.modules() if isinstance(module, torch.nn.BatchNorm2d))
    configuration = model.checkpoint_state()["model_configuration"]
    assert configuration["head_replace"] == "linear"
    assert configuration["freeze_encoder"] is True
    assert configuration["encoder_batchnorm"] == "eval"


@pytest.mark.parametrize("class_names", [("rat", "mouse"), ("shrew", "rat", "mouse")])
def test_fresh_linear_head_preserves_all_encoder_state(classifier_api, encoder_source, class_names):
    """Keep Classify.conv/pooling and reset linear even when the class count already matches."""
    expected = _encoder_state(encoder_source.network, "linear")
    labels = {name: index for index, name in enumerate(sorted(class_names))}
    model = _initialize(classifier_api, encoder_source, label_names=labels)
    _assert_same_state(_encoder_state(model.network, "linear"), expected)
    head = model.network.model[-1].linear
    assert head.out_features == len(class_names)
    assert head.in_features == encoder_source.network.model[-1].linear.in_features
    original = encoder_source.network.model[-1].linear.weight
    assert head.weight.shape != original.shape or not torch.equal(head.weight, original)
    assert model.label_names == {name: index for index, name in enumerate(sorted(class_names))}
    assert model.network.names == {index: name for name, index in model.label_names.items()}
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all("projector" not in name for name in model.state_dict())
    assert all(not isinstance(module, YoloSupConModel) for module in model.modules())


def test_full_random_head_replaces_classify_conv_but_not_backbone(classifier_api, encoder_source):
    """Full replacement deliberately discards the convolution inside Classify."""
    model = _initialize(classifier_api, encoder_source, head_replace="full")
    _assert_same_state(
        _encoder_state(model.network, "full"), _encoder_state(encoder_source.network, "full")
    )
    head = model.network.model[-1]
    assert not torch.equal(head.conv.conv.weight, encoder_source.network.model[-1].conv.conv.weight)
    assert torch.equal(head.conv.bn.running_mean, torch.zeros_like(head.conv.bn.running_mean))
    assert torch.equal(head.conv.bn.running_var, torch.ones_like(head.conv.bn.running_var))
    assert head.linear.out_features == 2


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_head_checkpoint_initializes_only_the_requested_component(
    classifier_api, supcon_source, native_source, head_replace
):
    """A donor head cannot overwrite the retained SupCon backbone or feature convolution."""
    model = _initialize(
        classifier_api, supcon_source, head_replace=head_replace,
        head_source="yolo_checkpoint", head_checkpoint=native_source.path,
    )
    _assert_same_state(
        _encoder_state(model.network, head_replace), _encoder_state(supcon_source.network, head_replace)
    )
    _assert_same_state(
        _replacement(model.network, head_replace).state_dict(),
        _replacement(native_source.network, head_replace).state_dict(),
    )
    if head_replace == "full":
        assert model.network.model[-1].drop.p == native_source.network.model[-1].drop.p


def test_linear_transfer_allows_a_different_backbone_when_features_match(
    classifier_api, supcon_source, tiny_architecture, tmp_path
):
    """Compatibility applies to the selected linear layer, not unrelated donor backbone shapes."""
    donor = _native_checkpoint(tiny_architecture, tmp_path / "wide-donor.pt", width=16)
    model = _initialize(
        classifier_api, supcon_source, head_source="yolo_checkpoint", head_checkpoint=donor.path
    )
    _assert_same_state(model.network.model[-1].linear.state_dict(), donor.network.model[-1].linear.state_dict())
    _assert_same_state(_encoder_state(model.network, "linear"), _encoder_state(supcon_source.network, "linear"))


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_incompatible_head_dimensions_fail_clearly(
    classifier_api, supcon_source, tiny_architecture, tmp_path, head_replace
):
    """Incompatible feature widths are rejected instead of partially transferring weights."""
    donor = _native_checkpoint(tiny_architecture, tmp_path / "incompatible.pt", width=16)
    if head_replace == "linear":
        yolo = YOLO(str(donor.path), task="classify")
        yolo.model.model[-1].linear = torch.nn.Linear(7, 2)
        yolo.save(donor.path)
    with pytest.raises(ValueError, match="compatible|shape|features|channels"):
        _initialize(
            classifier_api, supcon_source, head_replace=head_replace,
            head_source="yolo_checkpoint", head_checkpoint=donor.path,
        )


@pytest.mark.parametrize("names", [{0: "rat", 1: "mouse"}, {0: "mouse", 1: "shrew"}, {0: "0", 1: "1"}])
def test_head_checkpoint_requires_matching_class_identity_and_order(
    classifier_api, supcon_source, tiny_architecture, tmp_path, names
):
    """Matching output dimensions alone do not establish matching label semantics."""
    donor = _native_checkpoint(tiny_architecture, tmp_path / "wrong-labels.pt", names=names)
    with pytest.raises(ValueError, match="class|label"):
        _initialize(
            classifier_api, supcon_source, head_source="yolo_checkpoint", head_checkpoint=donor.path
        )


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_forward_returns_raw_logits_in_train_and_eval(classifier_api, supcon_source, head_replace):
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


@pytest.mark.parametrize("head_replace", ["linear", "full"])
@pytest.mark.parametrize("freeze,batchnorm", [(True, "eval"), (True, "adapt"), (False, "eval"), (False, "adapt")])
def test_freeze_policy_controls_parameters_buffers_and_modes_across_ce_steps(
    classifier_api, supcon_source, head_replace, freeze, batchnorm
):
    """Frozen/eval is truly fixed; adaptation changes BN buffers, not frozen weights."""
    model = _initialize(
        classifier_api, supcon_source, head_replace=head_replace,
        freeze_encoder=freeze, encoder_batchnorm=batchnorm,
    )
    model.network.model[-1].drop.p = 0.75
    head = _replacement(model.network, head_replace)
    head_modules = {id(module) for module in head.modules()}
    head_parameters = {id(parameter) for parameter in head.parameters()}
    encoder_parameters = [parameter for parameter in model.network.parameters() if id(parameter) not in head_parameters]
    encoder_bn = [
        module for module in model.network.modules()
        if isinstance(module, torch.nn.BatchNorm2d) and id(module) not in head_modules
    ]
    encoder_dropout = [
        module for module in model.network.modules()
        if isinstance(module, torch.nn.Dropout) and id(module) not in head_modules
    ]
    assert encoder_bn and encoder_dropout
    before_encoder = _encoder_state(model.network, head_replace)
    before_parameters = [parameter.detach().clone() for parameter in encoder_parameters]
    before_bn = [(module.running_mean.clone(), module.running_var.clone()) for module in encoder_bn]
    before_head = [parameter.detach().clone() for parameter in head.parameters()]
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.05
    )
    for _ in range(3):
        model.eval()
        assert all(not module.training for module in model.modules())
        assert model.train() is model
        assert all(module.training for module in head.modules())
        assert all(parameter.requires_grad for parameter in head.parameters())
        assert all(parameter.requires_grad == (not freeze) for parameter in encoder_parameters)
        assert all(module.training == (batchnorm == "adapt") for module in encoder_bn)
        assert all(module.training == (not freeze) for module in encoder_dropout)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(torch.rand(4, 3, 16, 16)), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        optimizer.step()
    changed_encoder = [not torch.equal(before, after) for before, after in zip(before_parameters, encoder_parameters, strict=True)]
    assert any(changed_encoder) == (not freeze)
    if freeze and batchnorm == "eval":
        _assert_same_state(_encoder_state(model.network, head_replace), before_encoder)
    if freeze:
        assert all(parameter.grad is None for parameter in encoder_parameters)
    assert any(not torch.equal(before, after) for before, after in zip(before_head, head.parameters(), strict=True))
    changed_bn = [
        not torch.equal(mean, module.running_mean) or not torch.equal(var, module.running_var)
        for (mean, var), module in zip(before_bn, encoder_bn, strict=True)
    ]
    assert any(changed_bn) == (batchnorm == "adapt")


@pytest.mark.parametrize("head_replace", ["linear", "full"])
@pytest.mark.parametrize("freeze,batchnorm", [(True, "eval"), (True, "adapt"), (False, "eval"), (False, "adapt")])
def test_classifier_checkpoint_round_trip_preserves_logits_state_and_policy(
    classifier_api, supcon_source, tmp_path, head_replace, freeze, batchnorm
):
    """Restore a standalone CE model without the original source checkpoint or YAML."""
    model = _initialize(
        classifier_api, supcon_source, label_names={"mouse": 0, "rat": 1, "shrew": 2},
        head_replace=head_replace, freeze_encoder=freeze, encoder_batchnorm=batchnorm,
    )
    model.train()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
    images = torch.rand(4, 3, 16, 16)
    torch.nn.functional.cross_entropy(model(images), torch.tensor([0, 1, 2, 0])).backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        expected = model(images)
    payload = model.checkpoint_state()
    assert payload["checkpoint_type"] == "classifier"
    assert payload["label_names"] == model.label_names
    assert payload["model_configuration"]["head_replace"] == head_replace
    assert payload["model_configuration"]["freeze_encoder"] == freeze
    assert payload["model_configuration"]["encoder_batchnorm"] == batchnorm
    path = tmp_path / "ce-best.pt"
    torch.save({**payload, "epoch": 3, "optimizer_state_dict": optimizer.state_dict()}, path)
    supcon_source.path.unlink()
    supcon_source.architecture.unlink()
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path, device="cpu")
    assert not restored.training
    assert restored.label_names == {"mouse": 0, "rat": 1, "shrew": 2}
    _assert_same_state(restored.state_dict(), model.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected)
    restored.train()
    restored_head = _replacement(restored.network, head_replace)
    head_ids = {id(parameter) for parameter in restored_head.parameters()}
    assert all(parameter.requires_grad == (id(parameter) in head_ids or not freeze) for parameter in restored.network.parameters())
    head_modules = {id(module) for module in restored_head.modules()}
    encoder_bn = [module for module in restored.network.modules() if isinstance(module, torch.nn.BatchNorm2d) and id(module) not in head_modules]
    assert encoder_bn and all(module.training == (batchnorm == "adapt") for module in encoder_bn)


def test_full_checkpoint_head_dropout_survives_portable_restoration(
    classifier_api, supcon_source, native_source, tmp_path
):
    """Head behavior not represented by tensors must survive serialization too."""
    model = _initialize(
        classifier_api, supcon_source, head_replace="full",
        head_source="yolo_checkpoint", head_checkpoint=native_source.path,
    )
    path = tmp_path / "donor-ce.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    assert restored.network.model[-1].drop.p == native_source.network.model[-1].drop.p


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_classifier_checkpoint_can_initialize_another_classifier(
    classifier_api, supcon_source, tmp_path, head_replace
):
    """An existing standalone CE checkpoint is an explicit encoder source, not an implicit resume."""
    first = _initialize(classifier_api, supcon_source)
    path = tmp_path / "classifier-source.pt"
    torch.save(first.checkpoint_state(), path)
    source = SimpleNamespace(path=path, source_format="classifier")
    second = _initialize(classifier_api, source, head_replace=head_replace)
    _assert_same_state(_encoder_state(second.network, head_replace), _encoder_state(first.network, head_replace))
    assert not torch.equal(second.network.model[-1].linear.weight, first.network.model[-1].linear.weight)


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
    architecture.write_text(yaml.safe_dump({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
        "head": [[[-1], 1, "Detect", [2]]],
    }))
    path = tmp_path / "detector.pt"
    YOLO(str(architecture), task="detect").save(path)
    with pytest.raises(ValueError, match="classif"):
        if location == "encoder":
            _initialize(
                classifier_api, SimpleNamespace(path=path, source_format="ultralytics")
            )
        else:
            _initialize(
                classifier_api, supcon_source,
                head_source="yolo_checkpoint", head_checkpoint=path,
            )


@pytest.mark.parametrize("wrong_format", ["ultralytics", "classifier"])
def test_explicit_source_format_does_not_fall_back_to_another_loader(
    classifier_api, supcon_source, wrong_format
):
    """A SupCon payload is not silently autodetected after an explicit format mismatch."""
    with pytest.raises(ValueError, match="format|checkpoint|classifier"):
        _initialize(classifier_api, supcon_source, source_format=wrong_format)


def test_native_checkpoint_cannot_be_read_as_supcon(classifier_api, native_source):
    """An explicit SupCon source format must not silently unpickle a native model."""
    with pytest.raises(ValueError, match="supcon|format|checkpoint"):
        _initialize(classifier_api, native_source, source_format="supcon")


def test_restore_rejects_supcon_instead_of_reinitializing_a_head(classifier_api, supcon_source):
    """Restoration and new classifier initialization are intentionally separate operations."""
    with pytest.raises(ValueError, match="classifier"):
        classifier_api.YoloClassifierModel.from_checkpoint(supcon_source.path)


def test_initialization_is_required_rather_than_guessed(classifier_api):
    """Model construction requires explicit, prepared initialization settings."""
    with pytest.raises(TypeError, match="initialization"):
        classifier_api.YoloClassifierModel.initialize(label_names={"mouse": 0, "rat": 1})


def test_missing_source_checkpoint_propagates_file_error(classifier_api, tmp_path):
    """A missing local checkpoint must not trigger downloads or another loader fallback."""
    source = SimpleNamespace(path=tmp_path / "missing.pt", source_format="supcon")
    with pytest.raises(FileNotFoundError):
        _initialize(classifier_api, source)


# The checkpoint fixture is read-only across examples; every example builds a
# fresh classifier. No mutable model or optimizer is shared between examples.
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(class_names=st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8), min_size=2, max_size=5, unique=True))
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
            _initialize(classifier_api, SimpleNamespace(path=path, source_format="ultralytics"))
        else:
            _initialize(
                classifier_api, supcon_source,
                head_source="yolo_checkpoint", head_checkpoint=path,
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
        modules = [module for module in yolo.model.modules() if isinstance(module, Conv)]
    for module in modules:
        module.conv = fuse_conv_and_bn(module.conv, module.bn)
        del module.bn
        module.forward = module.forward_fuse
    yolo.save(path)
    return SimpleNamespace(path=path, source_format="ultralytics")


@pytest.mark.parametrize(
    "component,boundary", [("backbone", "linear"), ("backbone", "full"), ("head", "linear")]
)
def test_fused_retained_encoder_is_rejected_before_classifier_is_returned(
    classifier_api, native_source, tmp_path, component, boundary
):
    """Retained fused state cannot silently become a CE checkpoint that will not restore."""
    source = _fused_native_checkpoint(native_source, tmp_path / "fused-encoder.pt", component)
    with pytest.raises(ValueError, match="fused|fusion"):
        _initialize(classifier_api, source, head_replace=boundary)


def test_fused_discarded_head_does_not_prevent_full_replacement(
    classifier_api, native_source, tmp_path
):
    """Only retained components matter; a fused head being discarded is harmless."""
    source = _fused_native_checkpoint(native_source, tmp_path / "discard-head.pt", "head")
    model = _initialize(classifier_api, source, head_replace="full")
    _assert_same_state(_encoder_state(model.network, "full"), _encoder_state(native_source.network, "full"))
    path = tmp_path / "unfused-ce.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    _assert_same_state(restored.state_dict(), model.state_dict())


def test_fused_full_head_donor_is_rejected(classifier_api, supcon_source, native_source, tmp_path):
    """Copying a fused Classify module would make the final classifier unrestorable."""
    donor = _fused_native_checkpoint(native_source, tmp_path / "fused-head-donor.pt", "head")
    with pytest.raises(ValueError, match="fused|fusion"):
        _initialize(
            classifier_api, supcon_source, head_replace="full",
            head_source="yolo_checkpoint", head_checkpoint=donor.path,
        )


@pytest.mark.parametrize("component,boundary", [("all", "linear"), ("backbone", "full")])
def test_fused_unused_donor_components_do_not_prevent_head_transfer(
    classifier_api, supcon_source, native_source, tmp_path, component, boundary
):
    """Do not reject fused donor layers outside the actual transferred component."""
    donor = _fused_native_checkpoint(native_source, tmp_path / "fused-unused-donor.pt", component)
    model = _initialize(
        classifier_api, supcon_source, head_replace=boundary,
        head_source="yolo_checkpoint", head_checkpoint=donor.path,
    )
    _assert_same_state(_encoder_state(model.network, boundary), _encoder_state(supcon_source.network, boundary))
    _assert_same_state(
        _replacement(model.network, boundary).state_dict(),
        _replacement(native_source.network, boundary).state_dict(),
    )


def test_ce_checkpoint_is_not_supcon_even_though_metadata_keys_overlap(
    classifier_api, supcon_source, tmp_path
):
    """Reject the CE discriminator before passing CE options into the SupCon constructor."""
    model = _initialize(classifier_api, supcon_source)
    path = tmp_path / "classifier-not-supcon.pt"
    torch.save(model.checkpoint_state(), path)
    with pytest.raises(ValueError, match="supcon|SupCon|format"):
        _initialize(classifier_api, SimpleNamespace(path=path, source_format="supcon"))


def test_fusion_after_initialization_cannot_export_an_unrestorable_payload(
    classifier_api, supcon_source
):
    """The exposed native network must not bypass portability checks when fused later."""
    model = _initialize(classifier_api, supcon_source)
    model.network.fuse(verbose=False)
    with pytest.raises(ValueError, match="fused|fusion"):
        model.checkpoint_state()


@pytest.fixture
def activation_defaults(monkeypatch):
    """Isolate Ultralytics' process-global default activation in activation regressions."""
    from ultralytics.nn.modules import Conv

    monkeypatch.setattr(Conv, "default_act", torch.nn.SiLU())


def test_full_donor_activation_survives_round_trip(
    activation_defaults, classifier_api, supcon_source, tiny_architecture, tmp_path, monkeypatch
):
    """Copying a ReLU donor onto SiLU features must not silently restore the donor as SiLU."""
    from ultralytics.nn.modules import Conv

    architecture = tmp_path / "relu-donor.yaml"
    document = yaml.safe_load(tiny_architecture.read_text())
    document["activation"] = "nn.ReLU()"
    architecture.write_text(yaml.safe_dump(document))
    donor = _native_checkpoint(architecture, tmp_path / "relu-donor.pt")
    monkeypatch.setattr(Conv, "default_act", torch.nn.SiLU())
    model = _initialize(
        classifier_api, supcon_source, head_replace="full",
        head_source="yolo_checkpoint", head_checkpoint=donor.path,
    ).eval()
    assert isinstance(model.network.model[0].act, torch.nn.SiLU)
    assert isinstance(model.network.model[-1].conv.act, torch.nn.ReLU)
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        expected = model(images)
    path = tmp_path / "relu-donor-ce.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    assert isinstance(restored.network.model[-1].conv.act, torch.nn.ReLU)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


def test_random_full_head_activation_is_independent_of_encoder_yaml(
    activation_defaults, classifier_api, tiny_architecture, tmp_path, monkeypatch
):
    """A fresh SiLU head on a saved ReLU encoder must round-trip with both behaviors intact."""
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
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        expected = model(images)
    path = tmp_path / "mixed-activation-ce.pt"
    torch.save(model.checkpoint_state(), path)
    previous_default = Conv.default_act
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    assert Conv.default_act is previous_default
    assert isinstance(restored.network.model[0].act, torch.nn.ReLU)
    assert isinstance(restored.network.model[-1].conv.act, torch.nn.SiLU)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "activation",
    [
        torch.nn.SiLU(inplace=True), torch.nn.ReLU(inplace=True),
        torch.nn.ReLU6(inplace=True), torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Hardswish(inplace=True), torch.nn.GELU(approximate="tanh"), torch.nn.Identity(),
    ],
)
def test_supported_head_activation_options_round_trip(classifier_api, supcon_source, tmp_path, activation):
    """Use safe, explicit metadata for known activations and their constructor options."""
    model = _initialize(classifier_api, supcon_source, head_replace="full").eval()
    model.network.model[-1].conv.act = deepcopy(activation)
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        expected = model(images)
    path = tmp_path / "activation-options.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    assert type(restored.network.model[-1].conv.act) is type(activation)
    assert repr(restored.network.model[-1].conv.act) == repr(activation)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


def test_unknown_head_activation_is_rejected_instead_of_pickled(classifier_api, supcon_source):
    """Unsupported module behavior must not silently enter a weights-only checkpoint."""
    model = _initialize(classifier_api, supcon_source)
    model.network.model[-1].conv.act = torch.nn.PReLU()
    with pytest.raises(ValueError, match="activation"):
        model.checkpoint_state()


@pytest.mark.parametrize("component", ["backbone", "head"])
def test_native_batchnorm_options_survive_portable_restoration(
    classifier_api, native_source, tmp_path, component
):
    """BN eps/momentum are behavior, not state_dict tensors, and must not revert to defaults."""
    yolo = YOLO(str(native_source.path), task="classify")
    batchnorm = yolo.model.model[0].bn if component == "backbone" else yolo.model.model[-1].conv.bn
    batchnorm.eps = 0.02
    batchnorm.momentum = 0.3
    path = tmp_path / "nondefault-bn-source.pt"
    yolo.save(path)
    model = _initialize(classifier_api, SimpleNamespace(path=path, source_format="ultralytics")).eval()
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        expected = model(images)
    path = tmp_path / "nondefault-bn-ce.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    actual = restored.network.model[0].bn if component == "backbone" else restored.network.model[-1].conv.bn
    assert actual.eps == 0.02 and actual.momentum == 0.3
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["backbone", "head"])
def test_native_dropout_options_survive_portable_restoration(
    classifier_api, native_source, tmp_path, component
):
    """Training can change Dropout options without updating the architecture YAML."""
    yolo = YOLO(str(native_source.path), task="classify")
    dropout = yolo.model.model[1] if component == "backbone" else yolo.model.model[-1].drop
    dropout.p = 0.27
    dropout.inplace = False
    path = tmp_path / "nondefault-dropout-source.pt"
    yolo.save(path)
    model = _initialize(classifier_api, SimpleNamespace(path=path, source_format="ultralytics"))
    # The official native loader normalizes inplace flags. Check standalone CE
    # restoration against the current model, including later runtime overrides.
    current = model.network.model[1] if component == "backbone" else model.network.model[-1].drop
    current.inplace = False
    path = tmp_path / "nondefault-dropout-ce.pt"
    torch.save(model.checkpoint_state(), path)
    restored = classifier_api.YoloClassifierModel.from_checkpoint(path)
    actual = restored.network.model[1] if component == "backbone" else restored.network.model[-1].drop
    assert actual.p == 0.27 and actual.inplace is False
