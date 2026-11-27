"""The classifier head is always new: ``linear`` or ``full``.

``linear`` puts only a new final layer on top of the encoder's features. ``full``
also renews everything after the pooling, which an encoder offers through the
optional ``reinitialize_post_pooling`` method; for YOLO that is the convolution of
the ``Classify`` block. Everything before it keeps the checkpoint's weights.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import yaml
from classifier_task_test_support import native_checkpoint, sgd, train_supcon
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from smartrodent.classifier_task import ClassifierTask

ARCHITECTURE = {
    "nc": 2,
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "nn.Dropout", [0.6]]],
    "head": [[-1, 1, "Classify", [2]]],
}
LABELS = {"mouse": 0, "rat": 1}


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _spec(**parameters) -> dict:
    """Describe the YOLO encoder as a component spec."""
    return {"name": "YoloEncoder", "parameters": parameters}


def _task(encoder, *, labels=LABELS, **options) -> ClassifierTask:
    """Build the production task with the head option under test."""
    return ClassifierTask(encoder, label_names=labels, optimizer=sgd(), **options)


def _state(module: nn.Module) -> dict:
    """Snapshot a module's state."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _without_block(state: dict) -> dict:
    """Drop the part of a YOLO encoder state that ``full`` renews."""
    last = max(int(k.split(".")[2]) for k in state if k.startswith("network.model."))
    prefix = f"network.model.{last}.conv."
    return {k: v for k, v in state.items() if not k.startswith(prefix)}


def _assert_same(actual: dict, expected: dict) -> None:
    """Require exact preservation, not approximate similarity."""
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


@pytest.fixture
def native_initializations(monkeypatch):
    """Observe native reset results so bounded-but-zero or custom initialization fails."""
    snapshots = {}
    originals = {layer: layer.reset_parameters for layer in (nn.Linear, nn.Conv2d)}

    def reset(layer):
        originals[type(layer)](layer)
        snapshots[id(layer)] = deepcopy(layer.state_dict())

    for layer in originals:
        monkeypatch.setattr(layer, "reset_parameters", reset)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        yield snapshots


@pytest.fixture(params=["supcon", "ultralytics"])
def encoder_source(request, tmp_path):
    """The same contracts for an encoder from a SupCon ``.ckpt`` and from a native file."""
    from smartrodent.encoders import YoloEncoder
    from smartrodent.supcon_task import SupConTask

    if request.param == "supcon":
        _, result = train_supcon(tmp_path, encoder=_spec(architecture=ARCHITECTURE))
        encoder = SupConTask.load_from_checkpoint(
            result.best_checkpoint, map_location="cpu"
        ).encoder
        return SimpleNamespace(
            spec={"checkpoint": str(result.best_checkpoint)}, encoder=encoder
        )
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(yaml.safe_dump(ARCHITECTURE))
    donor = native_checkpoint(architecture, tmp_path / "native-cls.pt")
    return SimpleNamespace(
        spec=_spec(model_name=str(donor.path)),
        encoder=YoloEncoder(model_name=str(donor.path)),
    )


@pytest.mark.parametrize("classes", [("rat", "mouse"), ("shrew", "rat", "mouse")])
def test_linear_head_is_a_fresh_layer_on_an_untouched_encoder(
    encoder_source, classes, native_initializations
):
    """The encoder keeps every weight; only the final layer is new, whatever the count."""
    expected = _state(encoder_source.encoder)
    labels = {name: index for index, name in enumerate(sorted(classes))}
    task = _task(encoder_source.spec, labels=labels, head="linear")

    _assert_same(_state(task.encoder), expected)
    assert id(task.head) in native_initializations
    _assert_same(task.head.state_dict(), native_initializations[id(task.head)])
    assert torch.count_nonzero(task.head.weight)
    assert task.head.in_features == encoder_source.encoder.feature_dim
    assert task.head.out_features == len(classes)
    assert task.label_names == labels
    assert all("projector" not in name for name in task.state_dict())


def test_full_head_renews_only_the_post_pooling_convolution(
    encoder_source, native_initializations
):
    """Everything before the pooling keeps the checkpoint; the Classify conv is new."""
    expected = _state(encoder_source.encoder)
    task = _task(encoder_source.spec, head="full")

    _assert_same(_without_block(_state(task.encoder)), _without_block(expected))
    conv = task.encoder.network.model[-1].conv
    original = encoder_source.encoder.network.model[-1].conv
    assert not torch.equal(conv.conv.weight, original.conv.weight)
    assert torch.equal(conv.bn.running_mean, torch.zeros_like(conv.bn.running_mean))
    assert torch.equal(conv.bn.running_var, torch.ones_like(conv.bn.running_var))
    assert torch.equal(conv.bn.weight, torch.ones_like(conv.bn.weight))
    assert torch.equal(conv.bn.bias, torch.zeros_like(conv.bn.bias))
    assert conv.bn.eps == 0.001 and conv.bn.momentum == 0.03
    for layer in (conv.conv, task.head):
        assert id(layer) in native_initializations
        _assert_same(layer.state_dict(), native_initializations[id(layer)])
        assert torch.count_nonzero(layer.weight)
        bound = layer.weight[0].numel() ** -0.5
        assert layer.weight.abs().max() <= bound
    assert task.head.out_features == 2


@settings(max_examples=15, deadline=None)
@given(
    class_names=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=2,
        max_size=5,
        unique=True,
    ),
    head=st.sampled_from(["linear", "full"]),
)
def test_head_width_follows_the_supplied_label_mapping(class_names, head):
    """The task keeps supplied IDs instead of sorting them, and sizes the head to them."""
    expected = {name: index for index, name in enumerate(class_names)}
    task = _task(_spec(architecture=ARCHITECTURE), labels=expected, head=head)
    assert task.label_names == expected
    assert task.head.out_features == len(class_names)


class _TwoPart(nn.Module):
    """A custom encoder with a body and a post-pooling part that can renew itself."""

    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 4)
        self.post = nn.Linear(4, 4)

    def reinitialize_post_pooling(self) -> None:
        """Renew only the part after the body."""
        self.post.reset_parameters()

    def forward(self, images):
        """Return one feature vector per image."""
        return self.post(self.body(images.mean(dim=(2, 3))))


class _Plain(nn.Module):
    """A custom encoder that offers no post-pooling interface."""

    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 4)

    def forward(self, images):
        """Return one feature vector per image."""
        return self.body(images.mean(dim=(2, 3)))


def test_full_head_works_through_the_encoder_interface_for_custom_models():
    """With ``full`` the encoder renews exactly its post-pooling part."""
    spec = {"name": f"{__name__}:_TwoPart"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        linear = _task(spec, head="linear")
        torch.manual_seed(1)
        full = _task(spec, head="full")

    before, after = _state(linear.encoder), _state(full.encoder)
    assert torch.equal(before["body.weight"], after["body.weight"])
    assert not torch.equal(before["post.weight"], after["post.weight"])
    assert full.head.out_features == 2


def test_full_head_is_rejected_for_an_encoder_without_the_interface():
    """The capability is checked at construction, before any training."""
    with pytest.raises(ValueError, match="post-pooling"):
        _task({"name": f"{__name__}:_Plain"}, head="full")


def test_unknown_head_is_rejected():
    """Only ``linear`` and ``full`` exist."""
    with pytest.raises(ValueError, match="Unsupported head"):
        _task({"name": f"{__name__}:_Plain"}, head="partial")


@pytest.mark.parametrize("head", ["linear", "full"])
def test_native_export_carries_the_new_head(head):
    """The native model reproduces the Lightning probabilities, whichever head."""
    task = _task(
        _spec(architecture=ARCHITECTURE),
        labels={"a": 0, "b": 1, "c": 2},
        head=head,
        native_checkpoint=True,
    ).eval()
    native = task.native_model()
    images = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        output = native(images)
        probabilities = output[0] if isinstance(output, (tuple, list)) else output
        expected = task(images).softmax(dim=1)
    torch.testing.assert_close(probabilities, expected, atol=1e-5, rtol=1e-4)
    assert native.names == {0: "a", 1: "b", 2: "c"}


def test_native_best_pt_loads_with_yolo_after_a_full_head_run(tmp_path):
    """StageRunner writes a ``best.pt`` that ``YOLO(path)`` loads, with the renewed block."""
    from classifier_task_test_support import fit, runner
    from ultralytics import YOLO

    task = _task(
        _spec(architecture=ARCHITECTURE),
        labels={"mouse": 0, "rat": 1, "shrew": 2},
        head="full",
        native_checkpoint=True,
    )
    stage, artifacts = runner(tmp_path / "run", epochs=1, native=True)
    fit(stage, task)

    native = YOLO(str(artifacts.native_checkpoint)).model.eval()
    assert list(native.names.values()) == ["mouse", "rat", "shrew"]
    images = torch.rand(2, 3, 32, 32)
    best = ClassifierTask.load_from_checkpoint(
        artifacts.best_checkpoint, map_location="cpu"
    ).eval()
    with torch.inference_mode():
        output = native(images)
        probabilities = output[0] if isinstance(output, (tuple, list)) else output
        torch.testing.assert_close(
            probabilities, best(images).softmax(dim=1), atol=1e-4, rtol=1e-3
        )


@pytest.fixture
def activation_defaults(monkeypatch):
    """Isolate Ultralytics' process-global default activation."""
    from ultralytics.nn.modules import Conv

    monkeypatch.setattr(Conv, "default_act", nn.SiLU())


def test_renewed_block_keeps_the_activation_of_the_new_head_not_the_encoder_yaml(
    activation_defaults, tmp_path, monkeypatch
):
    """A fresh SiLU block on a saved ReLU encoder keeps both activations."""
    from ultralytics.nn.modules import Conv

    document = deepcopy(ARCHITECTURE)
    document["activation"] = "nn.ReLU()"
    relu = tmp_path / "relu-encoder.yaml"
    relu.write_text(yaml.safe_dump(document))
    source = native_checkpoint(relu, tmp_path / "relu-encoder.pt")
    monkeypatch.setattr(Conv, "default_act", nn.SiLU())

    task = _task(_spec(model_name=str(source.path)), head="full").eval()

    assert isinstance(task.encoder.network.model[0].act, nn.ReLU)
    assert isinstance(task.encoder.network.model[-1].conv.act, nn.SiLU)


@pytest.mark.parametrize(
    "head_layer,message",
    [([-1, 1, "Conv", [4, 1]], "Classify"), ([-1, 1, "Classify", []], "arguments")],
)
def test_native_export_rejects_an_architecture_without_a_usable_classify_head(
    head_layer, message
):
    """A malformed architecture fails clearly instead of with an IndexError."""
    task = _task(_spec(architecture=ARCHITECTURE), native_checkpoint=True)
    task.encoder.network.yaml["head"][-1] = head_layer
    with pytest.raises(ValueError, match=message):
        task.native_model()
