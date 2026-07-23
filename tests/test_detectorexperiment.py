import pytest

import smartrodent.detection as detection
from smartrodent.base import DetectorBase
from smartrodent.detection import DetectionExperiment


class ExperimentDummyDetector(DetectorBase):
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        ExperimentDummyDetector.instances.append(self)

    def detect(self, path, out, *args, **kwargs):
        self.detect_call = (path, out)
        return ["result"]

    def write_detections_json(self, results, json_path):
        del results, json_path


@pytest.fixture
def experiment_detector(monkeypatch):
    ExperimentDummyDetector.instances = []
    monkeypatch.setattr(
        detection, "ExperimentDummyDetector", ExperimentDummyDetector, raising=False
    )
    return ExperimentDummyDetector


def make_experiment(tmp_path, **kwargs):
    return DetectionExperiment(
        tmp_path,
        "dataset",
        "experiment",
        0.05,
        "ExperimentDummyDetector",
        **kwargs,
    )


def test_invalid_detector_type_raises():
    with pytest.raises(ValueError, match="Unknown detector type"):
        DetectionExperiment("runs", "data", "exp", 0.1, "DetectorBase")

    with pytest.raises(ValueError, match="Unknown detector type"):
        DetectionExperiment("runs", "data", "exp", 0.1, "MissingDetector")


def test_init_defaults(tmp_path, experiment_detector):
    default_experiment = make_experiment(tmp_path)
    assert default_experiment.detector_args == []
    assert default_experiment.detector_kwargs == {}
    assert default_experiment.batchsize == 1

    experiment = make_experiment(
        tmp_path,
        detector_args=["positional"],
        detector_kwargs={"crop": True, "batchsize": 4, "model_name": "weights.pt"},
    )

    assert experiment.detector_args == ["positional"]
    assert experiment.detector_kwargs == {"crop": True, "model_name": "weights.pt"}
    assert experiment.batchsize == 4


def test_build_detector_passes_shared_configuration(tmp_path, experiment_detector):
    experiment = make_experiment(
        tmp_path,
        detector_args=["positional"],
        detector_kwargs={"crop": True, "batchsize": 4},
    )
    project = tmp_path / "project"

    detector = experiment.build_detector(project)

    assert detector.args == ("positional",)
    assert detector.kwargs == {
        "name": "boxed",
        "batchsize": 4,
        "img_outname": "boxed",
        "conf": 0.05,
        "project": project,
        "crop": True,
    }


def test_run_detector_on_group_creates_output_and_returns_detector_result(
    tmp_path, sample_paths, experiment_detector
):
    experiment = make_experiment(tmp_path, detector_kwargs={"crop": False})

    result = experiment.run_detector_on_group("group-a", sample_paths)

    out = tmp_path / "detect005" / "experiment" / "dataset" / "group-a"
    assert result == ["result"]
    assert out.is_dir()
    assert experiment_detector.instances[-1].detect_call == (sample_paths, out)


def test_conf_to_string(tmp_path, experiment_detector):
    experiment = make_experiment(tmp_path)

    assert experiment.conf_to_string(0.05) == "005"
    assert experiment.conf_to_string(1.0) == "10"


def test_load_experiment_config_supports_yaml_aliases(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
image_dir: images
runs_dir: runs
dataset_name: test-data
batchsize: 2
confs: [0.1]
class_sets:
  rodents: &rodents
    - mouse
    - rat
experiments:
  - name: prompted
    detector: YOLOE_Detector
    kwargs:
      classes: *rodents
""".lstrip()
    )

    config = DetectionExperiment.load_experiment_config(config_path)

    assert config["experiments"][0]["kwargs"]["classes"] == ["mouse", "rat"]
    assert (
        config["experiments"][0]["kwargs"]["classes"] is config["class_sets"]["rodents"]
    )


def test_load_experiment_config_rejects_non_yaml_path(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    with pytest.raises(ValueError, match="must be a YAML file"):
        DetectionExperiment.load_experiment_config(config_path)


@pytest.mark.parametrize("content", ["", "- not\n- a\n- mapping\n"])
def test_load_experiment_config_requires_mapping(tmp_path, content):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)

    with pytest.raises(ValueError, match="top-level mapping"):
        DetectionExperiment.load_experiment_config(config_path)


def test_load_experiment_config_requires_fields(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("experiments: []\n")

    with pytest.raises(ValueError, match="missing required fields"):
        DetectionExperiment.load_experiment_config(config_path)


def test_image_groups_skips_files_empty_dirs_and_sorts(tmp_path, experiment_detector):
    experiment = make_experiment(tmp_path)
    root = tmp_path / "images"
    root.mkdir()
    (root / "file.txt").write_text("skip")
    (root / "empty").mkdir()
    group_b = root / "b"
    group_a = root / "a"
    group_b.mkdir()
    group_a.mkdir()
    (group_b / "2.jpg").write_text("")
    (group_b / "1.jpg").write_text("")
    (group_a / "a.jpg").write_text("")

    groups = list(experiment.image_groups(root))

    assert groups == [
        ("a", [group_a / "a.jpg"]),
        ("b", [group_b / "1.jpg", group_b / "2.jpg"]),
    ]
