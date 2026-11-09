"""Verify the model-testing script's orchestration with a stubbed tester."""

from unittest.mock import Mock

import scripts.evaluate_yolo_model as evaluate_script


def test_script_copies_config_before_testing(tmp_path, monkeypatch):
    config_path = tmp_path / "test_yolo_classifier_config.yaml"
    config_path.write_text("# original testing configuration\nmodel_path: best.pt\n")
    output_path = tmp_path / "results"
    output_path.mkdir()

    def run_test():
        assert (output_path / config_path.name).read_bytes() == config_path.read_bytes()

    tester = Mock(output_path=output_path, metrics_file_name="metrics.json")
    tester.test.side_effect = run_test
    factory = Mock()
    factory.from_config.return_value = tester
    monkeypatch.setattr(evaluate_script, "YoloTester", factory)

    evaluate_script.main(config_path)

    factory.from_config.assert_called_once_with(config_path)
    tester.test.assert_called_once()
