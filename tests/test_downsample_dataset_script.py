"""Per-species downsampling through the script's entry point."""

import pandas as pd
import yaml

from scripts.downsample_dataset import main


def write_records(root, species, n):
    path = root / "input" / species / "records.csv"
    path.parent.mkdir(parents=True)
    records = pd.DataFrame({"id": range(n), "image_path": [f"{i}.jpg" for i in range(n)]})
    records.to_csv(path, index=False)
    return records


def test_caps_large_classes_and_keeps_small_ones(tmp_path):
    write_records(tmp_path, "Rattus rattus", 50)
    small = write_records(tmp_path, "Mus musculus", 5)
    config_path = tmp_path / "downsample.yaml"
    config_path.write_text(yaml.safe_dump({
        "input_dir": str(tmp_path / "input"),
        "output_directory": str(tmp_path / "output"),
        "max_images_per_class": 10,
    }))

    main(config_path)

    out = tmp_path / "output"
    sampled = pd.read_csv(out / "Rattus rattus" / "records.csv")
    assert len(sampled) == 10 and sampled["id"].is_monotonic_increasing
    pd.testing.assert_frame_equal(pd.read_csv(out / "Mus musculus" / "records.csv"), small)
    assert (out / config_path.name).read_bytes() == config_path.read_bytes()
