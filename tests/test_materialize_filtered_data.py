"""Behavior tests for materializing filtered CSVs and image symlinks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

from scripts.materialize_filtered_data import main, parse_args


def _write_config(input_dir, output_directory=None, deciding_prefix="rejected"):
    """Write an isolated materialization config beside the input directory."""
    config_path = input_dir.parent / "materialize.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_dir": str(input_dir),
                "output_directory": str(output_directory or input_dir.parent),
                "deciding_column_name_start": deciding_prefix,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _prepare(tmp_path, flags=None):
    """Create an input CSV and real images outside the output directories."""
    species_dir = tmp_path / "filtered" / "mouse"
    species_dir.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    paths = [images / f"image_{index}.jpg" for index in range(4)]
    for path in paths:
        path.write_bytes(b"image")
    records = pd.DataFrame(
        {
            "image_path": [str(path) for path in paths],
            "photo_id": [1, 2, 3, 4],
            "rejected_animal": [False, True, False, True],
            "rejected_bird": [False, False, True, True],
            "rejected_snake": [False, False, False, False],
            "kept_animal": [True, True, False, False],
            "kept_bird": [False, True, True, False],
            "kept_snake": [True, False, True, False],
        }
    )
    if flags is not None:
        records["rejected_animal"] = flags
        records = records.drop(columns=["rejected_bird", "rejected_snake"])
    csv_path = species_dir / "records.csv"
    records.to_csv(csv_path, index=False)
    return csv_path, records, paths


def test_partitions_preserve_records_and_link_targets(tmp_path):
    csv_path, records, paths = _prepare(tmp_path)
    main(_write_config(csv_path.parent.parent))
    for partition, indices in (("kept", [0]), ("rejected", [1, 2, 3])):
        directory = tmp_path / partition / "mouse"
        actual = pd.read_csv(directory / csv_path.name)
        pd.testing.assert_frame_equal(
            actual, records.iloc[indices].reset_index(drop=True)
        )
        assert {path.name for path in directory.iterdir()} == {csv_path.name, "imgs"}
        assert {path.name for path in (directory / "imgs").iterdir()} == {
            paths[index].name for index in indices
        }
        for index in indices:
            link = directory / "imgs" / paths[index].name
            assert link.is_symlink()
            assert link.resolve() == paths[index]
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), records)


@pytest.mark.parametrize(
    ("flags", "expected_rejected"),
    [
        (["True", "False", None, " true "], [1, 4]),
        ([False] * 4, []),
        ([True] * 4, [1, 2, 3, 4]),
    ],
)
def test_boolean_text_missing_flags_and_empty_partitions(
    tmp_path, flags, expected_rejected
):
    csv_path, _, _ = _prepare(tmp_path, flags)
    main(_write_config(csv_path.parent.parent))
    rejected = pd.read_csv(tmp_path / "rejected" / "mouse" / "records.csv")
    kept = pd.read_csv(tmp_path / "kept" / "mouse" / "records.csv")
    assert rejected["photo_id"].tolist() == expected_rejected
    assert len(kept) + len(rejected) == 4
    assert kept.columns.tolist() == rejected.columns.tolist()
    for partition in ("kept", "rejected"):
        image_dir = tmp_path / partition / "mouse" / "imgs"
        assert image_dir.is_dir()
        expected_count = len(kept) if partition == "kept" else len(rejected)
        assert len(list(image_dir.iterdir())) == expected_count


def test_no_rejection_columns_keeps_all_rows(tmp_path):
    csv_path, records, _ = _prepare(tmp_path)
    records = records[["image_path", "photo_id"]]
    records.to_csv(csv_path, index=False)
    main(_write_config(csv_path.parent.parent))
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "kept" / "mouse" / "records.csv"), records
    )
    assert pd.read_csv(tmp_path / "rejected" / "mouse" / "records.csv").empty


def test_only_rejected_prefix_columns_control_partition(tmp_path):
    csv_path, records, _ = _prepare(tmp_path)
    records["rejected_animal"] = False
    records["rejected_bird"] = False
    records["rejected_snake"] = [False, False, False, True]
    # Neither kept flags nor the old suffix naming convention reject a row.
    records["animal_rejected"] = True
    records.to_csv(csv_path, index=False)
    main(_write_config(csv_path.parent.parent))
    kept = pd.read_csv(tmp_path / "kept" / "mouse" / "records.csv")
    rejected = pd.read_csv(tmp_path / "rejected" / "mouse" / "records.csv")
    assert kept["photo_id"].tolist() == [1, 2, 3]
    assert rejected["photo_id"].tolist() == [4]


def test_only_records_csv_multiple_species_relative_paths_duplicates_and_reruns(
    tmp_path, monkeypatch
):
    csv_path, records, paths = _prepare(tmp_path)
    monkeypatch.chdir(tmp_path)
    records["image_path"] = [str(path.relative_to(tmp_path)) for path in paths]
    records = pd.concat([records, records.iloc[[0]]], ignore_index=True)
    records.to_csv(csv_path, index=False)
    other_species = csv_path.parent.parent / "rat"
    other_species.mkdir()
    records.to_csv(other_species / "records.csv", index=False)
    # Malformed unrelated CSVs must never be read or materialized.
    (other_species / "other.csv").write_text("unrelated\n", encoding="utf-8")
    (csv_path.parent / "additional.csv").write_text("unrelated\n", encoding="utf-8")
    (csv_path.parent.parent / "notes.txt").write_text("notes", encoding="utf-8")
    for _ in range(2):
        main(_write_config(csv_path.parent.parent))
    for species in ("mouse", "rat"):
        kept = tmp_path / "kept" / species
        assert pd.read_csv(kept / "records.csv")["photo_id"].tolist() == [1, 1]
        assert (kept / "imgs" / paths[0].name).resolve() == paths[0]
        assert {path.name for path in kept.iterdir()} == {"records.csv", "imgs"}
        assert {path.name for path in (kept / "imgs").iterdir()} == {paths[0].name}
        rejected = tmp_path / "rejected" / species
        assert not (rejected / "additional.csv").exists()
        assert not (rejected / "other.csv").exists()


@pytest.mark.parametrize("conflict", ["file", "other_link", "broken_link"])
def test_existing_conflicts_are_not_overwritten(tmp_path, conflict):
    csv_path, _, paths = _prepare(tmp_path)
    destination = tmp_path / "kept" / "mouse" / "imgs" / paths[0].name
    destination.parent.mkdir(parents=True)
    if conflict == "file":
        destination.write_bytes(b"original")
    elif conflict == "other_link":
        destination.symlink_to(paths[1])
    else:
        destination.symlink_to(tmp_path / "missing.jpg")
    with pytest.raises(FileExistsError, match="Conflicting output image"):
        main(_write_config(csv_path.parent.parent))
    if conflict == "file":
        assert destination.read_bytes() == b"original"
    elif conflict == "other_link":
        assert destination.resolve() == paths[1]
    else:
        assert destination.is_symlink()
        assert destination.readlink() == tmp_path / "missing.jpg"


@pytest.mark.parametrize("invalid", ["missing_column", "null_path", "missing_image"])
def test_invalid_image_paths_fail_explicitly(tmp_path, invalid):
    csv_path, records, paths = _prepare(tmp_path)
    if invalid == "missing_column":
        records = records.drop(columns="image_path")
        error = ValueError
    elif invalid == "null_path":
        records.loc[0, "image_path"] = None
        error = ValueError
    else:
        paths[0].unlink()
        error = FileNotFoundError
    records.to_csv(csv_path, index=False)
    with pytest.raises(error):
        main(_write_config(csv_path.parent.parent))


def test_image_named_records_csv_is_separate_from_output_csv(tmp_path):
    csv_path, records, _ = _prepare(tmp_path)
    image = tmp_path / "images" / "records.csv"
    image.write_bytes(b"image")
    records.loc[0, "image_path"] = str(image)
    records.to_csv(csv_path, index=False)
    main(_write_config(csv_path.parent.parent))
    directory = tmp_path / "kept" / "mouse"
    link = directory / "imgs" / "records.csv"
    assert link.is_symlink()
    assert link.resolve() == image
    assert image.read_bytes() == b"image"
    assert not (directory / "records.csv").is_symlink()
    pd.testing.assert_frame_equal(
        pd.read_csv(directory / "records.csv"), records.iloc[[0]].reset_index(drop=True)
    )


def test_parse_args(tmp_path, monkeypatch):
    config_path = tmp_path / "materialize.yaml"
    monkeypatch.setattr("sys.argv", ["materialize_filtered_data.py", str(config_path)])
    assert parse_args().config == config_path


def test_invalid_input_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        main(_write_config(tmp_path / "missing"))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    main(_write_config(empty_dir))
    assert not (tmp_path / "kept").exists()
    assert not (tmp_path / "rejected").exists()


def test_config_relative_input_dir_uses_working_directory(tmp_path, monkeypatch):
    csv_path, _, _ = _prepare(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "materialize.yaml"
    config_path.write_text(
        "input_dir: filtered\noutput_directory: materialized\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    main(config_path)
    assert (tmp_path / "materialized" / "kept" / "mouse" / csv_path.name).is_file()
    assert (
        tmp_path / "materialized" / config_path.name
    ).read_bytes() == config_path.read_bytes()


def test_missing_config_and_input_dir_setting(tmp_path):
    config_path = tmp_path / "materialize.yaml"
    with pytest.raises(FileNotFoundError):
        main(config_path)
    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="input_dir"):
        main(config_path)


def test_explicit_output_directory_and_exact_config_copy(tmp_path):
    csv_path, records, paths = _prepare(tmp_path)
    output = tmp_path / "results" / "materialized"
    config_path = _write_config(csv_path.parent.parent, output)
    # Comments and formatting must survive the snapshot, not just YAML values.
    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write("# configuration used for this run\n")
    original_config = config_path.read_bytes()
    main(config_path)
    assert (output / config_path.name).read_bytes() == original_config
    assert not (output / config_path.name).is_symlink()
    assert not (tmp_path / "kept").exists()
    assert not (tmp_path / "rejected").exists()
    assert (output / "kept" / "mouse" / "imgs" / paths[0].name).is_symlink()
    assert (output / "kept" / "mouse" / "imgs" / paths[0].name).resolve() == paths[0]
    assert pd.read_csv(output / "rejected" / "mouse" / "records.csv")[
        "photo_id"
    ].tolist() == [2, 3, 4]
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), records)
    main(config_path)
    assert (output / config_path.name).read_bytes() == original_config


def test_configured_deciding_prefix_is_preserved(tmp_path):
    csv_path, records, _ = _prepare(tmp_path)
    records["excluded_custom"] = [False, False, False, True]
    records.to_csv(csv_path, index=False)
    main(_write_config(csv_path.parent.parent, deciding_prefix="excluded_"))
    kept = pd.read_csv(tmp_path / "kept" / "mouse" / "records.csv")
    rejected = pd.read_csv(tmp_path / "rejected" / "mouse" / "records.csv")
    assert kept["photo_id"].tolist() == [1, 2, 3]
    assert rejected["photo_id"].tolist() == [4]


@pytest.mark.parametrize("conflict", ["different_config", "directory", "symlink"])
def test_config_snapshot_conflicts_are_not_overwritten(tmp_path, conflict):
    csv_path, _, _ = _prepare(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    config_path = _write_config(csv_path.parent.parent, output)
    snapshot = output / config_path.name
    if conflict == "different_config":
        snapshot.write_bytes(b"previous config")
    elif conflict == "directory":
        snapshot.mkdir()
    else:
        snapshot.symlink_to(tmp_path / "missing_config.yaml")
    with pytest.raises(FileExistsError, match="Conflicting output config"):
        main(config_path)
    assert not (output / "kept").exists()
    if conflict == "different_config":
        assert snapshot.read_bytes() == b"previous config"
    elif conflict == "directory":
        assert snapshot.is_dir()
    else:
        assert snapshot.is_symlink()
        assert not snapshot.exists()


def test_output_directory_is_required(tmp_path):
    csv_path, _, _ = _prepare(tmp_path)
    config_path = tmp_path / "materialize.yaml"
    config_path.write_text(
        yaml.safe_dump({"input_dir": str(csv_path.parent.parent)}), encoding="utf-8"
    )
    with pytest.raises(KeyError, match="output_directory"):
        main(config_path)


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=20))
def test_partition_matches_any_rejected_flag(flags):
    # Each generated case gets its own isolated filesystem.
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        species = root / "filtered" / "mouse"
        species.mkdir(parents=True)
        image = root / "image.jpg"
        image.write_bytes(b"image")
        records = pd.DataFrame(
            {
                "image_path": [str(image)] * len(flags),
                "photo_id": range(len(flags)),
                "rejected_animal": [flag[0] for flag in flags],
                "rejected_bird": [flag[1] for flag in flags],
            }
        )
        records.to_csv(species / "records.csv", index=False)
        main(_write_config(species.parent))
        kept = pd.read_csv(root / "kept" / "mouse" / "records.csv")
        rejected = pd.read_csv(root / "rejected" / "mouse" / "records.csv")
        assert kept["photo_id"].tolist() == [
            i for i, pair in enumerate(flags) if not any(pair)
        ]
        assert rejected["photo_id"].tolist() == [
            i for i, pair in enumerate(flags) if any(pair)
        ]
