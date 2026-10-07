"""Behavior tests for reconciling human-sorted kept/rejected images."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

from scripts.reconcile_filtered_data import main, parse_args


def _prepare(root, naming="mixed", species_names=("mouse",)):
    """Create original records, human partitions, and both YAML configurations."""
    original_root = root / "original"
    corrected_root = root / "corrected"
    images_root = root / "images"
    images_root.mkdir()
    records_by_species = {}
    for species in species_names:
        source_dir = original_root / species
        source_dir.mkdir(parents=True)
        for partition in ("kept", "rejected"):
            (corrected_root / partition / species).mkdir(parents=True)
        image_paths = []
        for index in range(4):
            image = images_root / f"{species}_{index}.jpg"
            image.write_bytes(b"image")
            image_paths.append(str(image))
            partition = "kept" if index in (0, 1) else "rejected"
            (corrected_root / partition / species / image.name).symlink_to(image)
        columns = {
            "image_path": image_paths,
            "photo_id": [1, 2, 3, 4],
            "rejected_animal": [False, True, False, False],
            "bird_rejected": [False, False, True, False],
            "kept_animal": [True, False, True, True],
            "bird_kept": [True, True, False, True],
        }
        records = pd.DataFrame(columns)
        if naming == "prefix":
            records = records.rename(
                columns={"bird_rejected": "rejected_bird", "bird_kept": "kept_bird"}
            )
        elif naming == "suffix":
            records = records.rename(
                columns={
                    "rejected_animal": "animal_rejected",
                    "kept_animal": "animal_kept",
                }
            )
        else:
            assert naming == "mixed"
        # Duplicate metadata rows for one image must receive the same correction.
        records = pd.concat([records, records.iloc[[1]]], ignore_index=True)
        records.to_csv(source_dir / "records.csv", index=False)
        records_by_species[species] = records
        # Stale materialized metadata and backups must not be used as inputs.
        (corrected_root / "kept" / species / "records.csv").write_text(
            "stale\n", encoding="utf-8"
        )
        (source_dir / "backup.csv").write_text("unrelated\n", encoding="utf-8")
    (original_root / "notes.txt").write_text("not a species", encoding="utf-8")
    materialization_config = corrected_root / "materialize_filtered_data.yaml"
    materialization_config.write_text(
        yaml.safe_dump(
            {"input_dir": str(original_root), "output_directory": "old_output"}
        ),
        encoding="utf-8",
    )
    config_path = root / "reconcile.yaml"
    config_path.write_text(
        "# human correction input\n"
        + yaml.safe_dump({"input_directory": str(corrected_root)}),
        encoding="utf-8",
    )
    return config_path, original_root, corrected_root, records_by_species


@pytest.mark.parametrize("naming", ["prefix", "suffix", "mixed"])
def test_corrections_preserve_originals_images_and_configs(
    tmp_path, monkeypatch, naming
):
    config_path, original_root, corrected_root, originals = _prepare(
        tmp_path, naming, ("mouse", "rat")
    )
    clock = Mock()
    clock.now.side_effect = [
        datetime(2026, 7, 8, 14, 30, 25),
        datetime(2026, 7, 8, 14, 30, 26),
    ]
    monkeypatch.setattr("scripts.reconcile_filtered_data.datetime", clock)
    before_links = {path: path.readlink() for path in corrected_root.glob("*/*/*.jpg")}
    main(config_path)
    for species, original in originals.items():
        output_dir = corrected_root / species
        assert {path.name for path in output_dir.iterdir()} == {
            "records_uncorrected.csv",
            "records_corrected_143025_08072026.csv",
        }
        corrected = pd.read_csv(output_dir / "records_corrected_143025_08072026.csv")
        assert corrected["human_corrected_kept"].tolist() == [
            False,
            True,
            False,
            False,
            True,
        ]
        assert corrected["human_corrected_rejected"].tolist() == [
            False,
            False,
            False,
            True,
            False,
        ]
        pd.testing.assert_frame_equal(
            corrected.drop(
                columns=["human_corrected_kept", "human_corrected_rejected"]
            ),
            original,
        )
        source_path = original_root / species / "records.csv"
        assert (
            output_dir / "records_uncorrected.csv"
        ).read_bytes() == source_path.read_bytes()
        pd.testing.assert_frame_equal(pd.read_csv(source_path), original)
    assert (corrected_root / config_path.name).read_bytes() == config_path.read_bytes()
    assert {
        path: path.readlink() for path in corrected_root.glob("*/*/*.jpg")
    } == before_links


@pytest.mark.parametrize("inconsistency", ["both", "neither", "broken_link"])
def test_ambiguous_or_unmatched_images_fail_before_species_outputs(
    tmp_path, inconsistency
):
    config_path, _, corrected_root, _ = _prepare(tmp_path)
    kept = corrected_root / "kept" / "mouse"
    rejected = corrected_root / "rejected" / "mouse"
    image = kept / "mouse_0.jpg"
    if inconsistency == "both":
        (rejected / image.name).symlink_to(image.resolve())
        message = "both=.*mouse_0.jpg"
    elif inconsistency == "neither":
        image.unlink()
        message = "neither=.*mouse_0.jpg"
    else:
        image.unlink()
        image.symlink_to(tmp_path / "missing.jpg")
        message = "neither=.*mouse_0.jpg"
    with pytest.raises(ValueError, match=message):
        main(config_path)
    assert not (corrected_root / "mouse").exists()


def test_all_kept_requires_every_kept_flag_and_rejected_requires_any(tmp_path):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    source = original_root / "mouse" / "records.csv"
    records = pd.read_csv(source)
    records.loc[3, "bird_kept"] = False
    records.loc[1, "rejected_animal"] = False
    records.loc[4, "rejected_animal"] = False
    records.to_csv(source, index=False)
    main(config_path)
    corrected = pd.read_csv(
        next((corrected_root / "mouse").glob("records_corrected_*.csv"))
    )
    assert not corrected["human_corrected_kept"].any()
    assert not corrected["human_corrected_rejected"].any()


@pytest.mark.parametrize(
    "invalid", ["image_column", "null_image", "rejected_columns", "kept_columns"]
)
def test_invalid_records_or_missing_flag_families_fail(tmp_path, invalid):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    source = original_root / "mouse" / "records.csv"
    records = pd.read_csv(source)
    if invalid == "image_column":
        records = records.drop(columns="image_path")
        error = KeyError
    elif invalid == "null_image":
        records.loc[0, "image_path"] = None
        error = TypeError
    elif invalid == "rejected_columns":
        records = records.drop(columns=["rejected_animal", "bird_rejected"])
        error = ValueError
    else:
        records = records.drop(columns=["kept_animal", "bird_kept"])
        error = ValueError
    records.to_csv(source, index=False)
    with pytest.raises(error):
        main(config_path)
    assert not (corrected_root / "mouse").exists()


def test_text_flags_and_previous_correction_columns(tmp_path):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    source = original_root / "mouse" / "records.csv"
    records = pd.read_csv(source)
    for column in ("rejected_animal", "bird_rejected", "kept_animal", "bird_kept"):
        records[column] = records[column].map({True: " TRUE ", False: " false "})
    records["human_corrected_kept"] = False
    records["human_corrected_rejected"] = True
    records.to_csv(source, index=False)
    main(config_path)
    corrected = pd.read_csv(
        next((corrected_root / "mouse").glob("records_corrected_*.csv"))
    )
    assert corrected["human_corrected_kept"].tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]
    assert corrected["human_corrected_rejected"].tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize("flag", [None, "maybe"])
def test_only_explicit_true_flags_count(tmp_path, flag):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    source = original_root / "mouse" / "records.csv"
    records = pd.read_csv(source)
    records["kept_animal"] = records["kept_animal"].astype("object")
    records["rejected_animal"] = records["rejected_animal"].astype("object")
    records.loc[3, "kept_animal"] = flag
    records.loc[[1, 4], "rejected_animal"] = flag
    records.to_csv(source, index=False)
    main(config_path)
    corrected = pd.read_csv(
        next((corrected_root / "mouse").glob("records_corrected_*.csv"))
    )
    assert not corrected["human_corrected_kept"].any()
    assert not corrected["human_corrected_rejected"].any()


def test_images_do_not_require_a_supported_extension(tmp_path):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    source = original_root / "mouse" / "records.csv"
    records = pd.read_csv(source)
    records.loc[0, "image_path"] = str(tmp_path / "extensionless_image")
    records.to_csv(source, index=False)
    kept = corrected_root / "kept" / "mouse"
    (kept / "mouse_0.jpg").rename(kept / "extensionless_image")
    main(config_path)
    assert (corrected_root / "mouse" / "records_uncorrected.csv").is_file()


def test_regular_images_and_absent_empty_species_partition(tmp_path):
    config_path, _, corrected_root, _ = _prepare(tmp_path)
    kept = corrected_root / "kept" / "mouse"
    rejected = corrected_root / "rejected" / "mouse"
    for path in list(rejected.iterdir()):
        path.rename(kept / path.name)
    rejected.rmdir()
    image = kept / "mouse_0.jpg"
    contents = image.read_bytes()
    image.unlink()
    image.write_bytes(contents)
    main(config_path)
    corrected = pd.read_csv(
        next((corrected_root / "mouse").glob("records_corrected_*.csv"))
    )
    assert not corrected["human_corrected_rejected"].any()


def test_unrelated_files_and_species_are_ignored(tmp_path):
    config_path, _, corrected_root, _ = _prepare(tmp_path)
    (corrected_root / "kept" / "unknown_species").mkdir()
    (corrected_root / "kept" / "mouse" / "extra.jpg").write_bytes(b"unrelated")
    main(config_path)
    assert (corrected_root / "mouse" / "records_uncorrected.csv").is_file()
    assert not (corrected_root / "unknown_species").exists()


@pytest.mark.parametrize(
    "missing",
    ["corrected", "partition", "materialization_config", "original_root", "records"],
)
def test_missing_inputs_fail(tmp_path, missing):
    config_path, original_root, corrected_root, _ = _prepare(tmp_path)
    if missing == "corrected":
        config_path.write_text(
            yaml.safe_dump({"input_directory": str(tmp_path / "missing")}),
            encoding="utf-8",
        )
        error = FileNotFoundError
    elif missing == "partition":
        (corrected_root / "kept").rename(corrected_root / "removed_kept")
        error = ValueError
    elif missing == "materialization_config":
        (corrected_root / "materialize_filtered_data.yaml").unlink()
        error = FileNotFoundError
    elif missing == "original_root":
        original_root.rename(tmp_path / "removed_original")
        error = FileNotFoundError
    else:
        (original_root / "mouse" / "records.csv").unlink()
        error = FileNotFoundError
    with pytest.raises(error):
        main(config_path)


def test_relative_paths_and_config_already_in_output(tmp_path, monkeypatch):
    config_path, _, corrected_root, _ = _prepare(tmp_path)
    config_path = corrected_root / "reconcile.yaml"
    config_path.write_text("input_directory: corrected\n", encoding="utf-8")
    (corrected_root / "materialize_filtered_data.yaml").write_text(
        "input_dir: original\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    main(config_path)
    assert (corrected_root / "mouse" / "records_uncorrected.csv").is_file()


def test_same_second_rerun_does_not_overwrite_corrected_records(tmp_path, monkeypatch):
    config_path, _, corrected_root, _ = _prepare(tmp_path)
    clock = Mock()
    clock.now.return_value = datetime(2026, 7, 8, 14, 30, 25)
    monkeypatch.setattr("scripts.reconcile_filtered_data.datetime", clock)
    main(config_path)
    output = corrected_root / "mouse" / "records_corrected_143025_08072026.csv"
    original_contents = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(config_path)
    assert output.read_bytes() == original_contents


@pytest.mark.parametrize("arguments", [["settings.yaml"], []])
def test_parse_args(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["reconcile_filtered_data.py", *arguments])
    if arguments:
        assert parse_args().config == Path("settings.yaml")
    else:
        with pytest.raises(SystemExit) as error:
            parse_args()
        assert error.value.code == 2


@given(st.lists(st.tuples(*[st.booleans() for _ in range(5)]), min_size=1, max_size=20))
def test_generated_flags_match_exact_correction_conditions(cases):
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / "original" / "mouse"
        original.mkdir(parents=True)
        corrected = root / "corrected"
        for partition in ("kept", "rejected"):
            (corrected / partition / "mouse").mkdir(parents=True)
        rows = []
        for index, (rejected_a, rejected_b, kept_a, kept_b, human_kept) in enumerate(
            cases
        ):
            name = f"image_{index}.jpg"
            partition = "kept" if human_kept else "rejected"
            (corrected / partition / "mouse" / name).write_bytes(b"image")
            rows.append(
                {
                    "image_path": str(root / name),
                    "rejected_animal": rejected_a,
                    "bird_rejected": rejected_b,
                    "kept_animal": kept_a,
                    "bird_kept": kept_b,
                }
            )
        pd.DataFrame(rows).to_csv(original / "records.csv", index=False)
        (corrected / "materialize_filtered_data.yaml").write_text(
            yaml.safe_dump({"input_dir": str(original.parent)}), encoding="utf-8"
        )
        config_path = root / "reconcile.yaml"
        config_path.write_text(
            yaml.safe_dump({"input_directory": str(corrected)}), encoding="utf-8"
        )
        main(config_path)
        records = pd.read_csv(
            next((corrected / "mouse").glob("records_corrected_*.csv"))
        )
        assert records["human_corrected_kept"].tolist() == [
            (a or b) and human for a, b, _, _, human in cases
        ]
        assert records["human_corrected_rejected"].tolist() == [
            a and b and not human for _, _, a, b, human in cases
        ]
