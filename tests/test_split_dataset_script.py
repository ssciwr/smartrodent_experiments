"""Configured record discovery through the splitter script's public entry point."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.split_dataset import main


def write_config(root, pattern="kept/*/records_corrected.csv"):
    """Use the supplied splitter template with isolated input and output paths."""
    template = Path(__file__).resolve().parents[1] / "configs" / "dataset_splitter.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["paths"] = {
        "dataset_splitter_input": str(root / "input"),
        "records_glob": pattern,
        "dataset_splitter_output": str(root / "output"),
    }
    config_path = root / "splitter.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def write_records(path):
    """Write image rows with repeated iNaturalist observation identifiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame({"id": [i // 2 for i in range(20)], "photo.id": range(20)})
    records.to_csv(path, index=False)
    return records


def assert_split_preserves_records(path, original):
    """Check preservation and observation-level leakage prevention."""
    result = pd.read_csv(path)
    pd.testing.assert_frame_equal(result.drop(columns="dataset_split"), original)
    assert set(result["dataset_split"]) == {"train", "validation", "test"}
    assert result.groupby("id")["dataset_split"].nunique().eq(1).all()


def test_supplied_config_selects_corrected_kept_records():
    """The project config explicitly targets the corrected stage and its kept CSVs."""
    template = Path(__file__).resolve().parents[1] / "configs" / "dataset_splitter.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))

    assert config["paths"]["dataset_splitter_input"] == (
        "datasets/stage5_materialized_filtered_data_corrected"
    )
    assert config["paths"]["records_glob"] == ("kept/*/records_corrected.csv")
    assert config["data"]["dataset_splitter"]["group_columns"] == ["id"]


def test_reads_only_kept_corrected_records(tmp_path):
    """Neither uncorrected metadata nor rejected species enter the split."""
    config_path = write_config(tmp_path)
    records_path = (
        tmp_path / "input" / "kept" / "Mus musculus" / "records_corrected.csv"
    )
    records = write_records(records_path)
    # Deliberately invalid metadata makes accidental inclusion observable.
    (records_path.parent / "records.csv").write_text("wrong_column\nignored\n")
    rejected_dir = tmp_path / "input" / "rejected" / "Mus musculus"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "records_corrected.csv").write_text("wrong_column\nignored\n")
    write_records(
        tmp_path / "input" / "rejected" / "Rattus rattus" / "records_corrected.csv"
    )

    main(config_path)

    output_root = tmp_path / "output"
    assert_split_preserves_records(
        output_root / "Mus musculus" / "records.csv", records
    )
    assert not (output_root / "Rattus rattus").exists()
    assert (output_root / config_path.name).read_bytes() == config_path.read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(records_path), records)


@pytest.mark.parametrize(
    ("pattern", "relative_path"),
    [
        ("*/records.csv", "Mus musculus/records.csv"),
        ("accepted/*/metadata.csv", "accepted/Mus musculus/metadata.csv"),
        (
            "reviewed/accepted/*/metadata.csv",
            "reviewed/accepted/Mus musculus/metadata.csv",
        ),
    ],
)
def test_supports_configured_layouts(tmp_path, pattern, relative_path):
    """The partition, nesting, and filename can change without editing the script."""
    config_path = write_config(tmp_path, pattern)
    records = write_records(tmp_path / "input" / relative_path)

    main(config_path)

    assert_split_preserves_records(
        tmp_path / "output" / "Mus musculus" / "records.csv", records
    )


@pytest.mark.parametrize(
    "input_state", ["missing", "file", "empty", "no_matching_files"]
)
def test_invalid_input_fails_before_creating_output(tmp_path, input_state):
    """Missing roots and record matches must not silently produce empty outputs."""
    config_path = write_config(tmp_path)
    input_root = tmp_path / "input"
    if input_state == "missing":
        error = FileNotFoundError
    elif input_state == "file":
        input_root.write_text("not a directory")
        error = NotADirectoryError
    elif input_state == "empty":
        input_root.mkdir()
        error = FileNotFoundError
    elif input_state == "no_matching_files":
        write_records(input_root / "kept" / "Mus musculus" / "records.csv")
        error = FileNotFoundError
    else:
        raise AssertionError(f"Unknown test input state: {input_state}")

    with pytest.raises(error, match="input|directory|records"):
        main(config_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("existing_output", [False, True])
def test_duplicate_species_matches_fail_before_writing(tmp_path, existing_output):
    """A broad pattern must not overwrite one species' records with another CSV."""
    config_path = write_config(tmp_path, "*/*/records_corrected.csv")
    write_records(
        tmp_path / "input" / "kept" / "Mus musculus" / "records_corrected.csv"
    )
    write_records(
        tmp_path / "input" / "rejected" / "Mus musculus" / "records_corrected.csv"
    )

    output_root = tmp_path / "output"
    if existing_output:
        output_root.mkdir()
        (output_root / config_path.name).write_text("previous run configuration")

    with pytest.raises(ValueError, match="Multiple records files.*Mus musculus"):
        main(config_path)
    if existing_output:
        assert list(output_root.iterdir()) == [output_root / config_path.name]
        assert (
            output_root / config_path.name
        ).read_text() == "previous run configuration"
    else:
        assert not output_root.exists()


@pytest.mark.parametrize(
    "pattern", [None, 42, "", "   ", "/absolute/*.csv", "../*.csv"]
)
def test_invalid_globs_fail_before_writing(tmp_path, pattern):
    """The record pattern must be a nonempty glob relative to the input root."""
    config_path = write_config(tmp_path, pattern)
    (tmp_path / "input").mkdir()

    with pytest.raises(ValueError, match="records_glob"):
        main(config_path)
    assert not (tmp_path / "output").exists()


def test_glob_is_required_without_a_silent_default(tmp_path):
    """Omitting discovery configuration must not switch to uncorrected records."""
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["paths"]["records_glob"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_records(tmp_path / "input" / "Mus musculus" / "records.csv")

    with pytest.raises(KeyError, match="records_glob"):
        main(config_path)
    assert not (tmp_path / "output").exists()


def test_glob_matching_a_directory_fails_before_writing(tmp_path):
    """A metadata glob selects files, not directories that happen to match."""
    config_path = write_config(tmp_path)
    (tmp_path / "input" / "kept" / "Mus musculus" / "records_corrected.csv").mkdir(
        parents=True
    )

    with pytest.raises(ValueError, match="not a file"):
        main(config_path)
    assert not (tmp_path / "output").exists()


@settings(max_examples=20)
@given(species=st.from_regex(r"[A-Z][a-z]{1,12} [a-z]{1,12}", fullmatch=True))
def test_glob_preserves_generated_species_directory_names(species):
    """Spaces and varying species names are preserved in the output layout."""
    with TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        config_path = write_config(root)
        records = write_records(
            root / "input" / "kept" / species / "records_corrected.csv"
        )

        main(config_path)

        assert_split_preserves_records(
            root / "output" / species / "records.csv", records
        )
