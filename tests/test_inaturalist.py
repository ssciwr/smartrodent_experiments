from pathlib import Path

import pandas as pd
import pytest
import yaml

from smartrodent.inaturalist import InaturalistDataset


@pytest.fixture
def inat_config(tmp_path):
    """Return a minimal, complete iNaturalist configuration mapping."""
    return {
        "inaturalist": {
            "output_path": str(tmp_path / "dataset"),
            "species": ["Mus musculus"],
            "years": [],
            "first_year": 2020,
            "last_year": 2022,
            "quality_grade": "needs_id",
            "seed": 7,
            "max_img_num": 5,
            "allowed_licences": ["cc-by-nc", "cc0"],
        }
    }


@pytest.fixture
def config_file(tmp_path, inat_config):
    """Write ``inat_config`` to disk and return its path."""
    path = tmp_path / "dataset_test.yaml"
    path.write_text(yaml.safe_dump(inat_config), encoding="utf-8")
    return path


def test_inaturalist_constructor(tmp_path):
    output_path = tmp_path / "test"
    dataset = InaturalistDataset(
        output_path=output_path,
        species=["Mus musculus"],
        years=[2022],
        quality_grade="research",
    )

    assert dataset.output_path == output_path.resolve()
    assert dataset.species == ["Mus musculus"]
    assert dataset.years == [2022]
    assert dataset.quality_grade == "research"
    assert dataset.seed == 42
    assert dataset.max_img_num == 2000
    assert dataset.allowed_licenses == {"cc-by-nc"}
    assert dataset.config_path is None
    assert output_path.is_dir()
    assert list(output_path.iterdir()) == []


def test_inaturalist_constructor_year_range(tmp_path):
    dataset = InaturalistDataset(
        output_path=tmp_path / "test",
        species=["Mus musculus"],
        first_year=2020,
        last_year=2023,
    )

    assert dataset.years == [2020, 2021, 2022, 2023]


def test_inaturalist_constructor_copies_config(tmp_path, config_file):
    output_path = tmp_path / "test"
    dataset = InaturalistDataset(
        output_path=output_path,
        species=["Mus musculus"],
        years=[2022],
        config_path=config_file,
    )

    copied = output_path / config_file.name
    assert dataset.config_path == config_file.resolve()
    assert copied.read_text(encoding="utf-8") == config_file.read_text(encoding="utf-8")


def test_inaturalist_constructor_keeps_config_in_output_path(tmp_path, inat_config):
    output_path = tmp_path / "test"
    output_path.mkdir()
    config_path = output_path / "dataset_test.yaml"
    config_path.write_text(yaml.safe_dump(inat_config), encoding="utf-8")

    InaturalistDataset(
        output_path=output_path,
        species=["Mus musculus"],
        years=[2022],
        config_path=config_path,
    )

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == inat_config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "either years or first_year and last_year"),
        ({"first_year": 2020}, "either years or first_year and last_year"),
        ({"last_year": 2020}, "either years or first_year and last_year"),
        (
            {"years": [2022], "first_year": 2020, "last_year": 2023},
            "mutually exclusive",
        ),
        ({"first_year": 2023, "last_year": 2020}, "less than or equal"),
    ],
)
def test_inaturalist_constructor_rejects_invalid_years(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        InaturalistDataset(
            output_path=tmp_path / "test", species=["Mus musculus"], **kwargs
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"species": [], "years": [2022]}, "at least one scientific name"),
        (
            {"species": ["Mus musculus"], "years": [2022], "max_img_num": -1},
            "max_img_num",
        ),
    ],
)
def test_inaturalist_constructor_rejects_invalid_values(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        InaturalistDataset(output_path=tmp_path / "test", **kwargs)


def test_inaturalist_from_config(config_file, inat_config):
    dataset = InaturalistDataset.from_config(config_file)
    settings = inat_config["inaturalist"]

    assert dataset.output_path == Path(settings["output_path"]).resolve()
    assert dataset.species == settings["species"]
    assert dataset.years == [2020, 2021, 2022]
    assert dataset.quality_grade == "needs_id"
    assert dataset.seed == 7
    assert dataset.max_img_num == 5
    assert dataset.allowed_licenses == {"cc-by-nc", "cc0"}
    assert (dataset.output_path / config_file.name).is_file()


def test_inaturalist_from_config_uses_defaults(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inaturalist": {
                    "output_path": str(tmp_path / "dataset"),
                    "species": ["Mus musculus"],
                    "years": [2022],
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = InaturalistDataset.from_config(path)

    assert dataset.quality_grade == "research"
    assert dataset.seed == 42
    assert dataset.max_img_num == 2000
    assert dataset.allowed_licenses == {"cc-by-nc"}


@pytest.mark.parametrize("missing", ["output_path", "species"])
def test_inaturalist_from_config_requires_keys(tmp_path, inat_config, missing):
    settings = dict(inat_config["inaturalist"])
    del settings[missing]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump({"inaturalist": settings}), encoding="utf-8")

    with pytest.raises(ValueError, match=missing):
        InaturalistDataset.from_config(path)


@pytest.mark.parametrize("content", ["", "- just\n- a\n- list\n", "inaturalist: 42\n"])
def test_inaturalist_from_config_requires_mapping(tmp_path, content):
    path = tmp_path / "broken.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(TypeError):
        InaturalistDataset.from_config(path)


@pytest.mark.network
def test_inaturalist_downloads_two_real_images(tmp_path):
    """Download a small, real iNaturalist sample without an uncapped request."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        quality_grade="research",
        max_img_num=2,
    )

    dataset.download()

    species_path = dataset.output_path / "Mus musculus"
    records = pd.read_csv(species_path / "records.csv")
    images = list((species_path / "imgs").iterdir())
    assert not records.empty
    assert len(images) == 2
    assert all(image.stat().st_size > 0 for image in images)
