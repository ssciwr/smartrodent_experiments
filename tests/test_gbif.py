from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

import smartrodent.gbif as gbif_module
from smartrodent.gbif import GbifDataset


def make_dataset(tmp_path, **kwargs):
    return GbifDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        **kwargs,
    )


def test_from_config_loads_data_gbif_settings(tmp_path):
    config = {
        "data": {
            "gbif": {
                "output_path": str(tmp_path / "dataset"),
                "species": ["Mus musculus"],
                "years": [],
                "first_year": 2020,
                "last_year": 2022,
                "quality_grade": "research",
                "seed": 7,
                "maxlen": 5,
                "allowed_licenses": ["cc0", "cc-by"],
                "media_type": "StillImage",
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    dataset = GbifDataset.from_config(config_path)

    assert dataset.output_path == (tmp_path / "dataset").resolve()
    assert dataset.species == ["Mus musculus"]
    assert dataset.years == [2020, 2021, 2022]
    assert dataset.seed == 7
    assert dataset.max_img_num == 5
    assert dataset.allowed_licenses == {"cc0", "cc-by"}
    assert dataset.media_type == "StillImage"
    assert dataset.config_path == config_path.resolve()
    assert (dataset.output_path / config_path.name).is_file()


def test_from_config_uses_constructor_defaults(tmp_path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "gbif": {
                        "output_path": str(tmp_path / "dataset"),
                        "species": ["Mus musculus"],
                        "years": [2022],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = GbifDataset.from_config(config_path)

    assert dataset.seed == 42
    assert dataset.max_img_num == 2000
    assert dataset.allowed_licenses == {"cc-by-nc"}
    assert dataset.media_type == "StillImage"


@pytest.mark.parametrize("missing", ["output_path", "species"])
def test_from_config_requires_gbif_settings(tmp_path, missing):
    settings = {
        "output_path": str(tmp_path / "dataset"),
        "species": ["Mus musculus"],
        "years": [2022],
    }
    del settings[missing]
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(
        yaml.safe_dump({"data": {"gbif": settings}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=missing):
        GbifDataset.from_config(config_path)


@pytest.mark.parametrize(
    "content",
    ["", "- not\n- a\n- mapping\n", "data: 42\n", "data:\n  gbif: 42\n"],
)
def test_from_config_requires_data_gbif_mapping(tmp_path, content):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(TypeError):
        GbifDataset.from_config(config_path)


def test_get_all_records_returns_fixed_image_level_records(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    first_occurrence = {
        "key": 123,
        "scientificName": "Mus musculus Linnaeus, 1758",
        "acceptedScientificName": "Mus musculus Linnaeus, 1758",
        "taxonKey": 7429082,
        "species": "Mus musculus",
        "speciesKey": 7429082,
        "genus": "Mus",
        "family": "Muridae",
        "order": "Rodentia",
        "class": "Mammalia",
        "phylum": "Chordata",
        "kingdom": "Animalia",
        "year": 2022,
        "eventDate": "2022-03-04",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "countryCode": "DE",
        "decimalLatitude": 52.5,
        "decimalLongitude": 13.4,
        "references": "https://example.test/occurrences/123",
        "datasetKey": "dataset-1",
        "publishingOrgKey": "publisher-1",
        "media": [
            {
                "type": "StillImage",
                "format": "image/jpeg",
                "identifier": "https://example.test/first.jpg",
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "creator": "A. Photographer",
                "references": "https://example.test/media/first",
                "created": "2022-03-04",
            },
            {
                "type": "Sound",
                "identifier": "https://example.test/audio.mp3",
            },
            {"type": "StillImage", "identifier": None},
        ],
    }
    second_occurrence = {
        "key": 124,
        "datasetKey": "dataset-1",
        "media": [
            {
                "type": "StillImage",
                "identifier": "https://example.test/second.jpg",
            }
        ],
    }
    responses = [
        {"results": [first_occurrence], "endOfRecords": False},
        {"results": [second_occurrence], "endOfRecords": True},
    ]
    search = Mock(side_effect=responses)
    datasets = Mock(
        return_value={
            "citation": {"text": "Example dataset citation"},
            "doi": "10.1234/example",
        }
    )
    monkeypatch.setattr(gbif_module.occ, "search", search)
    monkeypatch.setattr(gbif_module.registry, "datasets", datasets)

    records = dataset._get_all_records(scientificName="Mus musculus")

    assert len(records) == 2
    assert tuple(records[0]) == dataset._record_columns()
    assert records[0]["source_group_id"] == 123
    assert records[0]["scientific_name"] == "Mus musculus Linnaeus, 1758"
    assert records[0]["image_url"] == "https://example.test/first.jpg"
    assert records[0]["image_creator"] == "A. Photographer"
    assert records[0]["dataset_citation"] == "Example dataset citation"
    assert records[0]["dataset_doi"] == "10.1234/example"
    assert records[1]["source_group_id"] == 124
    assert records[1]["scientific_name"] is None
    assert records[1]["image_format"] is None
    assert search.call_args_list[0].kwargs["offset"] == 0
    assert search.call_args_list[1].kwargs["offset"] == 1
    datasets.assert_called_once_with(uuid="dataset-1")


def test_get_all_records_accepts_plain_text_citation_and_missing_dataset(
    tmp_path, monkeypatch
):
    dataset = make_dataset(tmp_path)
    occurrences = [
        {
            "key": 1,
            "datasetKey": "dataset-1",
            "media": [
                {"type": "StillImage", "identifier": "https://example.test/1.jpg"}
            ],
        },
        {
            "key": 2,
            "media": [
                {"type": "StillImage", "identifier": "https://example.test/2.jpg"}
            ],
        },
    ]
    monkeypatch.setattr(
        gbif_module.occ,
        "search",
        Mock(return_value={"results": occurrences, "endOfRecords": True}),
    )
    monkeypatch.setattr(
        gbif_module.registry,
        "datasets",
        Mock(return_value={"citation": "Plain citation", "doi": None}),
    )

    records = dataset._get_all_records()

    assert records[0]["dataset_citation"] == "Plain citation"
    assert records[1]["dataset_key"] is None
    assert records[1]["dataset_citation"] is None
    assert records[1]["dataset_doi"] is None


def test_get_species_records_has_fixed_columns_when_empty(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(dataset, "_get_all_records", lambda **params: [])

    records = dataset._get_species_records("Mus musculus")

    assert isinstance(records, pd.DataFrame)
    assert records.empty
    assert tuple(records.columns) == dataset._record_columns()


def test_get_species_records_shuffles_and_caps_image_rows(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path, max_img_num=1)
    dataset.years = [2021, 2022]
    get_all_records = Mock(
        side_effect=[
            [{"source_group_id": 1, "image_url": "https://example.test/1.jpg"}],
            [{"source_group_id": 2, "image_url": "https://example.test/2.jpg"}],
        ]
    )
    monkeypatch.setattr(dataset, "_get_all_records", get_all_records)

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 1
    assert tuple(records.columns) == dataset._record_columns()
    assert [call.kwargs["year"] for call in get_all_records.call_args_list] == [
        2021,
        2022,
    ]
