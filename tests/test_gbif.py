import csv
import hashlib
import os
import time
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, create_autospec
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import pytest
import requests
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pygbif import occurrences as gbif_occurrences
from pygbif import species as gbif_species

import smartrodent.gbif as gbif_module
from smartrodent.gbif import GbifDataset
from smartrodent.gbif_archive import GbifArchiveReader
from smartrodent.licenses import (
    LICENSE_MANAGER_TYPES,
    CreativeCommonsLicenseManager,
    LicenseManagerBase,
)


class ExampleLicenseManager(LicenseManagerBase):
    """License manager used to exercise multi-family GBIF policies."""

    def normalize_license(self, raw_license: object) -> str | None:
        """Normalize the two representations understood by this test family."""
        if raw_license in {"example-open", "cc-by"}:
            return "example-open"
        return None


def make_dataset(tmp_path, **kwargs):
    return GbifDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        **kwargs,
    )


def make_image_record(record_id, image_license):
    """Create the minimum GBIF image record needed by policy tests."""
    return {
        "source_group_id": record_id,
        "image_url": f"https://example.test/{record_id}.jpg",
        "image_license": image_license,
    }


def valid_jpeg_bytes():
    """Return a minimal byte sequence with JPEG start and end markers."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00test-image\xff\xd9"


def valid_png_bytes():
    """Return a minimal byte sequence with the PNG signature."""
    return b"\x89PNG\r\n\x1a\ntest-image"


def make_download_record(
    record_id=123,
    image_url="https://example.test/image.jpg",
    scientific_name="Mus musculus",
    image_format="image/jpeg",
):
    """Create one image-level record for download tests."""
    return pd.Series(
        {
            "source_group_id": record_id,
            "scientific_name": scientific_name,
            "image_url": image_url,
            "image_format": image_format,
        }
    )


def make_image_response(
    content=None,
    *,
    status_code=200,
    content_type="image/jpeg",
    content_length=None,
):
    """Create a streamed HTTP response double containing image bytes."""
    if content is None:
        content = valid_jpeg_bytes()
    response = Mock()
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    if content_length is not None:
        response.headers["Content-Length"] = str(content_length)
    response.iter_content.side_effect = lambda chunk_size: iter([content])
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=response
        )
    return response


def test_constructor_builds_one_license_manager_per_configured_family(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(LICENSE_MANAGER_TYPES, "example", ExampleLicenseManager)
    allowed_licenses = {
        "creative-commons": ["cc0", "cc-by"],
        "example": ["example-open"],
    }

    dataset = make_dataset(tmp_path, allowed_licenses=allowed_licenses)

    assert len(dataset._license_managers) == 2
    managers_by_type = {type(manager): manager for manager in dataset._license_managers}
    assert managers_by_type[
        CreativeCommonsLicenseManager
    ].allowed_licenses == frozenset({"cc0", "cc-by"})
    assert managers_by_type[ExampleLicenseManager].allowed_licenses == frozenset(
        {"example-open"}
    )


def test_constructor_copies_allowed_license_configuration(tmp_path):
    configured_licenses = ["cc-by"]
    allowed_licenses = {"creative-commons": configured_licenses}

    dataset = make_dataset(tmp_path, allowed_licenses=allowed_licenses)
    configured_licenses.append("cc-by-nc")
    allowed_licenses["creative-commons"] = ["cc0"]

    (manager,) = dataset._license_managers
    assert manager.allowed_licenses == frozenset({"cc-by"})


@pytest.mark.parametrize(
    ("allowed_licenses", "expected_exception"),
    [
        (["cc-by"], TypeError),
        ("cc-by", TypeError),
        ({}, ValueError),
        ({42: ["cc-by"]}, TypeError),
        ({"unknown": ["example-open"]}, ValueError),
        ({"creative-commons": []}, ValueError),
        ({"creative-commons": "cc-by"}, TypeError),
    ],
)
def test_constructor_rejects_invalid_allowed_license_configuration(
    tmp_path, allowed_licenses, expected_exception
):
    with pytest.raises(expected_exception):
        make_dataset(tmp_path, allowed_licenses=allowed_licenses)


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
                "allowed_licenses": {
                    "creative-commons": ["cc0", "cc-by"],
                },
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
    (manager,) = dataset._license_managers
    assert isinstance(manager, CreativeCommonsLicenseManager)
    assert manager.allowed_licenses == frozenset({"cc0", "cc-by"})
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
    (manager,) = dataset._license_managers
    assert isinstance(manager, CreativeCommonsLicenseManager)
    assert manager.allowed_licenses == frozenset({"cc-by-nc"})
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


def test_get_all_records_returns_fixed_image_level_records(
    tmp_path, monkeypatch, bulk_api
):
    """The SDK archive preserves projection, media selection, and shared attribution."""
    dataset = make_dataset(tmp_path)
    occurrences = [
        {
            "gbifID": "123",
            "scientificName": "Mus musculus Linnaeus, 1758",
            "acceptedScientificName": "Mus musculus Linnaeus, 1758",
            "taxonKey": "7429082",
            "species": "Mus musculus",
            "speciesKey": "7429082",
            "genus": "Mus",
            "family": "Muridae",
            "order": "Rodentia",
            "class": "Mammalia",
            "phylum": "Chordata",
            "kingdom": "Animalia",
            "year": "2022",
            "eventDate": "2022-03-04",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "countryCode": "DE",
            "decimalLatitude": "52.5",
            "decimalLongitude": "13.4",
            "references": "https://example.test/occurrences/123",
            "datasetKey": "dataset-1",
            "publishingOrgKey": "publisher-1",
        },
        {"gbifID": "124", "datasetKey": "dataset-1"},
    ]
    media = [
        {
            "gbifID": "123",
            "type": "StillImage",
            "format": "image/jpeg",
            "identifier": "https://example.test/first.jpg",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "creator": "A. Photographer",
            "references": "https://example.test/media/first",
            "created": "2022-03-04",
        },
        {
            "gbifID": "123",
            "type": "Sound",
            "identifier": "https://example.test/audio.mp3",
        },
        {"gbifID": "123", "type": "StillImage", "identifier": None},
        {
            "gbifID": "124",
            "type": "StillImage",
            "identifier": "https://example.test/second.jpg",
        },
    ]
    bulk_api.archive = make_bulk_archive(occurrences=occurrences, media=media)
    datasets = Mock(
        return_value={
            "citation": {"text": "Example dataset citation"},
            "doi": "10.1234/example",
        }
    )
    monkeypatch.setattr(gbif_module.registry, "datasets", datasets)

    records = dataset._get_all_records_for_params(scientificName="Mus musculus")

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
    assert len(bulk_api.submissions()) == 1
    datasets.assert_called_once_with(uuid="dataset-1")


def test_get_all_records_accepts_plain_text_citation_and_missing_dataset(
    tmp_path,
    monkeypatch,
    bulk_api,
):
    """Missing optional source metadata stays absent; plain citations remain valid."""
    dataset = make_dataset(tmp_path)
    bulk_api.archive = make_bulk_archive(
        occurrences=[{"gbifID": "1", "datasetKey": "dataset-1"}, {"gbifID": "2"}],
        media=[
            {
                "gbifID": "1",
                "type": "StillImage",
                "identifier": "https://example.test/1.jpg",
            },
            {
                "gbifID": "2",
                "type": "StillImage",
                "identifier": "https://example.test/2.jpg",
            },
        ],
    )
    monkeypatch.setattr(
        gbif_module.registry,
        "datasets",
        Mock(return_value={"citation": "Plain citation", "doi": None}),
    )

    records = dataset._get_all_records_for_params(scientificName="Mus musculus")

    assert records[0]["dataset_citation"] == "Plain citation"
    assert records[1]["dataset_key"] is None
    assert records[1]["dataset_citation"] is None
    assert records[1]["dataset_doi"] is None


def test_get_species_records_has_fixed_columns_when_empty(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(dataset, "_get_all_records_for_params", lambda **params: [])

    records = dataset._get_species_records("Mus musculus")

    assert isinstance(records, pd.DataFrame)
    assert records.empty
    assert tuple(records.columns) == dataset._record_columns()


def test_get_species_records_shuffles_and_caps_image_rows(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path, max_img_num=1)
    dataset.years = [2021, 2022]
    get_all_records = Mock(
        return_value=[
            make_image_record(1, "cc-by-nc"),
            make_image_record(2, "cc-by-nc"),
        ]
    )
    monkeypatch.setattr(dataset, "_get_all_records_for_params", get_all_records)

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 1
    assert tuple(records.columns) == dataset._record_columns()
    get_all_records.assert_called_once_with(scientificName="Mus musculus")


def test_get_species_records_keeps_only_licenses_allowed_by_a_manager(
    tmp_path, monkeypatch
):
    dataset = make_dataset(
        tmp_path,
        allowed_licenses={"creative-commons": ["cc-by"]},
    )
    disallowed_sibling_image = make_image_record(
        2, "https://creativecommons.org/licenses/by-nc/4.0/"
    )
    disallowed_sibling_image["source_group_id"] = 1
    records_for_year = [
        make_image_record(
            1, "https://creativecommons.org/licenses/by/4.0/?source=gbif"
        ),
        disallowed_sibling_image,
        make_image_record(3, "not-a-license"),
        make_image_record(4, None),
        {
            "source_group_id": 5,
            "image_url": "https://example.test/5.jpg",
        },
    ]
    monkeypatch.setattr(
        dataset,
        "_get_all_records_for_params",
        Mock(return_value=records_for_year),
    )

    records = dataset._get_species_records("Mus musculus")

    assert records["source_group_id"].tolist() == [1]
    assert records.iloc[0]["image_license"] == records_for_year[0]["image_license"]


def test_get_species_records_accepts_a_license_from_any_configured_family(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(LICENSE_MANAGER_TYPES, "example", ExampleLicenseManager)
    dataset = make_dataset(
        tmp_path,
        allowed_licenses={
            "creative-commons": ["cc-by"],
            "example": ["example-open"],
        },
    )
    records_for_year = [
        # Both managers accept this value; the image row must still occur once.
        make_image_record(1, "cc-by"),
        make_image_record(2, "example-open"),
        make_image_record(3, "cc-by-nc"),
    ]
    monkeypatch.setattr(
        dataset,
        "_get_all_records_for_params",
        Mock(return_value=records_for_year),
    )

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 2
    assert set(records["source_group_id"]) == {1, 2}


def test_get_species_records_filters_all_years_before_applying_cap(
    tmp_path,
    monkeypatch,
):
    dataset = make_dataset(
        tmp_path,
        max_img_num=2,
        allowed_licenses={"creative-commons": ["cc-by"]},
    )
    dataset.years = [2021, 2022]
    get_all_records = Mock(
        return_value=[
            {**make_image_record(1, "cc-by"), "year": 2021},
            {**make_image_record(2, "cc-by-nc"), "year": 2021},
            {**make_image_record(3, "cc-by"), "year": 2022},
            {**make_image_record(4, "cc-by-nc"), "year": 2022},
        ]
    )
    monkeypatch.setattr(dataset, "_get_all_records_for_params", get_all_records)

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 2
    assert set(records["source_group_id"]) == {1, 3}
    assert set(records["year"]) == {2021, 2022}
    get_all_records.assert_called_once_with(scientificName="Mus musculus")


def test_get_species_records_returns_fixed_columns_when_all_licenses_rejected(
    tmp_path, monkeypatch
):
    dataset = make_dataset(
        tmp_path,
        allowed_licenses={"creative-commons": ["cc-by"]},
    )
    monkeypatch.setattr(
        dataset,
        "_get_all_records_for_params",
        Mock(
            return_value=[
                make_image_record(1, "cc-by-nc"),
                make_image_record(2, None),
            ]
        ),
    )

    records = dataset._get_species_records("Mus musculus")

    assert records.empty
    assert tuple(records.columns) == dataset._record_columns()


@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.lists(st.booleans(), min_size=1, max_size=15))
def test_get_species_records_returns_exactly_the_allowed_rows(
    tmp_path, monkeypatch, allowed_rows
):
    dataset = make_dataset(
        tmp_path,
        max_img_num=len(allowed_rows),
        allowed_licenses={"creative-commons": ["cc-by"]},
    )
    fetched_records = [
        make_image_record(
            record_id,
            "cc-by" if is_allowed else "cc-by-nc",
        )
        for record_id, is_allowed in enumerate(allowed_rows)
    ]
    monkeypatch.setattr(
        dataset,
        "_get_all_records_for_params",
        Mock(return_value=fetched_records),
    )

    records = dataset._get_species_records("Mus musculus")

    expected_ids = [
        record_id for record_id, is_allowed in enumerate(allowed_rows) if is_allowed
    ]
    assert sorted(records["source_group_id"].tolist()) == expected_ids


def test_download_image_uses_documented_gbif_cache_url(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = (
        "https://inaturalist-open-data.s3.amazonaws.com/photos/31610070/original.jpg"
    )
    record = make_download_record(
        record_id=2005380410,
        image_url=source_url,
    )
    response = make_image_response()
    session = Mock()
    session.get.return_value = response

    success = dataset._download_image(record, images_path, session)

    assert success is True
    session.get.assert_called_once_with(
        "https://api.gbif.org/v1/image/cache/occurrence/"
        "2005380410/media/568639b65b65ddb9090d3d6ef1abce14",
        stream=True,
        timeout=30,
        allow_redirects=False,
    )


def test_download_image_hashes_the_exact_publisher_url(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = "https://example.test/image?id=7&size=original"
    source_hash = hashlib.md5(source_url.encode()).hexdigest()
    session = Mock()
    session.get.return_value = make_image_response()

    assert dataset._download_image(
        make_download_record(image_url=source_url), images_path, session
    )
    requested_url = session.get.call_args.args[0]
    assert requested_url.endswith(f"/media/{source_hash}")


@pytest.mark.parametrize(
    ("source_url", "content", "content_type", "expected_suffix"),
    [
        (
            "https://example.test/image.jpg",
            valid_jpeg_bytes(),
            "image/jpeg",
            ".jpg",
        ),
        (
            "https://example.test/image.png",
            valid_png_bytes(),
            "image/png",
            ".png",
        ),
        (
            "https://example.test/image",
            valid_jpeg_bytes(),
            "image/jpeg",
            ".jpg",
        ),
    ],
)
def test_download_image_writes_valid_jpeg_and_png_payloads(
    tmp_path, source_url, content, content_type, expected_suffix
):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    session = Mock()
    session.get.return_value = make_image_response(content, content_type=content_type)

    success = dataset._download_image(
        make_download_record(image_url=source_url), images_path, session
    )

    source_hash = hashlib.md5(source_url.encode()).hexdigest()
    expected_path = images_path / f"123_{source_hash}{expected_suffix}"
    assert success is True
    assert expected_path.read_bytes() == content
    assert not list(images_path.glob("*.part"))


def test_download_image_normalizes_float_occurrence_id_for_filename(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = "https://example.test/image.jpg"
    session = Mock()
    session.get.return_value = make_image_response()

    success = dataset._download_image(
        make_download_record(record_id=123.0, image_url=source_url),
        images_path,
        session,
    )

    source_hash = hashlib.md5(source_url.encode()).hexdigest()
    assert success is True
    assert (images_path / f"123_{source_hash}.jpg").is_file()


def test_download_image_keeps_distinct_media_from_one_occurrence(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    first_url = "https://example.test/first.jpg"
    second_url = "https://example.test/second.jpg"
    session = Mock()
    session.get.side_effect = [make_image_response(), make_image_response()]

    first_success = dataset._download_image(
        make_download_record(image_url=first_url), images_path, session
    )
    second_success = dataset._download_image(
        make_download_record(image_url=second_url), images_path, session
    )

    expected_names = {
        f"123_{hashlib.md5(url.encode()).hexdigest()}.jpg"
        for url in (first_url, second_url)
    }
    assert first_success is True
    assert second_success is True
    assert {path.name for path in images_path.iterdir()} == expected_names


@pytest.mark.parametrize(
    "record",
    [
        make_download_record(record_id=None),
        make_download_record(record_id=float("nan")),
        make_download_record(image_url=None),
        make_download_record(image_url=""),
    ],
)
def test_download_image_rejects_records_without_cache_identifiers(tmp_path, record):
    dataset = make_dataset(tmp_path)
    session = Mock()

    success = dataset._download_image(record, tmp_path / "images", session)

    assert success is False
    session.get.assert_not_called()


def test_download_image_treats_an_existing_valid_image_as_success(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = "https://example.test/image.jpg"
    source_hash = hashlib.md5(source_url.encode()).hexdigest()
    existing_image = images_path / f"123_{source_hash}.jpg"
    existing_image.write_bytes(valid_jpeg_bytes())
    session = Mock()

    success = dataset._download_image(
        make_download_record(image_url=source_url), images_path, session
    )

    assert success is True
    assert existing_image.read_bytes() == valid_jpeg_bytes()
    session.get.assert_not_called()


def test_download_image_replaces_an_existing_invalid_image(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = "https://example.test/image.jpg"
    source_hash = hashlib.md5(source_url.encode()).hexdigest()
    existing_image = images_path / f"123_{source_hash}.jpg"
    existing_image.write_bytes(b"not an image")
    session = Mock()
    session.get.return_value = make_image_response()

    success = dataset._download_image(
        make_download_record(image_url=source_url), images_path, session
    )

    assert success is True
    assert existing_image.read_bytes() == valid_jpeg_bytes()
    session.get.assert_called_once()


def test_download_image_rejects_redirect_without_following_it(tmp_path):
    dataset = make_dataset(tmp_path)
    response = make_image_response(status_code=302)
    response.headers["Location"] = "http://169.254.169.254/latest/meta-data/"
    session = Mock()
    session.get.return_value = response

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert not list(tmp_path.rglob("*.jpg"))


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"<html>upstream error</html>", "text/html"),
        (b"not really a jpeg", "image/jpeg"),
    ],
)
def test_download_image_rejects_non_image_responses(tmp_path, content, content_type):
    dataset = make_dataset(tmp_path)
    session = Mock()
    session.get.return_value = make_image_response(content, content_type=content_type)

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    assert not list(tmp_path.rglob("*.jpg"))
    assert not list(tmp_path.rglob("*.part"))


def test_download_image_accepts_valid_image_with_generic_content_type(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    session = Mock()
    session.get.return_value = make_image_response(
        valid_jpeg_bytes(), content_type="application/octet-stream"
    )

    success = dataset._download_image(make_download_record(), images_path, session)

    assert success is True
    assert len(list(images_path.glob("*.jpg"))) == 1


def test_download_image_rejects_oversized_content_length(tmp_path):
    dataset = make_dataset(tmp_path)
    response = make_image_response(content_length=20 * 1024 * 1024 + 1)
    session = Mock()
    session.get.return_value = response

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    response.iter_content.assert_not_called()
    assert not list(tmp_path.rglob("*.part"))


def test_download_image_stops_when_stream_exceeds_size_limit(tmp_path):
    dataset = make_dataset(tmp_path)
    response = make_image_response()
    ten_mebibytes = b"a" * (10 * 1024 * 1024)
    response.iter_content.side_effect = lambda chunk_size: iter(
        [valid_jpeg_bytes() + ten_mebibytes, ten_mebibytes]
    )
    session = Mock()
    session.get.return_value = response

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    assert not list(tmp_path.rglob("*.jpg"))
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("status_code", [404, 500])
def test_download_image_reports_http_errors_as_failures(tmp_path, status_code):
    dataset = make_dataset(tmp_path)
    session = Mock()
    session.get.return_value = make_image_response(status_code=status_code)

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    assert not list(tmp_path.rglob("*.part"))


def test_download_image_reports_request_errors_as_failures(tmp_path):
    dataset = make_dataset(tmp_path)
    session = Mock()
    session.get.side_effect = requests.ConnectionError("cache unavailable")

    success = dataset._download_image(
        make_download_record(), tmp_path / "images", session
    )

    assert success is False
    assert not list(tmp_path.rglob("*.part"))


def test_download_image_removes_partial_file_after_stream_failure(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"

    def interrupted_stream(chunk_size):
        yield valid_jpeg_bytes()
        raise requests.ConnectionError("stream interrupted")

    response = make_image_response()
    response.iter_content.side_effect = interrupted_stream
    session = Mock()
    session.get.return_value = response

    success = dataset._download_image(make_download_record(), images_path, session)

    assert success is False
    assert not list(images_path.glob("*"))


def test_download_species_images_reports_every_record_in_order(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    records = pd.DataFrame(
        [
            make_download_record(
                11, "https://example.test/11-first.jpg", "Species one"
            ),
            make_download_record(
                11, "https://example.test/11-second.jpg", "Species one"
            ),
            make_download_record(12, "https://example.test/12.jpg", "Species two"),
        ],
        index=[7, 3, 9],
    )
    download_image = Mock(side_effect=[True, False, True])
    monkeypatch.setattr(dataset, "_download_image", download_image, raising=False)
    session = Mock()
    session_factory = Mock(return_value=session)
    monkeypatch.setattr(gbif_module.requests, "Session", session_factory)

    report = dataset._download_species_images(records, tmp_path / "images")

    expected = records.copy()
    expected["success"] = [True, False, True]
    pd.testing.assert_frame_equal(report, expected)
    session_factory.assert_called_once_with()
    assert [call.args[2] for call in download_image.call_args_list] == [
        session,
        session,
        session,
    ]


def test_download_species_images_returns_stable_empty_report(tmp_path):
    dataset = make_dataset(tmp_path)
    records = pd.DataFrame(columns=dataset._record_columns())

    report = dataset._download_species_images(records, tmp_path / "images")

    assert report.empty
    assert list(report.columns) == [*dataset._record_columns(), "success"]


@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=3),
            st.booleans(),
        ),
        min_size=1,
        max_size=15,
    )
)
def test_download_species_images_has_one_result_for_every_input_record(
    tmp_path, monkeypatch, group_ids_and_outcomes
):
    dataset = make_dataset(tmp_path)
    group_ids = [item[0] for item in group_ids_and_outcomes]
    outcomes = [item[1] for item in group_ids_and_outcomes]
    records = pd.DataFrame(
        [
            make_download_record(
                record_id=group_id,
                image_url=f"https://example.test/{index}.jpg",
                scientific_name=f"Species {index}",
            )
            for index, group_id in enumerate(group_ids)
        ],
        index=range(100, 100 + len(outcomes)),
    )
    monkeypatch.setattr(
        dataset,
        "_download_image",
        Mock(side_effect=outcomes),
        raising=False,
    )
    monkeypatch.setattr(gbif_module.requests, "Session", Mock(return_value=Mock()))

    report = dataset._download_species_images(records, tmp_path / "images")

    assert len(report) == len(records)
    pd.testing.assert_frame_equal(report.drop(columns="success"), records)
    assert report["success"].tolist() == outcomes


def test_download_writes_records_and_download_report_for_each_species(
    tmp_path, monkeypatch
):
    dataset = make_dataset(tmp_path)
    records = pd.DataFrame(
        [
            make_download_record(123, "https://example.test/first.jpg"),
            make_download_record(123, "https://example.test/second.jpg"),
        ]
    )
    report = records.copy()
    report["success"] = [True, False]
    monkeypatch.setattr(dataset, "_get_species_records", Mock(return_value=records))
    monkeypatch.setattr(dataset, "_download_species_images", Mock(return_value=report))
    dataset.logger = Mock()

    dataset.download()

    species_path = dataset.output_path / "Mus musculus"
    pd.testing.assert_frame_equal(pd.read_csv(species_path / "records.csv"), records)
    pd.testing.assert_frame_equal(
        pd.read_csv(species_path / "download_report.csv"),
        report.reset_index(names="index"),
    )
    assert (species_path / "imgs").is_dir()
    dataset.logger.info.assert_any_call(
        "Downloaded %s images for %s", 1, "Mus musculus"
    )


@pytest.fixture
def dataset(tmp_path):
    """Construct a GBIF downloader with only temporary output paths."""
    return make_dataset(tmp_path)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_download_processes_images_with_cached_or_fresh_records(
    dataset, monkeypatch, cached, empty
):
    """Even header-only caches are reused and reach the image phase."""
    records = pd.DataFrame({"id": pd.Series([] if empty else [123], dtype="int64")})
    species_path = dataset.output_path / "Mus musculus"
    records_path = species_path / "records.csv"
    if cached:
        species_path.mkdir()
        records.to_csv(records_path, index=False)
        original_bytes = records_path.read_bytes()

    retrieve = Mock(return_value=records)
    report = records.assign(success=True)
    download_images = Mock(return_value=report)
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    dataset.download()

    if cached:
        retrieve.assert_not_called()
        assert records_path.read_bytes() == original_bytes
    else:
        retrieve.assert_called_once_with("Mus musculus")
    pd.testing.assert_frame_equal(
        pd.read_csv(records_path), records, check_dtype=not empty
    )
    download_images.assert_called_once()
    received_records, images_path = download_images.call_args.args
    pd.testing.assert_frame_equal(received_records, records, check_dtype=not empty)
    assert images_path == species_path / "imgs"
    assert images_path.is_dir()
    pd.testing.assert_frame_equal(
        pd.read_csv(species_path / "download_report.csv"),
        report.reset_index(names="index"),
        check_dtype=not empty,
    )


def test_invalid_cache_is_not_treated_as_missing(dataset, monkeypatch):
    """An unreadable CSV must not trigger retrieval or image downloads."""
    species_path = dataset.output_path / "Mus musculus"
    species_path.mkdir()
    (species_path / "records.csv").write_text("", encoding="utf-8")
    retrieve = Mock()
    download_images = Mock()
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    with pytest.raises(pd.errors.EmptyDataError):
        dataset.download()

    retrieve.assert_not_called()
    download_images.assert_not_called()


@pytest.mark.parametrize("phase", ["retrieval", "images"])
def test_file_not_found_outside_cache_read_propagates(dataset, monkeypatch, phase):
    """Only missing record caches are handled, not later filesystem errors."""
    records = pd.DataFrame({"id": [123]})
    error = FileNotFoundError("phase failed")
    retrieve = (
        Mock(side_effect=error) if phase == "retrieval" else Mock(return_value=records)
    )
    download_images = Mock(side_effect=error)
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    with pytest.raises(FileNotFoundError, match="phase failed"):
        dataset.download()

    retrieve.assert_called_once_with("Mus musculus")
    if phase == "retrieval":
        download_images.assert_not_called()
    else:
        download_images.assert_called_once()


def test_gbif_cached_records_resume_images_and_refresh_report(tmp_path, monkeypatch):
    """The cached-record path preserves valid images and downloads missing ones."""
    dataset = make_dataset(tmp_path)
    urls = ["https://example.test/existing.jpg", "https://example.test/missing.jpg"]
    records = pd.DataFrame({"source_group_id": [123, 124], "image_url": urls})
    species_path = dataset.output_path / "Mus musculus"
    images_path = species_path / "imgs"
    images_path.mkdir(parents=True)
    records.to_csv(species_path / "records.csv", index=False)
    digest = hashlib.md5(urls[0].encode(), usedforsecurity=False).hexdigest()
    existing_path = images_path / f"123_{digest}.jpg"
    existing_path.write_bytes(b"\xff\xd8\xffexisting image")
    (species_path / "download_report.csv").write_text(
        "outdated report", encoding="utf-8"
    )
    retrieve = Mock()
    download_cached = Mock(return_value=True)
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_cached_image", download_cached)

    dataset.download()

    retrieve.assert_not_called()
    download_cached.assert_called_once()
    assert download_cached.call_args.args[1] == urls[1]
    assert existing_path.read_bytes() == b"\xff\xd8\xffexisting image"
    report = pd.read_csv(species_path / "download_report.csv")
    assert report["success"].tolist() == [True, True]
    assert report["image_url"].tolist() == urls


@pytest.mark.parametrize("cached", [False, True])
def test_retrieve_records_does_not_download_images(dataset, monkeypatch, cached):
    """The records-only phase produces portable CSVs without image directories."""
    dataset.species = ["Mus musculus", "Rattus rattus"]
    records = pd.DataFrame({"id": [123]})
    if cached:
        for species in dataset.species:
            species_path = dataset.output_path / species
            species_path.mkdir()
            records.to_csv(species_path / "records.csv", index=False)
    retrieve = Mock(return_value=records)
    download_images = Mock()
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    dataset.retrieve_records()

    download_images.assert_not_called()
    if cached:
        retrieve.assert_not_called()
    else:
        assert [call.args[0] for call in retrieve.call_args_list] == dataset.species
    for species in dataset.species:
        species_path = dataset.output_path / species
        pd.testing.assert_frame_equal(
            pd.read_csv(species_path / "records.csv"), records
        )
        assert not (species_path / "imgs").exists()
        assert not (species_path / "download_report.csv").exists()


def test_download_images_uses_only_saved_records(dataset, monkeypatch):
    """Images can be downloaded independently from a supplied records CSV."""
    records = pd.DataFrame({"id": [123]})
    species_path = dataset.output_path / "Mus musculus"
    species_path.mkdir()
    records_path = species_path / "records.csv"
    records.to_csv(records_path, index=False)
    original_bytes = records_path.read_bytes()
    retrieve = Mock()
    report = records.assign(success=True)
    download_images = Mock(return_value=report)
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    dataset.download_images()

    retrieve.assert_not_called()
    download_images.assert_called_once()
    received_records, images_path = download_images.call_args.args
    pd.testing.assert_frame_equal(received_records, records)
    assert images_path == species_path / "imgs"
    assert images_path.is_dir()
    assert records_path.read_bytes() == original_bytes
    pd.testing.assert_frame_equal(
        pd.read_csv(species_path / "download_report.csv"),
        report.reset_index(names="index"),
    )


def test_download_images_reports_missing_records(dataset, monkeypatch):
    """A missing records file fails clearly rather than triggering retrieval."""
    retrieve = Mock()
    download_images = Mock()
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    with pytest.raises(
        FileNotFoundError, match="Missing records for Mus musculus.*retrieve_records"
    ):
        dataset.download_images()

    retrieve.assert_not_called()
    download_images.assert_not_called()
    assert not (dataset.output_path / "Mus musculus" / "imgs").exists()


def test_download_retrieves_all_records_before_any_images(dataset, monkeypatch):
    """Every species CSV is available before the first image is processed."""
    dataset.species = ["Mus musculus", "Rattus rattus"]
    records = pd.DataFrame({"id": [123]})
    events = []

    def retrieve(species):
        """Record retrieval order without making network requests."""
        events.append(("records", species))
        return records.copy()

    def download_images(received_records, images_path):
        """Check persisted state at the start of each image batch."""
        for species in dataset.species:
            assert (dataset.output_path / species / "records.csv").is_file()
        pd.testing.assert_frame_equal(received_records, records)
        events.append(("images", images_path.parent.name))
        return received_records.assign(success=True)

    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    dataset.download()

    assert events == [
        ("records", "Mus musculus"),
        ("records", "Rattus rattus"),
        ("images", "Mus musculus"),
        ("images", "Rattus rattus"),
    ]


def test_record_phase_failure_prevents_all_image_downloads(dataset, monkeypatch):
    """A later retrieval failure still leaves earlier CSVs available to inspect."""
    dataset.species = ["Mus musculus", "Rattus rattus"]
    records = pd.DataFrame({"id": [123]})
    retrieve = Mock(side_effect=[records, RuntimeError("retrieval failed")])
    download_images = Mock()
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    with pytest.raises(RuntimeError, match="retrieval failed"):
        dataset.download()

    download_images.assert_not_called()
    assert (dataset.output_path / "Mus musculus" / "records.csv").is_file()
    for species in dataset.species:
        assert not (dataset.output_path / species / "imgs").exists()


class BulkDownloadClock:
    """Advance polling time deterministically without sleeping."""

    def __init__(self):
        """Initialize a clock and its observed waits."""
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        """Return the current simulated time."""
        return self.now

    def sleep(self, seconds):
        """Record and advance a requested polling delay."""
        assert seconds > 0, "Polling must not spin without a delay"
        self.waits.append(seconds)
        self.now += seconds


class BulkDownloadApi:
    """Control the pygbif seam without network calls, real credentials, or waiting.

    The downloader, ZIP/XML reader, pandas joins, and filesystem remain real.
    This double replaces only the external SDK operations so tests can inspect
    per-species queries and reproduce failures without depending on live GBIF.

    Maintenance cost: this intentionally models only the pygbif contract used
    here, currently verified against 0.6.6. When upgrading pygbif, recheck
    download()'s (key, payload), download_meta()'s status, download_get()'s path
    dictionary, and the species/registry response shapes. Autospec in bulk_api
    checks call signatures, not these return values or their semantics. Passing
    these offline tests does not establish compatibility with a live GBIF job.
    """

    def __init__(self):
        """Provide synthetic matches, completed jobs, and a small DWCA."""
        # Override individual names to exercise synonym and unreliable-match cases.
        self.matches = {}
        # SDK operation history, not HTTP traffic; distinguishes jobs from polls.
        self.calls = []
        # Issued keys ensure polling/fetching uses the SDK result, not its payload.
        self.keys = []
        # Each poll advances this sequence, then repeats its last status. The guard
        # in download_meta catches runaway loops; production uses its own deadline.
        self.statuses = ["SUCCEEDED"]
        self.polls = 0
        # Actual ZIP bytes let archive conversion run normally, including bad inputs.
        self.archive = make_bulk_archive()
        # Inject SDK errors independently at submission, archive fetch, or polling.
        self.submission_error = None
        self.archive_error = None
        self.metadata_error = None
        # Distinct from transport errors: supply a malformed but returned response.
        self.malformed_metadata = None

    def name_backbone(self, scientificName=None, **kwargs):
        """Return a configured or default match for the requested name.

        Args:
            scientificName: Scientific name supplied to pygbif.
            **kwargs: Additional matching settings supported by pygbif.

        Returns:
            A synthetic v2 species-match response.
        """
        self.calls.append(("match", scientificName))
        return self.matches.get(scientificName, make_bulk_match(scientificName))

    def download(
        self,
        queries,
        format="SIMPLE_CSV",
        user=None,
        pwd=None,
        email=None,
        pred_type="and",
    ):
        """Submit a synthetic job using pygbif's documented return shape.

        Args:
            queries: Structured GBIF predicate.
            format: Requested download format.
            user: Optional account name; otherwise read from the environment.
            pwd: Optional password, never included in the returned payload.
            email: Optional notification address; otherwise read from the environment.
            pred_type: Predicate combiner for pygbif's string-query form.

        Returns:
            A pair containing the download key and the account-bearing payload.
        """
        assert isinstance(queries, dict), "Use a structured download predicate"
        body = {"predicate": queries, "format": format}
        self.calls.append(("download", body))
        if self.submission_error is not None:
            raise self.submission_error
        key = f"{len(self.submissions()):07d}-test-download"
        self.keys.append(key)
        # pygbif returns account details too: callers must not persist this blindly.
        payload = {
            **body,
            "creator": user if user is not None else os.environ.get("GBIF_USER"),
            "notification_address": [
                email if email is not None else os.environ.get("GBIF_EMAIL")
            ],
        }
        return key, payload

    def download_meta(self, key, **kwargs):
        """Return the next job status or simulate a metadata failure.

        Args:
            key: Download key returned by download().
            **kwargs: Additional pygbif request options.

        Returns:
            Synthetic job metadata.
        """
        assert key in self.keys, "Poll using the key returned by pygbif.download"
        self.calls.append(("metadata", key))
        if self.metadata_error is not None:
            raise self.metadata_error
        if self.malformed_metadata is not None:
            return self.malformed_metadata
        assert self.polls < 100, "Unbounded download polling"
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        return {"key": key, "status": status, "size": len(self.archive)}

    def download_get(self, key, path=".", **kwargs):
        """Write an archive as pygbif does, including partial bytes on failure.

        Args:
            key: Download key returned by download().
            path: Existing destination directory.
            **kwargs: Additional pygbif request options.

        Returns:
            Archive path, size, and key, as returned by pygbif.download_get().
        """
        assert key in self.keys, "Fetch using the key returned by pygbif.download"
        self.calls.append(("archive", key))
        destination = Path(path) / f"{key}.zip"
        if self.archive_error is not None:
            destination.write_bytes(self.archive[: max(1, len(self.archive) // 2)])
            raise self.archive_error
        destination.write_bytes(self.archive)
        return {"path": str(destination), "size": len(self.archive), "key": key}

    def source_dataset(self, uuid=None, **kwargs):
        """Return source attribution through pygbif's registry method.

        Args:
            uuid: Source dataset key.
            **kwargs: Additional pygbif registry options.

        Returns:
            Synthetic source dataset citation and DOI.
        """
        self.calls.append(("dataset", uuid))
        return {
            "citation": {"text": "Synthetic source dataset citation"},
            "doi": "10.1234/synthetic-source",
        }

    def submissions(self):
        """Return observed SDK download calls, excluding polls and archive fetches."""
        return [call for call in self.calls if call[0] == "download"]

    def submitted_body(self):
        """Return the latest SDK predicate and format, not a serialized HTTP body."""
        return self.submissions()[-1][1]


def make_bulk_match(name="Mus musculus", key="100", accepted_name=None):
    """Build a species-match response using the current v2 API structure."""
    usage = {"key": key, "name": name, "canonicalName": name, "rank": "SPECIES"}
    accepted = {
        "key": "100" if accepted_name else key,
        "name": accepted_name or name,
        "canonicalName": accepted_name or name,
        "rank": "SPECIES",
    }
    return {
        "usage": usage,
        "acceptedUsage": accepted,
        "classification": [
            {"key": accepted["key"], "name": accepted["name"], "rank": "SPECIES"}
        ],
        "diagnostics": {"matchType": "EXACT", "confidence": 100},
        "synonym": accepted_name is not None,
    }


def make_bulk_archive(
    occurrences=None, media=None, reverse_columns=False, omit=(), field_defaults=None
):
    """Build a small DWCA with explicit term mappings and linked media rows."""
    if occurrences is None:
        occurrences = [
            {
                "gbifID": "123",
                "scientificName": "Mus musculus Linnaeus, 1758",
                "acceptedScientificName": "Mus musculus Linnaeus, 1758",
                "taxonKey": "100",
                "speciesKey": "100",
                "species": "Mus musculus",
                "year": "2022",
                "datasetKey": "synthetic-dataset",
                "license": "http://creativecommons.org/publicdomain/zero/1.0/",
                "references": "https://example.test/occurrence/123",
            }
        ]
    if media is None:
        media = [
            {
                "gbifID": "123",
                "type": "StillImage",
                "identifier": "https://example.test/123.jpg",
                "license": "http://creativecommons.org/licenses/by-nc/4.0/",
                "creator": "Synthetic photographer",
                "references": "https://example.test/photo/123",
                "format": "image/jpeg",
            }
        ]
    buffer = BytesIO()
    namespace = "http://rs.tdwg.org/dwc/text/"
    root = ET.Element("archive", xmlns=namespace)
    with ZipFile(buffer, "w") as archive:
        for tag, filename, rows, row_type in [
            (
                "core",
                "occurrence.txt",
                occurrences,
                "http://rs.tdwg.org/dwc/terms/Occurrence",
            ),
            (
                "extension",
                "multimedia.txt",
                media,
                "http://rs.gbif.org/terms/1.0/Multimedia",
            ),
        ]:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            if reverse_columns:
                fields.reverse()
            if "gbifID" not in fields:
                fields.insert(0, "gbifID")
            table = ET.SubElement(
                root,
                tag,
                encoding="UTF-8",
                fieldsTerminatedBy="\\t",
                linesTerminatedBy="\\n",
                fieldsEnclosedBy='"',
                ignoreHeaderLines="1",
                rowType=row_type,
            )
            ET.SubElement(ET.SubElement(table, "files"), "location").text = filename
            ET.SubElement(
                table,
                "id" if tag == "core" else "coreid",
                index=str(fields.index("gbifID")),
            )
            for index, field in enumerate(fields):
                if field in {
                    "gbifID",
                    "taxonKey",
                    "speciesKey",
                    "species",
                    "datasetKey",
                }:
                    base = "http://rs.gbif.org/terms/1.0/"
                elif field in {
                    "identifier",
                    "type",
                    "license",
                    "creator",
                    "references",
                    "format",
                }:
                    base = "http://purl.org/dc/terms/"
                else:
                    base = "http://rs.tdwg.org/dwc/terms/"
                ET.SubElement(table, "field", index=str(index), term=base + field)
            for name, default in (field_defaults or {}).get(tag, {}).items():
                declaration = next(
                    (
                        field
                        for field in table.findall("field")
                        if field.attrib["term"].rsplit("/", 1)[-1] == name
                    ),
                    None,
                )
                if declaration is None:
                    declaration = ET.SubElement(
                        table, "field", term="http://rs.tdwg.org/dwc/terms/" + name
                    )
                declaration.set("default", default)
            stream = StringIO()
            writer = csv.DictWriter(
                stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            if filename not in omit:
                archive.writestr(filename, stream.getvalue())
        if "meta.xml" not in omit:
            archive.writestr("meta.xml", ET.tostring(root, encoding="utf-8"))
    return buffer.getvalue()


@pytest.mark.parametrize("table_tag", ["core", "extension"])
@pytest.mark.parametrize("default", ["WGS84", ""])
def test_archive_default_only_field_applies_to_every_media_row(
    tmp_path, table_tag, default
):
    """A constant XML field needs no physical column and preserves media joins."""
    media = [
        {"gbifID": "123", "identifier": "https://example.test/first.jpg"},
        {"gbifID": "123", "identifier": "https://example.test/second.jpg"},
    ]
    archive_path = tmp_path / "default-field.zip"
    with ZipFile(BytesIO(make_bulk_archive(media=media))) as source:
        descriptor = ET.fromstring(source.read("meta.xml"))
        table = descriptor.find(f"{{*}}{table_tag}")
        assert table is not None
        # GBIF declares the shared coordinate datum once, without a table index.
        ET.SubElement(
            table,
            "{http://rs.tdwg.org/dwc/text/}field",
            term="http://rs.tdwg.org/dwc/terms/geodeticDatum",
            default=default,
        )
        with ZipFile(archive_path, "w") as target:
            for name in source.namelist():
                content = (
                    ET.tostring(descriptor, encoding="utf-8")
                    if name == "meta.xml"
                    else source.read(name)
                )
                target.writestr(name, content)

    records = GbifArchiveReader(archive_path).read()

    column = "geodeticDatum" if table_tag == "core" else "media_geodeticDatum"
    actual = records[column].astype(object).where(records[column].notna(), None)
    assert actual.tolist() == [default or None, default or None]
    assert records["_core_id"].tolist() == ["123", "123"]
    assert records["media_identifier"].tolist() == [
        "https://example.test/first.jpg",
        "https://example.test/second.jpg",
    ]


@pytest.mark.parametrize("table_tag", ["core", "extension"])
@given(
    default=st.sampled_from(["WGS84", "EPSG:4326", ""]),
    values=st.lists(
        st.sampled_from(["", "EPSG:4258", "NA", " "]), min_size=1, max_size=8
    ),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_archive_indexed_defaults_fill_only_empty_values(
    tmp_path, table_tag, default, values
):
    """Indexed defaults fill empty cells, preserving explicit values and row count."""
    occurrences = [
        {"gbifID": str(index), "geodeticDatum": value}
        for index, value in enumerate(values)
    ]
    media = [
        {
            "gbifID": str(index),
            "identifier": f"https://example.test/{index}.jpg",
            "geodeticDatum": value,
        }
        for index, value in enumerate(values)
    ]
    path = tmp_path / "indexed-defaults.zip"
    path.write_bytes(
        make_bulk_archive(
            occurrences=occurrences,
            media=media,
            field_defaults={table_tag: {"geodeticDatum": default}},
        )
    )

    records = GbifArchiveReader(path).read()

    column = "geodeticDatum" if table_tag == "core" else "media_geodeticDatum"
    expected = [value if value != "" else (default or None) for value in values]
    actual = records[column].astype(object).where(records[column].notna(), None)
    assert actual.tolist() == expected
    assert records["_core_id"].tolist() == [str(index) for index in range(len(values))]


@pytest.mark.parametrize("empty_table", ["core", "extension"])
def test_archive_constant_defaults_do_not_create_rows_for_empty_tables(
    tmp_path, empty_table
):
    """Defaults preserve empty results and their declared columns."""
    path = tmp_path / "empty-defaults.zip"
    path.write_bytes(
        make_bulk_archive(
            occurrences=[] if empty_table == "core" else None,
            media=[] if empty_table == "extension" else None,
            field_defaults={
                "core": {"geodeticDatum": "WGS84"},
                "extension": {"identifier": "https://example.test/default.jpg"},
            },
        )
    )

    records = GbifArchiveReader(path).read()

    assert records.empty
    assert "geodeticDatum" in records.columns
    assert "media_identifier" in records.columns


@pytest.fixture
def bulk_api(monkeypatch):
    """Mock the installed SDK signatures and prevent any real network traffic."""
    api = BulkDownloadApi()
    monkeypatch.setenv("GBIF_USER", "synthetic-user")
    monkeypatch.setenv("GBIF_PWD", "synthetic-password")
    monkeypatch.setenv("GBIF_EMAIL", "synthetic@example.invalid")
    # Autospec enforces the installed SDK signatures, while side effects supply
    # controlled results. It cannot detect drift in response shapes or behavior;
    # recheck those against pygbif when changing the dependency version.
    for method in ["download", "download_meta", "download_get"]:
        monkeypatch.setattr(
            gbif_occurrences,
            method,
            create_autospec(
                getattr(gbif_occurrences, method), side_effect=getattr(api, method)
            ),
        )
    monkeypatch.setattr(
        gbif_species,
        "name_backbone",
        create_autospec(gbif_species.name_backbone, side_effect=api.name_backbone),
    )
    monkeypatch.setattr(
        gbif_module.registry,
        "datasets",
        create_autospec(gbif_module.registry.datasets, side_effect=api.source_dataset),
    )
    monkeypatch.setattr(
        gbif_occurrences,
        "search",
        Mock(side_effect=AssertionError("Use pygbif.download, not occurrence search")),
    )
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(
            side_effect=AssertionError(
                "Unexpected network traffic outside pygbif mocks"
            )
        ),
    )
    return api


@pytest.fixture
def bulk_clock(monkeypatch):
    """Replace sleeps and elapsed-time checks with a deterministic clock."""
    clock = BulkDownloadClock()
    monkeypatch.setattr(time, "sleep", clock.sleep)
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    return clock


def make_bulk_dataset(tmp_path, **overrides):
    """Construct the proposed bulk downloader through its public configuration."""
    config = {
        "output_path": tmp_path / "dataset",
        "species": ["Mus musculus"],
        "years": [2022],
        "poll_interval": 2,
        "max_wait_seconds": 6,
    }
    config.update(overrides)
    return GbifDataset(**config)


def bulk_predicate_values(predicate, key):
    """Collect values for a predicate field without depending on tree ordering."""
    values = set()
    if predicate.get("key") == key:
        values.update(str(value) for value in predicate.get("values", []))
        if "value" in predicate:
            values.add(str(predicate["value"]))
    for child in predicate.get("predicates", []):
        values.update(bulk_predicate_values(child, key))
    return values


@pytest.mark.parametrize("species_count", [1, 2, 20])
def test_bulk_download_submits_one_dwca_request_per_species(
    tmp_path, bulk_api, species_count
):
    """Each species gets its own SDK download covering exactly its years and media."""
    names = [f"Synthetic species{index}" for index in range(species_count)]
    keys = [str(100 + index) for index in range(species_count)]
    for name, key in zip(names, keys, strict=True):
        bulk_api.matches[name] = make_bulk_match(name, key=key)
    dataset = make_bulk_dataset(tmp_path, species=names, years=[2020, 2022])

    dataset.retrieve_records()

    submissions = bulk_api.submissions()
    assert len(submissions) == species_count
    assert len(set(bulk_api.keys)) == species_count
    requested_taxa = []
    for _, body in submissions:
        assert body["format"] == "DWCA"
        taxa = bulk_predicate_values(body["predicate"], "TAXON_KEY")
        assert len(taxa) == 1
        requested_taxa.append(taxa)
        assert bulk_predicate_values(body["predicate"], "YEAR") == {"2020", "2022"}
        assert bulk_predicate_values(body["predicate"], "MEDIA_TYPE") == {"StillImage"}
        assert not bulk_predicate_values(body["predicate"], "SCIENTIFIC_NAME")
    assert requested_taxa == [{key} for key in keys]
    assert [call[1] for call in bulk_api.calls if call[0] == "match"] == names
    assert [call[1] for call in bulk_api.calls if call[0] == "archive"] == bulk_api.keys
    for species in dataset.species:
        path = dataset.output_path / species
        assert (path / "records.csv").is_file()
        assert not (path / "imgs").exists()


def test_bulk_accepted_match_without_accepted_usage_uses_its_own_key(
    tmp_path, bulk_api
):
    """An already accepted name need not include a separate acceptedUsage object."""
    match = make_bulk_match()
    del match["acceptedUsage"]
    bulk_api.matches["Mus musculus"] = match
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    assert bulk_predicate_values(
        bulk_api.submitted_body()["predicate"], "TAXON_KEY"
    ) == {"100"}
    assert len(pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")) == 1


def test_bulk_synonyms_use_accepted_key_in_separate_jobs_and_keep_directory_labels(
    tmp_path, bulk_api
):
    """Each input label gets its own job, including synonyms of the same taxon."""
    names = ["Clethrionomys glareolus", "Myodes glareolus"]
    bulk_api.matches[names[0]] = make_bulk_match(
        names[0], key="101", accepted_name=names[1]
    )
    bulk_api.matches[names[1]] = make_bulk_match(names[1])
    occurrence = {
        "gbifID": "123",
        "speciesKey": "100",
        "taxonKey": "100",
        "species": names[1],
        "scientificName": names[1],
        "year": "2022",
    }
    bulk_api.archive = make_bulk_archive(occurrences=[occurrence])
    dataset = make_bulk_dataset(tmp_path, species=names)

    dataset.retrieve_records()

    assert len(bulk_api.submissions()) == len(names)
    for _, body in bulk_api.submissions():
        assert bulk_predicate_values(body["predicate"], "TAXON_KEY") == {"100"}
    for name in names:
        records = pd.read_csv(dataset.output_path / name / "records.csv")
        assert records["source_group_id"].tolist() == [123]


@pytest.mark.parametrize(
    "problem",
    ["ambiguous", "none", "fuzzy", "higher_rank", "wrong_rank", "missing_key"],
)
def test_bulk_rejects_unreliable_taxon_matches(tmp_path, bulk_api, problem):
    """An unreliable identification cannot silently broaden the download."""
    match = make_bulk_match()
    if problem == "ambiguous":
        match["diagnostics"]["note"] = "Multiple equal matches for name"
    elif problem == "none":
        match["diagnostics"]["matchType"] = "NONE"
    elif problem == "fuzzy":
        match["diagnostics"]["matchType"] = "FUZZY"
    elif problem == "higher_rank":
        match["diagnostics"]["matchType"] = "HIGHERRANK"
    elif problem == "wrong_rank":
        match["usage"]["rank"] = "GENUS"
        match["acceptedUsage"]["rank"] = "GENUS"
    else:
        match["synonym"] = True
        del match["acceptedUsage"]["key"]
    bulk_api.matches["Mus musculus"] = match
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match="Mus musculus"):
        dataset.retrieve_records()

    assert not bulk_api.submissions()
    assert not list(dataset.output_path.rglob("records.csv"))


@pytest.mark.parametrize("variable", ["GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_bulk_requires_nonblank_credentials_before_submission(
    tmp_path, bulk_api, monkeypatch, variable, value
):
    """Missing credentials produce actionable errors without creating a job."""
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match=variable):
        dataset.retrieve_records()

    assert not bulk_api.submissions()


def test_bulk_does_not_persist_or_log_credentials(tmp_path, bulk_api, caplog):
    """Account details returned by pygbif do not leak into records or logs."""
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    assert len(bulk_api.submissions()) == 1
    text = caplog.text + "\n".join(
        path.read_text(encoding="utf-8")
        for path in dataset.output_path.rglob("*")
        if path.is_file() and path.suffix in {".json", ".log", ".yaml", ".csv"}
    )
    assert "synthetic-password" not in text
    assert "synthetic-user" not in text
    assert "synthetic@example.invalid" not in text


@pytest.mark.parametrize("setting", ["poll_interval", "max_wait_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bulk_rejects_invalid_polling_configuration(tmp_path, setting, value):
    """Polling settings must be finite and positive."""
    with pytest.raises(ValueError, match=setting):
        make_bulk_dataset(tmp_path, **{setting: value})


def test_bulk_config_loads_download_settings(tmp_path):
    """YAML forwards non-secret bulk-download settings to the loader."""
    config = {
        "data": {
            "gbif": {
                "output_path": str(tmp_path / "dataset"),
                "species": ["Mus musculus"],
                "years": [2022],
                "poll_interval": 5,
                "max_wait_seconds": 60,
            }
        }
    }
    path = tmp_path / "bulk-config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    dataset = GbifDataset.from_config(path)

    assert dataset.poll_interval == 5
    assert dataset.max_wait_seconds == 60


def test_bulk_polling_waits_and_completes(tmp_path, bulk_api, bulk_clock):
    """Preparation and running states are polled at the configured interval."""
    bulk_api.statuses = ["PREPARING", "RUNNING", "SUCCEEDED"]
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    assert bulk_api.polls == 3
    assert bulk_clock.waits == [2, 2]
    assert len(bulk_api.submissions()) == 1


def test_bulk_timeout_prevents_archive_and_image_downloads(
    tmp_path, bulk_api, bulk_clock
):
    """An unfinished job times out without producing records or starting images."""
    bulk_api.statuses = ["RUNNING"]
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(TimeoutError):
        dataset.download()

    assert bulk_clock.now <= dataset.max_wait_seconds
    assert bulk_clock.waits
    assert len(bulk_api.submissions()) == 1
    assert not [call for call in bulk_api.calls if call[0] == "archive"]
    assert not list(dataset.output_path.rglob("records.csv"))
    assert not list(dataset.output_path.rglob("imgs"))


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "KILLED"])
def test_bulk_terminal_failure_prevents_image_downloads(tmp_path, bulk_api, status):
    """An unsuccessful job cannot be mistaken for a completed archive."""
    bulk_api.statuses = [status]
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(RuntimeError, match=status):
        dataset.download()

    assert len(bulk_api.submissions()) == 1
    assert not list(dataset.output_path.rglob("records.csv"))
    assert not list(dataset.output_path.rglob("imgs"))


@pytest.mark.parametrize(
    ("metadata", "expected_exception"),
    [({}, ValueError), ({"status": "UNRECOGNIZED"}, ValueError), ([], TypeError)],
)
def test_bulk_rejects_malformed_job_metadata(
    tmp_path,
    bulk_api,
    metadata,
    expected_exception,
):
    """Invalid responses or statuses fail explicitly rather than looping."""
    bulk_api.malformed_metadata = metadata
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(expected_exception, match="(?i)(status|download|metadata|link)"):
        dataset.retrieve_records()

    assert not list(dataset.output_path.rglob("records.csv"))


def test_bulk_metadata_http_error_prevents_record_creation(tmp_path, bulk_api):
    """An SDK polling error propagates without starting archive or image downloads."""
    bulk_api.metadata_error = requests.HTTPError("HTTP 503 while polling")
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(requests.HTTPError, match="503"):
        dataset.download()

    assert len(bulk_api.submissions()) == 1
    assert not [call for call in bulk_api.calls if call[0] == "archive"]
    assert not list(dataset.output_path.rglob("records.csv"))
    assert not list(dataset.output_path.rglob("imgs"))


def test_bulk_submission_error_propagates_without_record_creation(tmp_path, bulk_api):
    """An SDK submission error fails the current run without requiring recovery state."""
    bulk_api.submission_error = requests.Timeout("Submission response lost")
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(requests.Timeout):
        dataset.retrieve_records()

    assert len(bulk_api.submissions()) == 1
    assert not [call for call in bulk_api.calls if call[0] in {"metadata", "archive"}]
    assert not list(dataset.output_path.rglob("records.csv"))


def test_bulk_all_cached_records_need_no_network_or_credentials(
    tmp_path, bulk_api, monkeypatch
):
    """An existing readable CSV remains usable completely offline."""
    dataset = make_bulk_dataset(tmp_path)
    path = dataset.output_path / "Mus musculus" / "records.csv"
    path.parent.mkdir()
    pd.DataFrame({"source_group_id": [123]}).to_csv(path, index=False)
    original = path.read_bytes()
    monkeypatch.delenv("GBIF_USER")
    monkeypatch.delenv("GBIF_PWD")

    dataset.retrieve_records()

    assert not bulk_api.calls
    assert path.read_bytes() == original


def test_bulk_preserves_cached_species_when_retrieving_missing_species(
    tmp_path, bulk_api
):
    """A partial cache is preserved while only missing species are requested."""
    dataset = make_bulk_dataset(tmp_path, species=["Mus musculus", "Rattus rattus"])
    path = dataset.output_path / "Mus musculus" / "records.csv"
    path.parent.mkdir()
    pd.DataFrame({"source_group_id": [999]}).to_csv(path, index=False)
    original = path.read_bytes()
    bulk_api.matches["Rattus rattus"] = make_bulk_match("Rattus rattus", key="200")
    bulk_api.archive = make_bulk_archive(
        occurrences=[
            {
                "gbifID": "123",
                "speciesKey": "200",
                "taxonKey": "200",
                "species": "Rattus rattus",
                "scientificName": "Rattus rattus",
                "year": "2022",
            }
        ]
    )

    dataset.retrieve_records()

    assert path.read_bytes() == original
    assert len(bulk_api.submissions()) == 1
    assert bulk_predicate_values(
        bulk_api.submitted_body()["predicate"], "TAXON_KEY"
    ) == {"200"}
    assert [call[1] for call in bulk_api.calls if call[0] == "match"] == [
        "Rattus rattus"
    ]
    missing_records = pd.read_csv(dataset.output_path / "Rattus rattus" / "records.csv")
    assert missing_records["source_group_id"].tolist() == [123]


def test_bulk_archive_preserves_image_level_schema_and_attribution(tmp_path, bulk_api):
    """Archive conversion retains URLs, licenses, attribution, and source IDs."""
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert list(records.columns) == list(dataset._record_columns())
    assert len(records) == 1
    record = records.iloc[0]
    assert record["source_group_id"] == 123
    assert record["image_url"] == "https://example.test/123.jpg"
    assert record["image_license"] == "http://creativecommons.org/licenses/by-nc/4.0/"
    assert record["image_creator"] == "Synthetic photographer"
    assert record["image_references"] == "https://example.test/photo/123"
    assert record["image_format"] == "image/jpeg"
    assert record["occurrence_source_url"] == "https://example.test/occurrence/123"
    assert record["dataset_key"] == "synthetic-dataset"
    assert record["dataset_citation"] == "Synthetic source dataset citation"
    assert record["dataset_doi"] == "10.1234/synthetic-source"
    assert pd.isna(record["decimal_latitude"])
    assert not (dataset.output_path / "Mus musculus" / "imgs").exists()


@pytest.mark.parametrize(
    "image_license", [None, "", "http://creativecommons.org/licenses/by/4.0/"]
)
def test_bulk_never_substitutes_occurrence_license_for_image_license(
    tmp_path, bulk_api, image_license
):
    """An open occurrence does not make its photos eligible under another license."""
    media = [
        {
            "gbifID": "123",
            "identifier": "https://example.test/photo.jpg",
            "type": "StillImage",
            "license": image_license,
        }
    ]
    bulk_api.archive = make_bulk_archive(media=media)
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert records.empty
    assert list(records.columns) == list(dataset._record_columns())


@pytest.mark.parametrize("omitted", ["meta.xml", "occurrence.txt", "multimedia.txt"])
def test_bulk_rejects_archives_missing_required_tables(tmp_path, bulk_api, omitted):
    """An incomplete archive cannot be silently interpreted as empty results."""
    bulk_api.archive = make_bulk_archive(omit=[omitted])
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match=omitted):
        dataset.retrieve_records()

    assert not list(dataset.output_path.rglob("records.csv"))


@pytest.mark.parametrize(
    "problem",
    [
        "invalid_xml",
        "missing_core",
        "missing_media",
        "missing_location",
        "missing_coreid",
        "negative_index",
        "wide_index",
        "missing_field_index_and_default",
    ],
)
def test_bulk_rejects_invalid_archive_descriptors(tmp_path, bulk_api, problem):
    """Broken table mappings fail before any image records are saved."""
    buffer = BytesIO()
    with ZipFile(BytesIO(bulk_api.archive)) as source, ZipFile(buffer, "w") as target:
        descriptor = ET.fromstring(source.read("meta.xml"))
        core = descriptor.find("{*}core")
        media = descriptor.find("{*}extension")
        if problem == "missing_core":
            descriptor.remove(core)
        elif problem == "missing_media":
            descriptor.remove(media)
        elif problem == "missing_location":
            core.find("{*}files").remove(core.find("{*}files/{*}location"))
        elif problem == "missing_coreid":
            media.remove(media.find("{*}coreid"))
        elif problem == "negative_index":
            core.find("{*}id").set("index", "-1")
        elif problem == "wide_index":
            core.find("{*}id").set("index", "999")
        elif problem == "missing_field_index_and_default":
            del core.find("{*}field").attrib["index"]
        elif problem == "invalid_xml":
            pass  # Supply invalid bytes below rather than build invalid ElementTree nodes.
        else:
            raise AssertionError(f"Unhandled descriptor problem: {problem}")
        metadata = b"<unclosed" if problem == "invalid_xml" else ET.tostring(descriptor)
        for name in source.namelist():
            target.writestr(name, metadata if name == "meta.xml" else source.read(name))
    bulk_api.archive = buffer.getvalue()
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match="(?i)(archive|table|location|index|meta)"):
        dataset.retrieve_records()

    assert not list(dataset.output_path.rglob("records.csv"))


@pytest.mark.parametrize("empty_table", ["occurrence.txt", "multimedia.txt"])
def test_bulk_header_only_archive_tables_produce_empty_records(
    tmp_path, bulk_api, empty_table
):
    """Present tables with declared columns but no rows are valid empty results."""
    buffer = BytesIO()
    with ZipFile(BytesIO(bulk_api.archive)) as source, ZipFile(buffer, "w") as target:
        for name in source.namelist():
            content = source.read(name)
            if name == empty_table:
                content = content.splitlines(keepends=True)[0]
            target.writestr(name, content)
    bulk_api.archive = buffer.getvalue()
    dataset = make_bulk_dataset(tmp_path)

    dataset.retrieve_records()

    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert records.empty
    assert list(records.columns) == list(dataset._record_columns())


def test_bulk_rejects_corrupt_zip(tmp_path, bulk_api):
    """A successful job with a corrupt archive is not treated as usable data."""
    bulk_api.archive = b"not a zip archive"
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match="(?i)(archive|zip)"):
        dataset.retrieve_records()

    assert not list(dataset.output_path.rglob("records.csv"))


def test_bulk_rejects_unsafe_archive_locations(tmp_path, bulk_api):
    """A malicious table location cannot escape the downloader output area."""
    buffer = BytesIO()
    with ZipFile(BytesIO(bulk_api.archive)) as source, ZipFile(buffer, "w") as target:
        for name in source.namelist():
            content = source.read(name)
            if name == "meta.xml":
                content = content.replace(b"occurrence.txt", b"../outside.txt")
            target.writestr(name, content)
        target.writestr("../outside.txt", "malicious")
    bulk_api.archive = buffer.getvalue()
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match="(?i)(unsafe|path|location)"):
        dataset.retrieve_records()

    assert not (tmp_path / "outside.txt").exists()
    assert not (dataset.output_path / "outside.txt").exists()
    assert not list(dataset.output_path.rglob("records.csv"))


def test_bulk_partial_archive_transfer_does_not_create_records(tmp_path, bulk_api):
    """A failed SDK archive transfer is not mistaken for usable image records."""
    bulk_api.archive_error = requests.ConnectionError("Archive interrupted")
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(requests.ConnectionError, match="interrupted"):
        dataset.download()

    assert len(bulk_api.submissions()) == 1
    assert not list(dataset.output_path.rglob("records.csv"))
    assert not list(dataset.output_path.rglob("imgs"))


def test_bulk_csv_write_failure_does_not_leave_partial_records(
    tmp_path, bulk_api, monkeypatch
):
    """A failed CSV write cannot leave a partial records cache for the next run."""
    dataset = make_bulk_dataset(tmp_path)

    def interrupted_write(frame, destination, *args, **kwargs):
        """Write partial bytes before simulating a storage failure."""
        if isinstance(destination, (str, Path)):
            Path(destination).write_text("partial", encoding="utf-8")
        else:
            destination.write("partial")
        raise OSError("Synthetic disk full")

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_csv", interrupted_write)
        with pytest.raises(OSError, match="disk full"):
            dataset.retrieve_records()

    records_path = dataset.output_path / "Mus musculus" / "records.csv"
    assert not records_path.exists()


def test_bulk_records_feed_existing_image_download_workflow(
    tmp_path, bulk_api, monkeypatch
):
    """Converted records remain compatible with the unchanged image phase."""
    dataset = make_bulk_dataset(tmp_path)
    session = Mock()
    session.get.return_value = make_image_response()
    dataset.retrieve_records()
    monkeypatch.setattr(gbif_module.requests, "Session", Mock(return_value=session))

    dataset.download_images()

    session.get.assert_called_once()
    path = dataset.output_path / "Mus musculus"
    assert len(list((path / "imgs").glob("*.jpg"))) == 1
    report = pd.read_csv(path / "download_report.csv")
    assert report["success"].tolist() == [True]
    assert report["source_group_id"].tolist() == [123]


@pytest.mark.parametrize("preexisting", [False, True])
def test_bulk_script_loads_dotenv_without_overriding_environment(
    tmp_path, monkeypatch, preexisting
):
    """The entry point loads an explicit .env before constructing the loader."""
    from scripts import download_gbif_data as entrypoint

    env_path = tmp_path / ".env"
    env_path.write_text(
        "GBIF_USER=file-user\nGBIF_PWD=file-password\nGBIF_EMAIL=file@example.invalid\n",
        encoding="utf-8",
    )
    for variable in ["GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"]:
        monkeypatch.delenv(variable, raising=False)
    if preexisting:
        monkeypatch.setenv("GBIF_USER", "environment-user")
        monkeypatch.setenv("GBIF_PWD", "environment-password")
    client = Mock()
    observed = {}

    def construct(config_path):
        """Inspect credentials as visible when the loader is constructed."""
        import os

        observed.update(
            {
                name: os.environ.get(name)
                for name in ["GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"]
            }
        )
        return client

    monkeypatch.setattr(entrypoint.GbifDataset, "from_config", construct)

    entrypoint.main(tmp_path / "config.yaml", env_file=env_path)

    assert observed["GBIF_USER"] == ("environment-user" if preexisting else "file-user")
    assert observed["GBIF_PWD"] == (
        "environment-password" if preexisting else "file-password"
    )
    assert observed["GBIF_EMAIL"] == "file@example.invalid"
    client.download.assert_called_once_with()


def test_bulk_missing_explicit_env_file_fails_clearly(tmp_path, monkeypatch):
    """An explicitly selected nonexistent .env is not silently ignored."""
    from scripts import download_gbif_data as entrypoint

    construct = Mock()
    monkeypatch.setattr(entrypoint.GbifDataset, "from_config", construct)

    with pytest.raises(FileNotFoundError, match="missing.env"):
        entrypoint.main(tmp_path / "config.yaml", env_file=tmp_path / "missing.env")

    construct.assert_not_called()


def test_bulk_rejects_archive_without_image_identifiers(tmp_path, bulk_api):
    """Missing a required media column is an archive error, not an empty sample."""
    bulk_api.archive = make_bulk_archive(
        media=[
            {
                "gbifID": "123",
                "type": "StillImage",
                "license": "cc-by-nc",
            }
        ]
    )
    dataset = make_bulk_dataset(tmp_path)

    with pytest.raises(ValueError, match="(?i)identifier"):
        dataset.retrieve_records()

    assert not list(dataset.output_path.rglob("records.csv"))


@settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    years=st.lists(
        st.integers(min_value=1900, max_value=2026), min_size=1, max_size=6, unique=True
    )
)
def test_bulk_year_predicate_preserves_exact_year_set(bulk_api, years):
    """Arbitrary sparse year selections are never broadened to a range."""
    bulk_api.calls.clear()
    bulk_api.keys.clear()
    bulk_api.polls = 0
    with TemporaryDirectory() as output_path:
        dataset = make_bulk_dataset(Path(output_path), years=years)

        dataset.retrieve_records()

        body = bulk_api.submitted_body()
        assert bulk_predicate_values(body["predicate"], "YEAR") == {
            str(year) for year in years
        }
        assert len(bulk_api.submissions()) == 1


@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    accepted_keys=st.lists(
        st.sampled_from(["100", "200", "300"]), min_size=1, max_size=5
    )
)
def test_bulk_each_species_requests_its_accepted_taxon_key(bulk_api, accepted_keys):
    """Input species get separate jobs even when their accepted taxon keys coincide."""
    bulk_api.calls.clear()
    bulk_api.keys.clear()
    bulk_api.polls = 0
    bulk_api.matches.clear()
    names = [f"Synthetic species{index}" for index in range(len(accepted_keys))]
    for index, (name, accepted_key) in enumerate(
        zip(names, accepted_keys, strict=True)
    ):
        match = make_bulk_match(
            name, key=str(1000 + index), accepted_name=f"Resolved species{accepted_key}"
        )
        match["acceptedUsage"]["key"] = accepted_key
        match["classification"][0]["key"] = accepted_key
        bulk_api.matches[name] = match
    with TemporaryDirectory() as output_path:
        dataset = make_bulk_dataset(Path(output_path), species=names)

        dataset.retrieve_records()

        submissions = bulk_api.submissions()
        assert len(submissions) == len(names)
        assert [
            bulk_predicate_values(body["predicate"], "TAXON_KEY")
            for _, body in submissions
        ] == [{key} for key in accepted_keys]
        assert all(
            (dataset.output_path / name / "records.csv").is_file() for name in names
        )


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    licenses=st.lists(
        st.sampled_from(["cc-by-nc", "cc-by", "cc0", "", None]), max_size=12
    ),
    cap=st.integers(min_value=0, max_value=12),
    reverse_columns=st.booleans(),
)
def test_bulk_media_join_filter_and_sampling_properties(
    bulk_api, licenses, cap, reverse_columns
):
    """Joined eligible images retain their occurrence IDs across column orders."""
    bulk_api.calls.clear()
    bulk_api.keys.clear()
    bulk_api.polls = 0
    bulk_api.matches.clear()
    occurrences = [
        {
            "gbifID": str(key),
            "speciesKey": "100",
            "taxonKey": "100",
            "species": "Mus musculus",
            "year": "2022",
        }
        for key in [123, 124]
    ]
    media = [
        {
            "gbifID": str(123 + index % 2),
            "type": "StillImage",
            "identifier": f"https://example.test/{index}.jpg",
            "license": license,
        }
        for index, license in enumerate(licenses)
    ]
    # Keep required columns in the header even when there are no actual media.
    if not media:
        media = [
            {"gbifID": "123", "type": "StillImage", "identifier": "", "license": ""}
        ]
    eligible = {
        f"https://example.test/{index}.jpg": 123 + index % 2
        for index, license in enumerate(licenses)
        if license == "cc-by-nc"
    }
    bulk_api.archive = make_bulk_archive(
        occurrences=occurrences, media=media, reverse_columns=reverse_columns
    )
    with TemporaryDirectory() as output_path:
        dataset = make_bulk_dataset(Path(output_path), max_img_num=cap, seed=17)

        dataset.retrieve_records()

        records_path = dataset.output_path / "Mus musculus" / "records.csv"
        records = pd.read_csv(records_path)
        assert len(records) == min(cap, len(eligible))
        assert records["image_url"].is_unique
        assert set(records["image_url"]) <= set(eligible)
        for _, row in records.iterrows():
            assert row["source_group_id"] == eligible[row["image_url"]]
        # Independent downloads with the same seed must produce the same sample.
        repeated = make_bulk_dataset(
            Path(output_path) / "repeated", max_img_num=cap, seed=17
        )
        repeated.retrieve_records()
        pd.testing.assert_frame_equal(
            pd.read_csv(repeated.output_path / "Mus musculus" / "records.csv"), records
        )
        # Mapping the opposite column order must preserve that same output.
        bulk_api.archive = make_bulk_archive(
            occurrences=occurrences, media=media, reverse_columns=not reverse_columns
        )
        reordered = make_bulk_dataset(
            Path(output_path) / "reordered", max_img_num=cap, seed=17
        )
        reordered.retrieve_records()
        pd.testing.assert_frame_equal(
            pd.read_csv(reordered.output_path / "Mus musculus" / "records.csv"), records
        )
        assert len(bulk_api.submissions()) == 3
