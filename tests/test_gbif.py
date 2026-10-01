import hashlib
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import smartrodent.gbif as gbif_module
from smartrodent.gbif import GbifDataset
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
    managers_by_type = {
        type(manager): manager for manager in dataset._license_managers
    }
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

    records = dataset._get_all_records_for_params()

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
        side_effect=[
            [make_image_record(1, "cc-by-nc")],
            [make_image_record(2, "cc-by-nc")],
        ]
    )
    monkeypatch.setattr(dataset, "_get_all_records_for_params", get_all_records)

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 1
    assert tuple(records.columns) == dataset._record_columns()
    assert [call.kwargs["year"] for call in get_all_records.call_args_list] == [
        2021,
        2022,
    ]


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


def test_get_species_records_filters_each_year_before_applying_cap(
    tmp_path, monkeypatch
):
    dataset = make_dataset(
        tmp_path,
        max_img_num=2,
        allowed_licenses={"creative-commons": ["cc-by"]},
    )
    dataset.years = [2021, 2022]
    get_all_records = Mock(
        side_effect=[
            [
                make_image_record(1, "cc-by"),
                make_image_record(2, "cc-by-nc"),
            ],
            [
                make_image_record(3, "cc-by"),
                make_image_record(4, "cc-by-nc"),
            ],
        ]
    )
    monkeypatch.setattr(dataset, "_get_all_records_for_params", get_all_records)

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 2
    assert set(records["source_group_id"]) == {1, 3}
    assert [call.kwargs["year"] for call in get_all_records.call_args_list] == [
        2021,
        2022,
    ]


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
        record_id
        for record_id, is_allowed in enumerate(allowed_rows)
        if is_allowed
    ]
    assert sorted(records["source_group_id"].tolist()) == expected_ids


def test_download_image_uses_documented_gbif_cache_url(tmp_path):
    dataset = make_dataset(tmp_path)
    images_path = tmp_path / "images"
    images_path.mkdir()
    source_url = (
        "https://inaturalist-open-data.s3.amazonaws.com/"
        "photos/31610070/original.jpg"
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
    session.get.return_value = make_image_response(
        content, content_type=content_type
    )

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
def test_download_image_rejects_records_without_cache_identifiers(
    tmp_path, record
):
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
def test_download_image_rejects_non_image_responses(
    tmp_path, content, content_type
):
    dataset = make_dataset(tmp_path)
    session = Mock()
    session.get.return_value = make_image_response(
        content, content_type=content_type
    )

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

    success = dataset._download_image(
        make_download_record(), images_path, session
    )

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

    success = dataset._download_image(
        make_download_record(), images_path, session
    )

    assert success is False
    assert not list(images_path.glob("*"))


def test_download_species_images_reports_every_record_in_order(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        dataset, "_download_species_images", Mock(return_value=report)
    )
    dataset.logger = Mock()

    dataset.download()

    species_path = dataset.output_path / "Mus musculus"
    pd.testing.assert_frame_equal(
        pd.read_csv(species_path / "records.csv"), records
    )
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
        Mock(side_effect=error)
        if phase == "retrieval"
        else Mock(return_value=records)
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
    (species_path / "download_report.csv").write_text("outdated report", encoding="utf-8")
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
        pd.testing.assert_frame_equal(pd.read_csv(species_path / "records.csv"), records)
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
