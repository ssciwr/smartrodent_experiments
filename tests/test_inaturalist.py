import logging
import runpy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import yaml
from bundle_test_support import move_bundle, write_stage_records
from http_test_support import DROP

from smartrodent.inaturalist import InaturalistDataset


@pytest.fixture
def inat_config(tmp_path):
    """Return a minimal, complete iNaturalist configuration mapping."""
    return {
        "data": {
            "inaturalist": {
                "output_path": str(tmp_path / "dataset"),
                "species": ["Mus musculus"],
                "years": [],
                "first_year": 2020,
                "last_year": 2022,
                "quality_grade": "needs_id",
                "max_img_num": 5,
                "max_observations": 8,
                "allowed_licenses": ["cc-by-nc", "cc0"],
                "cache_file": str(tmp_path / "api_requests.db"),
                "ratelimit_path": str(tmp_path / "api_ratelimit.db"),
                "backoff_factor": 0.75,
                "max_retries": 2,
            }
        }
    }


@pytest.fixture
def config_file(tmp_path, inat_config):
    """Write ``inat_config`` to disk and return its path."""
    path = tmp_path / "dataset_test.yaml"
    path.write_text(yaml.safe_dump(inat_config), encoding="utf-8")
    return path


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
        (
            {"years": [2022], "first_year": 2020, "last_year": 2023},
            "mutually exclusive",
        ),
        # half a range next to explicit years must not be silently ignored
        ({"years": [2022], "first_year": 2020}, "mutually exclusive"),
        ({"first_year": 2023, "last_year": 2020}, "less than or equal"),
    ],
)
def test_inaturalist_constructor_rejects_invalid_years(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        InaturalistDataset(
            output_path=tmp_path / "test", species=["Mus musculus"], **kwargs
        )


def test_inaturalist_from_config(config_file):
    """The configured year range and licenses reach the API, and the file is kept."""
    dataset = InaturalistDataset.from_config(config_file)

    assert dataset.years == [2020, 2021, 2022]
    assert dataset.session.params["photo_license"] == "cc-by-nc,cc0"
    assert (dataset.output_path / config_file.name).is_file()


@pytest.mark.network
def test_inaturalist_downloads_two_real_images(tmp_path):
    """Download a small, real iNaturalist sample without an uncapped request."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        quality_grade="research",
        max_img_num=2,
        max_observations=10,
    )

    dataset.download()

    species_path = dataset.output_path / "Mus musculus"
    records = pd.read_csv(species_path / "records.csv")
    images = list((species_path / "imgs").iterdir())
    assert not records.empty
    assert len(images) == 2
    assert all(image.stat().st_size > 0 for image in images)


@pytest.fixture
def dataset(tmp_path):
    """Construct an iNaturalist downloader with only temporary output paths."""
    return InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_download_processes_images_with_cached_or_fresh_records(
    dataset, monkeypatch, cached, empty
):
    """Even header-only caches are reused and reach the image phase."""
    records = pd.DataFrame({"id": pd.Series([] if empty else [123], dtype="int64")})
    species_path = dataset.output_path / "Mus musculus"
    records_path = species_path / "records.csv"
    original_bytes = None
    if cached:
        species_path.mkdir()
        records.to_csv(records_path, index=False)
        original_bytes = records_path.read_bytes()

    retrieve = Mock(return_value=records)
    download_images = Mock(return_value=(len(records), records))
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    monkeypatch.setattr(dataset, "_download_species_images", download_images)

    dataset.download()

    if cached:
        retrieve.assert_not_called()
        assert original_bytes is not None
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


@pytest.mark.parametrize("cached", [False, True])
def test_inaturalist_resumes_images_without_redownloading_existing_files(
    tmp_path, monkeypatch, http_server, cached
):
    """Real image processing accepts photo rows and skips existing files."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset", species=["Mus musculus"], years=[2022]
    )
    records = pd.DataFrame(
        {
            "id": [123, 123],
            "photo_index": [0, 1],
            "photo.url": [
                http_server.url(f"/{index}/square.jpg") for index in range(2)
            ],
            "photo.license_code": ["cc-by-nc", "cc-by-nc"],
        }
    )
    species_path = dataset.output_path / "Mus musculus"
    images_path = species_path / "imgs"
    images_path.mkdir(parents=True)
    existing_path = images_path / "123_0.jpg"
    existing_path.write_bytes(b"existing image")
    if cached:
        records.to_csv(species_path / "records.csv", index=False)
    retrieve = Mock(return_value=records)
    monkeypatch.setattr(dataset, "_get_species_records", retrieve)
    http_server.serve("/1/large.jpg", (200, b"new image"))

    dataset.download()

    assert existing_path.read_bytes() == b"existing image"
    assert (images_path / "123_1.jpg").read_bytes() == b"new image"
    assert http_server.requested == ["/1/large.jpg"]
    if cached:
        retrieve.assert_not_called()
    else:
        retrieve.assert_called_once_with("Mus musculus")


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
    download_images = Mock(return_value=(1, records))
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
        return len(received_records), received_records

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


def test_inaturalist_no_observations_can_cross_phase_boundary(tmp_path, monkeypatch):
    """No observations produce a readable empty cache and no image requests."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=8,
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )
    observations = Mock(return_value={"results": []})
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", observations)

    dataset.retrieve_records()

    species_path = dataset.output_path / "Mus musculus"
    records = pd.read_csv(species_path / "records.csv")
    assert records.empty
    assert list(records.columns) == [
        "id",
        "photo_index",
        "photo.url",
        "photo.license_code",
    ]
    assert not (species_path / "imgs").exists()

    dataset.download_images()

    observations.assert_called_once()
    assert not any((species_path / "imgs").iterdir())


def _photo_observation(observation_id, licenses, year=2022):
    """Build an API observation with stable per-photo identity."""
    return {
        "id": observation_id,
        "observed_on": f"{year}-05-10",
        "species_guess": "Mus musculus",
        "photos": [
            {
                "id": observation_id * 10 + index,
                "url": f"https://example.test/{observation_id}/{index}/square.jpg",
                "license_code": license_code,
            }
            for index, license_code in enumerate(licenses)
        ],
    }


def test_inaturalist_v2_retrieval_combines_filters_and_honors_budget(
    tmp_path, monkeypatch
):
    """All filters share bounded random v2 requests rather than per-year crawls."""
    observations = [_photo_observation(index, ("cc-by",)) for index in range(1, 202)]

    def get_page(**params):
        """Return exactly the explicitly requested page from a fixed population."""
        page_size = params["per_page"]
        offset = (params["page"] - 1) * 200
        return {
            "results": observations[offset : offset + page_size],
            "total_results": len(observations),
        }

    get_observations_v2 = Mock(side_effect=get_page)
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations_v2", get_observations_v2
    )
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2020, 2022],
        max_observations=201,
        max_img_num=3,
        allowed_licenses=["cc-by", "cc0"],
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )

    dataset.retrieve_records()

    assert get_observations_v2.call_count == 2
    assert [call.kwargs["per_page"] for call in get_observations_v2.call_args_list] == [
        200,
        1,
    ]
    for page, call in enumerate(get_observations_v2.call_args_list, start=1):
        assert call.kwargs["session"] is dataset.session
        assert call.kwargs["year"] == [2020, 2022]
        assert call.kwargs["photos"] is True
        assert call.kwargs["order_by"] == "random"
        assert call.kwargs["page"] == page
        assert call.kwargs["fields"] == "all"
    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert len(records) == 3


def test_inaturalist_filters_licenses_before_truncating_photo_rows(
    tmp_path, monkeypatch
):
    """Disallowed photos cannot displace eligible photos in the retained sample."""
    observation = _photo_observation(101, (None, "cc-by-nc", "cc-by", "cc0"))
    get_observations_v2 = Mock(
        return_value={"results": [observation], "total_results": 1}
    )
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations_v2", get_observations_v2
    )
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=1,
        max_img_num=2,
        allowed_licenses=["cc-by", "cc0"],
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )

    dataset.retrieve_records()

    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert len(records) == 2
    assert set(records["photo.license_code"]) == {"cc-by", "cc0"}
    assert set(records["photo_index"]) == {2, 3}
    assert "photos" not in records.columns


def test_duplicate_observations_consume_the_explicit_budget(tmp_path, monkeypatch):
    """Duplicate random results do not permit retrieval beyond the configured budget."""
    duplicate = _photo_observation(101, ("cc-by",))
    observations = Mock(
        return_value={
            "results": [duplicate, duplicate, _photo_observation(102, ("cc-by",))],
            "total_results": 100,
        }
    )
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", observations)
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=3,
        allowed_licenses=["cc-by"],
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )

    dataset.retrieve_records()

    observations.assert_called_once()
    records = pd.read_csv(dataset.output_path / "Mus musculus" / "records.csv")
    assert not records.duplicated(["id", "photo_index"]).any()


@pytest.mark.parametrize("failure", [None, RuntimeError("download failed")])
def test_download_removes_response_cache_but_retains_rate_accounting(
    tmp_path, monkeypatch, failure
):
    """Full-download cleanup runs on both success and ordinary failure."""
    cache_file = tmp_path / "api_requests.db"
    ratelimit_path = tmp_path / "api_ratelimit.db"
    unrelated = tmp_path / "unrelated.db"
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=1,
        cache_file=cache_file,
        ratelimit_path=ratelimit_path,
    )
    cache_file.touch(exist_ok=True)
    ratelimit_path.touch(exist_ok=True)
    unrelated.write_bytes(b"unrelated")
    retrieve_records = Mock()
    download_images = Mock(side_effect=failure)
    monkeypatch.setattr(dataset, "retrieve_records", retrieve_records)
    monkeypatch.setattr(dataset, "download_images", download_images)

    if failure is None:
        dataset.download()
    else:
        with pytest.raises(RuntimeError, match="download failed"):
            dataset.download()

    retrieve_records.assert_called_once()
    download_images.assert_called_once()
    assert not cache_file.exists()
    assert ratelimit_path.exists()
    assert unrelated.read_bytes() == b"unrelated"


def _photo_rows(http_server, count):
    """Photo rows whose thumbnail URLs point at the local server."""
    return pd.DataFrame(
        {
            "id": [100 + index for index in range(count)],
            "photo_index": [0] * count,
            "photo.url": [
                http_server.url(f"/{100 + index}/square.jpg") for index in range(count)
            ],
        }
    )


def test_each_photo_retries_transient_failures_within_its_own_budget(
    tmp_path, http_server
):
    """One transient failure per photo fits a budget of one retry for each photo."""
    dataset = InaturalistDataset(
        tmp_path / "dataset",
        ["Mus musculus"],
        years=[2022],
        max_retries=1,
        backoff_factor=0,
    )
    http_server.serve("/100/large.jpg", (503, b""), (200, b"first"))
    http_server.serve("/101/large.jpg", DROP, (200, b"second"))

    count, returned = dataset._download_species_images(
        _photo_rows(http_server, 2), tmp_path
    )

    assert count == 2
    assert (tmp_path / "100_0.jpg").read_bytes() == b"first"
    assert (tmp_path / "101_0.jpg").read_bytes() == b"second"
    assert len(http_server.requested) == 4
    assert returned["image_path"].notna().all()


def test_persistent_photo_failure_raises_after_the_retry_budget(tmp_path, http_server):
    dataset = InaturalistDataset(
        tmp_path / "dataset",
        ["Mus musculus"],
        years=[2022],
        max_retries=1,
        backoff_factor=0,
    )
    http_server.serve("/100/large.jpg", (503, b""))

    with pytest.raises(requests.HTTPError):
        dataset._download_species_images(_photo_rows(http_server, 1), tmp_path)

    assert http_server.requested == ["/100/large.jpg", "/100/large.jpg"]
    assert not (tmp_path / "100_0.jpg").exists()


def test_record_request_errors_are_left_to_the_api_session(tmp_path, monkeypatch):
    """The pyinaturalist session already retries; failures are not retried again."""
    request = Mock(side_effect=requests.HTTPError("permanent failure"))
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", request)
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=1,
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )

    with pytest.raises(requests.HTTPError, match="permanent failure"):
        dataset.retrieve_records()

    request.assert_called_once()


def test_photo_rows_are_annotated_in_place_without_license_filtering(
    tmp_path, http_server
):
    """Image downloading handles only valid row data, not license policy."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    records = pd.DataFrame(
        {
            "id": [None, 123, 123, 123, 123, 123],
            "photo_index": [0, 0, 1, 2, 3, 4],
            "photo.url": [http_server.url("/square.jpg")] * 5 + [None],
            "photo.license_code": ["cc-by-nc"] * 4 + ["cc-by", "cc-by-nc"],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    images = tmp_path / "images"
    images.mkdir()
    (images / "123_0.jpg").write_bytes(b"existing")
    http_server.serve("/large.jpg", (200, b"new"))

    count, returned = dataset._download_species_images(records, images)

    assert returned is records
    assert count == 4
    assert len(http_server.requested) == 3
    assert pd.isna(records.at[10, "image_path"])
    for row, index in [(20, 0), (30, 1), (40, 2), (50, 3)]:
        assert records.at[row, "image_path"] == str(images / f"123_{index}.jpg")
    assert pd.isna(records.at[60, "image_path"])
    assert (images / "123_0.jpg").read_bytes() == b"existing"


def test_retrieval_flattens_each_photo_without_changing_the_row_schema(
    tmp_path, monkeypatch
):
    """Fresh v2 records retain the existing one-row-per-photo layout."""
    observations = Mock(
        return_value={
            "results": [
                {
                    "id": 123,
                    "photos": [
                        {"id": 11, "url": "first", "license_code": "cc-by-nc"},
                        {"id": 12, "url": "second", "license_code": "cc-by"},
                    ],
                }
            ],
            "total_results": 1,
        }
    )
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", observations)
    dataset = InaturalistDataset(
        tmp_path,
        ["Sorex minutus"],
        years=[2022],
        max_observations=1,
        allowed_licenses=["cc-by-nc", "cc-by"],
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )

    records = dataset._get_species_records("Sorex minutus").sort_values("photo_index")

    assert records["id"].tolist() == [123, 123]
    assert records["photo_index"].tolist() == [0, 1]
    assert records["photo.id"].tolist() == [11, 12]
    assert records["photo.url"].tolist() == ["first", "second"]
    assert "photos" not in records.columns


def test_missing_photo_columns_fail_explicitly(tmp_path):
    """An old observation-row table must not silently skip every image."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    with pytest.raises(KeyError, match="Missing photo columns"):
        dataset._download_species_images(pd.DataFrame({"id": [123]}), tmp_path)


def test_downloader_logs_through_the_package_logger_without_own_handlers(
    tmp_path, caplog
):
    """Handlers belong to the entry point; repeated construction must not stack them."""
    datasets = [
        InaturalistDataset(
            tmp_path / f"dataset_{index}", ["Mus musculus"], years=[2022]
        )
        for index in range(2)
    ]

    with caplog.at_level(logging.INFO):
        datasets[1].logger.info("probe message")

    assert caplog.messages.count("probe message") == 1
    assert all(dataset.logger.handlers == [] for dataset in datasets)


def test_download_script_logs_to_the_dataset_log_file(tmp_path, monkeypatch):
    """The entry point keeps the per-dataset log file that the class used to create.

    The script attaches its handlers to the ``smartrodent`` package logger. It cannot
    rely on ``logging.basicConfig``, which does nothing once the root logger already
    has handlers (as under pytest).
    """
    output_root = tmp_path / "downloaded"
    config_path = tmp_path / "download.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "inaturalist": {
                        "output_path": str(output_root),
                        "species": ["Mus musculus"],
                        "years": [2024],
                    }
                }
            }
        )
    )

    def download(dataset):
        dataset.logger.info("probe message")

    monkeypatch.setattr(InaturalistDataset, "download", download)
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "download_inaturalist_data.py"
    )
    monkeypatch.setattr("sys.argv", [str(script), str(config_path)])
    package_logger = logging.getLogger("smartrodent")
    handlers, level = list(package_logger.handlers), package_logger.level
    propagate = package_logger.propagate
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        for handler in set(package_logger.handlers) - set(handlers):
            handler.close()
            package_logger.removeHandler(handler)
        package_logger.setLevel(level)
        package_logger.propagate = propagate

    assert "probe message" in (output_root / "inaturalist_dataset.log").read_text()


def test_inaturalist_writes_shared_root_relative_paths_and_can_resume_after_move(
    bundle, tmp_path, monkeypatch
):
    """Original-image CSV references are portable and need no prefix migration."""
    dataset_root, records_by_species = bundle
    write_stage_records(dataset_root / "inaturalist", records_by_species)
    downloader = InaturalistDataset(
        output_path=dataset_root / "inaturalist",
        species=list(records_by_species),
        years=[2024],
    )
    unexpected_download = Mock(side_effect=AssertionError("Images already exist"))
    monkeypatch.setattr(downloader, "_download_photo", unexpected_download)

    try:
        downloader.download_images()
    finally:
        downloader.session.close()

    for species, original in records_by_species.items():
        actual = pd.read_csv(dataset_root / "inaturalist" / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all(not Path(path).is_absolute() for path in actual["image_path"])
    unexpected_download.assert_not_called()

    moved_root = move_bundle(dataset_root)
    resumed = InaturalistDataset(
        output_path=moved_root / "inaturalist",
        species=list(records_by_species),
        years=[2024],
    )
    monkeypatch.setattr(resumed, "_download_photo", unexpected_download)
    try:
        resumed.download_images()
    finally:
        resumed.session.close()
    for species, original in records_by_species.items():
        actual = pd.read_csv(moved_root / "inaturalist" / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all((moved_root / path).is_file() for path in actual["image_path"])
    unexpected_download.assert_not_called()
