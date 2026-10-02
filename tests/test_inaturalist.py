from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pyinaturalist.session import ClientSession

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
            "max_observations": 8,
            "allowed_licences": ["cc-by-nc", "cc0"],
            "cache_file": str(tmp_path / "api_requests.db"),
            "ratelimit_path": str(tmp_path / "api_ratelimit.db"),
            "backoff_factor": 0.75,
            "max_retries": 2,
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
    assert list(output_path.iterdir()) == [
        (dataset.output_path / "inaturalist_dataset.log").resolve()
    ]


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
        (
            {"species": ["Mus musculus"], "years": [2022], "max_observations": -1},
            "max_observations",
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
    assert dataset.max_observations == 8
    assert dataset.allowed_licenses == {"cc-by-nc", "cc0"}
    assert isinstance(dataset.session, ClientSession)
    assert Path(dataset.session.cache.db_path) == Path(settings["cache_file"])
    assert dataset.session.retries.total == 2
    assert dataset.session.retries.backoff_factor == 0.75
    assert dataset.session.params["photo_license"] == "cc-by-nc,cc0"
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


@given(observation_id=st.integers(min_value=1, max_value=2**31 - 1))
def test_inaturalist_photo_lists_survive_csv_round_trip(observation_id):
    """Cached nested photo metadata retains the values used by downloads."""
    photos = [
        {
            "url": f"https://example.test/{observation_id}/square.jpg",
            "license_code": "cc-by-nc",
        }
    ]
    records = pd.DataFrame({"id": [observation_id], "photos": [photos]})
    with TemporaryDirectory() as output_path:
        dataset = InaturalistDataset(
            output_path=output_path, species=["Mus musculus"], years=[2022]
        )
        species_path = dataset.output_path / "Mus musculus"
        species_path.mkdir()
        records.to_csv(species_path / "records.csv", index=False)
        dataset._get_species_records = Mock()
        dataset._download_species_images = Mock(return_value=(0, records))

        dataset.download()

        dataset._get_species_records.assert_not_called()
        received_records = dataset._download_species_images.call_args.args[0]
        pd.testing.assert_frame_equal(received_records, records)


@pytest.mark.parametrize("cached", [False, True])
def test_inaturalist_resumes_images_without_redownloading_existing_files(
    tmp_path, monkeypatch, cached
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
                f"https://example.test/{index}/square.jpg" for index in range(2)
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
    response = Mock(content=b"new image")
    get = Mock(return_value=response)
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    dataset.download()

    assert existing_path.read_bytes() == b"existing image"
    assert (images_path / "123_1.jpg").read_bytes() == b"new image"
    get.assert_called_once_with("https://example.test/1/large.jpg", timeout=30)
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
    image_request = Mock()
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", observations)
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", image_request)

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
    image_request.assert_not_called()
    assert (species_path / "imgs").is_dir()


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


def test_inaturalist_constructor_accepts_observation_budget_and_session_settings(
    tmp_path,
):
    """The observation budget and persistent session are explicit controls."""
    cache_file = tmp_path / "api_requests.db"
    ratelimit_path = tmp_path / "api_ratelimit.db"
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_img_num=3,
        max_observations=11,
        cache_file=cache_file,
        ratelimit_path=ratelimit_path,
        backoff_factor=0.75,
        max_retries=2,
    )

    assert dataset.max_img_num == 3
    assert dataset.max_observations == 11
    assert isinstance(dataset.session, ClientSession)
    assert Path(dataset.session.cache.db_path) == cache_file
    assert dataset.session.retries.total == 2
    assert dataset.session.retries.backoff_factor == 0.75


@pytest.mark.parametrize("max_observations", [1.5, True])
def test_inaturalist_rejects_non_integer_observation_budget(tmp_path, max_observations):
    """The API-request budget must be an integer and not a boolean."""
    with pytest.raises(ValueError, match="max_observations"):
        InaturalistDataset(
            output_path=tmp_path / "dataset",
            species=["Mus musculus"],
            years=[2022],
            max_observations=max_observations,
        )


@pytest.mark.parametrize("max_retries", [-1, 1.5, True])
def test_inaturalist_rejects_invalid_retry_budget(tmp_path, max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        InaturalistDataset(
            output_path=tmp_path / "dataset",
            species=["Mus musculus"],
            years=[2022],
            max_retries=max_retries,
        )


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


def test_photo_retry_budget_resets_after_each_successful_image(tmp_path, monkeypatch):
    """One transient failure on each photo stays within the per-photo budget."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_retries=1,
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )
    records = pd.DataFrame(
        {
            "id": [123, 456],
            "photo_index": [0, 0],
            "photo.url": [
                "https://example.test/first-square.jpg",
                "https://example.test/second-square.jpg",
            ],
        }
    )
    success_response = Mock(content=b"image")
    request = Mock(
        side_effect=[
            requests.HTTPError("first transient"),
            success_response,
            requests.HTTPError("second transient"),
            success_response,
        ]
    )
    sleep = Mock()
    images_path = tmp_path / "imgs"
    images_path.mkdir()
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", request)
    monkeypatch.setattr("smartrodent.inaturalist.time.sleep", sleep)

    count, returned = dataset._download_species_images(records, images_path)

    assert count == 2
    assert request.call_count == 4
    assert sleep.call_count == 2
    assert returned["image_path"].notna().all()


def test_permanent_api_failure_stops_after_max_retries(tmp_path, monkeypatch):
    """Permanent record API errors stop after the configured retry budget."""
    request = Mock(side_effect=requests.HTTPError("permanent failure"))
    sleep = Mock(side_effect=[None, None, AssertionError("retry loop was unbounded")])
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", request)
    monkeypatch.setattr("smartrodent.inaturalist.time.sleep", sleep)
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_observations=1,
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
        max_retries=2,
    )

    with pytest.raises(requests.HTTPError, match="permanent failure"):
        dataset.retrieve_records()

    assert request.call_count == 3
    assert sleep.call_count == 2


def test_permanent_photo_failure_stops_after_max_retries(tmp_path, monkeypatch):
    """Permanent image errors also stop after the configured retry budget."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset",
        species=["Mus musculus"],
        years=[2022],
        max_retries=1,
        cache_file=tmp_path / "api_requests.db",
        ratelimit_path=tmp_path / "api_ratelimit.db",
    )
    records = pd.DataFrame(
        {
            "id": [123],
            "photo_index": [0],
            "photo.url": ["https://example.test/square.jpg"],
        }
    )
    request = Mock(side_effect=requests.HTTPError("permanent photo failure"))
    sleep = Mock(side_effect=[None, AssertionError("retry loop was unbounded")])
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", request)
    monkeypatch.setattr("smartrodent.inaturalist.time.sleep", sleep)

    with pytest.raises(requests.HTTPError, match="permanent photo failure"):
        dataset._download_species_images(records, tmp_path / "imgs")

    assert request.call_count == 2
    sleep.assert_called_once_with(20)


def test_photo_rows_are_annotated_in_place_without_license_filtering(
    tmp_path, monkeypatch
):
    """Image downloading handles only valid row data, not license policy."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    records = pd.DataFrame(
        {
            "id": [None, 123, 123, 123, 123, 123],
            "photo_index": [0, 0, 1, 2, 3, 4],
            "photo.url": ["https://example.test/square.jpg"] * 5 + [None],
            "photo.license_code": ["cc-by-nc"] * 4 + ["cc-by", "cc-by-nc"],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    images = tmp_path / "images"
    images.mkdir()
    (images / "123_0.jpg").write_bytes(b"existing")
    get = Mock(return_value=Mock(content=b"new"))
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    count, returned = dataset._download_species_images(records, images)

    assert returned is records
    assert count == 4
    assert get.call_count == 3
    assert pd.isna(records.at[10, "image_path"])
    for row, index in [(20, 0), (30, 1), (40, 2), (50, 3)]:
        assert records.at[row, "image_path"] == str(images / f"123_{index}.jpg")
    assert pd.isna(records.at[60, "image_path"])
    assert (images / "123_0.jpg").read_bytes() == b"existing"


@given(cap=st.integers(min_value=0, max_value=5))
def test_photo_row_downloads_respect_cap(cap):
    """Only rows within the image cap acquire paths."""
    with TemporaryDirectory() as directory:
        dataset = InaturalistDataset(
            directory, ["Sorex minutus"], years=[2022], max_img_num=cap
        )
        records = pd.DataFrame(
            {
                "id": [123] * 5,
                "photo_index": list(range(5)),
                "photo.url": ["https://example.test/square.jpg"] * 5,
                "photo.license_code": ["cc-by-nc"] * 5,
            }
        )

        def save_photo(photo, images_path, observation_id, index):
            """Write a temporary image without network access."""
            del photo
            (images_path / f"{observation_id}_{index}.jpg").write_bytes(b"image")
            return True

        dataset._download_photo = save_photo
        count, returned = dataset._download_species_images(records, dataset.output_path)
        assert returned is records
        assert count == cap
        assert records["image_path"].notna().sum() == cap


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


def test_existing_photo_bypasses_download_helper(tmp_path, monkeypatch):
    """An existing image is annotated without invoking the download helper."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    records = pd.DataFrame(
        {
            "id": [123],
            "photo_index": [0],
            "photo.url": ["https://example.test/square.jpg"],
            "photo.license_code": ["cc-by-nc"],
        }
    )
    image_path = tmp_path / "123_0.jpg"
    image_path.write_bytes(b"existing")
    download_photo = Mock(side_effect=AssertionError("Must not be called"))
    monkeypatch.setattr(dataset, "_download_photo", download_photo)

    count, returned = dataset._download_species_images(records, tmp_path)

    download_photo.assert_not_called()
    assert count == 1
    assert returned is records
    assert records.at[0, "image_path"] == str(image_path)
    assert image_path.read_bytes() == b"existing"


def test_missing_photo_columns_fail_explicitly(tmp_path):
    """An old observation-row table must not silently skip every image."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    with pytest.raises(KeyError, match="Missing photo columns"):
        dataset._download_species_images(pd.DataFrame({"id": [123]}), tmp_path)


@pytest.mark.parametrize("error", [requests.HTTPError, requests.ConnectionError])
def test_failed_photo_is_retried_before_advancing(tmp_path, monkeypatch, error):
    """A transient image failure retries the same row before advancing."""
    dataset = InaturalistDataset(tmp_path, ["Sorex minutus"], years=[2022])
    records = pd.DataFrame(
        {
            "id": [123],
            "photo_index": [0],
            "photo.url": ["https://example.test/square.jpg"],
            "photo.license_code": ["cc-by-nc"],
        }
    )
    get = Mock(side_effect=[error("temporary"), Mock(content=b"image")])
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)
    monkeypatch.setattr("smartrodent.inaturalist.time.sleep", Mock())

    count, returned = dataset._download_species_images(records, tmp_path)

    assert count == 1
    assert returned.at[0, "image_path"] == str(tmp_path / "123_0.jpg")
    assert get.call_count == 2
