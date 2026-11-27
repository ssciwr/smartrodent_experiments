"""iNaturalist downloads against a fake observations API and a local image server.

Only the real boundaries are replaced: ``get_observations_v2`` (the iNaturalist
API) and the image host (a local HTTP server). The downloader is used through its
public methods, and tests check what it writes and which requests it makes.
"""

import logging

import pandas as pd
import pytest
import requests
import yaml
from bundle_test_support import move_bundle, write_stage_records
from http_test_support import DROP

from smartrodent.inaturalist import InaturalistDataset

PHOTO_COLUMNS = ["id", "photo_index", "photo.url", "photo.license_code"]


class FakeObservationsApi:
    """Answer ``get_observations_v2`` from a fixed population per species.

    ``errors`` maps a species to an exception raised for its requests. Every
    request's parameters are recorded in ``requests``. Pages are offset by 200, the
    downloader's full page size; the real API offsets by the requested ``per_page``,
    which only matters for a short last page of a random, deduplicated sample.
    """

    def __init__(self):
        self.observations: dict[str, list[dict]] = {}
        self.errors: dict[str, Exception] = {}
        self.requests: list[dict] = []

    def __call__(self, **params):
        self.requests.append(params)
        species = params["taxon_name"]
        if species in self.errors:
            raise self.errors[species]
        population = self.observations.get(species, [])
        start = (params["page"] - 1) * 200
        return {
            "results": population[start : start + params["per_page"]],
            "total_results": len(population),
        }


@pytest.fixture
def api(monkeypatch):
    """The fake iNaturalist API, in place of the real one."""
    fake = FakeObservationsApi()
    monkeypatch.setattr("smartrodent.inaturalist.get_observations_v2", fake)
    return fake


def observation(observation_id, licenses, *, server=None):
    """An API observation with one photo per license; URLs point at ``server``."""
    base = server.base_url if server is not None else "https://example.test"
    return {
        "id": observation_id,
        "observed_on": "2022-05-10",
        "species_guess": "Mus musculus",
        "photos": [
            {
                "id": observation_id * 10 + index,
                "url": f"{base}/{observation_id}/{index}/square.jpg",
                "license_code": license_code,
            }
            for index, license_code in enumerate(licenses)
        ],
    }


def make_dataset(tmp_path, **settings):
    """A downloader writing only below ``tmp_path``, retrying without delay."""
    options = {
        "output_path": tmp_path / "dataset",
        "species": ["Mus musculus"],
        "years": [2022],
        "cache_file": tmp_path / "api_requests.db",
        "ratelimit_path": tmp_path / "api_ratelimit.db",
        "backoff_factor": 0,
    }
    options.update(settings)
    return InaturalistDataset(**options)


def records_of(dataset, species="Mus musculus"):
    """The saved records of one species."""
    return pd.read_csv(dataset.output_path / species / "records.csv")


def save_records(dataset, rows, species="Mus musculus"):
    """Save photo rows as a species' records, as the retrieval phase would."""
    path = dataset.output_path / species / "records.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PHOTO_COLUMNS).to_csv(path, index=False)


def photo_rows(server, count):
    """Licensed photo rows whose thumbnail URLs point at the local server."""
    return [
        (100 + index, 0, server.url(f"/{100 + index}/square.jpg"), "cc-by-nc")
        for index in range(count)
    ]


# --- configuration ------------------------------------------------------------------


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


# --- the two phases -----------------------------------------------------------------


def test_download_saves_all_records_before_any_image(tmp_path, api, http_server):
    """Every species' records exist before the first image is requested."""
    dataset = make_dataset(tmp_path, species=["Mus musculus", "Rattus rattus"])
    api.observations = {
        "Mus musculus": [observation(1, ["cc-by-nc"], server=http_server)],
        "Rattus rattus": [observation(2, ["cc-by-nc"], server=http_server)],
    }

    def respond(path):
        for species in dataset.species:
            assert (dataset.output_path / species / "records.csv").is_file()
        return 200, path.encode()

    http_server.respond = respond

    dataset.download()

    assert [request["taxon_name"] for request in api.requests] == dataset.species
    for species, observation_id in [("Mus musculus", 1), ("Rattus rattus", 2)]:
        image = dataset.output_path / species / "imgs" / f"{observation_id}_0.jpg"
        assert image.read_bytes() == f"/{observation_id}/0/large.jpg".encode()
        assert records_of(dataset, species)["image_path"].tolist() == [
            f"dataset/{species}/imgs/{observation_id}_0.jpg"
        ]


def test_retrieve_records_requests_no_images(tmp_path, api, http_server):
    """The records-only phase writes CSVs and creates no image directories."""
    dataset = make_dataset(tmp_path)
    api.observations = {
        "Mus musculus": [observation(1, ["cc-by-nc"], server=http_server)]
    }

    dataset.retrieve_records()

    assert records_of(dataset)["id"].tolist() == [1]
    assert not (dataset.output_path / "Mus musculus" / "imgs").exists()
    assert http_server.requested == []


@pytest.mark.parametrize("empty", [False, True])
def test_saved_records_are_reused_without_api_requests(
    tmp_path, api, http_server, empty
):
    """Saved records, even header-only ones, are reused; delete them to retrieve again."""
    dataset = make_dataset(tmp_path)
    save_records(dataset, [] if empty else photo_rows(http_server, 1))
    http_server.serve("/100/large.jpg", (200, b"image"))

    dataset.download()

    assert api.requests == []
    assert (dataset.output_path / "Mus musculus" / "imgs").is_dir()
    assert len(records_of(dataset)) == (0 if empty else 1)
    assert http_server.requested == ([] if empty else ["/100/large.jpg"])


def test_unreadable_saved_records_are_not_treated_as_missing(tmp_path, api):
    """An empty file is a broken cache, not a reason to retrieve the species again."""
    dataset = make_dataset(tmp_path)
    species_path = dataset.output_path / "Mus musculus"
    species_path.mkdir(parents=True)
    (species_path / "records.csv").write_text("", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        dataset.download()

    assert api.requests == []
    assert not (species_path / "imgs").exists()


def test_download_images_requires_saved_records(tmp_path, api):
    """A missing records file fails clearly rather than triggering retrieval."""
    dataset = make_dataset(tmp_path)

    with pytest.raises(
        FileNotFoundError, match="Missing records for Mus musculus.*retrieve_records"
    ):
        dataset.download_images()

    assert api.requests == []
    assert not (dataset.output_path / "Mus musculus" / "imgs").exists()


@pytest.mark.parametrize(
    "error",
    [requests.HTTPError("permanent failure"), FileNotFoundError("disk failure")],
    ids=["api", "filesystem"],
)
def test_record_failure_keeps_earlier_records_and_downloads_no_images(
    tmp_path, api, error
):
    """Failures propagate after one request: the API session already retried."""
    dataset = make_dataset(tmp_path, species=["Mus musculus", "Rattus rattus"])
    api.observations = {"Mus musculus": [observation(1, ["cc-by-nc"])]}
    api.errors = {"Rattus rattus": error}

    with pytest.raises(type(error), match="failure"):
        dataset.download()

    assert [request["taxon_name"] for request in api.requests] == dataset.species
    assert records_of(dataset)["id"].tolist() == [1]
    assert not (dataset.output_path / "Rattus rattus" / "records.csv").exists()
    for species in dataset.species:
        assert not (dataset.output_path / species / "imgs").exists()


def test_no_observations_give_readable_empty_records(tmp_path, api):
    """No observations produce a header-only cache that the image phase accepts."""
    dataset = make_dataset(tmp_path, max_observations=8)

    dataset.retrieve_records()
    dataset.download_images()

    assert len(api.requests) == 1
    records = records_of(dataset)
    assert records.empty
    assert list(records.columns) == PHOTO_COLUMNS + ["image_path"]
    assert not any((dataset.output_path / "Mus musculus" / "imgs").iterdir())


@pytest.mark.parametrize("failure", [None, RuntimeError("download failed")])
def test_download_removes_response_cache_but_retains_rate_accounting(
    tmp_path, api, failure
):
    """Full-download cleanup runs on both success and ordinary failure."""
    dataset = make_dataset(tmp_path, max_observations=1)
    cache_file, ratelimit_path = dataset.cache_file, dataset.ratelimit_path
    unrelated = tmp_path / "unrelated.db"
    cache_file.touch(exist_ok=True)
    ratelimit_path.touch(exist_ok=True)
    unrelated.write_bytes(b"unrelated")
    if failure is not None:
        api.errors = {"Mus musculus": failure}

    if failure is None:
        dataset.download()
    else:
        with pytest.raises(RuntimeError, match="download failed"):
            dataset.download()

    assert not cache_file.exists()
    assert ratelimit_path.exists()
    assert unrelated.read_bytes() == b"unrelated"


# --- record retrieval ---------------------------------------------------------------


def test_retrieval_combines_filters_in_bounded_random_pages(tmp_path, api):
    """All filters share bounded random v2 requests rather than per-year crawls."""
    api.observations = {
        "Mus musculus": [observation(index, ["cc-by"]) for index in range(1, 202)]
    }
    dataset = make_dataset(
        tmp_path,
        years=[2020, 2022],
        max_observations=201,
        max_img_num=3,
        allowed_licenses=["cc-by", "cc0"],
    )

    dataset.retrieve_records()

    assert [request["per_page"] for request in api.requests] == [200, 1]
    for page, request in enumerate(api.requests, start=1):
        assert request["session"] is dataset.session
        assert request["year"] == [2020, 2022]
        assert request["photos"] is True
        assert request["order_by"] == "random"
        assert request["page"] == page
        assert request["fields"] == "all"
    assert len(records_of(dataset)) == 3


def test_retrieval_filters_licenses_before_capping_photo_rows(tmp_path, api):
    """Disallowed photos cannot displace eligible photos in the retained sample."""
    api.observations = {
        "Mus musculus": [observation(101, [None, "cc-by-nc", "cc-by", "cc0"])]
    }
    dataset = make_dataset(
        tmp_path, max_observations=1, max_img_num=2, allowed_licenses=["cc-by", "cc0"]
    )

    dataset.retrieve_records()

    records = records_of(dataset)
    assert set(records["photo.license_code"]) == {"cc-by", "cc0"}
    assert set(records["photo_index"]) == {2, 3}


def test_duplicate_observations_consume_the_explicit_budget(tmp_path, api):
    """Duplicate random results do not permit retrieval beyond the configured budget."""
    duplicate = observation(101, ["cc-by"])
    api.observations = {
        "Mus musculus": [duplicate, duplicate, observation(102, ["cc-by"])]
    }
    dataset = make_dataset(tmp_path, max_observations=3, allowed_licenses=["cc-by"])

    dataset.retrieve_records()

    assert len(api.requests) == 1
    assert not records_of(dataset).duplicated(["id", "photo_index"]).any()


def test_retrieval_writes_one_row_per_photo(tmp_path, api):
    """Fresh v2 records keep the one-row-per-photo layout, without the photo list."""
    api.observations = {
        "Mus musculus": [
            {
                "id": 123,
                "photos": [
                    {"id": 11, "url": "first", "license_code": "cc-by-nc"},
                    {"id": 12, "url": "second", "license_code": "cc-by"},
                ],
            }
        ]
    }
    dataset = make_dataset(
        tmp_path, max_observations=1, allowed_licenses=["cc-by-nc", "cc-by"]
    )

    dataset.retrieve_records()

    records = records_of(dataset).sort_values("photo_index")
    assert records["id"].tolist() == [123, 123]
    assert records["photo_index"].tolist() == [0, 1]
    assert records["photo.id"].tolist() == [11, 12]
    assert records["photo.url"].tolist() == ["first", "second"]
    assert "photos" not in records.columns


# --- image downloads ----------------------------------------------------------------


def test_existing_images_are_kept_and_not_requested_again(tmp_path, api, http_server):
    """A resumed download fetches only the missing large variants."""
    dataset = make_dataset(tmp_path)
    api.observations = {
        "Mus musculus": [observation(123, ["cc-by-nc"] * 2, server=http_server)]
    }
    images_path = dataset.output_path / "Mus musculus" / "imgs"
    images_path.mkdir(parents=True)
    (images_path / "123_0.jpg").write_bytes(b"existing image")
    http_server.serve("/123/1/large.jpg", (200, b"new image"))

    dataset.download()

    assert (images_path / "123_0.jpg").read_bytes() == b"existing image"
    assert (images_path / "123_1.jpg").read_bytes() == b"new image"
    assert http_server.requested == ["/123/1/large.jpg"]


def test_rows_without_identity_or_url_are_skipped_without_license_filtering(
    tmp_path, http_server
):
    """Image downloading handles only valid row data, not license policy."""
    dataset = make_dataset(tmp_path, species=["Sorex minutus"])
    url = http_server.url("/square.jpg")
    save_records(
        dataset,
        [
            (None, 0, url, "cc-by-nc"),
            (123, 1, url, "cc-by-nc"),
            (123, 2, url, "cc-by"),
            (123, 3, None, "cc-by-nc"),
        ],
        species="Sorex minutus",
    )
    http_server.serve("/large.jpg", (200, b"new"))

    dataset.download_images()

    paths = records_of(dataset, "Sorex minutus")["image_path"]
    assert paths.isna().tolist() == [True, False, False, True]
    assert paths.iloc[1:3].tolist() == [
        f"dataset/Sorex minutus/imgs/123_{index}.jpg" for index in (1, 2)
    ]
    assert len(http_server.requested) == 2


def test_downloads_stop_at_the_image_cap(tmp_path, http_server):
    """Existing and new images both count toward max_img_num; later rows get no path."""
    dataset = make_dataset(tmp_path, max_img_num=2)
    save_records(dataset, photo_rows(http_server, 3))
    images_path = dataset.output_path / "Mus musculus" / "imgs"
    images_path.mkdir()
    (images_path / "100_0.jpg").write_bytes(b"existing")
    http_server.serve("/101/large.jpg", (200, b"new"))

    dataset.download_images()

    assert http_server.requested == ["/101/large.jpg"]
    assert records_of(dataset)["image_path"].notna().tolist() == [True, True, False]


def test_each_photo_retries_transient_failures_within_its_own_budget(
    tmp_path, http_server
):
    """One transient failure per photo fits a budget of one retry for each photo."""
    dataset = make_dataset(tmp_path, max_retries=1)
    save_records(dataset, photo_rows(http_server, 2))
    http_server.serve("/100/large.jpg", (503, b""), (200, b"first"))
    http_server.serve("/101/large.jpg", DROP, (200, b"second"))

    dataset.download_images()

    images_path = dataset.output_path / "Mus musculus" / "imgs"
    assert (images_path / "100_0.jpg").read_bytes() == b"first"
    assert (images_path / "101_0.jpg").read_bytes() == b"second"
    assert len(http_server.requested) == 4


def test_persistent_photo_failure_raises_after_the_retry_budget(tmp_path, http_server):
    dataset = make_dataset(tmp_path, max_retries=1)
    save_records(dataset, photo_rows(http_server, 1))
    http_server.serve("/100/large.jpg", (503, b""))

    with pytest.raises(requests.HTTPError):
        dataset.download_images()

    assert http_server.requested == ["/100/large.jpg", "/100/large.jpg"]
    assert not (dataset.output_path / "Mus musculus" / "imgs" / "100_0.jpg").exists()


def test_records_without_photo_columns_fail_explicitly(tmp_path):
    """An old observation-row table must not silently skip every image."""
    dataset = make_dataset(tmp_path)
    species_path = dataset.output_path / "Mus musculus"
    species_path.mkdir(parents=True)
    pd.DataFrame({"id": [123]}).to_csv(species_path / "records.csv", index=False)

    with pytest.raises(KeyError, match="Missing photo columns"):
        dataset.download_images()


def test_records_keep_shared_root_relative_paths_and_resume_after_move(
    bundle, http_server
):
    """Image references are portable: a moved tree resumes without downloading."""
    dataset_root, records_by_species = bundle
    write_stage_records(dataset_root / "inaturalist", records_by_species)

    def download_and_check(root):
        InaturalistDataset(
            output_path=root / "inaturalist",
            species=list(records_by_species),
            years=[2024],
        ).download_images()
        for species, original in records_by_species.items():
            actual = pd.read_csv(root / "inaturalist" / species / "records.csv")
            assert actual["image_path"].tolist() == original["image_path"].tolist()
            assert all((root / path).is_file() for path in actual["image_path"])

    download_and_check(dataset_root)
    download_and_check(move_bundle(dataset_root))

    # Every image already existed, so nothing was requested.
    assert http_server.requested == []


# --- logging ------------------------------------------------------------------------


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
