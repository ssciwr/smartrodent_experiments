from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

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
    if cached:
        species_path.mkdir()
        records.to_csv(records_path, index=False)
        original_bytes = records_path.read_bytes()

    retrieve = Mock(return_value=records)
    download_images = Mock(return_value=len(records))
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
        dataset._download_species_images = Mock(return_value=0)

        dataset.download()

        dataset._get_species_records.assert_not_called()
        received_records = dataset._download_species_images.call_args.args[0]
        pd.testing.assert_frame_equal(received_records, records)


@pytest.mark.parametrize("cached", [False, True])
def test_inaturalist_resumes_images_without_redownloading_existing_files(
    tmp_path, monkeypatch, cached
):
    """Real image processing accepts cached photo lists and skips existing files."""
    dataset = InaturalistDataset(
        output_path=tmp_path / "dataset", species=["Mus musculus"], years=[2022]
    )
    photos = [
        {"url": f"https://example.test/{index}/square.jpg", "license_code": "cc-by-nc"}
        for index in range(2)
    ]
    records = pd.DataFrame({"id": [123], "photos": [photos]})
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
    download_images = Mock(return_value=1)
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
        return len(received_records)

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
        output_path=tmp_path / "dataset", species=["Mus musculus"], years=[2022]
    )
    observations = Mock(return_value={"results": []})
    image_request = Mock()
    monkeypatch.setattr("smartrodent.inaturalist.get_observations", observations)
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", image_request)

    dataset.retrieve_records()

    species_path = dataset.output_path / "Mus musculus"
    records = pd.read_csv(species_path / "records.csv")
    assert records.empty
    assert list(records.columns) == [
        "id",
        "photo_index",
        "photo.id",
        "photo.url",
        "photo.license_code",
    ]
    assert not (species_path / "imgs").exists()

    dataset.download_images()

    observations.assert_called_once()
    image_request.assert_not_called()
    assert (species_path / "imgs").is_dir()


def make_dataset(output_path, **settings):
    """Build a downloader whose output is isolated from project data."""
    return InaturalistDataset(
        output_path=output_path, species=["Mus musculus"], years=[2022], **settings
    )


def make_photo(photo_id, license_code="cc-by-nc"):
    """Return representative photo metadata with a distinct identifier."""
    return {
        "id": photo_id,
        "url": f"https://example.test/{photo_id}/square.jpg",
        "license_code": license_code,
        "attribution": "Example photographer",
    }


def test_retrieval_duplicates_observation_metadata_per_photo(tmp_path, monkeypatch):
    """Each photo retains its own fields and its observation's metadata."""
    dataset = make_dataset(tmp_path)
    observations = [
        {
            "id": 123,
            "observed_on": "2022-03-04",
            "taxon": {"name": "Mus musculus", "id": 456},
            "photos": [make_photo(11), make_photo(12, "cc0")],
        },
        {"id": 124, "photos": []},
        {"id": 125},
    ]
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations",
        Mock(return_value={"results": observations}),
    )

    records = dataset._get_species_records("Mus musculus")

    assert len(records) == 2
    assert "photos" not in records.columns
    assert records["id"].tolist() == [123, 123]
    assert records["taxon.name"].tolist() == ["Mus musculus"] * 2
    assert records["taxon.id"].tolist() == [456] * 2
    assert records["observed_on"].tolist() == ["2022-03-04"] * 2
    assert records["photo_index"].tolist() == [0, 1]
    assert records["photo.id"].tolist() == [11, 12]
    assert records["photo.license_code"].tolist() == ["cc-by-nc", "cc0"]
    assert records["photo.attribution"].tolist() == ["Example photographer"] * 2


@pytest.mark.parametrize("observations", [[], [{"id": 123, "photos": []}]])
def test_empty_flat_records_cross_csv_boundary(tmp_path, monkeypatch, observations):
    """No photos produce a readable header-only flat CSV and no requests."""
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations",
        Mock(return_value={"results": observations}),
    )
    get = Mock()
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    dataset.retrieve_records()
    records = pd.read_csv(tmp_path / "Mus musculus" / "records.csv")
    dataset.download_images()

    assert records.empty
    assert {"id", "photo_index", "photo.id", "photo.url", "photo.license_code"} <= set(
        records.columns
    )
    assert "photos" not in records.columns
    get.assert_not_called()


def test_flat_records_download_and_resume_original_filenames(tmp_path, monkeypatch):
    """CSV round-trips preserve photo indices, licenses, URLs, and resume paths."""
    dataset = make_dataset(tmp_path)
    observations = [
        {"id": 123, "photos": [make_photo(11), make_photo(12, "cc0"), make_photo(13)]}
    ]
    fetch = Mock(return_value={"results": observations})
    monkeypatch.setattr("smartrodent.inaturalist.get_observations", fetch)
    dataset.retrieve_records()
    records_path = tmp_path / "Mus musculus" / "records.csv"
    original_bytes = records_path.read_bytes()
    images_path = tmp_path / "Mus musculus" / "imgs"
    images_path.mkdir()
    (images_path / "123_0.jpg").write_bytes(b"existing image")
    get = Mock(return_value=Mock(content=b"new image"))
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    dataset.download()

    assert (images_path / "123_0.jpg").read_bytes() == b"existing image"
    assert not (images_path / "123_1.jpg").exists()
    assert (images_path / "123_2.jpg").read_bytes() == b"new image"
    get.assert_called_once_with("https://example.test/13/large.jpg", timeout=30)
    fetch.assert_called_once()
    assert records_path.read_bytes() == original_bytes


def test_flat_records_respect_download_cap(tmp_path, monkeypatch):
    """Multiple photos on one observation cannot exceed the image limit."""
    dataset = make_dataset(tmp_path, max_img_num=2)
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations",
        Mock(
            return_value={
                "results": [
                    {
                        "id": 123,
                        "photos": [make_photo(11), make_photo(12), make_photo(13)],
                    }
                ]
            }
        ),
    )
    get = Mock(return_value=Mock(content=b"image"))
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    dataset.download()

    assert get.call_count == 2
    assert sorted(p.name for p in (tmp_path / "Mus musculus" / "imgs").iterdir()) == [
        "123_0.jpg",
        "123_1.jpg",
    ]


def test_missing_flat_photo_url_is_skipped_after_csv_round_trip(tmp_path, monkeypatch):
    """Missing URL cells become NaN on CSV read and must not be requested."""
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(
        "smartrodent.inaturalist.get_observations",
        Mock(
            return_value={
                "results": [
                    {
                        "id": 123,
                        "photos": [
                            {"id": 11, "license_code": "cc-by-nc"},
                            make_photo(12),
                        ],
                    }
                ]
            }
        ),
    )
    get = Mock(return_value=Mock(content=b"image"))
    monkeypatch.setattr("smartrodent.inaturalist.requests.get", get)

    dataset.download()

    get.assert_called_once_with("https://example.test/12/large.jpg", timeout=30)
    assert not (tmp_path / "Mus musculus" / "imgs" / "123_0.jpg").exists()


@given(photo_counts=st.lists(st.integers(min_value=0, max_value=5), max_size=8))
def test_flattening_preserves_every_photo_and_observation_identity(photo_counts):
    """Arbitrary photo counts retain each photo exactly once with its owner."""
    observations = [
        {
            "id": owner,
            "photos": [make_photo(owner * 10 + index) for index in range(count)],
        }
        for owner, count in enumerate(photo_counts, start=1)
    ]
    with TemporaryDirectory() as directory:
        dataset = make_dataset(directory)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dataset.logger, "disabled", True)
            patch.setattr(
                "smartrodent.inaturalist.get_observations",
                Mock(return_value={"results": observations}),
            )
            records = dataset._get_species_records("Mus musculus")
            repeated = dataset._get_species_records("Mus musculus")
        assert len(records) == sum(photo_counts)
        assert "photos" not in records.columns
        actual = set(
            records[["id", "photo_index", "photo.id"]].itertuples(
                index=False, name=None
            )
        )
        expected = {
            (owner, index, owner * 10 + index)
            for owner, count in enumerate(photo_counts, start=1)
            for index in range(count)
        }
        assert actual == expected
        pd.testing.assert_frame_equal(records, repeated)
