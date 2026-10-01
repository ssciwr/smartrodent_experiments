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
    assert list(records.columns) == ["id", "photos"]
    assert not (species_path / "imgs").exists()

    dataset.download_images()

    observations.assert_called_once()
    image_request.assert_not_called()
    assert (species_path / "imgs").is_dir()
