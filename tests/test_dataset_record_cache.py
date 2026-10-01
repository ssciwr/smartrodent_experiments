"""Record caches affect retrieval, not the image-download phase."""

import hashlib
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from smartrodent.gbif import GbifDataset
from smartrodent.inaturalist import InaturalistDataset


@pytest.fixture(params=[GbifDataset, InaturalistDataset])
def dataset(request, tmp_path):
    """Construct either downloader with only temporary output paths."""
    return request.param(
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
    report = records.assign(success=True)
    download_images = Mock(
        return_value=report if isinstance(dataset, GbifDataset) else len(records)
    )
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
    if isinstance(dataset, GbifDataset):
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


def test_gbif_cached_records_resume_images_and_refresh_report(tmp_path, monkeypatch):
    """The cached-record path preserves valid images and downloads missing ones."""
    dataset = GbifDataset(
        output_path=tmp_path / "dataset", species=["Mus musculus"], years=[2022]
    )
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
    download_images = Mock(
        return_value=report if isinstance(dataset, GbifDataset) else 1
    )
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
    if isinstance(dataset, GbifDataset):
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
        if isinstance(dataset, GbifDataset):
            return received_records.assign(success=True)
        else:
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
