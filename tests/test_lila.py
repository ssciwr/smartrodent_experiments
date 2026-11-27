"""LILA record selection and downloading against a fake LILA server."""

import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import requests
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from smartrodent.lila import LilaDataset

JPEG = b"\xff\xd8\xff fake jpeg"
TAXONOMY = {
    "ship rat": {
        "scientific_name": "rattus rattus",
        "species": "rattus rattus",
        "genus": "rattus",
    },
    "rat": {"scientific_name": "rattus", "genus": "rattus"},
    "cat": {
        "scientific_name": "felis catus",
        "species": "felis catus",
        "genus": "felis",
    },
    "empty": {"scientific_name": ""},
}


class FakeLila:
    """Fake LILA index, taxonomy, metadata zips and image bucket; unknown paths 404.

    It answers a real local HTTP server, so downloads go through the retrying
    session. ``errors`` maps a path to a status, or to a list of statuses used one
    per request before the file is served.
    """

    def __init__(self, base_url):
        self.base_url = base_url
        self.index_url = f"{base_url}/lila/index.csv"
        self.taxonomy_url = f"{base_url}/lila/taxonomy.csv"
        self.files, self.errors, self.requested = {}, {}, []
        self.index, self.taxonomy = [], []

    def add(self, name, short, metadata, json_names=None):
        metadata_path = f"/lila/{short}.json.zip"
        self.index.append(
            {
                "name": name,
                "short_name": short,
                "image_base_url_gcp": f"{self.base_url}/bucket/{short}",
                "metadata_url_gcp": self.base_url + metadata_path,
            }
        )
        self.taxonomy += [
            {"dataset_name": name, "query": q, **r} for q, r in TAXONOMY.items()
        ]
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            for json_name in json_names or [f"{short}.json"]:
                zipped.writestr(json_name, json.dumps(metadata))
        self.files[metadata_path] = archive.getvalue()
        for image in metadata["images"]:
            self.files[f"/bucket/{short}/{image['file_name']}"] = JPEG
        self.files["/lila/index.csv"] = (
            pd.DataFrame(self.index).to_csv(index=False).encode()
        )
        self.files["/lila/taxonomy.csv"] = (
            pd.DataFrame(self.taxonomy).to_csv(index=False).encode()
        )

    def respond(self, path):
        """Answer one request for ``path`` with its scripted error or its file."""
        self.requested.append(path)
        error = self.errors.get(path)
        if isinstance(error, list) and error:
            return error.pop(0), b""
        elif isinstance(error, int):
            return error, b""
        elif path in self.files:
            return 200, self.files[path]
        else:
            return 404, b""


def alpha_metadata():
    """a1-a2 ship rat (one sequence), a3 rat, a4 cat + ship rat, a5 empty."""

    def image(i, seq, cam):
        return {"id": i, "file_name": f"{cam}/{i}.jpg", "seq_id": seq}

    return {
        "categories": [
            {"id": 1, "name": "ship rat"},
            {"id": 2, "name": "rat"},
            {"id": 3, "name": "cat"},
            {"id": 4, "name": "empty"},
            {"id": 5, "name": "#ref!"},
        ],
        "images": [
            image("a1", "s1", "cam1"),
            image("a2", "s1", "cam1"),
            image("a3", "s2", "cam1"),
            image("a4", "s3", "cam2"),
            image("a5", "s4", "cam2"),
        ],
        "annotations": [
            {"image_id": "a1", "category_id": 1},
            {"image_id": "a2", "category_id": 1},
            {"image_id": "a3", "category_id": 2},
            {"image_id": "a4", "category_id": 3, "bbox": [1, 2, 3, 4]},
            {"image_id": "a4", "category_id": 1, "bbox": [5, 6, 7, 8]},
            {"image_id": "a5", "category_id": 4},
            {"image_id": "a5", "category_id": 5},
        ],
    }


def make(lila, root, datasets, classes, max_img_num=2500):
    return LilaDataset(
        root / "lila",
        datasets,
        classes,
        ["cdla"],
        lila.index_url,
        lila.taxonomy_url,
        max_img_num=max_img_num,
        max_retries=2,
        backoff_factor=0,
    )


def records(dataset, class_name):
    return pd.read_csv(dataset.output_path / class_name / "records.csv")


@pytest.fixture
def lila(http_server):
    server = FakeLila(http_server.base_url)
    http_server.respond = server.respond
    return server


@pytest.fixture
def alpha(tmp_path, lila):
    lila.add("Alpha Traps", "alpha", alpha_metadata())
    return make(
        lila,
        tmp_path,
        {"Alpha Traps": "cdla"},
        ["Rattus rattus", "Rattus", "Felis catus", "empty"],
    )


def test_most_specific_class_wins_and_multi_class_images_go_to_each(alpha):
    alpha.retrieve_records()

    def ids(class_name):
        return {p.split("/")[1] for p in records(alpha, class_name)["photo.id"]}

    assert ids("Rattus rattus") == {"a1", "a2", "a4"}
    assert ids("Rattus") == {"a3"}
    assert ids("Felis catus") == {"a4"}
    assert ids("empty") == {"a5"}
    cat = records(alpha, "Felis catus").iloc[0]
    rat = records(alpha, "Rattus rattus").set_index("photo.id").loc["Alpha Traps/a4"]
    assert json.loads(cat["bbox"]) == [[1, 2, 3, 4]]
    assert json.loads(rat["bbox"]) == [[5, 6, 7, 8]]
    assert cat["id"] == rat["id"] == "Alpha Traps/s3"


def test_genus_class_collects_species_labels_when_species_is_not_configured(
    tmp_path, lila
):
    lila.add("Alpha Traps", "alpha", alpha_metadata())
    dataset = make(lila, tmp_path, {"Alpha Traps": "cdla"}, ["Rattus", "Mus musculus"])

    dataset.retrieve_records()

    assert len(records(dataset, "Rattus")) == 4
    assert records(dataset, "Mus musculus").empty


def test_disallowed_license_and_empty_datasets(tmp_path, lila):
    lila.add("Alpha Traps", "alpha", alpha_metadata())
    lila.add("Beta Traps", "beta", alpha_metadata())
    lila.add(
        "Gamma Traps", "gamma", {"categories": [], "images": [], "annotations": []}
    )
    dataset = make(
        lila,
        tmp_path,
        {"Alpha Traps": "cdla", "Beta Traps": "secret", "Gamma Traps": "cdla"},
        ["Rattus rattus"],
    )

    dataset.retrieve_records()

    assert set(records(dataset, "Rattus rattus")["dataset"]) == {"Alpha Traps"}
    assert "/lila/beta.json.zip" not in lila.requested


def test_reruns_keep_existing_records_and_use_the_cache(alpha, lila):
    alpha.retrieve_records()
    (alpha.output_path / "Felis catus" / "records.csv").unlink()
    pd.DataFrame({"id": ["kept"]}).to_csv(
        alpha.output_path / "Rattus" / "records.csv", index=False
    )
    lila.requested.clear()

    alpha.retrieve_records()
    alpha.retrieve_records()

    assert lila.requested == []
    assert len(records(alpha, "Felis catus")) == 1
    assert records(alpha, "Rattus")["id"].tolist() == ["kept"]


def test_download_drops_withheld_images_and_reuses_existing_ones(alpha, lila):
    del lila.files["/bucket/alpha/cam1/a2.jpg"]  # withheld by LILA

    alpha.download()

    rats = records(alpha, "Rattus rattus")
    assert {p.split("/")[1] for p in rats["photo.id"]} == {"a1", "a4"}
    assert rats["image_path"].iloc[0].startswith("lila/Rattus rattus/imgs/alpha_")
    assert all(
        (alpha.output_path.parent / p).read_bytes() == JPEG for p in rats["image_path"]
    )
    lila.requested.clear()
    alpha.download_images()
    assert lila.requested == []


def test_failures_raise(tmp_path, alpha, lila):
    with pytest.raises(FileNotFoundError):
        alpha.download_images()
    alpha.retrieve_records()
    lila.errors["/bucket/alpha/cam1/a3.jpg"] = 500
    with pytest.raises(requests.HTTPError):
        alpha.download_images()
    with pytest.raises(KeyError):
        make(
            lila, tmp_path / "x", {"Unknown Traps": "cdla"}, ["Rattus"]
        ).retrieve_records()


def test_transient_server_errors_are_retried(alpha, lila):
    """A busy bucket or index does not abort a download that would succeed on retry."""
    lila.errors["/lila/index.csv"] = [503]
    lila.errors["/bucket/alpha/cam1/a3.jpg"] = [503, 502]

    alpha.download()

    assert lila.requested.count("/lila/index.csv") == 2
    assert lila.requested.count("/bucket/alpha/cam1/a3.jpg") == 3
    assert records(alpha, "Rattus")["image_path"].tolist() == [
        "lila/Rattus/imgs/alpha_cam1__a3.jpg"
    ]


def test_metadata_zip_must_hold_one_json(tmp_path, lila):
    lila.add("Alpha Traps", "alpha", alpha_metadata(), json_names=["a.json", "b.json"])
    with pytest.raises(ValueError):
        make(lila, tmp_path, {"Alpha Traps": "cdla"}, ["Rattus"]).retrieve_records()


def test_from_config_reads_lila_section_and_copies_config(tmp_path):
    config = {
        "data": {
            "lila": {
                "output_path": str(tmp_path / "lila"),
                "datasets": {"Alpha Traps": "cdla"},
                "classes": ["Rattus"],
                "allowed_licenses": ["cdla"],
                "index_url": "https://lila.test/index.csv",
                "taxonomy_url": "https://lila.test/taxonomy.csv",
                "max_img_num": 7,
            }
        }
    }
    config_path = tmp_path / "lila.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert LilaDataset.from_config(config_path).max_img_num == 7
    assert (tmp_path / "lila" / "lila.yaml").read_bytes() == config_path.read_bytes()


def sequences(sizes):
    """One ship-rat image per frame, in sequences of the given sizes."""
    images = [
        {"id": f"{s}_{f}", "file_name": f"{s}_{f}.jpg", "seq_id": f"s{s}"}
        for s, size in enumerate(sizes)
        for f in range(size)
    ]
    return {
        "categories": [{"id": 1, "name": "ship rat"}],
        "images": images,
        "annotations": [{"image_id": i["id"], "category_id": 1} for i in images],
    }


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.integers(1, 6), min_size=1, max_size=8),
    st.lists(st.integers(1, 6), min_size=1, max_size=8),
    st.integers(1, 20),
)
def test_cap_applies_per_class_and_dataset_with_whole_sequences(
    module_http_server, alpha_sizes, beta_sizes, cap
):
    server = FakeLila(module_http_server.base_url)
    module_http_server.respond = server.respond
    server.add("Alpha Traps", "alpha", sequences(alpha_sizes))
    server.add("Beta Traps", "beta", sequences(beta_sizes))
    with TemporaryDirectory() as directory:
        dataset = make(
            server,
            Path(directory),
            {"Alpha Traps": "cdla", "Beta Traps": "cdla"},
            ["Rattus rattus"],
            max_img_num=cap,
        )
        dataset.retrieve_records()
        rows = records(dataset, "Rattus rattus")

    for name, sizes in (("Alpha Traps", alpha_sizes), ("Beta Traps", beta_sizes)):
        taken = rows[rows["dataset"] == name].groupby("id").size()
        assert taken.sum() == min(cap, sum(sizes))
        # Whole sequences are taken; only the last one may be cut.
        assert (
            sum(
                taken[f"{name}/s{s}"] < size
                for s, size in enumerate(sizes)
                if f"{name}/s{s}" in taken
            )
            <= 1
        )
