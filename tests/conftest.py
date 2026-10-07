"""Shared temporary-image fixtures for inference and dataset creation tests."""

import pytest
from PIL import Image


@pytest.fixture
def images(tmp_path):
    """Create two differently colored images with identical base filenames."""
    paths = []
    for species, color in [("mouse", "red"), ("shrew", "blue")]:
        directory = tmp_path / species
        directory.mkdir()
        path = directory / "observation.png"
        image = Image.new("RGB", (20, 10), "black")
        image.paste(color, (5, 2, 15, 8))
        image.save(path)
        paths.append(path)
    return paths
