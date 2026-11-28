"""The shared protocols reject incomplete implementations when they are created."""

import pytest

from smartrodent.base import Configurable, DatasetLoader, Filterable


@pytest.mark.parametrize("protocol", [Configurable, Filterable, DatasetLoader])
def test_incomplete_implementation_cannot_be_created(protocol):
    """A missing method fails loudly instead of inheriting a stub that does nothing."""

    class Incomplete(protocol):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
