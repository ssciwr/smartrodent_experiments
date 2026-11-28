"""Retrying HTTP sessions against a real local server."""

import pytest
import requests
from http_test_support import DROP
from hypothesis import given, settings
from hypothesis import strategies as st

from smartrodent.http_session import retrying_session


@settings(max_examples=25, deadline=None)
@given(
    failures=st.integers(0, 3),
    max_retries=st.integers(0, 3),
    status=st.sampled_from([429, 500, 502, 503, 504]),
)
def test_transient_server_errors_are_retried_up_to_the_budget(
    module_http_server, failures, max_retries, status
):
    """A request succeeds iff its transient failures fit into the retry budget."""
    server = module_http_server
    server.reset()
    server.serve("/image.jpg", *[(status, b"")] * failures, (200, b"image"))

    with retrying_session(max_retries=max_retries, backoff_factor=0) as session:
        response = session.get(server.url("/image.jpg"), timeout=5)

    if failures <= max_retries:
        assert response.content == b"image"
    else:
        # The final error response is returned, so callers keep raising HTTPError.
        with pytest.raises(requests.HTTPError):
            response.raise_for_status()
    assert len(server.requested) == min(failures, max_retries) + 1


def test_not_found_is_not_retried(http_server):
    """A missing resource is a final answer, e.g. an image LILA withholds."""
    with retrying_session(max_retries=3, backoff_factor=0) as session:
        response = session.get(http_server.url("/missing.jpg"), timeout=5)

    assert response.status_code == 404
    assert http_server.requested == ["/missing.jpg"]


def test_dropped_connection_is_retried(http_server):
    """A connection closed without an answer counts as a transient failure."""
    http_server.serve("/image.jpg", DROP, (200, b"image"))

    with retrying_session(max_retries=1, backoff_factor=0) as session:
        response = session.get(http_server.url("/image.jpg"), timeout=5)

    assert response.content == b"image"
    assert http_server.requested == ["/image.jpg", "/image.jpg"]
