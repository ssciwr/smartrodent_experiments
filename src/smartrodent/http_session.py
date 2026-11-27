"""HTTP sessions that retry transient failures, shared by the dataset downloaders."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def retrying_session(max_retries: int, backoff_factor: float) -> requests.Session:
    """Create a session that retries connection errors and transient server errors.

    Rate limiting (429) and the usual 5xx statuses are retried with exponential
    backoff. Other statuses, such as 404 for an image a server withholds, are
    returned at once.

    Args:
        max_retries: Retries after the first attempt of each request.
        backoff_factor: Base of the exponential delay between retries, in seconds;
            0 retries immediately.

    Returns:
        A session whose requests retry within the budget. When the budget is
        spent on error statuses, the last response is returned, so callers still
        raise ``requests.HTTPError`` through ``raise_for_status()``; persistent
        connection errors raise ``requests.ConnectionError``.
    """
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        # Without this, urllib3 raises its own RetryError instead of handing back
        # the final error response.
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
