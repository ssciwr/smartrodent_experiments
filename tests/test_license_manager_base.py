import pytest

from smartrodent.licenses import LicenseManagerBase


class ExampleLicenseManager(LicenseManagerBase):
    """Minimal concrete manager used to exercise the base-class contract."""

    def normalize_license(self, raw_license: object) -> str | None:
        if raw_license == "Example One":
            return "example-one"
        if raw_license == "Example Two":
            return "example-two"
        return None


def test_allowed_licenses_are_stored_immutably():
    manager = ExampleLicenseManager(["example-one"])

    assert manager.allowed_licenses == frozenset({"example-one"})


def test_allowed_licenses_reject_string_input():
    with pytest.raises(TypeError, match="sequence"):
        ExampleLicenseManager("example-one")


def test_allowed_licenses_reject_non_sequence_collection():
    with pytest.raises(TypeError, match="sequence"):
        ExampleLicenseManager({"example-one"})


def test_allowed_licenses_reject_empty_sequence():
    with pytest.raises(ValueError, match="at least one"):
        ExampleLicenseManager([])


def test_allowed_licenses_reject_non_string_entry():
    with pytest.raises(TypeError, match="non-empty strings"):
        ExampleLicenseManager(["example-one", 2])


def test_allowed_licenses_reject_empty_string_entry():
    with pytest.raises(ValueError, match="non-empty strings"):
        ExampleLicenseManager(["example-one", ""])


def test_normalize_url_accepts_http_url():
    normalized = ExampleLicenseManager.normalize_url(
        "http://example.com/licenses/example/1.0/"
    )

    assert normalized == "http://example.com/licenses/example/1.0"


def test_normalize_url_accepts_https_url():
    normalized = ExampleLicenseManager.normalize_url(
        "https://example.com/licenses/example/1.0/"
    )

    assert normalized == "https://example.com/licenses/example/1.0"


def test_normalize_url_normalizes_scheme_and_hostname_case():
    normalized = ExampleLicenseManager.normalize_url(
        "HTTPS://EXAMPLE.COM/licenses/example/1.0"
    )

    assert normalized == "https://example.com/licenses/example/1.0"


def test_normalize_url_removes_query_string():
    normalized = ExampleLicenseManager.normalize_url(
        "https://example.com/licenses/example/1.0?source=provider"
    )

    assert normalized == "https://example.com/licenses/example/1.0"


def test_normalize_url_removes_fragment():
    normalized = ExampleLicenseManager.normalize_url(
        "https://example.com/licenses/example/1.0#details"
    )

    assert normalized == "https://example.com/licenses/example/1.0"


def test_normalize_url_strips_surrounding_whitespace():
    normalized = ExampleLicenseManager.normalize_url(
        "  https://example.com/licenses/example/1.0/  "
    )

    assert normalized == "https://example.com/licenses/example/1.0"


def test_normalize_url_preserves_port():
    normalized = ExampleLicenseManager.normalize_url(
        "https://example.com:8443/licenses/example/1.0/"
    )

    assert normalized == "https://example.com:8443/licenses/example/1.0"


def test_normalize_url_formats_ipv6_hostname():
    normalized = ExampleLicenseManager.normalize_url(
        "https://[2001:db8::1]/licenses/example/1.0/"
    )

    assert normalized == "https://[2001:db8::1]/licenses/example/1.0"


def test_normalize_url_rejects_non_http_scheme():
    assert ExampleLicenseManager.normalize_url("ftp://example.com/license") is None


def test_normalize_url_rejects_non_string_value():
    assert ExampleLicenseManager.normalize_url(42) is None


def test_normalize_url_rejects_empty_string():
    assert ExampleLicenseManager.normalize_url("  ") is None


def test_normalize_url_rejects_url_without_hostname():
    assert ExampleLicenseManager.normalize_url("https:///licenses/example") is None


def test_normalize_url_rejects_malformed_url():
    assert ExampleLicenseManager.normalize_url("not a URL") is None


def test_normalize_url_handles_parser_error():
    assert ExampleLicenseManager.normalize_url("https://[invalid") is None


def test_normalize_allowed_license_returns_allowed_value():
    manager = ExampleLicenseManager(["example-one"])

    assert manager.normalize_allowed_license("Example One") == "example-one"


def test_normalize_allowed_license_rejects_disallowed_value():
    manager = ExampleLicenseManager(["example-one"])

    assert manager.normalize_allowed_license("Example Two") is None


def test_normalize_allowed_license_rejects_unrecognized_value():
    manager = ExampleLicenseManager(["example-one"])

    assert manager.normalize_allowed_license("Unknown") is None
