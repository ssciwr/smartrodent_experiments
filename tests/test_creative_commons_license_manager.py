import pytest

from smartrodent.licenses import CreativeCommonsLicenseManager


@pytest.fixture
def manager() -> CreativeCommonsLicenseManager:
    return CreativeCommonsLicenseManager(
        [
            "cc0",
            "cc-by",
            "cc-by-sa",
            "cc-by-nd",
            "cc-by-nc",
            "cc-by-nc-sa",
            "cc-by-nc-nd",
        ]
    )


def test_normalize_license_recognizes_cc_by_url(manager):
    assert (
        manager.normalize_license("https://creativecommons.org/licenses/by/4.0/")
        == "cc-by"
    )


def test_normalize_license_recognizes_cc_by_sa_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        == "cc-by-sa"
    )


def test_normalize_license_recognizes_cc_by_nd_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-nd/4.0/"
        )
        == "cc-by-nd"
    )


def test_normalize_license_recognizes_cc_by_nc_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-nc/4.0/"
        )
        == "cc-by-nc"
    )


def test_normalize_license_recognizes_cc_by_nc_sa_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-nc-sa/4.0/"
        )
        == "cc-by-nc-sa"
    )


def test_normalize_license_recognizes_cc_by_nc_nd_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-nc-nd/4.0/"
        )
        == "cc-by-nc-nd"
    )


def test_normalize_license_recognizes_cc0_url(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/publicdomain/zero/1.0/"
        )
        == "cc0"
    )


def test_normalize_license_accepts_http_url(manager):
    assert (
        manager.normalize_license("http://creativecommons.org/licenses/by/4.0/")
        == "cc-by"
    )


def test_normalize_license_ignores_license_version(manager):
    assert (
        manager.normalize_license("https://creativecommons.org/licenses/by/2.0/")
        == "cc-by"
    )


def test_normalize_license_accepts_legalcode_suffix(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by-nc/4.0/legalcode"
        )
        == "cc-by-nc"
    )


def test_normalize_license_rejects_non_creative_commons_url_host(manager):
    assert (
        manager.normalize_license("https://example.com/licenses/by/4.0/") is None
    )


def test_normalize_license_rejects_incomplete_creative_commons_url(manager):
    assert (
        manager.normalize_license("https://creativecommons.org/licenses/by/")
        is None
    )


def test_normalize_license_rejects_unknown_creative_commons_url_suffix(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by/4.0/deed"
        )
        is None
    )


def test_normalize_license_rejects_invalid_creative_commons_url_version(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/by/latest/"
        )
        is None
    )


def test_normalize_license_rejects_unknown_creative_commons_url_category(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/unknown/by/4.0/"
        )
        is None
    )


def test_normalize_license_rejects_unknown_creative_commons_url_code(manager):
    assert (
        manager.normalize_license(
            "https://creativecommons.org/licenses/sampling/1.0/"
        )
        is None
    )


def test_normalize_license_recognizes_hyphenated_compact_code(manager):
    assert manager.normalize_license("CC-BY-NC-4.0") == "cc-by-nc"


def test_normalize_license_recognizes_underscored_compact_code(manager):
    assert manager.normalize_license("CC_BY_NC_4_0") == "cc-by-nc"


def test_normalize_license_recognizes_spaced_compact_code(manager):
    assert manager.normalize_license("CC BY-NC 4.0") == "cc-by-nc"


def test_normalize_license_recognizes_descriptive_name(manager):
    assert (
        manager.normalize_license("Creative Commons Attribution 4.0 International")
        == "cc-by"
    )


def test_normalize_license_recognizes_descriptive_noncommercial_sharealike_name(
    manager,
):
    assert (
        manager.normalize_license(
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
        )
        == "cc-by-nc-sa"
    )


def test_normalize_license_recognizes_descriptive_no_derivatives_name(manager):
    assert (
        manager.normalize_license(
            "Creative Commons Attribution-NoDerivatives 4.0"
        )
        == "cc-by-nd"
    )


def test_normalize_license_rejects_descriptive_name_without_attribution(manager):
    assert manager.normalize_license("Creative Commons Unknown License") is None


def test_normalize_license_rejects_conflicting_descriptive_name(manager):
    assert (
        manager.normalize_license(
            "Creative Commons Attribution-NoDerivatives-ShareAlike 4.0"
        )
        is None
    )


def test_normalize_license_recognizes_cc0_compact_code(manager):
    assert manager.normalize_license("CC0 1.0") == "cc0"


def test_normalize_license_recognizes_cc0_descriptive_name(manager):
    assert (
        manager.normalize_license("Creative Commons Public Domain Zero 1.0")
        == "cc0"
    )


def test_normalize_license_rejects_none(manager):
    assert manager.normalize_license(None) is None


def test_normalize_license_rejects_non_string_value(manager):
    assert manager.normalize_license(42) is None


def test_normalize_license_rejects_empty_string(manager):
    assert manager.normalize_license("") is None


def test_normalize_license_rejects_all_rights_reserved(manager):
    assert manager.normalize_license("© All rights reserved") is None


def test_normalize_license_rejects_non_creative_commons_license(manager):
    assert manager.normalize_license("MIT License") is None


def test_normalize_license_rejects_invalid_cc_combination(manager):
    assert manager.normalize_license("CC-BY-SA-NC-4.0") is None


def test_constructor_rejects_unsupported_allowed_license():
    with pytest.raises(ValueError, match="unsupported"):
        CreativeCommonsLicenseManager(["cc-by", "cc-sampling-plus"])


def test_normalize_allowed_license_rejects_disallowed_cc_license():
    manager = CreativeCommonsLicenseManager(["cc-by"])

    assert (
        manager.normalize_allowed_license(
            "https://creativecommons.org/licenses/by-nc/4.0/"
        )
        is None
    )
