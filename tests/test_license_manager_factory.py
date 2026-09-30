import pytest

from smartrodent.licenses import (
    CreativeCommonsLicenseManager,
    LicenseManagerFactory,
)


def test_manager_types_contains_creative_commons_family():
    assert "creative-commons" in LicenseManagerFactory.manager_types()


def test_manager_types_maps_creative_commons_to_expected_class():
    manager_types = LicenseManagerFactory.manager_types()

    assert manager_types["creative-commons"] is CreativeCommonsLicenseManager


def test_create_returns_creative_commons_manager():
    manager = LicenseManagerFactory.create("creative-commons", ["cc-by"])

    assert isinstance(manager, CreativeCommonsLicenseManager)


def test_create_passes_allowed_licenses_to_manager():
    manager = LicenseManagerFactory.create(
        "creative-commons", ["cc-by", "cc-by-nc"]
    )

    assert manager.allowed_licenses == frozenset({"cc-by", "cc-by-nc"})


def test_create_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown license family"):
        LicenseManagerFactory.create("unknown", ["example-license"])


def test_create_rejects_empty_family():
    with pytest.raises(ValueError, match="family"):
        LicenseManagerFactory.create("", ["cc-by"])


def test_create_rejects_non_string_family():
    with pytest.raises(TypeError, match="family"):
        LicenseManagerFactory.create(42, ["cc-by"])
