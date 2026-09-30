import pytest

from smartrodent.licenses import (
    LICENSE_MANAGER_TYPES,
    CreativeCommonsLicenseManager,
    LicenseManagerBase,
    create_license_manager,
    register_license_manager,
)


class ExampleLicenseManager(LicenseManagerBase):
    """Minimal manager used to test runtime registration."""

    def normalize_license(self, raw_license: object) -> str | None:
        if raw_license == "Example":
            return "example"
        return None


class UnrelatedClass:
    pass


@pytest.fixture(autouse=True)
def restore_license_manager_types():
    original_manager_types = LICENSE_MANAGER_TYPES.copy()
    yield
    LICENSE_MANAGER_TYPES.clear()
    LICENSE_MANAGER_TYPES.update(original_manager_types)


def test_registry_contains_creative_commons_family():
    assert "creative-commons" in LICENSE_MANAGER_TYPES


def test_registry_maps_creative_commons_to_expected_class():
    assert LICENSE_MANAGER_TYPES["creative-commons"] is CreativeCommonsLicenseManager


def test_register_license_manager_adds_family():
    register_license_manager("example", ExampleLicenseManager)

    assert LICENSE_MANAGER_TYPES["example"] is ExampleLicenseManager


def test_create_license_manager_returns_creative_commons_manager():
    manager = create_license_manager("creative-commons", ["cc-by"])

    assert isinstance(manager, CreativeCommonsLicenseManager)


def test_create_license_manager_passes_allowed_licenses():
    manager = create_license_manager(
        "creative-commons", ["cc-by", "cc-by-nc"]
    )

    assert manager.allowed_licenses == frozenset({"cc-by", "cc-by-nc"})


def test_create_license_manager_uses_runtime_registration():
    register_license_manager("example", ExampleLicenseManager)

    manager = create_license_manager("example", ["example"])

    assert isinstance(manager, ExampleLicenseManager)


def test_register_license_manager_rejects_non_class_manager_type():
    with pytest.raises(TypeError, match="subclass"):
        register_license_manager("example", ExampleLicenseManager(["example"]))


def test_register_license_manager_rejects_unrelated_class():
    with pytest.raises(TypeError, match="subclass"):
        register_license_manager("example", UnrelatedClass)


def test_register_license_manager_rejects_base_class():
    with pytest.raises(TypeError, match="subclass"):
        register_license_manager("example", LicenseManagerBase)


def test_register_license_manager_rejects_duplicate_family():
    with pytest.raises(ValueError, match="already registered"):
        register_license_manager("creative-commons", CreativeCommonsLicenseManager)


def test_register_license_manager_rejects_empty_family():
    with pytest.raises(ValueError, match="family"):
        register_license_manager("", ExampleLicenseManager)


def test_register_license_manager_rejects_whitespace_only_family():
    with pytest.raises(ValueError, match="family"):
        register_license_manager("   ", ExampleLicenseManager)


def test_register_license_manager_rejects_non_string_family():
    with pytest.raises(TypeError, match="family"):
        register_license_manager(42, ExampleLicenseManager)


def test_create_license_manager_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown license family"):
        create_license_manager("unknown", ["example"])


def test_create_license_manager_rejects_empty_family():
    with pytest.raises(ValueError, match="family"):
        create_license_manager("", ["cc-by"])


def test_create_license_manager_rejects_whitespace_only_family():
    with pytest.raises(ValueError, match="family"):
        create_license_manager("   ", ["cc-by"])


def test_create_license_manager_rejects_non_string_family():
    with pytest.raises(TypeError, match="family"):
        create_license_manager(42, ["cc-by"])
