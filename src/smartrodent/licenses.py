"""License normalization and allow-list policy abstractions."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from urllib.parse import urlsplit, urlunsplit


class LicenseManagerBase(ABC):
    """Base class for license-family normalization and allow-list policy.

    Subclasses interpret raw license values for one license family. This base
    class owns the configured allow-list, provides family-independent HTTP URL
    normalization, and applies the allow-list to normalized identifiers.

    Args:
        allowed_licenses: Non-empty sequence of canonical license identifiers.

    Raises:
        TypeError: If the collection is not a sequence or contains a non-string
            entry.
        ValueError: If the collection is empty or contains an empty string.
    """

    def __init__(self, allowed_licenses: Sequence[str]) -> None:
        if isinstance(allowed_licenses, str):
            raise TypeError("allowed_licenses must be a sequence, not a string")
        elif not isinstance(allowed_licenses, Sequence):
            raise TypeError("allowed_licenses must be a sequence")
        elif not allowed_licenses:
            raise ValueError("allowed_licenses must contain at least one license")
        else:
            self._validate_license_entries(allowed_licenses)

        self.allowed_licenses = frozenset(allowed_licenses)

    @staticmethod
    def _validate_license_entries(allowed_licenses: Sequence[str]) -> None:
        """Validate the individual identifiers in an allow-list."""
        for license_identifier in allowed_licenses:
            if not isinstance(license_identifier, str):
                raise TypeError("license identifiers must be non-empty strings")
            elif not license_identifier.strip():
                raise ValueError("license identifiers must be non-empty strings")
            else:
                continue

    @staticmethod
    def normalize_url(raw_url: object) -> str | None:
        """Normalize an HTTP(S) URL without interpreting its license family.

        Scheme and hostname are lowercased, surrounding whitespace, query
        parameters, fragments, and trailing path slashes are removed. Invalid
        values and URLs without an HTTP(S) scheme or hostname return ``None``.

        Args:
            raw_url: Potential URL value supplied by a data provider.

        Returns:
            The normalized URL, or ``None`` when the value is not a supported
            URL.
        """
        if not isinstance(raw_url, str):
            return None

        candidate = raw_url.strip()
        if not candidate:
            return None

        try:
            parsed_url = urlsplit(candidate)
            port = parsed_url.port
        except ValueError:
            return None

        scheme = parsed_url.scheme.casefold()
        hostname = parsed_url.hostname
        if scheme not in {"http", "https"} or hostname is None:
            return None
        else:
            normalized_hostname = hostname.casefold()

        if ":" in normalized_hostname:
            normalized_hostname = f"[{normalized_hostname}]"

        normalized_netloc = normalized_hostname
        if port is not None:
            normalized_netloc = f"{normalized_netloc}:{port}"

        normalized_path = parsed_url.path.rstrip("/")
        return urlunsplit((scheme, normalized_netloc, normalized_path, "", ""))

    @abstractmethod
    def normalize_license(self, raw_license: object) -> str | None:
        """Convert a raw family-specific value into a canonical identifier."""

    def normalize_allowed_license(self, raw_license: object) -> str | None:
        """Normalize a raw value and return it only when it is allowed.

        Args:
            raw_license: Provider-specific license value.

        Returns:
            The canonical identifier when recognized and configured as allowed,
            otherwise ``None``.
        """
        normalized_license = self.normalize_license(raw_license)
        if (
            normalized_license is None
            or normalized_license not in self.allowed_licenses
        ):
            return None
        else:
            return normalized_license


class CreativeCommonsLicenseManager(LicenseManagerBase):
    """Normalize and authorize supported Creative Commons licenses.

    The manager recognizes official Creative Commons URLs, common compact
    codes, and descriptive Creative Commons names. It does not interpret other
    license families. Normalization is independent of the configured allowlist;
    use :meth:`normalize_allowed_license` when acceptance policy is required.
    """

    def __init__(self, allowed_licenses: Sequence[str]) -> None:
        super().__init__(allowed_licenses)
        unsupported_licenses = self.allowed_licenses - self.supported_licenses()
        if unsupported_licenses:
            unsupported = ", ".join(sorted(unsupported_licenses))
            raise ValueError(f"unsupported Creative Commons license: {unsupported}")

    @staticmethod
    def supported_licenses() -> frozenset[str]:
        """Return the canonical Creative Commons identifiers understood here."""
        return frozenset(
            {
                "cc0",
                "cc-by",
                "cc-by-sa",
                "cc-by-nd",
                "cc-by-nc",
                "cc-by-nc-sa",
                "cc-by-nc-nd",
            }
        )

    def normalize_license(self, raw_license: object) -> str | None:
        """Normalize a supported Creative Commons representation.

        Args:
            raw_license: URL, compact code, or descriptive license name.

        Returns:
            A canonical Creative Commons identifier, or ``None`` when the value
            is missing, malformed, or belongs to another license family.
        """
        if not isinstance(raw_license, str):
            return None

        license_value = raw_license.strip().casefold()
        if not license_value:
            return None

        normalized_license = self._normalize_url_license(license_value)
        if normalized_license is not None:
            return normalized_license

        normalized_license = self._normalize_compact_code(license_value)
        if normalized_license is not None:
            return normalized_license

        return self._normalize_descriptive_name(license_value)

    def _normalize_url_license(self, license_value: str) -> str | None:
        """Normalize an official Creative Commons license URL."""
        # Reject values that are not structurally valid HTTP(S) URLs before
        # applying any Creative Commons-specific interpretation.
        normalized_url = self.normalize_url(license_value)
        if normalized_url is None:
            return None

        # A matching path on another host does not identify an official
        # Creative Commons license.
        parsed_url = urlsplit(normalized_url)
        if parsed_url.hostname not in {
            "creativecommons.org",
            "www.creativecommons.org",
        }:
            return None

        path_parts = [part.casefold() for part in parsed_url.path.split("/") if part]

        # Official paths have the shape category/code/version, optionally
        # followed by the document selector "legalcode".
        if len(path_parts) not in {3, 4}:
            return None
        if len(path_parts) == 4 and path_parts[3] != "legalcode":
            return None

        # Versions are numeric identifiers such as 1.0 or 4.0. The version is
        # validated but omitted from this project's canonical license code.
        if re.fullmatch(r"\d+(?:\.\d+)?", path_parts[2]) is None:
            return None

        # Standard licenses use /licenses/<code>/<version>, while CC0 uses the
        # distinct /publicdomain/zero/<version> namespace.
        if path_parts[0] == "licenses":
            canonical_license = f"cc-{path_parts[1]}"
        elif path_parts[:2] == ["publicdomain", "zero"]:
            canonical_license = "cc0"
        else:
            return None

        # A syntactically valid Creative Commons URL can still name a license
        # outside the set implemented by this manager.
        if canonical_license not in self.supported_licenses():
            return None
        return canonical_license

    def _normalize_compact_code(self, license_value: str) -> str | None:
        """Normalize common compact Creative Commons identifiers."""
        code_match = re.fullmatch(
            r"(?:cc[-_ ]*)?"
            r"(?P<license>by(?:[-_ ](?:nc(?:[-_ ](?:nd|sa))?|nd|sa))?|zero|0)"
            r"(?:[-_ ]*\d+(?:[._-]\d+)?)?",
            license_value,
        )
        if code_match is None:
            return None

        matched_code = re.sub(r"[-_ ]+", "-", code_match.group("license"))
        if matched_code in {"0", "zero"}:
            canonical_license = "cc0"
        else:
            canonical_license = f"cc-{matched_code}"

        return canonical_license

    def _normalize_descriptive_name(self, license_value: str) -> str | None:
        """Normalize descriptive names from the Creative Commons family."""
        if not license_value.startswith("creative commons"):
            return None

        if "public domain" in license_value and "zero" in license_value:
            return "cc0"
        if "attribution" not in license_value:
            return None

        has_noncommercial = (
            "noncommercial" in license_value or "non-commercial" in license_value
        )
        has_no_derivatives = (
            "noderivatives" in license_value
            or "no derivatives" in license_value
            or "no-derivatives" in license_value
        )
        has_share_alike = (
            "sharealike" in license_value
            or "share alike" in license_value
            or "share-alike" in license_value
        )
        if has_no_derivatives and has_share_alike:
            return None

        canonical_license = "cc-by"
        if has_noncommercial:
            canonical_license += "-nc"
        if has_no_derivatives:
            canonical_license += "-nd"
        elif has_share_alike:
            canonical_license += "-sa"

        return canonical_license


# This mutable module-level registry is an intentional exception to the
# project's usual rule: runtime registration is the extension mechanism for
# adding license families without modifying manager creation logic.
LICENSE_MANAGER_TYPES: dict[str, type[LicenseManagerBase]] = {
    "creative-commons": CreativeCommonsLicenseManager,
}


def _validate_license_family(family: object) -> None:
    """Validate a registry family identifier."""
    if not isinstance(family, str):
        raise TypeError("license family must be a string")
    elif not family.strip():
        raise ValueError("license family must not be empty")
    else:
        return


def register_license_manager(
    family: str, manager_type: type[LicenseManagerBase]
) -> None:
    """Register a manager class for a new license family.

    Args:
        family: Exact family identifier used by configuration.
        manager_type: Concrete manager class associated with ``family``.

    Raises:
        TypeError: If the family is not a string or the manager is not a proper
            subclass of :class:`LicenseManagerBase`.
        ValueError: If the family is empty or already registered.
    """
    _validate_license_family(family)
    if not isinstance(manager_type, type):
        raise TypeError("manager_type must be a LicenseManagerBase subclass")
    elif manager_type is LicenseManagerBase:
        raise TypeError("manager_type must be a LicenseManagerBase subclass")
    elif not issubclass(manager_type, LicenseManagerBase):
        raise TypeError("manager_type must be a LicenseManagerBase subclass")
    elif family in LICENSE_MANAGER_TYPES:
        raise ValueError(f"license family is already registered: {family}")
    else:
        LICENSE_MANAGER_TYPES[family] = manager_type


def create_license_manager(
    family: str, allowed_licenses: Sequence[str]
) -> LicenseManagerBase:
    """Instantiate the manager registered for a license family.

    Args:
        family: Exact family identifier used by configuration.
        allowed_licenses: Canonical identifiers accepted by the manager.

    Returns:
        The configured manager for ``family``.

    Raises:
        TypeError: If ``family`` is not a string.
        ValueError: If ``family`` is empty or is not registered.
    """
    _validate_license_family(family)
    manager_type = LICENSE_MANAGER_TYPES.get(family)
    if manager_type is None:
        raise ValueError(f"Unknown license family: {family}")
    return manager_type(allowed_licenses)
