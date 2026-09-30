"""License normalization and allow-list policy abstractions."""

from __future__ import annotations

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
        if scheme not in {"http", "https"}:
            return None
        elif hostname is None:
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
        if normalized_license is None:
            return None
        elif normalized_license not in self.allowed_licenses:
            return None
        else:
            return normalized_license
