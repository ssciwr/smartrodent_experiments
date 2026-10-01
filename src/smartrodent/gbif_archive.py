"""Read the occurrence and multimedia tables needed from a GBIF DWCA."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

import pandas as pd


class GbifArchiveReader:
    """Join GBIF media to occurrences without extracting archive members.

    XML supplies table locations, column meanings, field defaults, and join indices.
    Pandas handles the tabular data; this is not a general-purpose DWCA reader.

    Args:
        archive_path: Path to the ZIP produced by pygbif.download_get().
    """

    def __init__(self, archive_path: str | Path):
        """Keep the archive location; reading is explicit."""
        self.archive_path = Path(archive_path)

    def read(self) -> pd.DataFrame:
        """Return one joined row per media item, with media fields prefixed.

        Returns:
            Occurrence fields and separate ``media_`` fields, linked by the
            descriptor's core identifiers rather than physical column order.

        Raises:
            ValueError: If the ZIP or descriptor is invalid, a required table
                or column is missing, or a table location is unsafe.
        """
        try:
            with ZipFile(self.archive_path) as archive:
                if "meta.xml" not in archive.namelist():
                    raise ValueError("GBIF archive is missing meta.xml")
                descriptor = ET.fromstring(archive.read("meta.xml"))
                core = descriptor.find("{*}core")
                media = next(
                    (
                        table
                        for table in descriptor.findall("{*}extension")
                        if table.get("rowType", "").endswith("/Multimedia")
                    ),
                    None,
                )
                if core is None or media is None:
                    raise ValueError(
                        "GBIF archive requires occurrence and multimedia tables"
                    )
                occurrences = self._read_table(archive, core, "id")
                images = self._read_table(archive, media, "coreid")
                if "identifier" not in images.columns:
                    raise ValueError("GBIF multimedia table is missing identifier")
        except (BadZipFile, ET.ParseError) as exc:
            raise ValueError("Invalid GBIF ZIP archive or meta.xml") from exc

        # Keep occurrence licenses/references distinct from their media counterparts.
        return images.add_prefix("media_").merge(
            occurrences,
            left_on="media__core_id",
            right_on="_core_id",
            how="inner",
            validate="many_to_one",
        )

    def _read_table(
        self,
        archive: ZipFile,
        table: ET.Element,
        id_tag: str,
    ) -> pd.DataFrame:
        """Load one table using the descriptor's positions and parsing settings."""
        location = table.findtext("{*}files/{*}location")
        if not location:
            raise ValueError("GBIF archive table has no location")
        path = PurePosixPath(location)
        if path.is_absolute() or ".." in path.parts or "\\" in location:
            raise ValueError(f"Unsafe GBIF archive table location: {location}")
        if location not in archive.namelist():
            raise ValueError(f"GBIF archive is missing {location}")
        identifier = table.find(f"{{*}}{id_tag}")
        if identifier is None or identifier.get("index") is None:
            raise ValueError(f"GBIF table {location} is missing its {id_tag} index")
        id_index = int(identifier.attrib["index"])
        fields = {}
        defaults = {}
        for field in table.findall("{*}field"):
            name = field.attrib["term"].rsplit("/", 1)[-1]
            index = field.get("index")
            if index is None and "default" not in field.attrib:
                raise ValueError(
                    f"GBIF table {location} field {name} has neither index nor default"
                )
            if index is not None:
                fields[int(index)] = name
            if "default" in field.attrib:
                defaults[name] = field.attrib["default"]
        indices = [id_index, *fields]
        if min(indices) < 0:
            raise ValueError(f"Invalid column index in GBIF table {location}")

        separator = table.get("fieldsTerminatedBy", "\\t").replace("\\t", "\t")
        enclosure = table.get("fieldsEnclosedBy", '"')
        with archive.open(location) as stream:
            try:
                frame = pd.read_csv(
                    stream,
                    sep=separator,
                    header=None,
                    skiprows=int(table.get("ignoreHeaderLines", "0")),
                    encoding=table.get("encoding", "UTF-8"),
                    quotechar=enclosure or '"',
                    quoting=csv.QUOTE_MINIMAL if enclosure else csv.QUOTE_NONE,
                    dtype=str,
                    keep_default_na=False,
                )
            except pd.errors.EmptyDataError:
                # Header-only tables are valid empty results, not missing tables.
                frame = pd.DataFrame(columns=range(max(indices) + 1))
        if max(indices) >= len(frame.columns):
            raise ValueError(f"Column index exceeds the width of GBIF table {location}")
        core_ids = frame[id_index]
        frame = frame.rename(columns=fields)
        frame["_core_id"] = core_ids
        # Apply defaults before normalizing empty cells: explicit values must survive.
        for name, default in defaults.items():
            if name in frame.columns:
                frame[name] = frame[name].replace("", default)
            else:
                # Constant descriptor fields have no physical column in the table.
                frame[name] = default
        return frame.replace("", None)
