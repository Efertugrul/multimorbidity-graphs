from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat


def xport_dimensions(path: Path) -> tuple[int, list[str]]:
    _, metadata = pyreadstat.read_xport(
        str(path),
        metadataonly=True,
        encoding="latin1",
        disable_datetime_conversion=True,
    )
    row_count = getattr(metadata, "number_rows", None)
    if row_count is None:
        row_count = sum(len(chunk) for chunk in iter_xport(path, 50000))
    return int(row_count), list(metadata.column_names)


def variable_metadata(path: Path) -> pd.DataFrame:
    _, metadata = pyreadstat.read_xport(
        str(path),
        metadataonly=True,
        encoding="latin1",
        disable_datetime_conversion=True,
    )
    labels = getattr(metadata, "column_names_to_labels", {}) or {}
    readstat_types = getattr(metadata, "readstat_variable_types", {}) or {}
    storage_widths = getattr(metadata, "variable_storage_width", {}) or {}
    display_widths = getattr(metadata, "variable_display_width", {}) or {}
    formats = getattr(metadata, "original_variable_types", {}) or {}
    rows: list[dict[str, Any]] = []
    for position, name in enumerate(metadata.column_names, start=1):
        rows.append(
            {
                "position": position,
                "variable": name,
                "label": labels.get(name),
                "readstat_type": readstat_types.get(name),
                "storage_width": storage_widths.get(name),
                "display_width": display_widths.get(name),
                "source_format": formats.get(name),
            }
        )
    return pd.DataFrame(rows)


def iter_xport(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    reader = pd.read_sas(
        path,
        format="xport",
        iterator=True,
        chunksize=chunk_size,
        encoding="latin-1",
    )
    yield from reader


def read_selected(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame, _ = pyreadstat.read_xport(
        str(path),
        usecols=list(dict.fromkeys(columns)),
        encoding="latin1",
        disable_datetime_conversion=True,
    )
    return frame
