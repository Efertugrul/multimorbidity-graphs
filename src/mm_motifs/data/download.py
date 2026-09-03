from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

from mm_motifs.config import ProjectConfig
from mm_motifs.runtime import raw_xpt_path, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_brfss(
    config: ProjectConfig,
    force: bool = False,
    keep_archive: bool = False,
) -> Path:
    destination = raw_xpt_path(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(".download.json")
    if destination.exists() and not force:
        return destination

    year = str(config.year)
    url = config.analysis["data"]["urls"][year]
    archive = destination.parent / f"LLCP{year}XPT.zip"
    partial_archive = archive.with_suffix(".zip.part")
    if force or not archive.exists():
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial_archive.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        os.replace(partial_archive, archive)

    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            name for name in bundle.namelist() if name.strip().lower().endswith(".xpt")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one XPT file in {archive}, found {len(candidates)}")
        partial_xpt = destination.with_suffix(".XPT.part")
        with bundle.open(candidates[0]) as source, partial_xpt.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(partial_xpt, destination)

    metadata = {
        "year": config.year,
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "xpt_sha256": _sha256(destination),
        "xpt_bytes": destination.stat().st_size,
    }
    write_json(metadata, metadata_path)
    if not keep_archive:
        archive.unlink()
    return destination
