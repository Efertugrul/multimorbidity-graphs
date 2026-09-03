from __future__ import annotations

from mm_motifs.config import load_config
from mm_motifs.runtime import output_directory, raw_xpt_path


def test_config_digest_and_paths_are_deterministic() -> None:
    first = load_config()
    second = load_config()
    assert first.digest == second.digest
    assert output_directory(first, "audit") == output_directory(second, "audit")
    assert raw_xpt_path(first).name == "LLCP2024.XPT"


def test_year_override_changes_data_path_and_digest() -> None:
    discovery = load_config()
    replication = load_config(year=2023)
    assert replication.year == 2023
    assert raw_xpt_path(replication).name == "LLCP2023.XPT"
    assert discovery.digest != replication.digest
