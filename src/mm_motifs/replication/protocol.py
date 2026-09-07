from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from mm_motifs.features.populations import primary_geography_codes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_motif(
    nodes_json: str,
    edges_json: str,
) -> tuple[str, str, str]:
    nodes = sorted(str(value) for value in json.loads(nodes_json))
    edges = sorted(
        tuple(sorted((str(source), str(target))))
        for source, target in json.loads(edges_json)
    )
    payload = json.dumps(
        {"nodes": nodes, "edges": [list(edge) for edge in edges]},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload, digest, f"motif_{digest[:16]}"


def _verify_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing protocol checksums: {checksum_path}")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe protocol checksum path: {relative}")
        expected[path.as_posix()] = digest
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != actual:
        raise ValueError("Protocol checksum inventory mismatch")
    for relative, digest in expected.items():
        if _sha256(directory / relative) != digest:
            raise ValueError(f"Protocol checksum mismatch: {relative}")


def _verify_lock(directory: Path) -> str:
    lock = json.loads(
        (directory / "protocol.lock.json").read_text(encoding="utf-8")
    )
    files = lock["payload_files"]
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "protocol.lock.json"}
    }
    if set(files) != actual:
        raise ValueError("Protocol lock inventory mismatch")
    for relative, digest in files.items():
        if _sha256(directory / relative) != digest:
            raise ValueError(f"Protocol lock mismatch: {relative}")
    encoded = "".join(
        f"{files[relative]}  {relative}\n" for relative in sorted(files)
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != lock["protocol_sha256"]:
        raise ValueError("Protocol aggregate digest mismatch")
    if lock["protocol_id"] != f"phase4_2023_{digest[:12]}":
        raise ValueError("Protocol ID does not match its aggregate digest")
    return digest


def _project_root(directory: Path) -> Path:
    for candidate in [directory, *directory.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / ".git"
        ).exists():
            return candidate
    raise FileNotFoundError("Protocol is not inside its source repository")


def _git_blob(project_root: Path, commit: str, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe discovery source path: {relative}")
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise ValueError(
            f"Missing discovery source at {commit}:{path.as_posix()}"
        )
    return completed.stdout


def _validate_discovery_provenance(
    protocol: dict[str, Any],
    directory: Path,
    vocabulary: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> None:
    source = protocol["discovery_source"]
    project_root = _project_root(directory)
    archive = str(source["archive"]).rstrip("/")
    commit = str(source["source_commit"])
    archive_commit = str(source["archive_commit"])
    paths = {
        "manifest_sha256": f"{archive}/manifest.json",
        "motif_catalog_sha256": (
            f"{archive}/motif_catalog_enriched.csv.gz"
        ),
        "ses_results_sha256": f"{archive}/exploratory_ses_permutation.csv",
    }
    blobs = {
        name: _git_blob(project_root, archive_commit, relative)
        for name, relative in paths.items()
    }
    for name, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != str(source[name]):
            raise ValueError(f"Discovery source checksum mismatch: {name}")
    source_manifest = json.loads(blobs["manifest_sha256"])
    if (
        source_manifest["config_digest"] != source["config_digest"]
        or source_manifest["git_commit"] != commit
        or int(source_manifest["dataset_year"]) != 2024
    ):
        raise ValueError("Discovery manifest provenance changed")
    catalog = pd.read_csv(
        io.BytesIO(blobs["motif_catalog_sha256"]),
        compression="gzip",
        low_memory=False,
    )
    qualifying = catalog[
        catalog["primary_support_x"].astype(bool)
        & catalog["p_boot_support_20pct"].ge(0.90)
    ].copy()
    if set(qualifying["motif_id"]) != set(vocabulary["motif_id"]):
        raise ValueError(
            "Frozen vocabulary is not the complete qualifying discovery set"
        )
    source_columns = {
        "node_count": "node_count",
        "edge_count": "edge_count",
        "node_labels": "node_labels",
        "edge_list": "canonical_edges",
        "occurrence_class_id": "occurrence_class_id",
        "is_closed": "is_closed",
        "is_maximal": "is_maximal",
        "support_count": "discovery_support_count",
        "support_fraction": "discovery_support_fraction",
        "distinct_state_support_count": (
            "discovery_distinct_state_support_count"
        ),
        "paired_state_count_x": "discovery_paired_state_count",
        "p_boot_support_20pct": "discovery_p_boot_support_20pct",
        "bootstrap_support_mean": "discovery_bootstrap_support_mean",
        "bootstrap_support_025": "discovery_bootstrap_support_025",
        "bootstrap_support_975": "discovery_bootstrap_support_975",
    }
    source_frame = qualifying[["motif_id", *source_columns]].rename(
        columns={
            column: f"_source_{column}" for column in source_columns
        }
    )
    merged = vocabulary.merge(
        source_frame,
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    for source_column, frozen_column in source_columns.items():
        source_values = merged[f"_source_{source_column}"]
        frozen_values = merged[frozen_column]
        if pd.api.types.is_numeric_dtype(source_values):
            if not np.allclose(
                source_values,
                frozen_values,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Vocabulary provenance mismatch: {frozen_column}"
                )
        elif not source_values.astype(str).equals(
            frozen_values.astype(str)
        ):
            raise ValueError(
                f"Vocabulary provenance mismatch: {frozen_column}"
            )
    ses = pd.read_csv(io.BytesIO(blobs["ses_results_sha256"]))
    surviving = ses[ses["permutation_max_t_p_value"].le(0.05)].copy()
    if set(surviving["motif_id"]) != set(hypotheses["motif_id"]):
        raise ValueError(
            "Confirmatory family is not the complete maxT-surviving set"
        )
    hypothesis_source_columns = {
        "support_count": "discovery_support_count",
        "lower_support_count": "discovery_lower_support_count",
        "higher_support_count": "discovery_higher_support_count",
        "paired_density_residual_difference": (
            "discovery_density_residual_difference"
        ),
        "paired_density_residual_t": (
            "discovery_studentized_statistic"
        ),
        "permutation_max_t_p_value": "discovery_max_t_p_value",
    }
    hypothesis_source = surviving[
        ["motif_id", *hypothesis_source_columns]
    ].rename(
        columns={
            column: f"_source_{column}"
            for column in hypothesis_source_columns
        }
    )
    hypothesis_merged = hypotheses.merge(
        hypothesis_source,
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    for source_column, frozen_column in hypothesis_source_columns.items():
        if not np.allclose(
            hypothesis_merged[f"_source_{source_column}"],
            hypothesis_merged[frozen_column],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Hypothesis provenance mismatch: {frozen_column}"
            )
    expected_sign = np.where(
        surviving["paired_density_residual_difference"].gt(0),
        1,
        -1,
    )
    source_signs = dict(zip(surviving["motif_id"], expected_sign, strict=True))
    observed_signs = hypotheses.set_index("motif_id")[
        "direction_sign"
    ].astype(int)
    if any(
        int(observed_signs[motif_id]) != int(direction)
        for motif_id, direction in source_signs.items()
    ):
        raise ValueError("Confirmatory direction provenance mismatch")


def _validate_motifs(
    protocol: dict[str, Any],
    directory: Path,
    condition_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = protocol["methodological_replication"]
    vocabulary = pd.read_csv(directory / settings["registry"])
    expected_count = int(settings["frozen_motif_count"])
    if len(vocabulary) != expected_count:
        raise ValueError("Frozen methodological motif count changed")
    if vocabulary["motif_id"].duplicated().any():
        raise ValueError("Frozen methodological motif IDs are not unique")
    if not vocabulary["discovery_p_boot_support_20pct"].ge(0.90).all():
        raise ValueError("Methodological vocabulary violates stability rule")
    if not vocabulary["discovery_support_fraction"].ge(0.20).all():
        raise ValueError("Methodological vocabulary violates support rule")
    if not vocabulary["discovery_paired_state_count"].ge(
        int(settings["primary_2023_reproduction"]["minimum_state_pairs"])
    ).all():
        raise ValueError("Methodological vocabulary lacks state opportunity")
    for row in vocabulary.itertuples(index=False):
        payload, digest, motif_id = _canonical_motif(
            str(row.node_labels),
            str(row.canonical_edges),
        )
        if (
            payload != row.canonical_payload
            or digest != row.canonical_sha256
            or motif_id != row.motif_id
        ):
            raise ValueError(f"Canonical motif mismatch: {row.motif_id}")
        nodes = json.loads(row.node_labels)
        edges = json.loads(row.canonical_edges)
        if (
            len(nodes) != int(row.node_count)
            or len(edges) != int(row.edge_count)
            or not set(nodes).issubset(condition_names)
        ):
            raise ValueError(f"Motif shape mismatch: {row.motif_id}")
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        if not nx.is_connected(graph):
            raise ValueError(f"Disconnected frozen motif: {row.motif_id}")
        minimum_support = math.ceil(
            0.20 * 2 * int(row.discovery_paired_state_count)
        )
        if int(row.discovery_support_count) < minimum_support:
            raise ValueError(f"Motif support mismatch: {row.motif_id}")
    families = pd.read_csv(directory / settings["family_registry"])
    if len(families) != int(settings["frozen_occurrence_family_count"]):
        raise ValueError("Frozen occurrence-family count changed")
    motif_to_class = vocabulary.set_index("motif_id")[
        "occurrence_class_id"
    ].to_dict()
    members: list[str] = []
    for row in families.itertuples(index=False):
        class_members = json.loads(row.member_motif_ids)
        if (
            row.representative_motif_id not in class_members
            or len(class_members) != int(row.robust_motif_count)
            or any(
                motif_to_class.get(motif_id) != row.occurrence_class_id
                for motif_id in class_members
            )
        ):
            raise ValueError(
                f"Occurrence-family mismatch: {row.occurrence_class_id}"
            )
        members.extend(class_members)
    if len(members) != len(set(members)) or set(members) != set(
        vocabulary["motif_id"]
    ):
        raise ValueError("Occurrence families do not partition vocabulary")
    return vocabulary, families


def _validate_hypotheses(
    protocol: dict[str, Any],
    directory: Path,
    vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    settings = protocol["confirmatory_replication"]
    hypotheses = pd.read_csv(directory / settings["registry"])
    if (
        len(hypotheses) != int(settings["hypothesis_count"])
        or hypotheses["motif_id"].duplicated().any()
        or hypotheses["hypothesis_id"].duplicated().any()
        or hypotheses["family_order"].astype(int).tolist() != [1, 2, 3]
        or not hypotheses["alternative"].eq(
            "direction_sign_times_effect_greater_than_zero"
        ).all()
        or not hypotheses["effect_contrast"].eq(
            "lower_minus_higher_density_residual"
        ).all()
    ):
        raise ValueError("Confirmatory hypothesis family changed")
    expected = pd.DataFrame(settings["frozen_hypotheses"]).sort_values(
        "hypothesis_id"
    )
    observed = hypotheses[
        [
            "hypothesis_id",
            "motif_id",
            "expected_enrichment",
            "direction_sign",
        ]
    ].sort_values("hypothesis_id")
    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    motif_lookup = vocabulary.set_index("motif_id")
    for row in hypotheses.itertuples(index=False):
        if row.motif_id not in motif_lookup.index:
            raise ValueError("Confirmatory motif is absent from vocabulary")
        motif = motif_lookup.loc[row.motif_id]
        if (
            row.canonical_sha256 != motif["canonical_sha256"]
            or row.edge_list != motif["canonical_edges"]
            or float(row.discovery_p_boot_support_20pct) < 0.90
        ):
            raise ValueError(
                f"Confirmatory motif provenance mismatch: {row.motif_id}"
            )
        expected_sign = -1 if row.expected_enrichment == "higher" else 1
        if int(row.direction_sign) != expected_sign:
            raise ValueError(f"Hypothesis direction mismatch: {row.motif_id}")
    multiplicity = settings["multiplicity"]
    if (
        multiplicity["method"] != "holm"
        or int(multiplicity["family_size"]) != 3
        or float(multiplicity["alpha"]) != 0.05
    ):
        raise ValueError("Confirmatory multiplicity rule changed")
    return hypotheses


def _validate_target_contract(
    protocol: dict[str, Any],
    analysis: dict[str, Any],
    graph: dict[str, Any],
    target_codes: set[int],
) -> None:
    education = analysis["socioeconomic"]["definitions"]["education_binary"]
    if (
        int(analysis["data"]["year"]) != 2023
        or set(analysis["socioeconomic"]["definitions"])
        != {"education_binary"}
        or education["source_by_year"]["2023"] != "_EDUCAG"
        or education["groups"]["lower"] != [1, 2]
        or education["groups"]["higher"] != [3, 4]
        or protocol["ses"]["income_analysis"] != "prohibited"
        or set(analysis["phase25"]["geography_codes"]) != target_codes
        or int(analysis["phase25"]["minimum_population_n"]) != 1000
    ):
        raise ValueError("Frozen 2023 target analysis changed")
    settings = graph["phase4"]
    bootstrap = settings["bootstrap"]
    edge_criteria = graph["edge_criteria"]
    node_criteria = graph["node_criteria"]
    if (
        graph["estimator"]["name"] != "phase4_confirmatory_replication"
        or int(edge_criteria["minimum_complete_n"]) != 500
        or int(edge_criteria["minimum_condition_cases"]) != 30
        or int(edge_criteria["minimum_cooccurring_cases"]) != 10
        or edge_criteria["require_fdr"] is not False
        or int(node_criteria["minimum_valid_n"]) != 500
        or int(node_criteria["minimum_cases"]) != 30
        or float(settings["point_phi_threshold"]) != 0.12
        or float(settings["bootstrap_selection_threshold"]) != 0.12
        or float(settings["minimum_bootstrap_selection_probability"])
        != 0.90
        or int(bootstrap["replicates"]) != 500
        or int(bootstrap["seed"]) != 20230902
        or bootstrap["type"] != "bootstrap"
        or bootstrap["mse"] is not True
        or bootstrap["require_all_replicates_valid"] is not True
        or settings["motif_discovery"]["rerun_gspan"] is not False
        or settings["motif_discovery"]["permit_new_motifs"] is not False
    ):
        raise ValueError("Frozen 2023 graph target changed")
    population = protocol["population"]
    motif_evaluability = protocol["motifs"]["evaluability_rule"]
    edge_bootstrap = protocol["edge_bootstrap"]
    if (
        population["confirmatory_decision_set"]
        != "primary_2023_jurisdictions_only"
        or population["sensitivity_can_rescue_primary_failure"] is not False
        or population["sensitivity_is_descriptive_only"] is not True
        or population["graph_usability"][
            "require_every_point_eligible_dyad_bootstrap_complete"
        ]
        is not True
        or motif_evaluability["required_edges_only"] is not True
        or motif_evaluability[
            "each_required_edge_must_be_point_eligible_in_both_ses_graphs"
        ]
        is not True
        or motif_evaluability[
            "extra_motif_internal_dyads_need_not_be_eligible"
        ]
        is not True
        or edge_bootstrap["job_universe"]
        != "point_eligible_dyads_only"
        or edge_bootstrap["checkpoints"]["result_row_statuses"]
        != ["ok", "partial", "state_error"]
        or edge_bootstrap["checkpoints"][
            "permit_zero_valid_replicates_for_failure_rows"
        ]
        is not True
    ):
        raise ValueError("Phase 4 evaluability or decision set changed")
    confirmatory = protocol["confirmatory_replication"]
    calibration = confirmatory["calibration"]
    if (
        int(confirmatory["permutations"]) != 99999
        or int(confirmatory["seed"]) != 20230904
        or confirmatory["rng"] != "numpy_PCG64"
        or confirmatory["sign_rule"]
        != "one_rademacher_sign_per_state_shared_across_all_hypotheses"
        or confirmatory["monte_carlo_p_value"] != "plus_one_correction"
        or confirmatory["statistic_definition"][
            "sample_variance_denominator"
        ]
        != "state_pair_count_minus_one"
        or confirmatory["one_sided_exceedance_comparison"]
        != "greater_than_or_equal_including_ties"
        or confirmatory["state_order"] != "ascending_numeric_fips"
        or confirmatory["rng_draw"]
        != "integers_0_or_1_int8_then_transform_2x_minus_1"
        or confirmatory["matched_sensitivity_signs"]
        != "subset_primary_sign_bank_by_fips"
        or calibration["name"]
        != "studentized_rademacher_wild_sign_flip"
        or calibration["exact_only_under"]
        != (
            "componentwise_state_contrast_sign_symmetry_or_"
            "within_state_lower_higher_label_exchangeability"
        )
        or calibration[
            "exchangeability_is_not_guaranteed_by_observational_education_groups"
        ]
        is not True
        or confirmatory["hypothesis_replication_rule"][
            "primary_analysis_only"
        ]
        is not True
    ):
        raise ValueError("Phase 4 inferential endpoint changed")


def _validate_execution_sources(
    protocol: dict[str, Any],
    directory: Path,
) -> None:
    project_root = _project_root(directory)
    execution = protocol["execution"]
    input_integrity = execution["input_integrity"]
    pairs = [
        (
            directory
            / protocol["frozen_snapshots"]["target_analysis"]["path"],
            project_root / execution["target_analysis_config"],
        ),
        (
            directory / protocol["frozen_snapshots"]["target_graph"]["path"],
            project_root / execution["target_graph_config"],
        ),
        (
            directory
            / protocol["frozen_snapshots"]["condition_registry"]["path"],
            project_root / execution["condition_registry"],
        ),
        (
            directory
            / protocol["frozen_snapshots"]["bootstrap_script"]["path"],
            project_root / "scripts/phi_bootstrap.R",
        ),
    ]
    for frozen, active in pairs:
        if _sha256(frozen) != _sha256(active):
            raise ValueError(f"Executable source differs from snapshot: {active}")
    if (
        execution["command"] != "multimorbidity-motifs phase4"
        or execution["permit_skip_flags"] is not False
        or execution["reject_any_config_snapshot_mismatch"] is not True
        or execution["require_clean_tagged_release_commit"] is not True
        or input_integrity["require_downloader_metadata"] is not True
        or input_integrity["require_exact_frozen_source_url"] is not True
        or input_integrity["require_metadata_xpt_sha256_match"] is not True
        or input_integrity[
            "require_metadata_xpt_byte_count_match"
        ]
        is not True
        or int(input_integrity["require_expected_row_count"]) != 433323
        or input_integrity["require_all_analysis_variables"] is not True
    ):
        raise ValueError("Frozen Phase 4 execution command changed")


def _validate_release_context(
    protocol: dict[str, Any],
    directory: Path,
) -> None:
    project_root = _project_root(directory)
    tag = str(protocol["protocol"]["release_tag"])
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", f"refs/tags/{tag}"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tag_type.returncode or tag_type.stdout.strip() != "tag":
        raise ValueError("Phase 4 requires its annotated release tag")
    tag_commit = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head_commit != tag_commit:
        raise ValueError("Phase 4 must run from its tagged release commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise ValueError("Phase 4 requires a clean tagged worktree")


def validate_replication_protocol(
    directory: str | Path,
    require_release_context: bool = False,
) -> dict[str, Any]:
    root = Path(directory).resolve()
    _verify_checksums(root)
    aggregate_digest = _verify_lock(root)
    protocol = yaml.safe_load(
        (root / "protocol.yaml").read_text(encoding="utf-8")
    )
    metadata = protocol["protocol"]
    lock_record = json.loads(
        (root / "protocol.lock.json").read_text(encoding="utf-8")
    )
    if (
        metadata["status"] != "FROZEN_BEFORE_2023_ANALYSIS"
        or int(metadata["target_year"]) != 2023
        or bool(metadata["analytic_2023_data_used_during_freeze"])
        or bool(lock_record["analytic_2023_data_used_during_freeze"])
        or lock_record["release_tag"] != metadata["release_tag"]
        or metadata["immutable_freeze_requirement"][
            "containing_commit_must_be_tagged"
        ]
        is not True
    ):
        raise ValueError("Protocol was not frozen before 2023 analysis")
    for snapshot in protocol["frozen_snapshots"].values():
        if _sha256(root / snapshot["path"]) != snapshot["source_sha256"]:
            raise ValueError(f"Frozen source changed: {snapshot['path']}")
    conditions = yaml.safe_load(
        (root / protocol["frozen_snapshots"]["condition_registry"]["path"])
        .read_text(encoding="utf-8")
    )
    condition_names = {
        condition["canonical_name"] for condition in conditions["conditions"]
    }
    if (
        conditions["registry_version"] != "0.1.0"
        or len(condition_names) != 10
        or any(
            "2023" not in condition["source_by_year"]
            for condition in conditions["conditions"]
        )
    ):
        raise ValueError("Frozen condition registry changed")
    analysis = yaml.safe_load(
        (
            root
            / protocol["frozen_snapshots"]["target_analysis"]["path"]
        ).read_text(encoding="utf-8")
    )
    graph = yaml.safe_load(
        (
            root / protocol["frozen_snapshots"]["target_graph"]["path"]
        ).read_text(encoding="utf-8")
    )
    if (
        bool(protocol["motifs"]["rerun_gspan_in_2023"])
        or bool(protocol["motifs"]["permit_2023_motif_discovery"])
    ):
        raise ValueError("The replication protocol cannot rerun motif discovery")
    vocabulary, families = _validate_motifs(
        protocol,
        root,
        condition_names,
    )
    hypotheses = _validate_hypotheses(
        protocol,
        root,
        vocabulary,
    )
    target = pd.read_csv(root / protocol["population"]["target_registry"])
    discovery = pd.read_csv(
        root / protocol["population"]["discovery_registry"]
    )
    target_codes = set(target["geography_code"].astype(int))
    discovery_codes = set(discovery["geography_code"].astype(int))
    if (
        target_codes != set(primary_geography_codes())
        or len(discovery_codes) != 47
        or not discovery_codes.issubset(target_codes)
        or target_codes & {66, 72, 78}
    ):
        raise ValueError("Frozen jurisdiction registry changed")
    _validate_target_contract(
        protocol,
        analysis,
        graph,
        target_codes,
    )
    _validate_execution_sources(protocol, root)
    _validate_discovery_provenance(
        protocol,
        root,
        vocabulary,
        hypotheses,
    )
    if require_release_context:
        _validate_release_context(protocol, root)
    return {
        "protocol_id": json.loads(
            (root / "protocol.lock.json").read_text(encoding="utf-8")
        )["protocol_id"],
        "protocol_sha256": aggregate_digest,
        "methodological_motif_count": len(vocabulary),
        "occurrence_family_count": len(families),
        "confirmatory_hypothesis_count": len(hypotheses),
        "target_jurisdiction_count": len(target_codes),
        "discovery_jurisdiction_count": len(discovery_codes),
    }
