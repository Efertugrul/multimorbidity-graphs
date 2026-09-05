from __future__ import annotations

import argparse
from collections.abc import Sequence

from mm_motifs.config import load_config
from mm_motifs.data.download import download_brfss
from mm_motifs.workflows.audit import run_audit
from mm_motifs.workflows.phase25 import run_phase25
from mm_motifs.workflows.phase26 import run_phase26
from mm_motifs.workflows.phase3 import run_phase3
from mm_motifs.workflows.prototype import run_prototype


def _add_config_arguments(
    parser: argparse.ArgumentParser,
    analysis_default: str = "configs/analysis.yaml",
    graph_default: str = "configs/graph.yaml",
) -> None:
    parser.add_argument("--analysis", default=analysis_default)
    parser.add_argument("--conditions", default="configs/conditions.yaml")
    parser.add_argument("--graph", default=graph_default)
    parser.add_argument("--year", type=int)
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimorbidity-motifs",
        description="BRFSS multimorbidity graph Phases 0–3",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official BRFSS XPT data")
    _add_config_arguments(download)
    download.add_argument("--force", action="store_true")
    download.add_argument("--keep-archive", action="store_true")

    audit = subparsers.add_parser("audit", help="Run the Phase 1 BRFSS audit")
    _add_config_arguments(audit)

    prototype = subparsers.add_parser(
        "prototype",
        help="Run Phase 2 population graph generation",
    )
    _add_config_arguments(prototype)
    prototype.add_argument("--no-persist-harmonized", action="store_true")

    phase02 = subparsers.add_parser(
        "phase02",
        help="Download data and run Phases 1–2",
    )
    _add_config_arguments(phase02)
    phase02.add_argument("--skip-download", action="store_true")
    phase02.add_argument("--keep-archive", action="store_true")
    phase02.add_argument("--no-persist-harmonized", action="store_true")

    phase25 = subparsers.add_parser(
        "phase25",
        help="Run all-state Phase 2.5 graph calibration",
    )
    _add_config_arguments(
        phase25,
        analysis_default="configs/analysis_phase25.yaml",
        graph_default="configs/graph_phase25.yaml",
    )
    phase25.add_argument("--workers", type=int, default=4)
    phase25.add_argument("--skip-bootstrap", action="store_true")

    phase26 = subparsers.add_parser(
        "phase26",
        help="Run education-primary phi stability calibration",
    )
    _add_config_arguments(
        phase26,
        analysis_default="configs/analysis_phase25.yaml",
        graph_default="configs/graph_phase26.yaml",
    )
    phase26.add_argument("--workers", type=int, default=4)

    phase3 = subparsers.add_parser(
        "phase3",
        help="Run exploratory stable-graph motif analysis",
    )
    _add_config_arguments(
        phase3,
        analysis_default="configs/analysis_phase25.yaml",
        graph_default="configs/graph_phase3.yaml",
    )
    phase3.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(
        args.analysis,
        args.conditions,
        args.graph,
        year=args.year,
    )
    if args.command == "download":
        path = download_brfss(config, force=args.force, keep_archive=args.keep_archive)
        print(path)
    elif args.command == "audit":
        print(run_audit(config, verbose=args.verbose))
    elif args.command == "prototype":
        print(
            run_prototype(
                config,
                verbose=args.verbose,
                persist_harmonized=False if args.no_persist_harmonized else None,
            )
        )
    elif args.command == "phase02":
        if not args.skip_download:
            download_brfss(config, keep_archive=args.keep_archive)
        print(run_audit(config, verbose=args.verbose))
        print(
            run_prototype(
                config,
                verbose=args.verbose,
                persist_harmonized=False if args.no_persist_harmonized else None,
            )
        )
    elif args.command == "phase25":
        print(
            run_phase25(
                config,
                verbose=args.verbose,
                workers=max(1, args.workers),
                run_bootstrap=not args.skip_bootstrap,
            )
        )
    elif args.command == "phase26":
        print(
            run_phase26(
                config,
                verbose=args.verbose,
                workers=max(1, args.workers),
            )
        )
    elif args.command == "phase3":
        print(
            run_phase3(
                config,
                verbose=args.verbose,
                workers=max(1, args.workers),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
