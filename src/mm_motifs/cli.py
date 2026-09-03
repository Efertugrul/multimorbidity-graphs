from __future__ import annotations

import argparse
from collections.abc import Sequence

from mm_motifs.config import load_config
from mm_motifs.data.download import download_brfss
from mm_motifs.workflows.audit import run_audit
from mm_motifs.workflows.prototype import run_prototype


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis", default="configs/analysis.yaml")
    parser.add_argument("--conditions", default="configs/conditions.yaml")
    parser.add_argument("--graph", default="configs/graph.yaml")
    parser.add_argument("--year", type=int)
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimorbidity-motifs",
        description="BRFSS multimorbidity graph Phases 0–2",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
