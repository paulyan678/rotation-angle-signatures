"""Command-line interface for the rotation-angle research workflows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the first research-workflow command parser."""
    parser = argparse.ArgumentParser(prog="rotation-patterns")
    parser.add_argument(
        "--version",
        action="store_true",
        help="report the research toolkit version",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments for the evolving toolkit."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print("rotation-patterns research prototype")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
