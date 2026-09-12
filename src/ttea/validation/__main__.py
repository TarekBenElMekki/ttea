"""Command-line entry point for autonomous T/TEA validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import load_latest_report, result_to_dict, text_report
from .runner import ValidationRunner, exit_code
from .suites import build_registry


REPORT_ROOT = Path("data/validation")


def main() -> None:
    parser = argparse.ArgumentParser(description="T/TEA autonomous system validation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered validation suites")
    run_parser = commands.add_parser("run", help="run a registered validation suite")
    run_parser.add_argument("--suite", required=True)
    run_parser.add_argument("--json", action="store_true", help="print the complete JSON result")
    run_parser.add_argument("--wait-for-resources", type=float, default=0.0, metavar="SECONDS")
    commands.add_parser("latest", help="display the most recent report without running validation")
    arguments = parser.parse_args()
    registry = build_registry()
    if arguments.command == "list":
        print("Registered T/TEA validation suites:")
        for name in registry.names():
            print(name)
        return
    if arguments.command == "latest":
        report = load_latest_report(REPORT_ROOT)
        if report is None:
            print("No validation report is available.")
            raise SystemExit(2)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    try:
        result = ValidationRunner(registry, REPORT_ROOT).run(arguments.suite, arguments.wait_for_resources)
    except ValueError as error_value:
        parser.error(str(error_value))
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True) if arguments.json else text_report(result), end="" if arguments.json else "")
    raise SystemExit(exit_code(result))


if __name__ == "__main__":
    main()
