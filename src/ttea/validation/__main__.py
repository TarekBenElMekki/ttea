"""Command-line entry point for autonomous T/TEA validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import CampaignManager, campaign_status_text, supervise_campaign
from .campaign_models import campaign_to_dict
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
    campaign_parser = commands.add_parser("campaign", help="manage persistent validation campaigns")
    campaign_commands = campaign_parser.add_subparsers(dest="campaign_command", required=True)
    start_parser = campaign_commands.add_parser("start", help="start a validation campaign")
    start_parser.add_argument("--through", required=True)
    start_parser.add_argument("--detach", action="store_true")
    start_parser.add_argument("--poll-interval", type=float, default=30.0, metavar="SECONDS")
    start_parser.add_argument("--max-duration", type=float, default=6 * 60 * 60, metavar="SECONDS")
    status_parser = campaign_commands.add_parser("status", help="show the current campaign")
    status_parser.add_argument("--json", action="store_true")
    campaign_commands.add_parser("cancel", help="cancel the current campaign")
    campaign_commands.add_parser("resume", help="resume an interrupted or blocked campaign")
    campaign_commands.add_parser("list", help="list persisted campaigns")
    supervise_parser = campaign_commands.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise_parser.add_argument("--campaign-id", required=True)
    supervise_parser.add_argument("--claim-token", required=True)
    arguments = parser.parse_args()
    registry = build_registry()
    if arguments.command == "campaign":
        manager = CampaignManager(registry=registry, report_root=REPORT_ROOT)
        try:
            if arguments.campaign_command == "start":
                campaign = manager.start(arguments.through, arguments.poll_interval, arguments.max_duration, arguments.detach)
                alive = manager.store.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
                print(campaign_status_text(campaign, alive), end="")
                return
            if arguments.campaign_command == "_supervise":
                supervise_campaign(arguments.campaign_id, arguments.claim_token, manager.store, registry)
                return
            if arguments.campaign_command == "status":
                campaign, alive = manager.status()
                if campaign is None:
                    print("No validation campaign exists.")
                    raise SystemExit(2)
                print(json.dumps({**campaign_to_dict(campaign), "supervisor_alive": alive}, indent=2, sort_keys=True)
                      if arguments.json else campaign_status_text(campaign, alive))
                return
            if arguments.campaign_command == "cancel":
                campaign = manager.cancel()
                print(campaign_status_text(campaign, False), end="")
                return
            if arguments.campaign_command == "resume":
                campaign = manager.resume()
                print(campaign_status_text(campaign, True), end="")
                return
            for campaign in manager.store.list():
                alive = manager.store.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
                print(f"{campaign.campaign_id}  {campaign.state.value:<22} {'alive' if alive else 'not running'}")
            return
        except (OSError, RuntimeError, ValueError) as error_value:
            parser.error(str(error_value))
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
