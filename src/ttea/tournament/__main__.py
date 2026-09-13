"""Controlled T/TEA tournament command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .catalog import build_survival_manifest
from .execution import SurvivalStore
from .manifest import ManifestMismatchError
from .planning import TournamentIntegrityError
from .reporting import text_report
from .store import TournamentStore, TournamentStoreError
from .supervisor import TournamentSupervisorManager, supervise_tournament
from .survival import survival_report_text
from .watch import watch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the controlled T/TEA local-model tournament")
    parser.add_argument("--root", type=Path, default=Path("data/tournaments"), help="persisted tournament root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for command in ("status", "report", "manifest"):
        child = commands.add_parser(command)
        child.add_argument("tournament_id", nargs="?")
    start = commands.add_parser("start")
    start.add_argument("--through", choices=("survival",))
    start.add_argument("--allow-model-download", action="store_true")
    start.add_argument("--detach", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("tournament_id", nargs="?")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("tournament_id", nargs="?")
    monitor = commands.add_parser("watch")
    monitor.add_argument("--refresh", type=float, default=2.0)
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--json", action="store_true")
    supervise = commands.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise.add_argument("--tournament-id", required=True)
    supervise.add_argument("--claim-token", required=True)
    return parser


def _resolve(store: TournamentStore, tournament_id: str | None):
    return store.load(tournament_id) if tournament_id else store.latest()


def _status_text(campaign, alive: bool, allow_download: bool) -> str:
    completed, total = campaign.progress
    return "\n".join((
        f"Tournament: {campaign.tournament_id}", f"State: {campaign.state.value}",
        "Phase: SURVIVAL", f"Round: {campaign.current_round_id or '(none)'}",
        f"Candidate: {campaign.current_candidate_id or '(none)'}",
        f"Progress: {completed}/{total}", f"Supervisor: {'alive' if alive else 'not running'}",
        f"Download authorized: {'YES' if allow_download else 'NO'}",
        f"Last reason: {campaign.last_reason or '(none)'}",
    )) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    store = TournamentStore(arguments.root)
    survival_store = SurvivalStore(arguments.root)
    manager = TournamentSupervisorManager(survival_store)
    try:
        if arguments.command == "start":
            if arguments.through is None:
                print("execution not enabled without explicit --through survival")
                return 2
            campaign = manager.start(arguments.through, arguments.allow_model_download, arguments.detach)
            alive = manager.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
            print(_status_text(campaign, alive, arguments.allow_model_download), end="")
            return 0
        if arguments.command == "_supervise":
            supervise_tournament(arguments.tournament_id, arguments.claim_token, survival_store)
            return 0
        if arguments.command == "resume":
            campaign = manager.resume(arguments.tournament_id)
            policy = survival_store.load_policy(campaign.tournament_id)
            print(_status_text(campaign, True, policy.allow_model_download), end="")
            return 0
        if arguments.command == "cancel":
            campaign = manager.cancel(arguments.tournament_id)
            policy = survival_store.load_policy(campaign.tournament_id)
            print(_status_text(campaign, False, policy.allow_model_download), end="")
            return 0
        if arguments.command == "watch":
            return watch(arguments.root, refresh_seconds=arguments.refresh,
                         once=arguments.once, json_output=arguments.json)
        if arguments.command == "list":
            executions = survival_store.list()
            if executions:
                print("ID\tSTATE\tUPDATED\tPROGRESS")
                for item in executions:
                    completed, total = item.progress
                    print(f"{item.tournament_id}\t{item.state.value}\t{item.updated_at.isoformat()}\t{completed}/{total}")
                return 0
            tournaments = store.list()
            if not tournaments:
                print("No persisted tournaments.")
                return 0
            print("ID\tSTATE\tUPDATED\tPROGRESS")
            for item in tournaments:
                print(f"{item.tournament_id}\t{item.state.value}\t{item.updated_at.isoformat()}\t{item.progress_completed}/{item.progress_total}")
            return 0
        if arguments.command == "status":
            candidate = (survival_store.load(arguments.tournament_id) if arguments.tournament_id
                         and survival_store.execution_path(arguments.tournament_id).is_file()
                         else survival_store.latest() if arguments.tournament_id is None else None)
            if candidate is not None:
                alive = manager.process_matches(candidate.supervisor_pid, candidate.supervisor_started_epoch)
                policy = survival_store.load_policy(candidate.tournament_id)
                print(_status_text(candidate, alive, policy.allow_model_download), end="")
                return 0
        tournament = _resolve(store, getattr(arguments, "tournament_id", None))
        if arguments.command == "manifest" and tournament is None:
            manifest = build_survival_manifest()
            print(f"Default survival manifest digest: {manifest.digest}")
            print(f"Candidates: {len(manifest.candidates)}")
            print("Executable round: survival_4k")
            return 0
        if tournament is None:
            print("No tournament is available.")
            return 1
        if arguments.command == "status":
            print(f"Tournament: {tournament.tournament_id}")
            print(f"State: {tournament.state.value}")
            print(f"Phase: {tournament.current_phase.value if tournament.current_phase else '(none)'}")
            print(f"Round: {tournament.current_round_id or '(none)'}")
            print(f"Candidate: {tournament.current_candidate_id or '(none)'}")
            print(f"Task: {tournament.current_task_id or '(none)'}")
            print(f"Progress: {tournament.progress_completed}/{tournament.progress_total}")
            print(f"Resource wait: {tournament.resource_wait_reason or '(none)'}")
            print(f"Last reason: {tournament.last_reason or '(none)'}")
            return 0
        if arguments.command == "report":
            if survival_store.execution_path(tournament.tournament_id).is_file():
                campaign = survival_store.load(tournament.tournament_id)
                print(survival_report_text(campaign, store.load_manifest(tournament.tournament_id)), end="")
            else:
                report_path = store.report_text_path(tournament.tournament_id)
                print(report_path.read_text(encoding="utf-8") if report_path.is_file()
                      else text_report(tournament, store.load_manifest(tournament.tournament_id)), end="")
            return 0
        manifest = store.load_manifest(tournament.tournament_id)
        print(f"Tournament: {tournament.tournament_id}")
        print(f"Manifest digest: {tournament.manifest_digest}")
        print(f"Snapshot: {store.manifest_snapshot_path(tournament.tournament_id)}")
        print(f"Candidates: {len(manifest.candidates)}")
        print(f"Rounds: {len(manifest.rounds)}")
        print(f"Tasks: {len(manifest.tasks)}")
        return 0
    except (TournamentStoreError, ManifestMismatchError, TournamentIntegrityError, RuntimeError, ValueError) as exc:
        print(f"Tournament error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
