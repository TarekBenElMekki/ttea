"""Read-only tournament framework CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .manifest import ManifestMismatchError, SAFETY_CANDIDATE, TournamentManifest
from .planning import TournamentIntegrityError
from .reporting import text_report
from .store import TournamentStore, TournamentStoreError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the inert T/TEA tournament framework")
    parser.add_argument("--root", type=Path, default=Path("data/tournaments"), help="persisted tournament root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for command in ("status", "report", "manifest"):
        child = subparsers.add_parser(command)
        child.add_argument("tournament_id", nargs="?")
    subparsers.add_parser("start")
    return parser


def _resolve(store: TournamentStore, tournament_id: str | None):
    if tournament_id:
        return store.load(tournament_id)
    return store.latest()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    store = TournamentStore(arguments.root)
    try:
        if arguments.command == "start":
            print("execution not enabled in framework milestone")
            return 2
        if arguments.command == "list":
            tournaments = store.list()
            if not tournaments:
                print("No persisted tournaments.")
                return 0
            print("ID\tSTATE\tUPDATED\tPROGRESS")
            for item in tournaments:
                print(f"{item.tournament_id}\t{item.state.value}\t{item.updated_at.isoformat()}\t{item.progress_completed}/{item.progress_total}")
            return 0
        tournament = _resolve(store, arguments.tournament_id)
        if arguments.command == "manifest" and tournament is None:
            manifest = TournamentManifest((SAFETY_CANDIDATE,))
            print(f"Default framework manifest digest: {manifest.digest}")
            print(f"Validation-only candidate: {SAFETY_CANDIDATE.variant.model}")
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
            report_path = store.report_text_path(tournament.tournament_id)
            if report_path.is_file():
                print(report_path.read_text(encoding="utf-8"), end="")
            else:
                print(text_report(tournament, store.load_manifest(tournament.tournament_id)), end="")
            return 0
        manifest = store.load_manifest(tournament.tournament_id)
        print(f"Tournament: {tournament.tournament_id}")
        print(f"Manifest digest: {tournament.manifest_digest}")
        print(f"Snapshot: {store.manifest_snapshot_path(tournament.tournament_id)}")
        print(f"Candidates: {len(manifest.candidates)}")
        print(f"Rounds: {len(manifest.rounds)}")
        print(f"Tasks: {len(manifest.tasks)}")
        return 0
    except (TournamentStoreError, ManifestMismatchError, TournamentIntegrityError) as exc:
        print(f"Tournament data error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
