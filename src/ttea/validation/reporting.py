"""Machine-readable and operator-readable validation reports."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .models import Evidence, FrozenArray, FrozenJson, FrozenObject, ValidationRunResult


def _thaw(value: FrozenJson | Evidence) -> Any:
    if isinstance(value, FrozenObject):
        return {item[0]: _thaw(item[1]) for item in value}
    if isinstance(value, FrozenArray):
        return [_thaw(item) for item in value]
    return value


def result_to_dict(result: ValidationRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "suite": result.suite,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "duration_seconds": result.duration_seconds,
        "overall_status": result.overall_status.value,
        "cleanup_verified": result.cleanup_verified,
        "interrupted": result.interrupted,
        "environment_summary": _thaw(result.environment_summary),
        "report_paths": list(result.report_paths),
        "scenario_results": [
            {
                "scenario_id": item.scenario_id,
                "status": item.status.value,
                "reasons": list(item.reasons),
                "started_at": item.started_at.isoformat(),
                "finished_at": item.finished_at.isoformat(),
                "duration_seconds": item.duration_seconds,
                "evidence": _thaw(item.evidence),
            }
            for item in result.scenario_results
        ],
    }


def text_report(result: ValidationRunResult) -> str:
    lines = ["T/TEA Validation", "================", f"Suite: {result.suite}", ""]
    for item in result.scenario_results:
        lines.append(f"{item.status.value:<8} {item.scenario_id}")
        for reason in item.reasons:
            lines.append(f"         {reason}")
    lines.extend(("", f"Result: {result.overall_status.value}", f"Cleanup verified: {'YES' if result.cleanup_verified else 'NO'}"))
    return "\n".join(lines) + "\n"


def write_reports(result: ValidationRunResult, report_root: Path) -> ValidationRunResult:
    directory = report_root / result.run_id
    directory.mkdir(parents=True, exist_ok=False)
    json_path, text_path = directory / "report.json", directory / "report.txt"
    with_paths = replace(result, report_paths=(str(json_path.resolve()), str(text_path.resolve())))
    json_path.write_text(json.dumps(result_to_dict(with_paths), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(text_report(with_paths), encoding="utf-8")
    return with_paths


def latest_report_path(report_root: Path) -> Path | None:
    if not report_root.is_dir():
        return None
    candidates = [path for path in report_root.iterdir() if path.is_dir() and (path / "report.json").is_file()]
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)) / "report.json" if candidates else None


def load_latest_report(report_root: Path) -> dict[str, Any] | None:
    path = latest_report_path(report_root)
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("latest validation report is not a JSON object")
    return value
