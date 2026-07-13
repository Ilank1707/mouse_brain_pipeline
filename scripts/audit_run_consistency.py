#!/usr/bin/env python
"""Read-only run-consistency audit (Task 5).

Verifies that a completed run is internally consistent:
  * every candidate has a non-empty, unique candidate_id (no silent collapse)
  * all status counts sum to the candidate total
  * inside + outside injection-mask counts equal the total (core => analysis mask)
  * every candidate maps to exactly one peak optical plane (or is flagged unassigned)
  * included_in_count stays false unless a human 'cell' label / validated model confirms
  * coordinate exports reconcile with all_candidates.csv (no candidate disappears)
  * green and red candidate IDs are never mixed
  * (WARNING) candidate generation is not highly suppression-sensitive

Structural violations are HARD ERRORS (exit code 1). Suppression sensitivity is a
WARNING only. This script ONLY reads the run and writes a JSON report; it never
changes any candidate, status, mask, threshold or raw TIFF, and never targets a
count.

Example (PowerShell):
  python scripts/audit_run_consistency.py `
    --run-dir "C:/mouse_brain_work/candidates/runs/section070_20260706_151305"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.review import read_csv_rows
from mouse_brain_pipeline.run_consistency import audit_run


def _resolve_inputs(args):
    if args.candidates:
        candidates_csv = Path(args.candidates)
        base = candidates_csv.parent
    elif args.run_dir:
        base = Path(args.run_dir)
        candidates_csv = base / "all_candidates.csv"
    else:
        return None, None
    return candidates_csv, base


def _read_exports(base: Path) -> dict:
    """Map each coordinate-export CSV to its set of candidate_ids (if present)."""
    export_dir = base / "coordinate_exports"
    exports = {}
    if not export_dir.is_dir():
        return exports
    for path in sorted(export_dir.glob("*_coordinates.csv")):
        ids = {str(r.get("candidate_id", "")) for r in read_csv_rows(path)
               if r.get("candidate_id")}
        exports[path.name] = ids
    return exports


def _planes_per_section(config_path) -> int:
    if not config_path:
        return 7
    try:
        from mouse_brain_pipeline.config import load_config  # noqa: PLC0415

        return int(load_config(config_path).acquisition.planes_per_section)
    except Exception:  # pragma: no cover - config optional
        return 7


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only run-consistency audit (changes nothing; exit 1 on a "
                    "hard error, 0 otherwise).")
    p.add_argument("--config", "-c", default=None,
                   help="Optional config.yml (only for planes_per_section).")
    p.add_argument("--run-dir", default=None,
                   help="Completed run folder with all_candidates.csv + coordinate_exports/.")
    p.add_argument("--candidates", default=None,
                   help="Explicit all_candidates.csv (overrides --run-dir).")
    p.add_argument("--out-dir", default=None,
                   help="Where to write run_consistency_report.json (default: the run dir).")
    args = p.parse_args(argv)

    candidates_csv, base = _resolve_inputs(args)
    if candidates_csv is None:
        print("ERROR: provide --run-dir or --candidates.")
        return 2
    if not candidates_csv.is_file():
        print(f"ERROR: all_candidates.csv not found: {candidates_csv}")
        return 2

    candidates = read_csv_rows(candidates_csv)
    if not candidates:
        print(f"ERROR: no candidates in {candidates_csv}")
        return 2

    exports = _read_exports(base)
    planes = _planes_per_section(args.config)
    report = audit_run(candidates, exports=exports or None, planes_per_section=planes)

    out_dir = Path(args.out_dir) if args.out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "run_consistency_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Run consistency audit : {candidates_csv} ({report['n_candidates']} candidates)")
    print(f"Coordinate exports    : {len(exports)} file(s)"
          if exports else "Coordinate exports    : none found (export checks skipped)")
    print("Read-only -- no candidate, status, mask, threshold or TIFF is changed.")
    print("=" * 72)
    for check in report["checks"]:
        mark = ("PASS " if check["passed"]
                else ("ERROR" if check["level"] == "error" else "WARN "))
        print(f"  [{mark}] {check['name']}: {check['detail']}")

    print("-" * 72)
    if report["ok"]:
        print(f"OK: no hard errors. {report['n_warnings']} warning(s). "
              f"Report -> {report_path}")
        return 0
    print("!" * 72)
    print(f"FAILED: {report['n_errors']} hard error(s), {report['n_warnings']} warning(s).")
    for err in report["errors"]:
        print(f"  ERROR {err['name']}: {err['detail']}")
    print(f"Report -> {report_path}")
    print("!" * 72)
    return 1


if __name__ == "__main__":
    sys.exit(main())
