"""Isolated per-run output directories for candidate detection.

Every real detection run gets its own directory under
``<work_dir>/candidates/runs/<run_id>`` so the outputs of different attempts are
never mixed in one folder. The renderer is then pointed at one exact run
directory and never "finds the newest CSV" while a run is active.

``<work_dir>/candidates/latest_run.json`` records the newest *successfully
completed* run, and is only written after a run finishes. This module never
deletes or overwrites a previous run's files.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Sub-folders every run directory owns. Each run is fully self-contained.
RUN_SUBDIRS = ("coordinate_exports", "qc", "review_patches", "seven_plane_qc")

LATEST_RUN_FILENAME = "latest_run.json"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_run_name(name: str) -> str:
    """Make a filesystem-safe run name (no spaces / separators)."""
    cleaned = _SAFE.sub("_", str(name).strip()).strip("_")
    return cleaned or "run"


def make_run_id(run_name: str | None, first_section: int | None, *, now=None) -> str:
    """Run id: an explicit ``run_name`` or ``<UTC-timestamp>_section<NNN>``."""
    if run_name:
        return sanitize_run_name(run_name)
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    if first_section is not None:
        return f"{stamp}_section{int(first_section):03d}"
    return stamp


def runs_root(work_dir) -> Path:
    return Path(work_dir) / "candidates" / "runs"


def candidates_root(work_dir) -> Path:
    return Path(work_dir) / "candidates"


def create_run_dir(work_dir, run_id: str) -> Path:
    """Create a fresh, empty run directory + sub-folders.

    Refuses to reuse a non-empty existing directory so a previous run is never
    silently overwritten.
    """
    run_dir = runs_root(work_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            f"Choose a different --run-name to avoid overwriting a previous run."
        )
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def config_hash(config_path) -> str | None:
    """SHA-256 of the config file, so attempts can be compared exactly."""
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def code_version(project_root=None) -> str:
    """Best-effort code version: git commit if available, else package version."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
            return f"git:{commit}{suffix}"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from . import __version__  # noqa: PLC0415

        return f"package:{__version__}"
    except Exception:  # pragma: no cover - version optional
        return "unknown"


def write_latest_run(work_dir, run_dir, payload: dict) -> Path:
    """Record the newest successfully completed run (called only on success)."""
    root = candidates_root(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "run_dir": str(Path(run_dir).resolve()),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path = root / LATEST_RUN_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def is_single_section(processed_sections) -> bool:
    """One processed section is a single section -- never a whole-brain count."""
    return len(list(processed_sections)) <= 1


def read_latest_run(work_dir) -> dict | None:
    path = candidates_root(work_dir) / LATEST_RUN_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
