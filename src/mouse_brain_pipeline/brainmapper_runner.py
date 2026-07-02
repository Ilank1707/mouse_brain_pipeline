"""Wrapper that builds and (optionally) runs the Brainmapper command.

HARD SAFETY RULES enforced here:
  * A signal channel is NEVER used as the background channel.
  * Brainmapper is refused unless a REAL ``background_dir`` is configured AND a
    BrainGlobe ``orientation`` has been explicitly confirmed.
  * The exact command is generated and displayed before anything runs.
  * ``--dry-run`` prints the command and stops.

Brainmapper (brainglobe-workflows / Cellfinder) expects a separate anatomical or
autofluorescence background channel. This dataset has two *signal* channels, so
full execution stays disabled until the user supplies that background.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .utilities import LOG, ensure_dir, has_space_in_path


@dataclass
class BrainmapperPlan:
    command: list[str]
    output_dir: Path
    blockers: list[str]
    warnings: list[str]

    @property
    def runnable(self) -> bool:
        return not self.blockers


def _versions() -> dict[str, str | None]:
    import importlib

    out: dict[str, str | None] = {"python": sys.version}
    for name in ("cellfinder", "brainglobe_workflows", "brainglobe_atlasapi", "brainreg", "numpy"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


def build_plan(
    config: Config,
    start_plane: int | None = None,
    end_plane: int | None = None,
    output_subdir: str = "brainmapper",
) -> BrainmapperPlan:
    """Construct the Brainmapper command and list any blockers."""
    blockers: list[str] = []
    warnings: list[str] = []

    if not config.data.has_background:
        blockers.append(
            "No background_dir configured. Brainmapper requires a SEPARATE anatomical/"
            "autofluorescence background channel. A signal channel will NOT be used as "
            "background. Provide data.background_dir to enable."
        )
    if not config.registration.orientation:
        blockers.append(
            "registration.orientation is not set. The BrainGlobe orientation must be "
            "explicitly confirmed (e.g. 'asr') before registration -- do not guess it."
        )
    if shutil.which("brainmapper") is None:
        blockers.append(
            "`brainmapper` CLI not found on PATH. Install brainglobe-workflows + cellfinder "
            "in a clean Python 3.11 environment (see README)."
        )

    if not config.data.green_signal_dir or not config.data.channel_2_signal_dir:
        blockers.append("Both signal directories (green_signal_dir, channel_2_signal_dir) must be set.")

    out_dir = config.work_dir / output_subdir

    # Space-in-path checks (Cellfinder dislikes spaces).
    for label, p in (
        ("green_signal_dir", config.data.green_signal_dir),
        ("channel_2_signal_dir", config.data.channel_2_signal_dir),
        ("background_dir", config.data.background_dir),
        ("work_dir", str(out_dir)),
    ):
        if has_space_in_path(p):
            warnings.append(f"{label} contains a space: {p!r}. Move it to a space-free path.")

    vz, vy, vx = config.acquisition.voxel_size_zyx
    command = [
        "brainmapper",
        "-s", str(config.data.green_signal_dir),
        "-s", str(config.data.channel_2_signal_dir),
        "-b", str(config.data.background_dir),
        "-o", str(out_dir),
        "-v", str(vz), str(vy), str(vx),
        "--orientation", str(config.registration.orientation),
        "--atlas", str(config.registration.atlas),
    ]
    if start_plane is not None:
        command += ["--start-plane", str(start_plane)]
    if end_plane is not None:
        command += ["--end-plane", str(end_plane)]

    return BrainmapperPlan(command=command, output_dir=out_dir, blockers=blockers, warnings=warnings)


def display_plan(plan: BrainmapperPlan) -> None:
    print("=" * 70)
    print("BRAINMAPPER PLAN")
    print("=" * 70)
    print("Proposed command:\n")
    print("  " + " ".join(_quote(a) for a in plan.command))
    print()
    if plan.warnings:
        print("Warnings:")
        for w in plan.warnings:
            print(f"  [warn]  {w}")
    if plan.blockers:
        print("BLOCKERS (will NOT run):")
        for b in plan.blockers:
            print(f"  [BLOCK] {b}")
    else:
        print("No blockers -- command is runnable.")
    print("=" * 70)


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def run(
    config: Config,
    start_plane: int | None = None,
    end_plane: int | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> int:
    """Display the plan; run only if not dry_run, not blocked, and confirmed."""
    plan = build_plan(config, start_plane=start_plane, end_plane=end_plane)
    display_plan(plan)

    if not plan.runnable:
        print("Refusing to run Brainmapper (see blockers above).")
        return 1
    if dry_run:
        print("[dry-run] Command displayed; not executed.")
        return 0
    if not confirm:
        print("Pass --confirm (and remove --dry-run) to actually execute. Not running.")
        return 0

    out_dir = ensure_dir(plan.output_dir)
    (out_dir / "command.txt").write_text(" ".join(_quote(a) for a in plan.command) + "\n", encoding="utf-8")
    with open(out_dir / "software_versions.json", "w", encoding="utf-8") as fh:
        json.dump(_versions(), fh, indent=2)
    LOG.info("Captured command + software versions in %s", out_dir)

    print("Launching Brainmapper ... (this is a long-running job)")
    proc = subprocess.run(plan.command)
    return proc.returncode
