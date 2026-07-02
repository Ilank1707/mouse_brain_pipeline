"""Prepare a small contiguous pilot range without duplicating data.

Given a first section + count, this:
  * selects the contiguous section range,
  * verifies every required plane (1..N) is present in both signal channels,
  * writes ORDERED text-file lists per channel (and optionally safe symlinks),
  * prints the exact planes and the calculated relative-Z range,
  * refuses to continue if any plane is missing.

It never copies the (hundreds of GB of) raw TIFFs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import CHANNEL_2_SIGNAL, GREEN_SIGNAL
from .audit import index_channel
from .config import Config
from .filenames import expected_planes, global_plane, z_um
from .utilities import LOG, ensure_dir


@dataclass
class PilotPlan:
    first_section: int
    last_section: int
    sections: list[int]
    planes_per_section: int
    green_files: list[Path] = field(default_factory=list)
    channel_2_files: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    z_um_first: float = 0.0
    z_um_last: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.missing and bool(self.green_files) and bool(self.channel_2_files)

    @property
    def n_planes(self) -> int:
        return len(self.sections) * self.planes_per_section


def section_availability(requested: list[int], available: list[int] | set[int]) -> dict:
    """Partition requested sections without claiming absent sections were processed."""
    available_set = set(available)
    return {
        "requested": list(requested),
        "available": sorted(available_set),
        "processed": [section for section in requested if section in available_set],
        "skipped": [section for section in requested if section not in available_set],
    }


def plan_pilot(config: Config) -> PilotPlan:
    """Build (but do not write) the pilot plan from the configured section range."""
    regex = config.data.filename_regex
    ppl = config.acquisition.planes_per_section
    green = index_channel(GREEN_SIGNAL, config.data.green_signal_dir, regex)
    ch2 = index_channel(CHANNEL_2_SIGNAL, config.data.channel_2_signal_dir, regex)

    all_sections = sorted(green.sections | ch2.sections)
    if not all_sections:
        return PilotPlan(0, 0, [], ppl, missing=["no sections discovered in either signal channel"])

    first = config.pilot.first_section if config.pilot.first_section is not None else all_sections[0]
    count = max(1, config.pilot.number_of_sections)
    sections = list(range(first, first + count))
    last = sections[-1]

    plan = PilotPlan(first_section=first, last_section=last, sections=sections, planes_per_section=ppl)
    wanted = expected_planes(ppl)
    for section in sections:
        for plane in wanted:
            g = green.files.get((section, plane))
            c = ch2.files.get((section, plane))
            if g is None:
                plan.missing.append(f"{GREEN_SIGNAL}: section {section} plane {plane:02d}")
            else:
                plan.green_files.append(g)
            if c is None:
                plan.missing.append(f"{CHANNEL_2_SIGNAL}: section {section} plane {plane:02d}")
            else:
                plan.channel_2_files.append(c)

    plan.z_um_first = z_um(global_plane(first, first, 1, ppl), config.acquisition.voxel_size_z_um)
    plan.z_um_last = z_um(global_plane(last, first, ppl, ppl), config.acquisition.voxel_size_z_um)
    return plan


def write_pilot(config: Config, plan: PilotPlan, use_symlinks: bool = False, dry_run: bool = False) -> Path:
    """Write ordered file lists (and optional symlinks) for the pilot range."""
    out_dir = ensure_dir(config.work_dir / "pilot" / f"sections_{plan.first_section}-{plan.last_section}")
    green_list = out_dir / "green_signal_files.txt"
    ch2_list = out_dir / "channel_2_signal_files.txt"

    print("=" * 70)
    print("PILOT PLAN")
    print("=" * 70)
    print(f"sections     : {plan.first_section}..{plan.last_section} ({len(plan.sections)} sections)")
    print(f"planes/sect  : {plan.planes_per_section}  ->  {plan.n_planes} planes/channel")
    print(f"relative Z   : {plan.z_um_first:.1f} um .. {plan.z_um_last:.1f} um")
    print(f"green files  : {len(plan.green_files)}")
    print(f"channel_2    : {len(plan.channel_2_files)}")
    if plan.missing:
        print("-" * 70)
        print(f"REFUSING TO CONTINUE -- {len(plan.missing)} missing plane(s):")
        for m in plan.missing[:14]:
            print(f"  [missing] {m}")
        if len(plan.missing) > 14:
            print(f"  ... and {len(plan.missing) - 14} more")
        print("=" * 70)
        raise SystemExit(1)

    if dry_run:
        print(f"[dry-run] would write ordered lists to {out_dir}")
        print("=" * 70)
        return out_dir

    green_list.write_text("\n".join(str(p) for p in plan.green_files) + "\n", encoding="utf-8")
    ch2_list.write_text("\n".join(str(p) for p in plan.channel_2_files) + "\n", encoding="utf-8")
    LOG.info("Wrote %s and %s", green_list.name, ch2_list.name)

    if use_symlinks:
        _make_symlinks(out_dir, plan)

    print(f"Wrote ordered file lists to {out_dir}")
    print("=" * 70)
    return out_dir


def _make_symlinks(out_dir: Path, plan: PilotPlan) -> None:
    """Create read-only symlinks (no data copy). Falls back gracefully if unsupported."""
    for name, files in ((GREEN_SIGNAL, plan.green_files), (CHANNEL_2_SIGNAL, plan.channel_2_files)):
        link_dir = ensure_dir(out_dir / name)
        for src in files:
            dst = link_dir / src.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                os.symlink(src, dst)
            except OSError as exc:
                LOG.warning(
                    "Could not create symlink %s -> %s (%s). On Windows enable Developer "
                    "Mode or run as admin; the .txt lists are sufficient regardless.",
                    dst, src, exc,
                )
                return
