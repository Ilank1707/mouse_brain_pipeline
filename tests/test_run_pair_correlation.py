"""Tests for the pair-correlation and radial spatial-analysis wrappers.

Covers: both channels generated, no ``green_signal/green_signal`` duplication,
all four graphs per eligible status, the old radial script refusing to run
without confirmation, and existing outputs never being overwritten.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

import run_pair_correlation as rpc  # noqa: E402
import radial_candidate_analysis as radial_alias  # noqa: E402

REQUIRED_STATUSES = {
    "preliminary_pass",
    "preliminary_fail",
    "all_outside_injection",
    "manual_review",
}
GRAPH_FILENAMES = {
    "pair_correlation_g_r.png",
    "pair_density_per_mm2.png",
    "pair_correlation_inhomogeneous_g_r.png",
    "estimated_intensity_surface.png",
}


def _write_two_channel_run(tmp_path: Path):
    """Synthetic two-channel run with candidates in several statuses."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    channels = ("green_signal", "channel_2_signal")
    # >= 2 candidates in each of the four spatial statuses so all are eligible.
    status_counts = {
        "preliminary_rule_pass": 8,
        "preliminary_rule_fail": 6,
        "manual_review": 4,
    }
    rng = np.random.default_rng(2026)
    with (run_dir / "all_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "channel", "section", "x_global_px",
                        "y_global_px", "current_status"],
        )
        writer.writeheader()
        for channel in channels:
            index = 0
            for status, count in status_counts.items():
                for _ in range(count):
                    x, y = rng.uniform(5.0, 95.0, size=2)
                    writer.writerow({
                        "candidate_id": f"{channel}_{index}",
                        "channel": channel,
                        "section": 70,
                        "x_global_px": float(x),
                        "y_global_px": float(y),
                        "current_status": status,
                    })
                    index += 1

    metadata = {
        "array_order": "z,y,x",
        "crop_x_min_x_max_y_min_y_max": None,
        "source_image_dimensions": {c: {"height": 100, "width": 100} for c in channels},
        "acquisition": {"planes_per_section": 7},
    }
    (run_dir / "candidate_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for channel in channels:
        section_dir = run_dir / "qc" / f"{channel}_section_070"
        section_dir.mkdir(parents=True)
        np.save(section_dir / "tissue_mask.npy", np.ones((100, 100), dtype=bool))
        np.save(section_dir / "injection_analysis_exclusion_mask.npy",
                np.zeros((100, 100), dtype=bool))

    config = SimpleNamespace(
        acquisition=SimpleNamespace(voxel_size_y_um=1.004, voxel_size_x_um=1.004)
    )
    return run_dir, channels, config


def _generate(tmp_path, config, run_dir, *, timestamp="20260101_000000"):
    return rpc.generate_pair_correlation(
        config=config,
        run_dir=run_dir,
        section=70,
        out_dir=tmp_path / "spatial",
        timestamp=timestamp,
        simulations=2,
        bin_width_um=5.0,
        maximum_distance_um=30.0,
        intensity_bandwidth_um=40.0,
    )


# --------------------------------------------------------------------------- #
# 1. Both channels generated
# --------------------------------------------------------------------------- #
def test_both_channels_are_generated(tmp_path):
    run_dir, channels, config = _write_two_channel_run(tmp_path)
    root, rows = _generate(tmp_path, config, run_dir)

    assert root.name == "pair_correlation_20260101_000000"
    for channel in channels:
        channel_dir = root / channel
        assert channel_dir.is_dir()
        produced = {p.name for p in channel_dir.iterdir() if p.is_dir()}
        assert produced == REQUIRED_STATUSES
    assert {row["channel"] for row in rows} == set(channels)


# --------------------------------------------------------------------------- #
# 2. No green_signal/green_signal duplication
# --------------------------------------------------------------------------- #
def test_no_duplicated_channel_folder(tmp_path):
    run_dir, channels, config = _write_two_channel_run(tmp_path)
    root, rows = _generate(tmp_path, config, run_dir)

    for channel in channels:
        assert not (root / channel / channel).exists()
    # And no indexed graph path contains a doubled channel segment.
    for row in rows:
        for channel in channels:
            assert f"{channel}/{channel}" not in row["graph_path"].replace("\\", "/")


# --------------------------------------------------------------------------- #
# 3. All four graphs generated per eligible status
# --------------------------------------------------------------------------- #
def test_all_four_graphs_per_eligible_status(tmp_path):
    run_dir, channels, config = _write_two_channel_run(tmp_path)
    root, rows = _generate(tmp_path, config, run_dir)

    for channel in channels:
        for status in REQUIRED_STATUSES:
            status_dir = root / channel / status
            present = {p.name for p in status_dir.glob("*.png")}
            assert GRAPH_FILENAMES <= present, f"{channel}/{status} missing graphs"

    # The index CSV lists exactly the four graph types for every eligible status.
    by_key: dict[tuple[str, str], set] = {}
    for row in rows:
        by_key.setdefault((row["channel"], row["status"]), set()).add(row["graph_type"])
    expected_types = {
        "pair_correlation_g_r",
        "pair_density_per_mm2",
        "pair_correlation_inhomogeneous_g_r",
        "estimated_intensity_surface",
    }
    for (channel, status), types in by_key.items():
        assert types == expected_types, f"{channel}/{status} -> {types}"


def test_spatial_outputs_csv_has_required_columns(tmp_path):
    run_dir, _channels, config = _write_two_channel_run(tmp_path)
    root, _rows = _generate(tmp_path, config, run_dir)
    with (root / "spatial_analysis_outputs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "channel", "status", "candidate_count", "graph_type", "graph_path",
        ]
        assert any(row["channel"] == "green_signal" for row in reader)


# --------------------------------------------------------------------------- #
# 4. Old radial script refuses without confirmation
# --------------------------------------------------------------------------- #
def test_old_radial_alias_refuses_without_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(radial_alias.injection_centered, "main",
                        lambda argv: calls.append(argv) or 0)

    # No confirmation flag -> refuses (non-zero) and never delegates.
    code = radial_alias.main(["--run-dir", "somewhere"])
    assert code == 2
    assert calls == []


def test_old_radial_alias_delegates_when_confirmed(monkeypatch):
    calls = []
    monkeypatch.setattr(radial_alias.injection_centered, "main",
                        lambda argv: calls.append(argv) or 0)

    code = radial_alias.main(["--confirm-injection-centered", "--run-dir", "somewhere"])
    assert code == 0
    # The confirmation flag is stripped before delegating to the real analysis.
    assert calls == [["--run-dir", "somewhere"]]


# --------------------------------------------------------------------------- #
# 5. Existing outputs are not overwritten
# --------------------------------------------------------------------------- #
def test_existing_outputs_are_not_overwritten(tmp_path):
    run_dir, _channels, config = _write_two_channel_run(tmp_path)
    root, _rows = _generate(tmp_path, config, run_dir, timestamp="20260202_121212")
    assert root.is_dir()

    # A second run into the SAME timestamped root must refuse rather than clobber.
    with pytest.raises(FileExistsError):
        _generate(tmp_path, config, run_dir, timestamp="20260202_121212")
