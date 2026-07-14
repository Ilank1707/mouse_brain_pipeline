"""Synthetic tests for read-only green/red PROVISIONAL candidate histograms."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from mouse_brain_pipeline.channel_overlay_histograms import (  # noqa: E402
    OUTPUT_FILENAMES,
    classify_dominant_channel,
    run_channel_overlay_histograms,
    safe_snr_ratio,
)
from mouse_brain_pipeline.config import Config  # noqa: E402
import channel_overlay_histograms as cli  # noqa: E402


SECTION = 70
HEIGHT = 128
WIDTH = 136


def test_green_dominant_assignment_works():
    assert classify_dominant_channel(8.0, 2.0) == "green_dominant"


def test_red_dominant_assignment_works():
    assert classify_dominant_channel(2.0, 8.0) == "red_dominant"


def test_both_assignment_works():
    assert classify_dominant_channel(6.0, 5.0) == "both"


def test_unclear_assignment_works():
    assert classify_dominant_channel(2.9, 2.9) == "unclear"


def test_zero_and_near_zero_denominators_are_safe():
    assert safe_snr_ratio(5.0, 0.0) == float("inf")
    assert safe_snr_ratio(0.0, 0.0) == pytest.approx(1.0)
    assert safe_snr_ratio(5.0, 1e-15) == pytest.approx(5e15)
    assert classify_dominant_channel(8.0, 0.0) == "green_dominant"
    assert classify_dominant_channel(0.0, 8.0) == "red_dominant"


def _background(z_index: int):
    yy, xx = np.indices((HEIGHT, WIDTH))
    return np.where((yy + xx + z_index) % 2, 95, 105).astype(np.uint16)


def _write_channel(directory: Path, blobs):
    directory.mkdir(parents=True)
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    for z_index in range(7):
        image = _background(z_index)
        for cy, cx, intensity in blobs.get(z_index, []):
            image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = intensity
        tifffile.imwrite(
            directory / f"section_{SECTION:03d}_{z_index + 1:02d}.tif", image
        )


def _candidate(candidate_id, channel, x, y, status):
    return {
        "candidate_id": candidate_id,
        "channel": channel,
        "section": SECTION,
        "x_global_px": x,
        "y_global_px": y,
        "z_index": 3,
        "optical_plane": 4,
        "current_status": status,
    }


def _write_candidates(run_dir: Path):
    rows = [
        _candidate("g1", "green_signal", 26, 32, "preliminary_rule_pass"),
        _candidate("r1", "channel_2_signal", 108, 32, "preliminary_rule_fail"),
        _candidate("b1", "green_signal", 34, 96, "manual_review"),
        _candidate("u1", "channel_2_signal", 106, 96, "invalid_measurement"),
    ]
    run_dir.mkdir(parents=True)
    with (run_dir / "all_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _snapshot(root: Path):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def histogram_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("channel_overlay_histograms")
    green_dir = root / "raw_green"
    red_dir = root / "raw_red"
    run_dir = root / "completed_run"
    _write_channel(
        green_dir,
        {3: [(32, 26, 5000), (96, 34, 4500)]},
    )
    _write_channel(
        red_dir,
        {3: [(32, 108, 5000), (96, 34, 4500)]},
    )
    candidates = _write_candidates(run_dir)
    config = Config.from_dict(
        {
            "data": {
                "green_signal_dir": str(green_dir),
                "channel_2_signal_dir": str(red_dir),
                "work_dir": str(root / "work"),
            },
            "acquisition": {
                "planes_per_section": 7,
                "voxel_size_z_um": 6.0,
                "voxel_size_y_um": 1.0,
                "voxel_size_x_um": 1.0,
                "cut_thickness_um": 42.0,
            },
            "channel_overlay": {"qc_max_dim": 512},
        }
    )
    before = {
        "run": _snapshot(run_dir),
        "green": _snapshot(green_dir),
        "red": _snapshot(red_dir),
    }
    out_dir = root / "histogram_outputs"
    result = run_channel_overlay_histograms(
        config=config,
        run_dir=run_dir,
        section=SECTION,
        out_dir=out_dir,
        green_channel="green_signal",
        red_channel="channel_2_signal",
    )
    after = {
        "run": _snapshot(run_dir),
        "green": _snapshot(green_dir),
        "red": _snapshot(red_dir),
    }
    return {
        "root": root,
        "out_dir": out_dir,
        "result": result,
        "candidates": candidates,
        "before": before,
        "after": after,
    }


def test_outputs_are_written_with_measurements_and_suggested_filter_table(histogram_run):
    out_dir = histogram_run["out_dir"]
    assert set(OUTPUT_FILENAMES) <= {path.name for path in out_dir.iterdir()}
    assert all((out_dir / filename).stat().st_size > 0 for filename in OUTPUT_FILENAMES)

    with (out_dir / "channel_overlay_measurements.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {row["candidate_id"] for row in histogram_run["candidates"]}
    assert rows["g1"]["dominant_channel"] == "green_dominant"
    assert rows["r1"]["dominant_channel"] == "red_dominant"
    assert rows["b1"]["dominant_channel"] == "both"
    assert rows["u1"]["dominant_channel"] == "unclear"
    assert rows["g1"]["current_status"] == "preliminary_rule_pass"

    with (out_dir / "channel_overlay_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert summary
    assert {
        "original_channel",
        "original_status",
        "dominant_channel",
        "candidate_count",
        "percent_of_group",
        "suggested_action",
    } <= set(summary[0])
    assert {row["suggested_action"] for row in summary} <= {
        "keep_green_candidate",
        "keep_red_candidate",
        "possible_duplicate_or_both",
        "manual_review",
        "likely_filter_unclear",
    }


def test_original_run_and_raw_tiffs_are_not_modified(histogram_run):
    assert histogram_run["after"] == histogram_run["before"]


def test_cli_defaults_match_requested_thresholds():
    args = cli.build_parser().parse_args(
        [
            "--config", "config.yml",
            "--run-dir", "run",
            "--section", "70",
            "--out-dir", "out",
            "--green-channel", "green_signal",
            "--red-channel", "channel_2_signal",
        ]
    )
    assert args.ratio_threshold == pytest.approx(2.0)
    assert args.snr_threshold == pytest.approx(3.0)

