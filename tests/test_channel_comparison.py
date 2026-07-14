"""Focused synthetic tests for green/red PROVISIONAL candidate comparison.

The fixtures use small, deterministic 16-bit TIFF stacks.  Signal is placed at
the candidate's own XY/Z location, while a checkerboard background keeps local
noise finite and makes weak/background-only measurements unambiguous.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
yaml = pytest.importorskip("yaml")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from mouse_brain_pipeline.channel_comparison import (  # noqa: E402
    run_channel_comparison,
)
from mouse_brain_pipeline.config import Config  # noqa: E402
import compare_green_red_candidates as cli  # noqa: E402


SECTION = 70
HEIGHT = 128
WIDTH = 136
MIN_DOMINANCE_RATIO = 1.5
MIN_SNR = 3.0
MAX_MATCH_DISTANCE_UM = 5.0

REQUIRED_MEASUREMENT_COLUMNS = {
    "green_peak",
    "red_peak",
    "green_local_background",
    "red_local_background",
    "green_snr",
    "red_snr",
    "red_green_ratio",
    "nearest_opposite_channel_candidate_distance_um",
    "matched_opposite_channel_candidate_id",
    "dominant_channel",
}

REQUIRED_STATUS_COLUMNS = {
    "current_status",
    "original_status",
    "refined_candidate_status",
    "channel_comparison_decision",
}

REQUIRED_OUTPUTS = {
    "channel_comparison_candidates.csv",
    "channel_comparison_summary.csv",
    "green_dominant_candidates.csv",
    "red_dominant_candidates.csv",
    "both_channel_candidates.csv",
    "unclear_candidates.csv",
    "green_red_ratio_histograms.png",
    "green_vs_red_snr_scatter.png",
    "green_red_overlay_qc.png",
    "channel_comparison_summary.json",
}

CATEGORY_FILES = {
    "green_dominant": "green_dominant_candidates.csv",
    "red_dominant": "red_dominant_candidates.csv",
    "both": "both_channel_candidates.csv",
    "unclear": "unclear_candidates.csv",
}


def _background_plane(z_index: int):
    """Deterministic non-zero background with a finite local MAD."""
    yy, xx = np.indices((HEIGHT, WIDTH))
    return np.where((yy + xx + z_index) % 2, 95, 105).astype(np.uint16)


def _write_channel(directory: Path, blobs_by_z: dict[int, list[tuple[int, int, int]]]):
    """Write seven planes; blob tuples are ``(y, x, intensity)``."""
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    for z_index in range(7):
        image = _background_plane(z_index)
        for cy, cx, intensity in blobs_by_z.get(z_index, []):
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2
            image[disk] = np.uint16(intensity)
        tifffile.imwrite(
            directory / f"section_{SECTION:03d}_{z_index + 1:02d}.tif",
            image,
        )


def _candidate(candidate_id, channel, x, y, z_index, status):
    return {
        "candidate_id": candidate_id,
        "channel": channel,
        "section": SECTION,
        "x_local_px": x,
        "y_local_px": y,
        "x_global_px": x,
        "y_global_px": y,
        "z_index": z_index,
        "fixed_xy_peak_z_index": z_index,
        "optical_plane": z_index + 1,
        "current_status": status,
    }


def _write_candidates(run_dir: Path) -> list[dict]:
    """Create four signal cases plus an opposite-channel XY matching pair."""
    rows = [
        _candidate("g1", "green_signal", 26, 32, 3, "preliminary_rule_pass"),
        _candidate("r1", "channel_2_signal", 108, 32, 3, "preliminary_rule_fail"),
        # Identical XY but deliberately different Z: matching must be by XY.
        _candidate("b_g", "green_signal", 34, 96, 1, "manual_review"),
        _candidate("b_r", "channel_2_signal", 34, 96, 5, "suspect_injection_mask"),
        _candidate("u1", "green_signal", 106, 96, 3, "invalid_measurement"),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "all_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "candidate_run_metadata.json").write_text(
        json.dumps({"completed": True, "label": "PROVISIONAL candidates"}, indent=2),
        encoding="utf-8",
    )
    return rows


def _make_synthetic_inputs(root: Path) -> dict:
    green_dir = root / "raw_green"
    red_dir = root / "raw_red"
    run_dir = root / "completed_run"

    # Green-only at g1; both channels are strong at the two matched candidates.
    _write_channel(
        green_dir,
        {
            1: [(96, 34, 4000)],
            3: [(32, 26, 5000)],
            5: [(96, 34, 4000)],
        },
    )
    # Red-only at r1; the same both-channel signal at the matched candidates.
    _write_channel(
        red_dir,
        {
            1: [(96, 34, 4000)],
            3: [(32, 108, 5000)],
            5: [(96, 34, 4000)],
        },
    )
    candidates = _write_candidates(run_dir)

    raw_config = {
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
        # Explicit CLI/core thresholds are used below. These existing settings
        # keep the synthetic measurement geometry stable.
        "channel_overlay": {
            "snr_threshold": MIN_SNR,
            "dominance_ratio": MIN_DOMINANCE_RATIO,
            "qc_max_dim": 512,
        },
    }
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    config = Config.from_dict(raw_config, source_path=str(config_path))
    return {
        "root": root,
        "green_dir": green_dir,
        "red_dir": red_dir,
        "run_dir": run_dir,
        "candidates": candidates,
        "config": config,
        "config_path": config_path,
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_by_id(context: dict) -> dict[str, dict]:
    rows = _read_csv(context["out_dir"] / "channel_comparison_candidates.csv")
    return {row["candidate_id"]: row for row in rows}


def _snapshot_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def report_result(tmp_path_factory):
    context = _make_synthetic_inputs(
        tmp_path_factory.mktemp("channel_comparison_report")
    )
    before = {
        "run": _snapshot_tree(context["run_dir"]),
        "green": _snapshot_tree(context["green_dir"]),
        "red": _snapshot_tree(context["red_dir"]),
    }
    out_dir = context["root"] / "report_outputs"
    result = run_channel_comparison(
        config=context["config"],
        run_dir=context["run_dir"],
        sections=[SECTION],
        out_dir=out_dir,
        mode="report",
        min_dominance_ratio=MIN_DOMINANCE_RATIO,
        min_snr=MIN_SNR,
        max_match_distance_um=MAX_MATCH_DISTANCE_UM,
        render_plots=True,
    )
    after = {
        "run": _snapshot_tree(context["run_dir"]),
        "green": _snapshot_tree(context["green_dir"]),
        "red": _snapshot_tree(context["red_dir"]),
    }
    return {
        **context,
        "out_dir": out_dir,
        "result": result,
        "before": before,
        "after": after,
    }


# 1 ------------------------------------------------------------------------- #
def test_green_only_candidate_becomes_green_dominant(report_result):
    row = _audit_by_id(report_result)["g1"]
    assert row["dominant_channel"] == "green_dominant"
    assert float(row["green_snr"]) >= MIN_SNR
    assert float(row["green_peak"]) > float(row["red_peak"])


# 2 ------------------------------------------------------------------------- #
def test_red_only_candidate_becomes_red_dominant(report_result):
    row = _audit_by_id(report_result)["r1"]
    assert row["dominant_channel"] == "red_dominant"
    assert float(row["red_snr"]) >= MIN_SNR
    assert float(row["red_peak"]) > float(row["green_peak"])


# 3 ------------------------------------------------------------------------- #
def test_strong_candidate_in_both_channels_becomes_both(report_result):
    rows = _audit_by_id(report_result)
    for candidate_id in ("b_g", "b_r"):
        row = rows[candidate_id]
        assert row["dominant_channel"] == "both"
        assert float(row["green_snr"]) >= MIN_SNR
        assert float(row["red_snr"]) >= MIN_SNR


# 4 ------------------------------------------------------------------------- #
def test_weak_ambiguous_candidate_becomes_unclear(report_result):
    row = _audit_by_id(report_result)["u1"]
    assert row["dominant_channel"] == "unclear"
    assert float(row["green_snr"]) < MIN_SNR
    assert float(row["red_snr"]) < MIN_SNR


# 5 ------------------------------------------------------------------------- #
def test_opposite_channel_candidates_match_by_xy_distance(report_result):
    rows = _audit_by_id(report_result)
    assert rows["b_g"]["matched_opposite_channel_candidate_id"] == "b_r"
    assert rows["b_r"]["matched_opposite_channel_candidate_id"] == "b_g"
    assert float(rows["b_g"]["nearest_opposite_channel_candidate_distance_um"]) \
        == pytest.approx(0.0)
    assert float(rows["b_r"]["nearest_opposite_channel_candidate_distance_um"]) \
        == pytest.approx(0.0)


# 6 ------------------------------------------------------------------------- #
def test_report_mode_preserves_statuses_and_writes_complete_audit(report_result):
    out_dir = report_result["out_dir"]
    assert REQUIRED_OUTPUTS <= {path.name for path in out_dir.iterdir()}
    for name in REQUIRED_OUTPUTS:
        assert (out_dir / name).stat().st_size > 0

    audit_rows = _read_csv(out_dir / "channel_comparison_candidates.csv")
    assert audit_rows
    assert REQUIRED_MEASUREMENT_COLUMNS <= set(audit_rows[0])
    assert REQUIRED_STATUS_COLUMNS <= set(audit_rows[0])

    inputs = {row["candidate_id"]: row for row in report_result["candidates"]}
    audit = {row["candidate_id"]: row for row in audit_rows}
    assert set(audit) == set(inputs)
    for candidate_id, original in inputs.items():
        row = audit[candidate_id]
        assert row["channel"] == original["channel"]
        assert row["current_status"] == original["current_status"]
        assert row["original_status"] == original["current_status"]
        assert row["refined_candidate_status"] == original["current_status"]

    # The four class CSVs are a disjoint and exhaustive partition; candidates
    # remain in the audit regardless of their comparison outcome.
    partition_ids = []
    for dominant_channel, filename in CATEGORY_FILES.items():
        rows = _read_csv(out_dir / filename)
        assert all(row["dominant_channel"] == dominant_channel for row in rows)
        partition_ids.extend(row["candidate_id"] for row in rows)
    assert Counter(partition_ids) == Counter(inputs.keys())

    summary = json.loads(
        (out_dir / "channel_comparison_summary.json").read_text(encoding="utf-8")
    )
    assert isinstance(summary, dict)


# 7 ------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "omitted_flag",
    ("--min-dominance-ratio", "--min-snr", "--max-match-distance-um"),
)
def test_apply_mode_refuses_without_every_explicit_threshold(
    report_result, omitted_flag
):
    out_dir = report_result["root"] / f"missing_{omitted_flag[2:].replace('-', '_')}"
    threshold_args = {
        "--min-dominance-ratio": str(MIN_DOMINANCE_RATIO),
        "--min-snr": str(MIN_SNR),
        "--max-match-distance-um": str(MAX_MATCH_DISTANCE_UM),
    }
    argv = [
        "--config", str(report_result["config_path"]),
        "--run-dir", str(report_result["run_dir"]),
        "--section", str(SECTION),
        "--out-dir", str(out_dir),
        "--mode", "apply",
    ]
    for flag, value in threshold_args.items():
        if flag != omitted_flag:
            argv.extend((flag, value))

    with pytest.raises(SystemExit):
        cli.main(argv)
    assert not out_dir.exists()


def test_apply_mode_accepts_all_explicit_thresholds_and_keeps_all_candidates(
    report_result,
):
    out_dir = report_result["root"] / "apply_outputs"
    before = {
        "run": _snapshot_tree(report_result["run_dir"]),
        "green": _snapshot_tree(report_result["green_dir"]),
        "red": _snapshot_tree(report_result["red_dir"]),
    }
    rc = cli.main([
        "--config", str(report_result["config_path"]),
        "--run-dir", str(report_result["run_dir"]),
        "--section", str(SECTION),
        "--out-dir", str(out_dir),
        "--mode", "apply",
        "--min-dominance-ratio", str(MIN_DOMINANCE_RATIO),
        "--min-snr", str(MIN_SNR),
        "--max-match-distance-um", str(MAX_MATCH_DISTANCE_UM),
    ])
    assert rc == 0
    audit = _read_csv(out_dir / "channel_comparison_candidates.csv")
    assert {row["candidate_id"] for row in audit} == {
        row["candidate_id"] for row in report_result["candidates"]
    }
    assert all(REQUIRED_STATUS_COLUMNS <= set(row) for row in audit)
    after = {
        "run": _snapshot_tree(report_result["run_dir"]),
        "green": _snapshot_tree(report_result["green_dir"]),
        "red": _snapshot_tree(report_result["red_dir"]),
    }
    assert after == before


# 8 ------------------------------------------------------------------------- #
def test_existing_run_and_raw_tiffs_are_unmodified(report_result):
    assert report_result["after"] == report_result["before"]


def test_nonempty_output_directory_is_not_overwritten(report_result):
    out_dir = report_result["root"] / "existing_outputs"
    out_dir.mkdir()
    marker = out_dir / "keep.txt"
    marker.write_bytes(b"EXISTING-OUTPUT-DO-NOT-OVERWRITE")

    with pytest.raises(FileExistsError):
        run_channel_comparison(
            config=report_result["config"],
            run_dir=report_result["run_dir"],
            sections=[SECTION],
            out_dir=out_dir,
            mode="report",
            min_dominance_ratio=MIN_DOMINANCE_RATIO,
            min_snr=MIN_SNR,
            max_match_distance_um=MAX_MATCH_DISTANCE_UM,
            render_plots=True,
        )
    assert marker.read_bytes() == b"EXISTING-OUTPUT-DO-NOT-OVERWRITE"
    assert {path.name for path in out_dir.iterdir()} == {"keep.txt"}


def test_cli_section_flag_is_repeatable():
    args = cli.build_parser().parse_args([
        "--config", "config.yml",
        "--run-dir", "run",
        "--section", "70",
        "--section", "71",
        "--out-dir", "out",
        "--mode", "report",
    ])
    assert args.section == [70, 71]
    assert args.min_dominance_ratio is None
    assert args.min_snr is None
    assert args.max_match_distance_um is None
