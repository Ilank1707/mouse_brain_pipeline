"""Peak-assigned seven-plane render tests (run-scoped).

Proves:

2.  A completed run never reads candidates from an older run.
4.  The renderer uses only the supplied current run directory.
7.  No candidate appears in two main peak-assigned plane images.
8.  Support-view images may repeat candidates but carry a warning.
9.  Each QC plane image contains its title, count summary and legend.
10. Legend status counts match the candidates drawn on that plane.
11. Native images preserve the source TIFF dimensions.
12. Raw TIFF arrays and files remain unchanged.
Plus the run-dir refusal checks (missing metadata / crop / mismatch).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from mouse_brain_pipeline.filenames import DEFAULT_FILENAME_REGEX  # noqa: E402
from mouse_brain_pipeline.seven_plane_report import (  # noqa: E402
    RenderRefusedError,
    assigned_count_breakdown,
    peak_plane_header_lines,
    peak_plane_legend_entries,
    render_run,
    support_header_lines,
)

HEIGHT, WIDTH = 70, 110
CANDIDATE_HEADER = [
    "candidate_id", "channel", "section", "x_global_px", "y_global_px",
    "fixed_xy_peak_z_index", "fixed_xy_support_z_indices", "current_status",
    "preliminary_sampling_category", "inside_injection_analysis_exclusion",
    "invalid_coordinate", "manual_label", "model_validation_passed", "global_z_um",
]


def _make_config(green_dir, ch2_dir):
    settings = SimpleNamespace(mode="robust_tissue_percentile", minimum=0.0,
                               maximum=513.0, lower_percentile=0.5, upper_percentile=99.7)
    qc = SimpleNamespace(minimum_pixels=20, for_channel=lambda ch: settings)
    return SimpleNamespace(
        data=SimpleNamespace(green_signal_dir=str(green_dir),
                             channel_2_signal_dir=str(ch2_dir),
                             filename_regex=DEFAULT_FILENAME_REGEX),
        qc_display=qc,
        detection=SimpleNamespace(padding_values=[0.0]),
        acquisition=SimpleNamespace(planes_per_section=7),
    )


def _write_tiffs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = np.random.default_rng(0)
    for plane in range(1, 8):
        img = rng.integers(50, 400, size=(HEIGHT, WIDTH)).astype(np.uint16)
        img[20:40, 30:70] = 9000
        path = directory / f"section_070_{plane:02d}.tif"
        tifffile.imwrite(path, img)
        paths.append(path)
    return paths


def _candidates(specs):
    """specs: list of (id, peak, status, support)."""
    out = []
    for cid, peak, status, support in specs:
        out.append({
            "candidate_id": cid, "channel": "green_signal", "section": "70",
            "x_global_px": "55", "y_global_px": "35", "fixed_xy_peak_z_index": str(peak),
            "fixed_xy_support_z_indices": support, "current_status": status,
            "preliminary_sampling_category": status,
            "inside_injection_analysis_exclusion": "False", "invalid_coordinate": "False",
            "manual_label": "", "model_validation_passed": "", "global_z_um": "0",
        })
    return out


def _write_run(tmp_path, candidates, *, crop_mode="full_xy_section", crop=None,
               processed=(70,), dims=True, name="run"):
    green = tmp_path / name / "green_tiffs"
    _write_tiffs(green)
    run_dir = tmp_path / name / "run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "all_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_HEADER)
        writer.writeheader()
        for c in candidates:
            writer.writerow({k: c.get(k, "") for k in CANDIDATE_HEADER})
    meta = {
        "run_timestamp_utc": "2026-06-30T00:00:00+00:00",
        "crop_mode": crop_mode, "crop_x_min_x_max_y_min_y_max": crop,
        "processed_sections": list(processed),
        "candidate_counts_by_channel": {"green_signal": len(candidates)},
        "one_section_not_whole_brain": len(processed) <= 1,
    }
    if dims:
        meta["source_image_dimensions"] = {"green_signal": {"height": HEIGHT, "width": WIDTH}}
    (run_dir / "candidate_run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    config = _make_config(green, tmp_path / name / "ch2_tiffs")
    _write_tiffs(tmp_path / name / "ch2_tiffs")
    return run_dir, config


# 7 + 11 + 12 --------------------------------------------------------------- #
def test_peak_assignment_no_double_count_dims_and_raw_untouched(tmp_path):
    from PIL import Image

    specs = [("c1", 0, "preliminary_rule_pass", "0;1"),
             ("c2", 3, "manual_review", "2;3;4"),
             ("c3", 6, "artifact", "5;6")]
    run_dir, config = _write_run(tmp_path, _candidates(specs))
    green_dir = Path(config.data.green_signal_dir)
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), os.stat(p).st_mtime_ns)
              for p in green_dir.glob("*.tif")}

    out = render_run(run_dir, "green_signal", 70, config=config, make_preview=False)

    # 7: each candidate assigned to exactly one plane; assigned total == unique.
    rec = out["reconciliation"]
    assert rec["assigned_total"] == 3 and out["unassigned"] == 0
    drawn = [r["candidates_assigned"] for r in out["metadata_rows"]]
    assert sum(drawn) == 3                      # no candidate drawn on two main images
    assert sorted(p for p, n in zip(range(1, 8), drawn) if n) == [1, 4, 7]

    # 11: native images keep the source TIFF dimensions.
    for row in out["metadata_rows"]:
        assert (row["native_saved_width"], row["native_saved_height"]) == (WIDTH, HEIGHT)
        assert row["resizing_occurred"] is False
        native = out["qc_dir"] / row["filename"]
        with Image.open(native) as im:
            assert im.size == (WIDTH, HEIGHT)

    # 12: the raw TIFFs are byte-for-byte unchanged.
    for path, (digest, mtime) in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert os.stat(path).st_mtime_ns == mtime


# 9 + 10 -------------------------------------------------------------------- #
def test_qc_image_has_header_legend_and_counts_match(tmp_path):
    from PIL import Image

    specs = [("c1", 3, "preliminary_rule_pass", "3"),
             ("c2", 3, "manual_review", "3"),
             ("c3", 3, "preliminary_rule_pass", "3")]
    run_dir, config = _write_run(tmp_path, _candidates(specs))
    out = render_run(run_dir, "green_signal", 70, config=config, make_preview=False)

    # The plane-04 image (z-index 3) holds all three candidates.
    assigned = [{"current_status": s, "inside_injection_analysis_exclusion": "False",
                 "invalid_coordinate": "False"} for _, _, s, _ in specs]
    header = peak_plane_header_lines("green_signal", 70, 4, assigned, 3,
                                     {"display_mode": "per_plane_robust",
                                      "display_min": 0, "display_max": 100})
    # 9: title + count summary lines present.
    assert header[0] == "green_signal section 070 - optical plane 04"
    assert any("unique candidates in full 7-plane stack: 3" in h for h in header)
    assert any("assigned to this plane: 3" in h for h in header)
    assert any("manual review among assigned: 1" in h for h in header)

    # 10: legend counts match the per-status candidates drawn on the plane.
    legend = peak_plane_legend_entries(assigned)
    legend_counts = {label: count for label, _rgb, _sym, count in legend}
    assert legend_counts["preliminary rule pass"] == 2
    assert legend_counts["manual review"] == 1

    # The QC PNG exists and is TALLER than native (white header + footer added).
    qc_png = out["qc_dir"] / "plane_04_peak_assigned_qc.png"
    with Image.open(qc_png) as im:
        assert im.size[0] == WIDTH and im.size[1] > HEIGHT


# 8 ------------------------------------------------------------------------- #
def test_support_views_repeat_candidates_with_warning(tmp_path):
    # One candidate supported on planes 2,3,4 (z-index 1,2,3 -> optical 2,3,4).
    specs = [("c1", 2, "preliminary_rule_pass", "1;2;3")]
    run_dir, config = _write_run(tmp_path, _candidates(specs))
    out = render_run(run_dir, "green_signal", 70, config=config, make_preview=False)

    supported = [r["candidates_supported"] for r in out["metadata_rows"]]
    # The single candidate is visible on THREE support planes (repeated) ...
    assert sum(supported) == 3
    # ... while it is peak-assigned to exactly one main image.
    assert sum(r["candidates_assigned"] for r in out["metadata_rows"]) == 1
    assert (out["support_dir"] / "plane_03_support_qc.png").is_file()
    warn = support_header_lines("green_signal", 70, 3, 1)
    assert any("DO NOT SUM" in line for line in warn)


# 2 + 4 --------------------------------------------------------------------- #
def test_renderer_uses_only_the_supplied_run_dir(tmp_path):
    new_dir, config = _write_run(
        tmp_path, _candidates([("new1", 1, "preliminary_rule_pass", "1"),
                               ("new2", 2, "manual_review", "2")]), name="new")
    # An OLDER run with a totally different candidate count sits alongside.
    _write_run(tmp_path, _candidates([("old%d" % i, 3, "artifact", "3") for i in range(50)]),
               name="old")
    out = render_run(new_dir, "green_signal", 70, config=config, make_preview=False)
    # Only the supplied run dir's two candidates are used.
    assert out["reconciliation"]["unique_total"] == 2
    rows = list(csv.DictReader(open(new_dir / "coordinate_exports" / "all_candidate_coordinates.csv")))
    assert {r["candidate_id"] for r in rows} == {"new1", "new2"}


# Refusals --------------------------------------------------------------- #
def test_render_refuses_missing_metadata(tmp_path):
    run_dir, config = _write_run(tmp_path, _candidates([("c1", 1, "preliminary_rule_pass", "1")]))
    (run_dir / "candidate_run_metadata.json").unlink()
    with pytest.raises(RenderRefusedError, match="metadata missing"):
        render_run(run_dir, "green_signal", 70, config=config, make_preview=False)


def test_render_refuses_cropped_run_for_full_section(tmp_path):
    run_dir, config = _write_run(
        tmp_path, _candidates([("c1", 1, "preliminary_rule_pass", "1")]),
        crop_mode="xy_crop", crop=[1000, 5000, 500, 4000])
    with pytest.raises(RenderRefusedError, match="CROPPED"):
        render_run(run_dir, "green_signal", 70, config=config, make_preview=False)
    # ...unless explicitly allowed.
    out = render_run(run_dir, "green_signal", 70, config=config,
                     make_preview=False, allow_cropped=True)
    assert out["reconciliation"]["unique_total"] == 1


def test_render_refuses_tiff_dimension_mismatch(tmp_path):
    run_dir, config = _write_run(tmp_path, _candidates([("c1", 1, "preliminary_rule_pass", "1")]))
    meta = json.loads((run_dir / "candidate_run_metadata.json").read_text())
    meta["source_image_dimensions"] = {"green_signal": {"height": 9999, "width": 8888}}
    (run_dir / "candidate_run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RenderRefusedError, match="dimensions"):
        render_run(run_dir, "green_signal", 70, config=config, make_preview=False)


def test_breakdown_counts():
    assigned = [
        {"current_status": "manual_review", "inside_injection_analysis_exclusion": "True",
         "invalid_coordinate": "False"},
        {"current_status": "preliminary_rule_pass", "inside_injection_analysis_exclusion": "False",
         "invalid_coordinate": "True"},
    ]
    b = assigned_count_breakdown(assigned)
    assert b["assigned"] == 2 and b["inside_injection"] == 1 and b["outside_injection"] == 1
    assert b["manual_review"] == 1 and b["invalid_coordinate"] == 1
