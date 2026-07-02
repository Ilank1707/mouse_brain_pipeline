"""Audit / manifest tests using small synthetic TIFFs.

These require numpy + tifffile; they skip cleanly when those are unavailable so
the pure-stdlib tests still run on a minimal interpreter.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from mouse_brain_pipeline.config import Config  # noqa: E402
from mouse_brain_pipeline.audit import run_audit  # noqa: E402


def write_tiff(path: Path, shape=(32, 32), dtype="uint16", value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(shape, value, dtype=dtype)
    tifffile.imwrite(str(path), arr)


def make_dataset(root: Path, sections=(70, 71), planes=range(1, 8),
                 shape=(32, 32), dtype="uint16"):
    green = root / "green"
    ch2 = root / "ch2"
    for s in sections:
        for p in planes:
            write_tiff(green / f"section_{s:03d}_{p:02d}.tif", shape, dtype)
            write_tiff(ch2 / f"section_{s:03d}_{p:02d}.tif", shape, dtype)
    return green, ch2


def make_config(green: Path, ch2: Path, work: Path) -> Config:
    return Config.from_dict(
        {
            "data": {
                "green_signal_dir": str(green),
                "channel_2_signal_dir": str(ch2),
                "background_dir": None,
                "work_dir": str(work),
            },
            "acquisition": {"planes_per_section": 7, "voxel_size_z_um": 6.0},
            "pilot": {"first_section": 70, "number_of_sections": 2},
        }
    )


def test_clean_dataset_has_no_errors(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=True, dry_run=True)
    assert res.errors == []
    assert res.exit_code == 0
    # 2 sections * 7 planes = 14 manifest rows.
    assert len(res.manifest_rows) == 14
    assert all(r["pair_valid"] for r in res.manifest_rows)


def test_z_um_in_manifest(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=False, dry_run=True)
    by_key = {(r["section"], r["plane"]): r for r in res.manifest_rows}
    assert by_key[(70, 1)]["z_um"] == 0.0
    assert by_key[(70, 7)]["z_um"] == 36.0
    assert by_key[(71, 1)]["z_um"] == 42.0  # one physical cut later


def test_seven_planes_required_missing_plane_is_error(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    # Remove one green plane -> missing-plane error AND unpaired error.
    (green / "section_070_04.tif").unlink()
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=False, dry_run=True)
    assert res.exit_code == 1
    assert any("missing plane 04" in e for e in res.errors)


def test_duplicate_plane_is_error(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    # Same basename in a nested subdir -> duplicate (section, plane) key.
    write_tiff(green / "extra" / "section_070_01.tif")
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=False, dry_run=True)
    assert res.exit_code == 1
    assert any("duplicate plane" in e for e in res.errors)


def test_channel_shape_mismatch_is_error(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    # Rewrite one channel_2 plane with a different shape.
    write_tiff(ch2 / "section_070_02.tif", shape=(16, 16))
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=True, dry_run=True)
    assert res.exit_code == 1
    assert any("Shape mismatch" in e for e in res.errors)
    row = next(r for r in res.manifest_rows if (r["section"], r["plane"]) == (70, 2))
    assert row["pair_valid"] is False


def test_unpaired_when_channel_2_missing(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    (ch2 / "section_071_05.tif").unlink()
    cfg = make_config(green, ch2, tmp_path / "work")
    res = run_audit(cfg, check_metadata=False, dry_run=True)
    assert res.exit_code == 1
    assert any("Unpaired" in e or "missing plane 05" in e for e in res.errors)


def test_outputs_written_when_not_dry_run(tmp_path):
    green, ch2 = make_dataset(tmp_path)
    cfg = make_config(green, ch2, tmp_path / "work")
    run_audit(cfg, check_metadata=False, dry_run=False)
    audit_dir = tmp_path / "work" / "audit"
    assert (audit_dir / "manifest.csv").is_file()
    assert (audit_dir / "missing_files.csv").is_file()
    assert (audit_dir / "dataset_summary.json").is_file()


def test_tile_overlap_duplicate_removal(tmp_path):
    pytest.importorskip("scipy")
    from mouse_brain_pipeline.candidate_detection import _nms_merge

    # Two detections of the same blob from overlapping tiles + one far away.
    candidates = [
        {"z_plane": 0, "y_px": 100, "x_px": 100, "score": 0.9},
        {"z_plane": 0, "y_px": 101, "x_px": 100, "score": 0.7},  # ~1 um from the first
        {"z_plane": 0, "y_px": 400, "x_px": 400, "score": 0.8},  # far away -> kept
    ]
    merged = _nms_merge(candidates, nms_distance_um=6.0, voxel_zyx=(6.0, 1.004, 1.004))
    assert len(merged) == 2  # the near-duplicate is merged away
