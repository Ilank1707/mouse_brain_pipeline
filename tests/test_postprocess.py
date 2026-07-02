"""Post-processing: new isolated run, no Cellfinder rerun, source left intact."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline import postprocess as pp  # noqa: E402
from mouse_brain_pipeline.candidate_detection import CANDIDATE_COLUMNS  # noqa: E402
from mouse_brain_pipeline.config import Config  # noqa: E402
from mouse_brain_pipeline.injection_overrides import save_channel_polygons  # noqa: E402


def _row(**over):
    row = {col: "" for col in CANDIDATE_COLUMNS}
    row.update({
        "channel": "green_signal", "section": "70", "fixed_xy_peak_z_index": "0",
        "invalid_coordinate": "False", "original_cellfinder_z_valid": "True",
        "measurement_valid": "True", "injection_mask_source": "automatic",
        "injection_mask_validated": "False", "injection_mask_qc_failed": "False",
    })
    row.update(over)
    return row


def _make_source_run(root: Path) -> Path:
    run = root / "source_run"
    (run / "qc" / "green_signal_section_070").mkdir(parents=True)

    rows = [
        # A: inside the (false) mask edge -> currently suspect; will be removed.
        _row(candidate_id="A", x_local_px="90", y_local_px="90",
             x_global_px="90", y_global_px="90", current_status="suspect_injection_mask",
             inside_injection_analysis_exclusion="True", inside_injection_site="True",
             preliminary_sampling_category="preliminary_rule_pass", preliminary_rule_reason=""),
        # B: inside the genuine core -> suspect; stays.
        _row(candidate_id="B", x_local_px="55", y_local_px="55",
             x_global_px="55", y_global_px="55", current_status="suspect_injection_mask",
             inside_injection_analysis_exclusion="True", inside_injection_site="True",
             preliminary_sampling_category="preliminary_rule_pass", preliminary_rule_reason=""),
        # C: outside the mask, a preliminary fail -> unchanged.
        _row(candidate_id="C", x_local_px="10", y_local_px="10",
             x_global_px="10", y_global_px="10", current_status="preliminary_rule_fail",
             inside_injection_analysis_exclusion="False", inside_injection_site="False",
             preliminary_sampling_category="preliminary_rule_fail",
             preliminary_rule_reason="too_small"),
    ]
    with open(run / "all_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    core = np.zeros((120, 120), dtype=bool)
    core[50:70, 50:70] = True
    analysis = np.zeros((120, 120), dtype=bool)
    analysis[40:100, 40:100] = True
    qc = run / "qc" / "green_signal_section_070"
    np.save(qc / "injection_core_mask.npy", core)
    np.save(qc / "injection_analysis_exclusion_mask.npy", analysis)
    np.save(qc / "tissue_mask.npy", np.ones((120, 120), dtype=bool))

    (run / "candidate_run_metadata.json").write_text(json.dumps({
        "processed_sections": [70], "crop_x_min_x_max_y_min_y_max": None,
        "candidate_counts_by_channel": {"green_signal": 3},
        "source_image_dimensions": {"green_signal": {"height": 120, "width": 120}},
        "acquisition": {"planes_per_section": 7},
        "injection_exclusion_by_channel": {"green_signal": {"mask_validated": False}},
    }), encoding="utf-8")
    return run


def _make_override(root: Path) -> Path:
    path = root / "ov.yml"
    # Polygon over candidate A (90, 90); leaves the core (B) untouched.
    save_channel_polygons(path, "green_signal", [],
                          [[[80, 80], [110, 80], [110, 110], [80, 110]]])
    return path


def _hash_tree(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _config(work_dir: Path) -> Config:
    cfg = Config.from_dict({"data": {"work_dir": str(work_dir)}}, source_path="config.yml")
    return cfg


def _read_new_statuses(run_dir: Path) -> dict:
    with open(run_dir / "all_candidates.csv", newline="", encoding="utf-8") as fh:
        return {r["candidate_id"]: r["current_status"] for r in csv.DictReader(fh)}


def test_postprocess_creates_new_run_and_reverts_removed_candidate(tmp_path):
    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")

    result = pp.postprocess_run(
        config=cfg, source_run_dir=source, new_run_name="maskfix_01",
        work_dir=cfg.work_dir, overrides_path=override)

    run_dir = result["run_dir"]
    assert run_dir == tmp_path / "work" / "candidates" / "runs" / "maskfix_01"
    assert (run_dir / "all_candidates.csv").is_file()

    statuses = _read_new_statuses(run_dir)
    # A left the mask -> back to its preliminary interpretation.
    assert statuses["A"] == "preliminary_rule_pass"
    # B still inside the genuine injection core.
    assert statuses["B"] == "suspect_injection_mask"
    # C untouched.
    assert statuses["C"] == "preliminary_rule_fail"


def test_source_run_is_never_modified(tmp_path):
    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")

    before = _hash_tree(source)
    pp.postprocess_run(config=cfg, source_run_dir=source, new_run_name="mf",
                       work_dir=cfg.work_dir, overrides_path=override)
    after = _hash_tree(source)
    assert before == after


def test_postprocess_does_not_rerun_cellfinder(tmp_path, monkeypatch):
    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")

    # Any attempt to run Cellfinder detection must blow up.
    import mouse_brain_pipeline.cellfinder_adapter as cfa

    def _boom(*a, **k):
        raise AssertionError("Cellfinder must not be called during post-processing")

    monkeypatch.setattr(cfa, "run_cellfinder_detection", _boom)

    result = pp.postprocess_run(config=cfg, source_run_dir=source, new_run_name="mf",
                                work_dir=cfg.work_dir, overrides_path=override)
    assert result["metadata"]["cellfinder_rerun"] is False


def test_existing_run_folder_is_not_overwritten(tmp_path):
    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")

    target = tmp_path / "work" / "candidates" / "runs" / "taken"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("do not touch", encoding="utf-8")

    with pytest.raises(FileExistsError):
        pp.postprocess_run(config=cfg, source_run_dir=source, new_run_name="taken",
                           work_dir=cfg.work_dir, overrides_path=override)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "do not touch"


def test_channel_labels_and_metadata_use_red_for_channel_2(tmp_path):
    from mouse_brain_pipeline import channel_display_name

    # Human label maps channel_2_signal -> red signal channel; internal name kept.
    assert channel_display_name("green_signal") == "green signal channel"
    assert channel_display_name("channel_2_signal") == "red signal channel"

    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")
    result = pp.postprocess_run(config=cfg, source_run_dir=source, new_run_name="labels",
                                work_dir=cfg.work_dir, overrides_path=override)
    labels = result["metadata"]["channel_display_names"]
    assert labels["green_signal"] == "green signal channel"


def test_radial_analysis_runs_on_postprocessed_run(tmp_path):
    source = _make_source_run(tmp_path)
    override = _make_override(tmp_path)
    cfg = _config(tmp_path / "work")
    result = pp.postprocess_run(config=cfg, source_run_dir=source, new_run_name="rad",
                                work_dir=cfg.work_dir, overrides_path=override)

    from mouse_brain_pipeline.radial_report import analyze_run

    summary = analyze_run(result["run_dir"], cfg, channel="green_signal", section=70,
                          center_xy=(60.0, 60.0), bin_width_um=20.0)
    for key in ("candidate_radial_coordinates", "radial_counts_by_status",
                "radial_density_vs_distance", "radial_count_vs_distance",
                "radial_fraction_vs_distance", "radial_cumulative_fraction"):
        assert Path(summary[key]).is_file()
