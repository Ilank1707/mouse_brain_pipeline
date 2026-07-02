"""Focused tests for candidate-to-candidate pair-correlation analysis."""

from __future__ import annotations

import inspect
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

import pair_correlation_analysis as pc  # noqa: E402


def _square_window(size: int = 400) -> pc.MaskWindow:
    return pc.MaskWindow(np.ones((size, size), dtype=bool))


def test_uniform_random_points_have_g_near_one_away_from_small_distances():
    window = _square_window()
    observed_rng = np.random.default_rng(101)
    points = window.random_points_um(900, observed_rng, (1.0, 1.0))
    edges = np.arange(0.0, 105.0, 5.0)

    result = pc.analyze_pair_correlation(
        points,
        window,
        edges,
        simulations=39,
        random_seed=202,
        voxel_yx_um=(1.0, 1.0),
    )

    middle = result["g_r"][4:16]
    assert np.all(np.isfinite(middle))
    assert 0.90 < float(np.mean(middle)) < 1.10


def test_clustered_points_have_g_above_one_at_short_distances():
    rng = np.random.default_rng(303)
    window = _square_window()
    centres = rng.uniform(60.0, 340.0, size=(8, 2))
    points = np.vstack(
        [centre + rng.normal(0.0, 3.0, size=(100, 2)) for centre in centres]
    )
    edges = np.arange(0.0, 105.0, 5.0)

    result = pc.analyze_pair_correlation(
        points,
        window,
        edges,
        simulations=29,
        random_seed=404,
        voxel_yx_um=(1.0, 1.0),
    )

    assert result["g_r"][0] > 1.0
    assert float(np.nanmean(result["g_r"][:3])) > 2.0


def test_regularly_spaced_points_have_g_below_one_at_short_distances():
    window = _square_window()
    coordinate = np.arange(10.0, 391.0, 20.0)
    xx, yy = np.meshgrid(coordinate, coordinate)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    edges = np.arange(0.0, 110.0, 10.0)

    result = pc.analyze_pair_correlation(
        points,
        window,
        edges,
        simulations=29,
        random_seed=505,
        voxel_yx_um=(1.0, 1.0),
    )

    assert result["g_r"][0] == pytest.approx(0.0)
    assert result["g_r"][0] < 1.0


def test_pair_count_excludes_self_pairs_and_counts_unordered_pairs_once():
    # Three unique inter-point distances: 3, 4 and 5.
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    counts = pc.pair_histogram(points, np.array([0.0, 3.5, 4.5, 5.5]))
    assert counts.tolist() == [1, 1, 1]
    assert int(counts.sum()) == 3

    # Two distinct candidate rows at the same XY are one non-self pair.
    duplicate_xy = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert pc.pair_histogram(duplicate_xy, np.array([0.0, 1.0])).tolist() == [1]


def test_pair_count_does_not_use_full_distance_matrix_functions(monkeypatch):
    import scipy.spatial
    import scipy.spatial.distance

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full pairwise distance matrix function was called")

    monkeypatch.setattr(scipy.spatial, "distance_matrix", forbidden)
    monkeypatch.setattr(scipy.spatial.distance, "cdist", forbidden)
    monkeypatch.setattr(scipy.spatial.distance, "pdist", forbidden)

    points = np.random.default_rng(606).uniform(0.0, 10_000.0, size=(12_500, 2))
    counts = pc.pair_histogram(points, np.arange(0.0, 55.0, 5.0))
    assert counts.shape == (10,)


def test_green_and_red_outputs_remain_separate(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True)
    out_dir = tmp_path / "pair_outputs"
    channels = ("green_signal", "channel_2_signal")

    with (run_dir / "all_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "channel",
                "section",
                "x_global_px",
                "y_global_px",
                "current_status",
            ],
        )
        writer.writeheader()
        for channel in channels:
            for index, (x, y) in enumerate(((10, 10), (20, 20), (30, 15))):
                writer.writerow(
                    {
                        "candidate_id": f"{channel}_{index}",
                        "channel": channel,
                        "section": 70,
                        "x_global_px": x,
                        "y_global_px": y,
                        "current_status": "preliminary_rule_pass",
                    }
                )

    metadata = {
        "array_order": "z,y,x",
        "crop_x_min_x_max_y_min_y_max": None,
        "source_image_dimensions": {
            channel: {"height": 50, "width": 50} for channel in channels
        },
        "acquisition": {"planes_per_section": 7},
    }
    (run_dir / "candidate_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    for channel in channels:
        section_dir = run_dir / "qc" / f"{channel}_section_070"
        section_dir.mkdir(parents=True)
        np.save(section_dir / "tissue_mask.npy", np.ones((50, 50), dtype=bool))
        np.save(
            section_dir / "injection_analysis_exclusion_mask.npy",
            np.zeros((50, 50), dtype=bool),
        )

    config = SimpleNamespace(
        acquisition=SimpleNamespace(voxel_size_y_um=1.004, voxel_size_x_um=1.004)
    )
    for channel in channels:
        pc.run_analysis(
            config=config,
            run_dir=run_dir,
            channel=channel,
            section=70,
            out_dir=out_dir,
            bin_width_um=5,
            maximum_distance_um=25,
            simulations=2,
            random_seed=123,
        )

    green = out_dir / "green_signal" / "all_outside_injection"
    red = out_dir / "channel_2_signal" / "all_outside_injection"
    assert (green / "pair_correlation_g_r.png").is_file()
    assert (red / "pair_correlation_g_r.png").is_file()
    assert green != red
    assert (out_dir / "pair_correlation_run.json").is_file()
    with (green / "pair_correlation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert {row["channel"] for row in csv.DictReader(handle)} == {
            "green_signal"
        }
    with (red / "pair_correlation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert {row["channel"] for row in csv.DictReader(handle)} == {
            "channel_2_signal"
        }


def test_existing_injection_centred_radial_analysis_is_not_used():
    source = inspect.getsource(pc)
    assert "mouse_brain_pipeline.radial_analysis" not in source
    assert "mouse_brain_pipeline.radial_report" not in source
    assert "injection_core_centroid" not in source


def test_cli_defaults_and_requested_status_series():
    parser = pc.build_parser()
    args = parser.parse_args(
        [
            "--run-dir",
            "run",
            "--channel",
            "green_signal",
            "--section",
            "70",
            "--out-dir",
            "out",
        ]
    )
    assert args.bin_width_um == 5.0
    assert args.maximum_distance_um == 500.0
    assert args.simulations == 99
    assert args.random_seed == 12345
    assert [status for status, _selector, _outside in pc.SERIES] == [
        "all_outside_injection",
        "preliminary_pass",
        "preliminary_fail",
        "manual_review",
        "invalid_measurement",
        "confirmed_cell",
        "artefact",
        "uncertain",
        "injection",
    ]
