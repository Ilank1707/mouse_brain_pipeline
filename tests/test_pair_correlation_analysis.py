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
    assert args.intensity_bandwidth_um == 200.0
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


# --------------------------------------------------------------------------- #
# Inhomogeneous pair-correlation analysis
# --------------------------------------------------------------------------- #


def _gradient_poisson_points(rng, size_um=400.0, target=4000):
    """Inhomogeneous Poisson points with a smooth linear intensity gradient."""
    xs: list[float] = []
    ys: list[float] = []
    while len(xs) < target:
        x = rng.uniform(0.0, size_um, 20_000)
        y = rng.uniform(0.0, size_um, 20_000)
        keep = rng.random(20_000) < (x + 50.0) / (size_um + 50.0)
        xs.extend(x[keep].tolist())
        ys.extend(y[keep].tolist())
    return np.column_stack((xs[:target], ys[:target]))


def test_inhomogeneous_poisson_gradient_has_g_inhom_near_one():
    window = _square_window(400)
    points = _gradient_poisson_points(np.random.default_rng(2024))
    edges = np.arange(0.0, 105.0, 5.0)

    result = pc.analyze_inhomogeneous_pair_correlation(
        points,
        window,
        edges,
        simulations=39,
        random_seed=7,
        voxel_yx_um=(1.0, 1.0),
        bandwidth_um=60.0,
    )

    middle = result["g_inhom_r"][3:16]
    assert np.all(np.isfinite(middle))
    # Reweighting by the estimated intensity removes the large-scale gradient, so
    # the inhomogeneous statistic sits near unity even though a homogeneous g(r)
    # would stay above one across these distances.
    assert 0.80 < float(np.nanmean(middle)) < 1.20


def test_inhomogeneous_clustered_pattern_exceeds_one_at_short_distances():
    rng = np.random.default_rng(11)
    window = _square_window(400)
    centres = rng.uniform(60.0, 340.0, size=(8, 2))
    points = np.vstack(
        [centre + rng.normal(0.0, 3.0, size=(120, 2)) for centre in centres]
    )
    edges = np.arange(0.0, 105.0, 5.0)

    result = pc.analyze_inhomogeneous_pair_correlation(
        points,
        window,
        edges,
        simulations=29,
        random_seed=5,
        voxel_yx_um=(1.0, 1.0),
        bandwidth_um=200.0,
    )

    # A large-scale intensity bandwidth cannot absorb fine (~3 µm) clustering.
    assert result["g_inhom_r"][0] > 1.0
    assert float(np.nanmean(result["g_inhom_r"][:3])) > 2.0


def test_weighted_pair_histogram_counts_unordered_pairs_once():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    weights = np.array([1.0, 2.0, 3.0])
    # Pair distances 3, 4, 5 with weights w_i*w_j = 2, 3, 6 respectively.
    counts = pc.weighted_pair_histogram(
        points, weights, np.array([0.0, 3.5, 4.5, 5.5])
    )
    assert counts.tolist() == pytest.approx([2.0, 3.0, 6.0])
    # Sum over unordered pairs equals sum_{i<j} w_i*w_j, never double-counted.
    assert float(counts.sum()) == pytest.approx(11.0)

    # Two distinct rows at one location are a single non-self weighted pair.
    duplicate_xy = np.array([[1.0, 1.0], [1.0, 1.0]])
    single = pc.weighted_pair_histogram(
        duplicate_xy, np.array([2.0, 3.0]), np.array([0.0, 1.0])
    )
    assert single.tolist() == pytest.approx([6.0])


def test_inhomogeneous_analysis_does_not_use_full_distance_matrix(monkeypatch):
    import scipy.spatial
    import scipy.spatial.distance

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full pairwise distance matrix function was called")

    monkeypatch.setattr(scipy.spatial, "distance_matrix", forbidden)
    monkeypatch.setattr(scipy.spatial.distance, "cdist", forbidden)
    monkeypatch.setattr(scipy.spatial.distance, "pdist", forbidden)

    window = _square_window(200)
    points = np.random.default_rng(1).uniform(0.0, 200.0, size=(800, 2))
    result = pc.analyze_inhomogeneous_pair_correlation(
        points,
        window,
        np.arange(0.0, 55.0, 5.0),
        simulations=5,
        random_seed=3,
        voxel_yx_um=(1.0, 1.0),
        bandwidth_um=50.0,
    )
    assert result["g_inhom_r"].shape == (10,)


def test_zero_intensity_pixels_are_handled_safely():
    window = _square_window(300)
    rng = np.random.default_rng(9)
    # Candidates confined to one corner leave the opposite corner near-zero.
    points = rng.uniform(10.0, 120.0, size=(400, 2))
    surface = pc.IntensitySurface.build(
        window,
        points,
        voxel_yx_um=(1.0, 1.0),
        bandwidth_um=40.0,
        grid_step_um=pc.resolve_grid_step_um(40.0),
    )
    far_corner = np.array([[280.0, 280.0]])
    lam = surface.evaluate(far_corner)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0.0)

    result = pc.analyze_inhomogeneous_pair_correlation(
        points,
        window,
        np.arange(0.0, 55.0, 5.0),
        simulations=5,
        random_seed=2,
        voxel_yx_um=(1.0, 1.0),
        bandwidth_um=40.0,
    )
    # No division by zero anywhere: reweighted sums stay finite.
    assert np.all(np.isfinite(result["observed_weighted_pair_sum"]))


def _write_two_channel_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
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
        rng = np.random.default_rng(77)
        for channel in channels:
            coordinates = rng.uniform(5.0, 95.0, size=(60, 2))
            for index, (x, y) in enumerate(coordinates):
                writer.writerow(
                    {
                        "candidate_id": f"{channel}_{index}",
                        "channel": channel,
                        "section": 70,
                        "x_global_px": float(x),
                        "y_global_px": float(y),
                        "current_status": "preliminary_rule_pass",
                    }
                )
    metadata = {
        "array_order": "z,y,x",
        "crop_x_min_x_max_y_min_y_max": None,
        "source_image_dimensions": {
            channel: {"height": 100, "width": 100} for channel in channels
        },
        "acquisition": {"planes_per_section": 7},
    }
    (run_dir / "candidate_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    for channel in channels:
        section_dir = run_dir / "qc" / f"{channel}_section_070"
        section_dir.mkdir(parents=True)
        np.save(section_dir / "tissue_mask.npy", np.ones((100, 100), dtype=bool))
        np.save(
            section_dir / "injection_analysis_exclusion_mask.npy",
            np.zeros((100, 100), dtype=bool),
        )
    config = SimpleNamespace(
        acquisition=SimpleNamespace(voxel_size_y_um=1.004, voxel_size_x_um=1.004)
    )
    return run_dir, channels, config


def test_homogeneous_and_inhomogeneous_outputs_use_distinct_filenames_per_channel(
    tmp_path,
):
    run_dir, channels, config = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "pair_outputs"
    for channel in channels:
        pc.run_analysis(
            config=config,
            run_dir=run_dir,
            channel=channel,
            section=70,
            out_dir=out_dir,
            bin_width_um=5,
            maximum_distance_um=30,
            simulations=3,
            random_seed=123,
            intensity_bandwidth_um=40.0,
        )

    homogeneous_names = {
        "pair_correlation.csv",
        "pair_correlation_g_r.png",
    }
    inhomogeneous_names = {
        "pair_correlation_inhomogeneous.csv",
        "pair_correlation_inhomogeneous_g_r.png",
        "inhomogeneous_analysis_summary.json",
        "estimated_intensity_surface.png",
    }
    # Homogeneous and inhomogeneous outputs never share a filename.
    assert homogeneous_names.isdisjoint(inhomogeneous_names)

    green = out_dir / "green_signal" / "all_outside_injection"
    red = out_dir / "channel_2_signal" / "all_outside_injection"
    assert green != red
    for status_dir in (green, red):
        for name in homogeneous_names | inhomogeneous_names:
            assert (status_dir / name).is_file(), f"missing {name} in {status_dir}"

    # Green and red inhomogeneous outputs stay separate and self-consistent.
    for status_dir, expected_channel in (
        (green, "green_signal"),
        (red, "channel_2_signal"),
    ):
        with (status_dir / "pair_correlation_inhomogeneous.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert {row["channel"] for row in rows} == {expected_channel}
            assert {row["intensity_bandwidth_um"] for row in rows} == {"40"}
            assert reader.fieldnames == pc.INHOMOGENEOUS_CSV_COLUMNS
        summary = json.loads(
            (status_dir / "inhomogeneous_analysis_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["intensity_bandwidth_um"] == 40.0
        assert summary["channel"] == expected_channel
