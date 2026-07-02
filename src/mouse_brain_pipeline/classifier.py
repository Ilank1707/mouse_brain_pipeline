"""Single-channel candidate-centred 3D classifier utilities."""

from __future__ import annotations

import platform
import random
from collections import Counter

BINARY_LABELS = {"cell": 1, "artefact": 0}
EXCLUDED_LABELS = {"injection", "uncertain"}


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def merge_candidates_and_labels(candidates: list[dict], labels: list[dict], channel: str) -> list[dict]:
    label_map = {
        (row.get("candidate_id", ""), row.get("channel", "")): row
        for row in labels
    }
    merged = []
    for candidate in candidates:
        if candidate.get("channel") != channel:
            continue
        row = dict(candidate)
        label = label_map.get((row.get("candidate_id", ""), channel), {})
        manual_label = str(label.get("manual_label", "")).strip().lower()
        if manual_label == "artifact":
            manual_label = "artefact"
        row["manual_label"] = manual_label
        row["reviewer"] = label.get("reviewer", "")
        merged.append(row)
    return merged


def binary_training_records(records: list[dict]) -> list[dict]:
    """Keep only human-labelled cell/artefact examples."""
    return [row for row in records if row.get("manual_label") in BINARY_LABELS]


def require_minimum_labels(records: list[dict], minimum_cells: int, minimum_artifacts: int) -> dict:
    counts = Counter(row.get("manual_label") for row in binary_training_records(records))
    if counts["cell"] < minimum_cells or counts["artefact"] < minimum_artifacts:
        raise ValueError(
            "Insufficient manual labels: "
            f"cells={counts['cell']}/{minimum_cells}, "
            f"artefacts={counts['artefact']}/{minimum_artifacts}. "
            "Injection and uncertain labels are retained but excluded from binary training."
        )
    return {"cell": counts["cell"], "artefact": counts["artefact"]}


def candidate_group_key(row: dict, group_by: str, spatial_tile_size_px: int = 512) -> str:
    if group_by == "mouse":
        return f"mouse:{row.get('mouse', 'unknown')}"
    if group_by == "section":
        return f"section:{row.get('section', 'unknown')}"
    if group_by == "spatial_tile":
        x = int(float(row.get("x_global_px", 0) or 0))
        y = int(float(row.get("y_global_px", 0) or 0))
        section = row.get("section", "unknown")
        return f"section:{section}:tile:{x // spatial_tile_size_px}:{y // spatial_tile_size_px}"
    raise ValueError(f"Unsupported grouping: {group_by!r}")


def grouped_train_validation_split(
    records: list[dict],
    *,
    group_by: str = "spatial_tile",
    spatial_tile_size_px: int = 512,
    validation_fraction: float = 0.2,
    seed: int = 20260625,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in records:
        key = candidate_group_key(row, group_by, spatial_tile_size_px)
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        return list(records), []
    n_validation = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:n_validation])
    train = [row for key, rows in groups.items() if key not in validation_keys for row in rows]
    validation = [row for key, rows in groups.items() if key in validation_keys for row in rows]
    return train, validation


def extract_candidate_patch(candidate: dict, channel_index, patch_size_xy_px: int, z_planes=7):
    """Read one channel only and return a normalized ``[1,Z,Y,X]`` tensor array."""
    import numpy as np
    import tifffile

    section = int(float(candidate["section"]))
    x = int(float(candidate["x_global_px"]))
    y = int(float(candidate["y_global_px"]))
    half = patch_size_xy_px // 2
    out_size = 2 * half + 1
    planes = {
        plane: path for (candidate_section, plane), path in channel_index.files.items()
        if candidate_section == section
    }
    if not planes:
        raise FileNotFoundError(f"No {channel_index.name} planes for section {section}")
    patches = []
    for plane in sorted(planes)[:z_planes]:
        with tifffile.TiffFile(str(planes[plane])) as tf:
            page = tf.pages[0]
            try:
                image = page.asarray(out="memmap")
            except (TypeError, ValueError):
                image = page.asarray()
            y0, y1 = max(0, y - half), min(image.shape[0], y + half + 1)
            x0, x1 = max(0, x - half), min(image.shape[1], x + half + 1)
            source = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
            target = np.zeros((out_size, out_size), dtype=np.float32)
            oy, ox = half - (y - y0), half - (x - x0)
            target[oy:oy + source.shape[0], ox:ox + source.shape[1]] = source
            patches.append(target)
    while len(patches) < z_planes:
        patches.append(np.zeros((out_size, out_size), dtype=np.float32))
    volume = np.stack(patches[:z_planes])
    valid = np.isfinite(volume) & (volume != 0)
    if valid.any():
        lo, hi = np.percentile(volume[valid], [1.0, 99.5])
        volume = np.clip((volume - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    else:
        volume.fill(0)
    return volume[None].astype(np.float32)


class CandidatePatchDataset:
    def __init__(self, records, channel_index, patch_size_xy_px, *, augment=False, z_planes=7):
        self.records = list(records)
        self.channel_index = channel_index
        self.patch_size_xy_px = int(patch_size_xy_px)
        self.augment = bool(augment)
        self.z_planes = int(z_planes)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        row = self.records[index]
        array = extract_candidate_patch(
            row, self.channel_index, self.patch_size_xy_px, self.z_planes
        )
        tensor = torch.from_numpy(array)
        if self.augment:
            tensor = augment_xy(tensor)
        label = BINARY_LABELS.get(row.get("manual_label"), -1)
        return tensor, label, row.get("candidate_id", "")


def augment_xy(tensor):
    """Apply one shared modest XY transform to every Z plane."""
    import math
    import torch
    import torch.nn.functional as functional

    if torch.rand(()) < 0.5:
        tensor = torch.flip(tensor, dims=(-1,))
    if torch.rand(()) < 0.5:
        tensor = torch.flip(tensor, dims=(-2,))
    angle = float(torch.empty(()).uniform_(-10.0, 10.0)) * math.pi / 180.0
    tx = float(torch.empty(()).uniform_(-2.0, 2.0)) * 2.0 / tensor.shape[-1]
    ty = float(torch.empty(()).uniform_(-2.0, 2.0)) * 2.0 / tensor.shape[-2]
    theta = tensor.new_tensor([[
        [math.cos(angle), -math.sin(angle), tx],
        [math.sin(angle), math.cos(angle), ty],
    ]])
    # Treat Z as 2D channels so every plane receives exactly the same transform.
    as_2d = tensor
    grid = functional.affine_grid(
        theta, (1, tensor.shape[1], tensor.shape[2], tensor.shape[3]),
        align_corners=False,
    )
    transformed = functional.grid_sample(
        as_2d, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    scale = float(torch.empty(()).uniform_(0.85, 1.15))
    return torch.clamp(transformed * scale, 0.0, 1.0)


def build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv3d(1, 12, kernel_size=3, padding=1),
        nn.BatchNorm3d(12),
        nn.ReLU(),
        nn.MaxPool3d((1, 2, 2)),
        nn.Conv3d(12, 24, kernel_size=3, padding=1),
        nn.BatchNorm3d(24),
        nn.ReLU(),
        nn.MaxPool3d((1, 2, 2)),
        nn.Conv3d(24, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(32, 2),
    )


def software_versions() -> dict:
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def validation_metrics(
    truths,
    probabilities,
    *,
    cell_threshold=0.80,
    artifact_threshold=0.20,
) -> dict:
    """Compute validation metrics without inventing values for missing classes."""
    import numpy as np

    truths = np.asarray(truths, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "n_validation": int(truths.size),
        "n_artifacts": int(np.sum(truths == 0)),
        "n_cells": int(np.sum(truths == 1)),
        "cell_precision": None,
        "cell_recall": None,
        "cell_f1": None,
        "pr_auc": None,
        "uncertain_fraction": None,
        "probability_quantiles": None,
        "limitations": [],
    }
    if truths.size == 0:
        result["limitations"].append("No held-out validation candidates.")
        return result
    tp = int(np.sum((truths == 1) & (predictions == 1)))
    fp = int(np.sum((truths == 0) & (predictions == 1)))
    fn = int(np.sum((truths == 1) & (predictions == 0)))
    if tp + fp:
        result["cell_precision"] = tp / (tp + fp)
    else:
        result["limitations"].append("Cell precision undefined: no predicted cells.")
    if tp + fn:
        result["cell_recall"] = tp / (tp + fn)
    else:
        result["limitations"].append("Cell recall undefined: no held-out cells.")
    precision = result["cell_precision"]
    recall = result["cell_recall"]
    if precision is not None and recall is not None and precision + recall:
        result["cell_f1"] = 2 * precision * recall / (precision + recall)
    if len(set(truths.tolist())) == 2:
        try:
            from sklearn.metrics import average_precision_score

            result["pr_auc"] = float(average_precision_score(truths, probabilities))
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            result["limitations"].append(f"PR-AUC unavailable: {exc}")
    else:
        result["limitations"].append(
            "PR-AUC undefined because the held-out set does not contain both classes."
        )
    uncertain = (
        (probabilities > artifact_threshold)
        & (probabilities < cell_threshold)
    )
    result["uncertain_fraction"] = float(np.mean(uncertain))
    result["probability_quantiles"] = {
        str(q): float(np.quantile(probabilities, q))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }
    return result


def classifier_state(
    candidate: dict,
    probability: float,
    *,
    manual_label: str = "",
    cell_threshold: float = 0.8,
    artifact_threshold: float = 0.2,
    model_validated: bool = False,
) -> tuple[str, bool]:
    manual_label = manual_label.strip().lower()
    if manual_label == "artifact":
        manual_label = "artefact"
    if manual_label == "cell":
        return "manual_cell", True
    if manual_label == "artefact":
        return "manual_artifact", False
    if manual_label == "injection":
        return "injection_site", False
    if manual_label == "uncertain" or not _truthy(candidate.get("measurement_valid")):
        return "manual_review", False
    if candidate.get("current_status") == "injection_site":
        return "injection_site", False
    if candidate.get("current_status") == "suspect_injection_mask":
        return "manual_review", False
    if probability >= cell_threshold:
        return "predicted_cell", bool(model_validated)
    if probability <= artifact_threshold:
        return "predicted_artifact", False
    return "manual_review", False
