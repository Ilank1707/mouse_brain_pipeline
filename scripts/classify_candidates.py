#!/usr/bin/env python
"""Apply one trained dye-specific 3D classifier to preliminary candidates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.classifier import (
    CandidatePatchDataset,
    build_model,
    classifier_state,
    merge_candidates_and_labels,
)
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import read_csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify preliminary 3D candidates.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True, choices=["green_signal", "channel_2_signal"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if checkpoint.get("channel") != args.channel:
        print(
            f"ERROR: model channel={checkpoint.get('channel')!r} cannot classify "
            f"{args.channel!r}.",
            file=sys.stderr,
        )
        return 2
    candidates_path = Path(args.candidates or config.work_dir / "candidates" / "all_candidates.csv")
    labels_path = Path(args.labels or config.work_dir / "candidates" / "manual_labels.csv")
    rows = merge_candidates_and_labels(
        read_csv_rows(candidates_path), read_csv_rows(labels_path), args.channel
    )
    directory = (
        config.data.green_signal_dir if args.channel == "green_signal"
        else config.data.channel_2_signal_dir
    )
    channel_index = index_channel(args.channel, directory, config.data.filename_regex)
    dataset = CandidatePatchDataset(
        rows, channel_index, checkpoint["patch_size_xy_px"], augment=False,
        z_planes=checkpoint.get("z_planes", 7),
    )
    loader = DataLoader(dataset, batch_size=config.classifier.batch_size, shuffle=False)
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities = {}
    with torch.no_grad():
        for volumes, _labels, ids in loader:
            probs = torch.softmax(model(volumes), dim=1)[:, 1].numpy()
            probabilities.update({candidate_id: float(prob) for candidate_id, prob in zip(ids, probs)})

    for row in rows:
        probability = probabilities[row["candidate_id"]]
        validation_passed = bool(checkpoint.get("validation_gate_passed", False))
        status, included = classifier_state(
            row, probability,
            manual_label=row.get("manual_label", ""),
            cell_threshold=config.classifier.cell_probability_threshold,
            artifact_threshold=config.classifier.artifact_probability_threshold,
            model_validated=validation_passed,
        )
        row["classifier_probability"] = round(probability, 6)
        row["classifier_model"] = str(Path(args.model).resolve())
        row["classifier_version"] = checkpoint.get("version", "")
        row["model_validation_passed"] = validation_passed
        row["final_decision"] = status
        row["current_status"] = status
        row["included_in_count"] = included
        if str(row.get("manual_label", "")).lower() == "injection":
            row["injection_assignment_source"] = "human_candidate_label"
    output = Path(
        args.output
        or config.work_dir / "candidates" / f"classified_candidates_{args.channel}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(
        [key for row in rows for key in row]
        + ["classifier_probability", "classifier_model", "classifier_version",
           "current_status", "included_in_count"]
    ))
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not checkpoint.get("validation_gate_passed", False):
        print("WARNING: model is not marked validated; predicted cells remain excluded from counts.")
    print(f"Wrote classified candidates to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
