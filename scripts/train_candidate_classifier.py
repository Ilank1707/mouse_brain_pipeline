#!/usr/bin/env python
"""Train one dye-specific candidate-level 3D PyTorch classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.classifier import (
    CandidatePatchDataset,
    binary_training_records,
    build_model,
    candidate_group_key,
    grouped_train_validation_split,
    merge_candidates_and_labels,
    require_minimum_labels,
    software_versions,
    validation_metrics,
)
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import read_csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a single-channel 3D candidate classifier.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True, choices=["green_signal", "channel_2_signal"])
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    config = load_config(args.config)
    candidates_path = Path(args.candidates or config.work_dir / "candidates" / "all_candidates.csv")
    labels_path = Path(args.labels or config.work_dir / "candidates" / "manual_labels.csv")
    records = merge_candidates_and_labels(
        read_csv_rows(candidates_path), read_csv_rows(labels_path), args.channel
    )
    try:
        counts = require_minimum_labels(
            records, config.classifier.minimum_cells, config.classifier.minimum_artifacts
        )
    except ValueError as exc:
        print(f"REFUSING TO TRAIN: {exc}", file=sys.stderr)
        return 2
    eligible = binary_training_records(records)
    train_rows, validation_rows = grouped_train_validation_split(
        eligible,
        group_by=config.classifier.group_by,
        spatial_tile_size_px=config.classifier.spatial_tile_size_px,
        validation_fraction=config.classifier.validation_fraction,
        seed=config.classifier.random_seed,
    )
    if not validation_rows:
        print(
            "WARNING: only one independent group is available. A trustworthy validation set "
            "is not possible; the model will be trained but remains unvalidated."
        )
    if len({row.get("section") for row in eligible}) == 1:
        print(
            "WARNING: all labels come from one section. Scientific validation requires "
            "held-out sections/regions from multiple brains."
        )

    directory = (
        config.data.green_signal_dir if args.channel == "green_signal"
        else config.data.channel_2_signal_dir
    )
    channel_index = index_channel(args.channel, directory, config.data.filename_regex)
    patch_px = int(round(
        config.classifier.patch_size_xy_um / config.acquisition.voxel_size_y_um
    ))
    if patch_px % 2 == 0:
        patch_px += 1
    train_dataset = CandidatePatchDataset(train_rows, channel_index, patch_px, augment=True)
    validation_dataset = CandidatePatchDataset(
        validation_rows, channel_index, patch_px, augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.classifier.batch_size, shuffle=True,
        num_workers=config.classifier.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.classifier.batch_size, shuffle=False,
        num_workers=config.classifier.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    class_weights = torch.tensor([
        len(eligible) / (2 * counts["artefact"]),
        len(eligible) / (2 * counts["cell"]),
    ], dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.classifier.learning_rate)
    history = []
    for epoch in range(config.classifier.epochs):
        model.train()
        losses = []
        for volumes, labels, _ids in train_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(volumes), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    confusion = np.zeros((2, 2), dtype=int)
    validation_probabilities = []
    validation_truths = []
    validation_ids = []
    model.eval()
    with torch.no_grad():
        for volumes, labels, ids in validation_loader:
            probabilities = torch.softmax(model(volumes.to(device)), dim=1)[:, 1].cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)
            validation_probabilities.extend(float(value) for value in probabilities)
            validation_truths.extend(int(value) for value in labels.numpy())
            validation_ids.extend(ids)
            for truth, prediction in zip(labels.numpy(), predictions):
                confusion[int(truth), int(prediction)] += 1
    validation_total = int(confusion.sum())
    validation_accuracy = (
        float(np.trace(confusion) / validation_total) if validation_total else None
    )
    detailed_metrics = validation_metrics(
        validation_truths,
        validation_probabilities,
        cell_threshold=config.classifier.cell_probability_threshold,
        artifact_threshold=config.classifier.artifact_probability_threshold,
    )
    spatial_groups = {
        candidate_group_key(
            row, config.classifier.group_by,
            config.classifier.spatial_tile_size_px,
        )
        for row in eligible
    }

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(
        args.output_dir or config.work_dir / "classifiers" / args.channel / version
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "channel": args.channel,
        "version": version,
        "patch_size_xy_px": patch_px,
        "z_planes": config.acquisition.planes_per_section,
        # Training completion never passes the scientific validation gate.
        "validated": False,
        "validation_gate_passed": False,
    }
    torch.save(checkpoint, out_dir / "model.pt")
    (out_dir / "training_configuration.json").write_text(json.dumps({
        "channel": args.channel,
        "classifier": vars(config.classifier),
        "label_counts": counts,
        "number_of_spatial_groups": len(spatial_groups),
        "device": str(device),
    }, indent=2), encoding="utf-8")
    (out_dir / "software_versions.json").write_text(
        json.dumps(software_versions(), indent=2), encoding="utf-8"
    )
    (out_dir / "split_candidate_ids.json").write_text(json.dumps({
        "training": [row["candidate_id"] for row in train_rows],
        "validation": [row["candidate_id"] for row in validation_rows],
    }, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "history": history,
        "validation_accuracy": validation_accuracy,
        "grouped_holdout_available": bool(validation_rows),
        "validation_gate_passed": False,
        **detailed_metrics,
    }, indent=2), encoding="utf-8")
    with open(out_dir / "validation_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "truth", "cell_probability", "predicted_class"],
        )
        writer.writeheader()
        for candidate_id, truth, probability in zip(
            validation_ids, validation_truths, validation_probabilities
        ):
            writer.writerow({
                "candidate_id": candidate_id,
                "truth": truth,
                "cell_probability": probability,
                "predicted_class": int(probability >= 0.5),
            })
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["truth/prediction", "artefact", "cell"])
        writer.writerow(["artefact", *confusion[0].tolist()])
        writer.writerow(["cell", *confusion[1].tolist()])
    print(f"Saved {args.channel} classifier bundle to {out_dir}")
    print("Model validation gate: NOT PASSED (training alone never validates a model).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
