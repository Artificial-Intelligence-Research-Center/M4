"""Evaluation utilities for multi-label image classification."""

import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import util.misc as misc


def _dataset_classes(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset.classes
    return dataset.classes


def _macro_auc(targets, probabilities):
    values = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size == 2:
            values.append(roc_auc_score(targets[:, index], probabilities[:, index]))
    return float(np.mean(values)) if values else float("nan"), len(values)


def _macro_average_precision(targets, probabilities):
    values = []
    for index in range(targets.shape[1]):
        if targets[:, index].sum() > 0:
            values.append(average_precision_score(targets[:, index], probabilities[:, index]))
    return float(np.mean(values)) if values else float("nan"), len(values)


@torch.no_grad()
def evaluate_multilabel(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    criterion = nn.BCEWithLogitsLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    model.eval()

    targets_all, predictions_all, probabilities_all, paths = [], [], [], []
    for images, targets, batch_paths in metric_logger.log_every(data_loader, 10, f"{mode}:"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True, dtype=torch.float32)
        with torch.cuda.amp.autocast():
            logits = model(images)
            if logits.shape != targets.shape:
                raise ValueError(
                    f"Multi-label logits shape {tuple(logits.shape)} does not match "
                    f"target shape {tuple(targets.shape)}"
                )
            loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= args.multilabel_threshold).to(torch.int64)
        metric_logger.update(loss=loss.item())
        targets_all.append(targets.to(torch.int64).cpu().numpy())
        predictions_all.append(predictions.cpu().numpy())
        probabilities_all.append(probabilities.float().cpu().numpy())
        paths.extend(batch_paths)

    targets = np.concatenate(targets_all, axis=0)
    predictions = np.concatenate(predictions_all, axis=0)
    probabilities = np.concatenate(probabilities_all, axis=0)
    if targets.shape[1] != num_class:
        raise ValueError(f"Expected {num_class} labels, received {targets.shape[1]}")

    metrics = {
        "subset_accuracy": accuracy_score(targets, predictions),
        "label_accuracy": float((targets == predictions).mean()),
        "micro_f1": f1_score(targets, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "micro_precision": precision_score(targets, predictions, average="micro", zero_division=0),
        "macro_precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "micro_recall": recall_score(targets, predictions, average="micro", zero_division=0),
        "macro_recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "hamming": hamming_loss(targets, predictions),
        "jaccard": jaccard_score(targets, predictions, average="macro", zero_division=0),
    }
    metrics["roc_auc"], valid_auc_classes = _macro_auc(targets, probabilities)
    metrics["average_precision"], valid_ap_classes = _macro_average_precision(targets, probabilities)
    score_values = [metrics[name] for name in ("macro_f1", "roc_auc", "average_precision")]
    metrics["score"] = float(np.mean([value for value in score_values if np.isfinite(value)]))

    if log_writer:
        for name, value in metrics.items():
            log_writer.add_scalar(f"perf/{name}", value, epoch)
    print(f"val loss: {metric_logger.meters['loss'].global_avg}")
    print(
        f"Subset accuracy: {metrics['subset_accuracy']:.4f}, "
        f"Micro/Macro F1: {metrics['micro_f1']:.4f}/{metrics['macro_f1']:.4f}, "
        f"Macro AUROC: {metrics['roc_auc']:.4f} ({valid_auc_classes}/{num_class} classes), "
        f"mAP: {metrics['average_precision']:.4f} ({valid_ap_classes}/{num_class} classes)"
    )
    metric_logger.synchronize_between_processes()

    classes = _dataset_classes(data_loader.dataset)
    frame = pd.DataFrame({"image_path": paths})
    for index, class_name in enumerate(classes):
        frame[f"{class_name}_score"] = probabilities[:, index]
        frame[f"{class_name}_pred"] = predictions[:, index]
        frame[f"{class_name}_true"] = targets[:, index]

    results_path = os.path.join(args.output_dir, args.task, f"metrics_{mode}.csv")
    file_exists = os.path.isfile(results_path)
    row = {"val_loss": metric_logger.meters["loss"].global_avg, **metrics}
    with open(results_path, "a", newline="", encoding="utf8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics["score"], frame
