"""Evaluator — 讀取 main_finetune 產出的評估結果 (設計文件 §5.6).

main_finetune 於 output_dir/<task_id>/metrics_{val,test}.csv 落地內建指標, 欄位:
    val_loss, accuracy, f1, roc_auc, hamming, jaccard, precision, recall,
    average_precision, kappa
另於 predictions_{val,test}.csv 落地逐樣本 (true_label + <cls>_score...)。

P5: 選優目標與報告指標由 EvalConfig 決定; primary/report metric 若為 MetricRegistry
註冊的自訂名稱, 則從 predictions_*.csv 讀 (y_true, y_prob) 計算。多 fold 彙整見
metric_registry.aggregate*。
"""
from __future__ import annotations

import csv
import os
from typing import Optional

from . import metric_registry as mreg
from .schemas import EvalConfig

_METRIC_ALIASES = {
    "auroc": "roc_auc", "auc": "roc_auc", "acc": "accuracy",
}


def _read_last_row(csv_path: str) -> Optional[dict]:
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: _to_float(v) for k, v in rows[-1].items()}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_predictions(task_dir: str, mode: str = "test"):
    """讀 predictions_{mode}.csv → (y_true, y_prob)。無 numpy/檔案時回傳 (None, None)。"""
    path = os.path.join(task_dir, f"predictions_{mode}.csv")
    if not os.path.isfile(path):
        return None, None
    try:
        import numpy as np
        import pandas as pd
        df = pd.read_csv(path)
        score_cols = [c for c in df.columns if c.endswith("_score")]
        if "true_label" not in df.columns or not score_cols:
            return None, None
        return df["true_label"].to_numpy(), df[score_cols].to_numpy()
    except Exception:
        return None, None


def _custom_metric(name: str, task_dir: str, mode: str):
    fn = mreg.get(name)
    if fn is None:
        return None
    y_true, y_prob = read_predictions(task_dir, mode)
    if y_true is None:
        return None
    try:
        return float(fn(y_true, y_prob))
    except Exception:
        return None


def compute_primary(metrics: dict, cfg: EvalConfig,
                    task_dir: Optional[str] = None, mode: str = "test") -> float:
    key = cfg.primary_metric
    if key == "score":  # 沿用 engine_finetune 的 (f1+roc_auc+kappa)/3
        try:
            return (metrics["f1"] + metrics["roc_auc"] + metrics["kappa"]) / 3.0
        except KeyError:
            return 0.0
    real = _METRIC_ALIASES.get(key, key)
    if real in metrics and isinstance(metrics[real], (int, float)):
        return float(metrics[real])
    # 自訂 metric: 從 predictions 計算
    if task_dir is not None:
        v = _custom_metric(real, task_dir, mode)
        if v is not None:
            return v
    return 0.0


def evaluate(task_dir: str, cfg: EvalConfig, mode: str = "test") -> dict:
    """回傳 dict(metrics, primary_score, ckpt_path)。

    metrics 含 metrics_*.csv 內建指標 + EvalConfig.report_metrics 中的自訂指標。
    """
    metrics = _read_last_row(os.path.join(task_dir, f"metrics_{mode}.csv")) or {}
    numeric = {k: v for k, v in metrics.items() if isinstance(v, float)}

    # report_metrics 中的自訂 (registry) 指標: 補算
    for m in cfg.report_metrics:
        real = _METRIC_ALIASES.get(m, m)
        if real not in numeric and mreg.get(real) is not None:
            v = _custom_metric(real, task_dir, mode)
            if v is not None:
                numeric[real] = v

    primary = compute_primary(numeric, cfg, task_dir=task_dir, mode=mode) if metrics or numeric else 0.0
    ckpt = os.path.join(task_dir, "checkpoint-best.pth")
    return {
        "metrics": numeric,
        "primary_score": primary,
        "ckpt_path": ckpt if os.path.exists(ckpt) else None,
    }
