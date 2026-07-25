"""MetricRegistry — 使用者自訂評估指標 (設計文件 §5.6, P5).

支援註冊 `callable(y_true, y_prob) -> float`。Evaluator 在 primary_metric /
report_metrics 指向自訂名稱時, 從 main_finetune 的 predictions_{mode}.csv 讀
(y_true, y_prob) 計算。內建幾個常見指標作為範例與相容退路。

隨任務型態切換預設 metric 集 (§5.6):
  classification → accuracy/f1/roc_auc/kappa (main_finetune 已算, 存 metrics_*.csv)
  regression     → mae/r2   (自訂, 從 predictions 計算)
  segmentation   → miou/dice (預留)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# name -> callable(y_true: np.ndarray, y_prob: np.ndarray) -> float
_REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn


def get(name: str):
    return _REGISTRY.get(name)


def names() -> list[str]:
    return list(_REGISTRY)


# ---- 內建範例指標 ---------------------------------------------------------
def _balanced_accuracy(y_true, y_prob) -> float:
    y_pred = np.asarray(y_prob).argmax(axis=-1)
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    recalls = [(y_pred[y_true == c] == c).mean() for c in classes if (y_true == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def _mae(y_true, y_prob) -> float:
    """回歸範例: y_prob 為預測值 (單欄)。"""
    yp = np.asarray(y_prob).reshape(-1)
    return float(np.abs(np.asarray(y_true).reshape(-1) - yp).mean())


def _r2(y_true, y_prob) -> float:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_prob).reshape(-1)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum() or 1e-12
    return float(1 - ss_res / ss_tot)


register("balanced_accuracy", _balanced_accuracy)
register("mae", _mae)
register("r2", _r2)


# 任務型態 → 預設 metric 集 (Evaluator/EvalConfig 可據此切換預設)
DEFAULT_METRICS = {
    "classification": ["accuracy", "f1", "roc_auc", "kappa"],
    "regression": ["mae", "r2"],
    "segmentation": ["miou", "dice"],
}


# ---- 多 fold 彙整 (§5.6 aggregation=mean_std) ------------------------------
def aggregate(scores: list[float]) -> dict:
    """把多 fold 的 primary_score 彙整成 mean/std/n/folds。"""
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "folds": []}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "n": int(arr.size), "folds": [float(s) for s in arr]}


def aggregate_metrics(per_fold: list[dict]) -> dict:
    """把多 fold 的 metrics dict 逐指標彙整成 {metric: {mean,std}}。"""
    out: dict[str, dict] = {}
    keys = set()
    for m in per_fold:
        keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    for k in keys:
        vals = [m[k] for m in per_fold if isinstance(m.get(k), (int, float))]
        a = aggregate(vals)
        out[k] = {"mean": a["mean"], "std": a["std"]}
    return out
