"""類別平衡分析 — 有效樣本數與建議的 class weight (Cui et al., CB Loss)。

counts / fractions 依**類別索引**排列, 與 DatasetFacts.class_labels (C0..Cn) 同序 —
決策層看得到分佈, 但看不到類別的真實名稱。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ._util import class_counts

_BETA = 0.999


class ClassBalanceOut(BaseModel):
    n_classes: int = 0
    counts: list[int] = Field(default_factory=list)        # 與 class_labels 同序
    fractions: list[float] = Field(default_factory=list)
    val_counts: list[int] = Field(default_factory=list)
    test_counts: list[int] = Field(default_factory=list)
    imbalance_ratio: float = 1.0
    gini: float = 0.0                                      # 0=完全平均
    effective_n: list[float] = Field(default_factory=list)
    suggested_weights: list[float] = Field(default_factory=list)  # 均值正規化為 1
    min_class_count: int = 0
    max_class_count: int = 0
    n_classes_under_50: int = 0
    recommend_weighted_loss: bool = False
    recommend_focal_loss: bool = False
    recommend_macro_f1_over_accuracy: bool = False


def run(root: str) -> ClassBalanceOut:
    import os

    train = class_counts(os.path.join(root, "train"))
    classes = sorted(train)
    counts = [train[c] for c in classes]
    out = ClassBalanceOut(n_classes=len(classes), counts=counts)
    if not counts:
        return out

    val = class_counts(os.path.join(root, "val"))
    test = class_counts(os.path.join(root, "test"))
    out.val_counts = [val.get(c, 0) for c in classes]
    out.test_counts = [test.get(c, 0) for c in classes]

    total = sum(counts) or 1
    out.fractions = [round(c / total, 5) for c in counts]
    nz = [c for c in counts if c > 0] or [1]
    out.min_class_count, out.max_class_count = min(nz), max(nz)
    out.imbalance_ratio = round(out.max_class_count / out.min_class_count, 3)
    out.n_classes_under_50 = sum(1 for c in counts if c < 50)

    # Gini: 0 = 完全平均, 越大越集中
    srt = sorted(counts)
    n = len(srt)
    cum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(srt))
    out.gini = round(cum / (n * total), 4) if total else 0.0

    # 有效樣本數 (1 - beta^n) / (1 - beta); 權重取倒數後正規化為均值 1
    eff = [(1.0 - _BETA ** c) / (1.0 - _BETA) if c > 0 else 1.0 for c in counts]
    out.effective_n = [round(e, 2) for e in eff]
    raw = [1.0 / e if e else 0.0 for e in eff]
    mean = (sum(raw) / len(raw)) or 1.0
    out.suggested_weights = [round(r / mean, 4) for r in raw]

    out.recommend_weighted_loss = out.imbalance_ratio >= 3.0
    out.recommend_focal_loss = (out.imbalance_ratio >= 10.0
                                or out.n_classes_under_50 > 0)
    out.recommend_macro_f1_over_accuracy = out.imbalance_ratio >= 3.0
    return out
