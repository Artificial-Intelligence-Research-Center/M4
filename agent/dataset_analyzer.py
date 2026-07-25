"""DatasetAnalyzer — 掃描 ImageFolder 結構產出 DatasetProfile (設計文件 §5.1).

純程式, 不呼叫 LLM。抽樣量測影像統計以控時。資料結構:
    <root>/{train,val,test}/<class>/*.{jpg,png,...}
"""
from __future__ import annotations

import os
import random
import statistics

from PIL import Image

from .schemas import DatasetProfile

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_SPLITS = ("train", "val", "test")


def _list_classes(split_dir: str) -> list[str]:
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def _count_split(split_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in _list_classes(split_dir):
        cdir = os.path.join(split_dir, cls)
        counts[cls] = sum(
            1 for f in os.listdir(cdir) if f.lower().endswith(_IMG_EXT)
        )
    return counts


def _sample_image_paths(root: str, k: int = 40) -> list[str]:
    paths: list[str] = []
    for split in _SPLITS:
        sdir = os.path.join(root, split)
        for cls in _list_classes(sdir):
            cdir = os.path.join(sdir, cls)
            for f in os.listdir(cdir):
                if f.lower().endswith(_IMG_EXT):
                    paths.append(os.path.join(cdir, f))
    random.Random(0).shuffle(paths)
    return paths[:k]


def analyze(root: str, task_type: str = "classification") -> DatasetProfile:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"資料根目錄不存在: {root}")

    train_counts = _count_split(os.path.join(root, "train"))
    val_counts = _count_split(os.path.join(root, "val"))
    test_counts = _count_split(os.path.join(root, "test"))

    # 類別以 train 為準 (train/val/test 應一致)
    class_names = sorted(train_counts) or sorted(val_counts) or sorted(test_counts)
    if not class_names:
        raise ValueError(f"在 {root} 找不到 ImageFolder 類別子資料夾 (train/<class>/*)")

    n_train = sum(train_counts.values())
    n_val = sum(val_counts.values())
    n_test = sum(test_counts.values())

    nonzero = [v for v in train_counts.values() if v > 0] or [1]
    imbalance = max(nonzero) / min(nonzero)

    # 抽樣量測影像統計
    widths, heights, grays = [], [], 0
    sampled = _sample_image_paths(root)
    for p in sampled:
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
                if im.mode in ("L", "1"):
                    grays += 1
        except Exception:
            continue

    def _stat(v: list[int]) -> dict:
        if not v:
            return {}
        return {"min": min(v), "max": max(v), "median": int(statistics.median(v))}

    size_stats = {"width": _stat(widths), "height": _stat(heights),
                  "n_sampled": len(widths)}
    is_gray = bool(sampled) and grays == len(sampled)

    # 5-fold: 若 root 名稱含 _fold 或上層目錄符合 5_fold_* 慣例
    has_kfold = "_fold" in os.path.basename(root).lower() or \
        "5_fold" in root.lower()

    return DatasetProfile(
        root=root,
        task_type=task_type,
        num_classes=len(class_names),
        class_names=class_names,
        class_counts=train_counts,
        n_train=n_train, n_val=n_val, n_test=n_test,
        has_kfold=has_kfold,
        image_size_stats=size_stats,
        is_grayscale=is_gray,
        imbalance_ratio=round(imbalance, 3),
    )
