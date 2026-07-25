"""分析器共用的取樣工具 — 全部跑在**資料平面**, 可以碰檔案與像素。

回傳給決策層的一律只有聚合數值 (見各分析器的 output_schema); 檔名與路徑不外流。
"""
from __future__ import annotations

import os
from typing import Iterator

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")


def classes_of(split_dir: str) -> list[str]:
    if not os.path.isdir(split_dir):
        return []
    return sorted(d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d)))


def class_counts(split_dir: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for cls in classes_of(split_dir):
        cdir = os.path.join(split_dir, cls)
        try:
            out[cls] = sum(1 for f in os.listdir(cdir)
                           if f.lower().endswith(IMG_EXT))
        except OSError:
            out[cls] = 0
    return out


def iter_images(root: str, split: str, per_class: int | None = None
                ) -> Iterator[tuple[str, str]]:
    """逐一產出 (class_name, abs_path)。per_class=None 表示不限量。"""
    sdir = os.path.join(root, split)
    for cls in classes_of(sdir):
        cdir = os.path.join(sdir, cls)
        try:
            names = sorted(f for f in os.listdir(cdir)
                           if f.lower().endswith(IMG_EXT))
        except OSError:
            continue
        for name in (names[:per_class] if per_class else names):
            yield cls, os.path.join(cdir, name)


def sample_paths(root: str, total: int = 200,
                 splits: tuple[str, ...] = SPLITS) -> list[tuple[str, str, str]]:
    """跨 split × class 均勻抽樣, 回傳 [(split, class, path)]。取檔名排序後的前幾張
    (不隨機), 讓同一資料集的分析結果可重現。"""
    cells: list[tuple[str, str]] = []
    for s in splits:
        for c in classes_of(os.path.join(root, s)):
            cells.append((s, c))
    if not cells:
        return []
    per = max(1, total // len(cells))
    out: list[tuple[str, str, str]] = []
    for s, c in cells:
        cdir = os.path.join(root, s, c)
        try:
            names = sorted(f for f in os.listdir(cdir)
                           if f.lower().endswith(IMG_EXT))
        except OSError:
            continue
        out += [(s, c, os.path.join(cdir, n)) for n in names[:per]]
    return out[:total]


def pct(values: list[float], q: float) -> float | None:
    """百分位 (不引入 numpy 依賴; 值少時退化為最近鄰)。"""
    if not values:
        return None
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]
