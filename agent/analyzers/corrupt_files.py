"""可讀性檢查 — 抽樣開圖, 回報「有幾張讀不開」而不是「哪幾張」。"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel

from ._util import sample_paths

_N = 300


class CorruptFilesOut(BaseModel):
    n_sampled: int = 0
    n_unreadable: int = 0          # PIL 無法開啟/解碼
    n_truncated: int = 0           # 可開啟但 verify() 失敗
    n_zero_bytes: int = 0
    unreadable_ratio: float = 0.0
    smallest_file_bytes: Optional[int] = None
    median_file_kb: Optional[float] = None
    blocking: bool = False         # 比例高到會讓訓練失敗


def run(root: str) -> CorruptFilesOut:
    from PIL import Image

    out = CorruptFilesOut()
    sizes: list[int] = []
    for _split, _cls, path in sample_paths(root, _N):
        out.n_sampled += 1
        try:
            size = os.path.getsize(path)
            sizes.append(size)
            if size == 0:
                out.n_zero_bytes += 1
                out.n_unreadable += 1
                continue
        except OSError:
            out.n_unreadable += 1
            continue
        try:
            with Image.open(path) as im:
                im.verify()           # 只驗結構, 不解碼全圖
        except Exception:             # noqa: BLE001
            out.n_truncated += 1
            out.n_unreadable += 1
            continue
        try:
            with Image.open(path) as im:
                im.convert("RGB").resize((32, 32))   # 真的能解碼嗎
        except Exception:             # noqa: BLE001
            out.n_unreadable += 1

    if out.n_sampled:
        out.unreadable_ratio = round(out.n_unreadable / out.n_sampled, 4)
        out.blocking = out.unreadable_ratio >= 0.02
    if sizes:
        sizes.sort()
        out.smallest_file_bytes = sizes[0]
        out.median_file_kb = round(sizes[len(sizes) // 2] / 1024.0, 1)
    return out
