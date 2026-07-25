"""Split 洩漏檢查 — train/val/test 之間有沒有**完全相同**的影像。

只回報「有幾組重複」, 永遠不回報是哪些檔案。要看是哪些檔案請在本機 UI 看。
"""
from __future__ import annotations

import os
from hashlib import md5

from pydantic import BaseModel

from ._util import SPLITS, iter_images

_MAX_PER_SPLIT = 8000          # 每個 split 最多雜湊這麼多張 (護欄)
_CHUNK = 1 << 20


def _hash_file(path: str) -> str | None:
    h = md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


class SplitLeakageOut(BaseModel):
    n_hashed_train: int = 0
    n_hashed_val: int = 0
    n_hashed_test: int = 0
    complete: bool = True             # 是否掃完全部 (False = 有做上限截斷)
    n_dup_train_val: int = 0
    n_dup_train_test: int = 0
    n_dup_val_test: int = 0
    n_dup_within_train: int = 0
    leak_ratio_test: float = 0.0      # test 中與 train 重複的比例
    has_leakage: bool = False
    # 重複樣本是否跨到不同類別 (標註不一致的訊號)
    n_dup_cross_class: int = 0


def run(root: str) -> SplitLeakageOut:
    out = SplitLeakageOut()
    # {split: {digest: [class, ...]}}
    tables: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS:
        table: dict[str, list[str]] = {}
        n = 0
        for cls, path in iter_images(root, split):
            if n >= _MAX_PER_SPLIT:
                out.complete = False
                break
            d = _hash_file(path)
            n += 1
            if d:
                table.setdefault(d, []).append(cls)
        tables[split] = table
        setattr(out, f"n_hashed_{split}", n)

    tr, va, te = tables["train"], tables["val"], tables["test"]
    tv, tt = set(tr) & set(va), set(tr) & set(te)
    out.n_dup_train_val = len(tv)
    out.n_dup_train_test = len(tt)
    out.n_dup_val_test = len(set(va) & set(te))
    out.n_dup_within_train = sum(len(v) - 1 for v in tr.values() if len(v) > 1)
    if out.n_hashed_test:
        out.leak_ratio_test = round(out.n_dup_train_test / out.n_hashed_test, 4)
    out.has_leakage = bool(out.n_dup_train_val or out.n_dup_train_test
                           or out.n_dup_val_test)

    cross = 0
    for d in tv | tt:
        labels = set(tr.get(d, [])) | set(va.get(d, [])) | set(te.get(d, []))
        if len(labels) > 1:
            cross += 1
    out.n_dup_cross_class = cross
    return out
