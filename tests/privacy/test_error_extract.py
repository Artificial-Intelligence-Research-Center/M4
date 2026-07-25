"""error_extract: 允許清單抽取 — 只吐已知樣板與數字, 不吐 log 原文。"""
from __future__ import annotations

import glob
import os
import re

from agent.analyzers import error_extract

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write(tmp: str, body: str) -> str:
    path = os.path.join(tmp, "log.txt")
    with open(path, "w", encoding="utf8") as f:
        f.write(body)
    return path


def test_known_templates(tmp_path=None):
    import tempfile
    tmp = tempfile.mkdtemp()
    cases = [
        ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate "
         "2.00 GiB. 512.00 MiB is free", "oom"),
        ("RuntimeError: shape '[24, 1024]' is invalid for input of size 12288",
         "shape_mismatch"),
        ("RuntimeError: Loss is nan, stopping training", "nan_loss"),
        ("ValueError: num_samples should be a positive integer value, but got 0",
         "dataloader"),
        ("RuntimeError: CUDA error: device-side assert triggered", "cuda"),
        ("size mismatch for head.weight: copying a param with shape "
         "torch.Size([3, 1024])", "checkpoint"),
        ("something entirely unexpected happened", "unknown"),
    ]
    for body, expect in cases:
        f = _write(tmp, f"Namespace(data_path='/data/SECRET_ds')\n{body}\n")
        facts = error_extract.extract(f)
        assert facts.error_class == expect, f"{body[:40]} → {facts.error_class}"
        blob = facts.model_dump_json()
        assert "SECRET_ds" not in blob, "log 中的路徑漏進 ErrorFacts"
        assert "/data/" not in blob

    f = _write(tmp, "torch.cuda.OutOfMemoryError: CUDA out of memory. "
                    "Tried to allocate 2.00 GiB. 512.00 MiB is free")
    facts = error_extract.extract(f)
    assert facts.requested_mb == 2048.0 and facts.free_mb == 512.0


def test_never_emits_paths_on_real_logs():
    """對 repo 內既有的真實 log 跑一遍: message_template 不得含路徑或副檔名。"""
    logs = glob.glob(os.path.join(_REPO, "runs", "*", "logs", "*.txt"))[:20]
    if not logs:
        return                        # 沒有既有 log 就跳過
    for path in logs:
        facts = error_extract.extract(path)
        blob = facts.model_dump_json()
        assert "/" not in facts.message_template, path
        assert not re.search(r"\.(jpg|jpeg|png|py|pth)\b", blob, re.I), path


def test_missing_log_is_safe():
    facts = error_extract.extract("/nonexistent/log.txt")
    assert facts.error_class == "unknown" and facts.n_lines_scanned == 0
