"""從訓練 log 抽出結構化失敗事實 (docs/data_firewall_design.md §8.1)。

**這是允許清單抽取, 不是黑名單過濾。** 只有比對到下列已知樣板的內容才會產生對應的
`message_template`; 其餘一律歸為 `unknown` 並丟棄。原始 log 含 `Namespace(...
data_path='/…/5_fold_XXX/…')` 與 traceback 中的影像路徑, 永遠不進 prompt。

正則只從 log 中抽「數字」與「樣板編號」— 不抽任何自由文字片段, 所以不可能夾帶。
"""
from __future__ import annotations

import os
import re
from typing import Optional

from ..privacy.facts import ErrorFacts

# 已知例外型別 (允許清單; 不在清單內一律不回報型別名)
_KNOWN_EXC = {
    "RuntimeError", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "FileNotFoundError", "OSError", "MemoryError",
    "AssertionError", "ZeroDivisionError", "NotImplementedError",
    "OutOfMemoryError", "CUDAOutOfMemoryError", "ImportError",
    "ModuleNotFoundError", "PermissionError", "StopIteration",
}

# (error_class, message_template, pattern) — template 是**固定字串**, 不含 log 內容
_TEMPLATES: list[tuple[str, str, re.Pattern]] = [
    ("oom", "CUDA out of memory",
     re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)),
    ("oom", "host memory exhausted (OOM killer)",
     re.compile(r"DefaultCPUAllocator: can't allocate|Killed\b.*python|"
                r"RuntimeError: \[enforce fail.*posix_memalign", re.I)),
    ("shape_mismatch", "tensor shape invalid for input size",
     re.compile(r"shape '\[[^\]]*\]' is invalid for input of size")),
    ("shape_mismatch", "batch size of input and target differ",
     re.compile(r"Expected input batch_size \(\d+\) to match target batch_size")),
    ("shape_mismatch", "matrix dimensions do not match",
     re.compile(r"mat1 and mat2 shapes cannot be multiplied|"
                r"size mismatch, m\d+:")),
    ("checkpoint", "checkpoint parameter shape does not match model",
     re.compile(r"size mismatch for [\w.]+: copying a param with shape")),
    ("checkpoint", "checkpoint file missing or unreadable",
     re.compile(r"(?:Error\(s\) in loading state_dict|"
                r"No such file or directory:.*\.pth)")),
    ("nan_loss", "loss became NaN or Inf and training stopped",
     re.compile(r"Loss is (?:nan|inf), stopping training|"
                r"loss is nan|Loss is NaN", re.I)),
    ("dataloader", "dataloader worker died",
     re.compile(r"DataLoader worker \(pid \d+\) is killed|"
                r"DataLoader worker \(pid\(s\) \d+\) exited unexpectedly")),
    ("dataloader", "empty dataset; num_samples must be positive",
     re.compile(r"num_samples should be a positive integer|"
                r"Found 0 files in subfolders")),
    ("dataloader", "image file missing or unreadable during loading",
     re.compile(r"(?:FileNotFoundError|UnidentifiedImageError|"
                r"cannot identify image file|image file is truncated)")),
    ("cuda", "CUDA device-side assert or kernel failure",
     re.compile(r"device-side assert triggered|CUDA error:|"
                r"no kernel image is available")),
    ("cuda", "no CUDA device available",
     re.compile(r"CUDA driver version is insufficient|"
                r"Torch not compiled with CUDA|no CUDA-capable device")),
    ("config", "invalid command line arguments",
     re.compile(r"error: unrecognized arguments|error: argument |"
                r"the following arguments are required")),
]

_SHAPE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ALLOC_RE = re.compile(r"Tried to allocate ([\d.]+)\s*([KMG])iB", re.I)
_FREE_RE = re.compile(r"([\d.]+)\s*([KMG])iB is free", re.I)
_EPOCH_RE = re.compile(r"^\s*Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/", re.M)
_EXC_RE = re.compile(r"^\s*(?:\w+\.)*(\w+(?:Error|Exception|Interrupt))\s*:",
                     re.M)

_UNIT = {"k": 1 / 1024.0, "m": 1.0, "g": 1024.0}


def _mb(value: str, unit: str) -> float:
    return round(float(value) * _UNIT[unit.lower()], 1)


def extract(log_path: Optional[str], n_lines: int = 200,
            epoch_curve: Optional[dict] = None) -> ErrorFacts:
    """讀 log 尾端 → ErrorFacts。讀不到 log 也會回傳一個合法物件 (error_class=unknown)。"""
    facts = ErrorFacts()
    text = ""
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, errors="ignore") as f:
                lines = f.readlines()
            tail = lines[-n_lines:]
            facts.n_lines_scanned = len(tail)
            text = "".join(tail)
        except OSError:
            pass

    cv = epoch_curve or {}
    tl, vl = list(cv.get("train_loss") or []), list(cv.get("val_loss") or [])
    facts.n_epochs_completed = max(len(tl), len(vl))
    facts.last_train_loss = float(tl[-1]) if tl else None
    facts.last_val_loss = float(vl[-1]) if vl else None
    if not text:
        return facts

    for error_class, template, pat in _TEMPLATES:
        if pat.search(text):
            facts.error_class = error_class          # type: ignore[assignment]
            facts.message_template = template
            break

    if m := _EXC_RE.search(text):
        name = m.group(1)
        if name in _KNOWN_EXC:
            facts.exc_type = name

    if facts.error_class == "oom":
        if m := _ALLOC_RE.search(text):
            facts.requested_mb = _mb(m.group(1), m.group(2))
        if m := _FREE_RE.search(text):
            facts.free_mb = _mb(m.group(1), m.group(2))
    if facts.error_class in ("shape_mismatch", "checkpoint"):
        for m in _SHAPE_RE.finditer(text):
            shape = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
            if 1 <= len(shape) <= 6 and shape not in facts.tensor_shapes:
                facts.tensor_shapes.append(shape)
            if len(facts.tensor_shapes) >= 4:
                break

    epochs = _EPOCH_RE.findall(text)
    if epochs:
        facts.at_epoch = int(epochs[-1][0])
        facts.at_step = int(epochs[-1][1])

    # 有內容但沒比對到任何樣板 → 明說「有東西被丟棄」, 讓決策層知道自己資訊不全
    facts.truncated = facts.error_class == "unknown"
    return facts
