"""CodeWorkspace — LLM 修改訓練程式的隔離工作區 (參考 aideml 的 workspace 概念).

原則: **絕不修改原始程式**。允許 LLM 修改程式時 (advisor.allow_code_edit=True),
把訓練原始碼 (根目錄 *.py + util/) 複製到 `<run_dir>/src/<tag>/`, 在副本上套用
LLM 提出的 exact find/replace 編輯, 該 trial 改跑副本的 main_finetune.py。
每個修改版本各自一個目錄, 落地保存以便重現與審查 (edits.json)。
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 可被 LLM 編輯/複製的訓練原始碼 (白名單; agent/ 本身不可改)
_COPY_DIRS = ["util"]


def _copy_sources(dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(_ROOT):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(_ROOT, f), os.path.join(dst, f))
    for d in _COPY_DIRS:
        src = os.path.join(_ROOT, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, d), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))


def _safe_rel(path: str) -> Optional[str]:
    """驗證編輯目標: 相對路徑、不可跳脫、只能是白名單內的 .py。"""
    p = os.path.normpath(path.strip().lstrip("/"))
    if p.startswith("..") or not p.endswith(".py"):
        return None
    top = p.split(os.sep)[0]
    if os.sep in p and top not in _COPY_DIRS:
        return None
    return p


def create(base_dir: str, tag: str, edits: list[dict],
           parent_code_dir: Optional[str] = None,
           allowed_files: Optional[tuple[str, ...]] = None) -> Optional[dict]:
    """建立修改版工作區並套用編輯。

    edits: [{"file": 相對路徑, "find": 原片段, "replace": 新片段}, ...]
    parent_code_dir: 若上一版已是修改版, 以它為基底 (編輯可累積)。
    allowed_files: 額外的檔案白名單 (相對路徑); 給定時, 只有清單內的檔案可被編輯
        (improve 階段限 main_finetune.py 用)。None = 沿用預設範圍 (根目錄 *.py + util/)。
    回傳 {"code_dir": ..., "applied": [...], "failed": [...]};
    全部編輯都套不上則清掉目錄回 None (視為除錯提案無效)。
    """
    code_dir = os.path.join(base_dir, tag)
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    if parent_code_dir and os.path.isdir(parent_code_dir):
        shutil.copytree(parent_code_dir, code_dir,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        _copy_sources(code_dir)

    applied, failed = [], []
    for e in edits or []:
        rel = _safe_rel(str(e.get("file", "")))
        find, replace = e.get("find", ""), e.get("replace", "")
        if not rel or not find:
            failed.append({**e, "why": "非法路徑或空 find"})
            continue
        if allowed_files is not None and rel not in allowed_files:
            failed.append({**e, "why": f"{rel} 不在允許清單 {list(allowed_files)}"})
            continue
        path = os.path.join(code_dir, rel)
        if not os.path.isfile(path):
            failed.append({**e, "why": f"{rel} 不存在"})
            continue
        with open(path, encoding="utf8") as f:
            src = f.read()
        if find not in src:
            failed.append({**e, "why": "find 片段不存在於檔案中"})
            continue
        with open(path, "w", encoding="utf8") as f:
            f.write(src.replace(find, replace, 1))
        applied.append({"file": rel, "find": find, "replace": replace})

    if not applied:
        shutil.rmtree(code_dir, ignore_errors=True)
        return None
    with open(os.path.join(code_dir, "edits.json"), "w", encoding="utf8") as f:
        json.dump({"applied": applied, "failed": failed}, f,
                  ensure_ascii=False, indent=2)
    return {"code_dir": code_dir, "applied": applied, "failed": failed}
