"""選項說明文件載入 — 供工作台「新完整流程」表單右側的 markdown 說明面板。

單一來源 `docs/run_options.md`，以 `<!-- key: X -->` 標記分節；每節渲染成 HTML，
回傳 `{key: html}`。key 命名慣例 `field.value`（如 `advisor.llm`）或 `field`
（欄位層級，如 `modality`）；前端解析時 `field.value` 找不到會退回 `field`，
再退回 `overview`。**編輯 markdown 檔即可改善說明，重新整理頁面就生效**（依 mtime 快取）。
"""
from __future__ import annotations

import html
import os
import re

try:
    import markdown as _md
except Exception:                       # pragma: no cover - markdown 應已安裝
    _md = None

_DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs",
                    "run_options.md")
_MARK = re.compile(r"^<!--\s*key:\s*([\w.\-]+)\s*-->\s*$", re.M)
_cache: dict = {"mtime": None, "docs": {}}


def _render(md_text: str) -> str:
    md_text = md_text.strip()
    if _md is not None:
        return _md.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return "<pre>" + html.escape(md_text) + "</pre>"


def load() -> dict:
    """回傳 {section_key: html}；檔案缺失回空 dict。依 mtime 快取，改檔即時生效。"""
    try:
        mt = os.path.getmtime(_DOC)
    except OSError:
        return {}
    if _cache["mtime"] == mt:
        return _cache["docs"]
    try:
        with open(_DOC, encoding="utf8") as f:
            raw = f.read()
    except OSError:
        return {}
    # re.split(捕獲組) → [preamble, key1, body1, key2, body2, ...]
    parts = _MARK.split(raw)
    docs: dict = {}
    it = iter(parts[1:])
    for key, body in zip(it, it):
        docs[key.strip()] = _render(body)
    _cache["mtime"] = mt
    _cache["docs"] = docs
    return docs
