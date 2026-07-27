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

_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
_MARK = re.compile(r"^<!--\s*key:\s*([\w.\-]+)\s*-->\s*$", re.M)
_cache: dict = {}                       # {path: (mtime, {key: html})}


def _render(md_text: str) -> str:
    md_text = md_text.strip()
    if _md is not None:
        return _md.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return "<pre>" + html.escape(md_text) + "</pre>"


def _load(path: str) -> dict:
    """讀某個 `<!-- key: X -->` 分節的 markdown → {key: html}；依 mtime 快取。"""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    ent = _cache.get(path)
    if ent and ent[0] == mt:
        return ent[1]
    try:
        with open(path, encoding="utf8") as f:
            raw = f.read()
    except OSError:
        return {}
    # re.split(捕獲組) → [preamble, key1, body1, key2, body2, ...]
    parts = _MARK.split(raw)
    docs: dict = {}
    it = iter(parts[1:])
    for key, body in zip(it, it):
        docs[key.strip()] = _render(body)
    _cache[path] = (mt, docs)
    return docs


def render(md_text: str) -> str:
    """把任意 markdown 文字渲染成 HTML (供前端顯示 report.md 等)。"""
    return _render(md_text or "")


def load() -> dict:
    """工作台「新完整流程」表單各選項的說明 (docs/run_options.md)。"""
    return _load(os.path.join(_DOCS_DIR, "run_options.md"))


def load_settings() -> dict:
    """整體設定頁各欄位的說明 (docs/settings_help.md)。"""
    return _load(os.path.join(_DOCS_DIR, "settings_help.md"))
