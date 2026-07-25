"""執行期哨兵 — 堵住繞過資料圍欄的 API 呼叫 (docs/data_firewall_design.md §7.3)。

「請大家都走 egress」不構成圍欄。本模組在 anthropic SDK 的 Messages 方法外面包一層,
檢查呼叫堆疊中是否有 `privacy/egress.py` 的 frame; 沒有就拋 EgressViolation。

如此一來, 未來任何人 (包含 LLM 自己修改程式時) 新增一條 prompt 建構路徑, 都會在第一次
執行時就失敗, 而不是靜默洩漏。
"""
from __future__ import annotations

import functools
import importlib.util
import os
import sys

from .errors import EgressViolation

_HERE = os.path.dirname(os.path.abspath(__file__))
# egress.py 的各種可能寫法 (原路徑 / realpath), 供逐 frame 比對
_EGRESS_FILES = {
    os.path.join(_HERE, "egress.py"),
    os.path.realpath(os.path.join(_HERE, "egress.py")),
}

_installed = False
_METHODS = ("create", "stream", "parse", "count_tokens")


def _called_from_egress() -> bool:
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_filename in _EGRESS_FILES:
            return True
        f = f.f_back
    return False


def _wrap(orig, what: str):
    @functools.wraps(orig)
    def guarded(*args, **kwargs):
        if not _called_from_egress():
            raise EgressViolation(
                f"偵測到未經資料圍欄的 Claude API 呼叫（{what}）。"
                f"所有呼叫必須經由 agent.privacy.egress — 這是資料圍欄的唯一出口，"
                f"直接呼叫 SDK 會繞過消毒與稽核。", label=what)
        return orig(*args, **kwargs)

    guarded._medclaw_guarded = True       # type: ignore[attr-defined]
    return guarded


def _patch(cls) -> int:
    n = 0
    for name in _METHODS:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_medclaw_guarded", False):
            continue
        setattr(cls, name, _wrap(orig, f"{cls.__name__}.{name}"))
        n += 1
    return n


def install() -> bool:
    """安裝哨兵。anthropic 未安裝時靜默略過 (那就沒有 API 可呼叫)。"""
    global _installed
    if _installed:
        return True
    if importlib.util.find_spec("anthropic") is None:
        return False
    targets = []
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
        targets += [Messages, AsyncMessages]
    except Exception:
        try:
            from anthropic.resources.messages import Messages
            targets.append(Messages)
        except Exception:
            return False
    try:    # beta 命名空間 (若存在) 同樣要堵
        from anthropic.resources.beta.messages import messages as _beta
        targets += [getattr(_beta, n) for n in ("Messages", "AsyncMessages")
                    if hasattr(_beta, n)]
    except Exception:
        pass

    for cls in targets:
        try:
            _patch(cls)
        except Exception:
            pass
    _installed = True
    return True


def is_installed() -> bool:
    return _installed
