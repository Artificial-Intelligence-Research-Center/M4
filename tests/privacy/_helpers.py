"""測試輔助 — canary 資料集與假的 Anthropic client。

canary 資料集刻意把「絕對不能外流的字串」放進目錄名、類別名與檔名; 之後只要在
送出的 payload 中找到任何一個, 就是圍欄破功。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

# 這些字串一個都不准出現在任何送往 API 的 payload 裡
CANARY_COLLECTION = "CANARYHOSP9527"
CANARY_DATASET = "PTNSET7788_seed42_fold0"
CANARY_CLASSES = ["PATIENTWANG_dr3", "LINMEIHUA_normal"]
CANARY_FILE_STEM = "CANARYTOKEN0001"

CANARY_TOKENS = [
    CANARY_COLLECTION, CANARY_DATASET, CANARY_CLASSES[0], CANARY_CLASSES[1],
    CANARY_FILE_STEM, "PATIENTWANG", "LINMEIHUA", "PTNSET7788",
]


def make_canary_dataset(base: str | None = None) -> str:
    """建一個最小但合法的 ImageFolder canary 資料集, 回傳資料集根目錄。"""
    from PIL import Image

    base = base or tempfile.mkdtemp(prefix="medclaw_canary_")
    root = os.path.join(base, CANARY_COLLECTION, CANARY_DATASET)
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        for cls in CANARY_CLASSES:
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                im = Image.new("RGB", (256, 256), (i * 20 % 256, 90, 140))
                im.save(os.path.join(d, f"{CANARY_FILE_STEM}_{split}_{i}.jpg"))
    return root


def cleanup(root: str) -> None:
    # root = <base>/<collection>/<dataset>; 連 base 一起刪
    shutil.rmtree(os.path.dirname(os.path.dirname(root)), ignore_errors=True)


# ---------------------------------------------------------------------------
# 假的 Anthropic client — 記錄所有 payload, 不連網。
# 注意: sentinel 只包裝真正的 anthropic SDK 類別, 所以這個 stub 可以正常被呼叫;
# 圍欄的掃描與稽核仍然照跑 (它在 egress 裡, 早於 client 呼叫)。
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text: str):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"


class _StreamCM:
    def __init__(self, text: str):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        yield self._text

    def get_final_message(self):
        return _Resp(self._text)


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kw):
        self._owner.record(kw)
        return _Resp(self._owner.reply)

    def stream(self, **kw):
        self._owner.record(kw)
        return _StreamCM(self._owner.reply)


class FakeClient:
    """記錄每次呼叫的完整 system+prompt。`payloads` 供測試斷言。"""

    def __init__(self, reply: str = "{}"):
        self.reply = reply
        self.calls: list[dict] = []
        self.messages = _Messages(self)

    def with_options(self, **kw):
        return self

    def record(self, kw: dict) -> None:
        self.calls.append(kw)

    @property
    def payloads(self) -> list[str]:
        out = []
        for c in self.calls:
            parts = []
            sysblk = c.get("system")
            if isinstance(sysblk, str):
                parts.append(sysblk)
            elif sysblk:
                parts += [b.get("text", "") for b in sysblk if isinstance(b, dict)]
            for m in c.get("messages") or []:
                content = m.get("content")
                if isinstance(content, str):
                    parts.append(content)
            out.append("\n".join(parts))
        return out

    def all_text(self) -> str:
        return "\n".join(self.payloads)


# ---------------------------------------------------------------------------
def find_tokens(text: str, tokens=CANARY_TOKENS) -> list[str]:
    low = text.lower()
    return [t for t in tokens if t.lower() in low]
