"""決策討論頻道 — LLM ⇄ 使用者的共享對話 + 中斷控制 (人機協作迴圈).

以檔案交握, 讓背景執行的 LoopController (worker thread) 與 web UI 共用:
  - conversation.jsonl : 逐則對話 (role=llm/user/system), append-only。
  - control.json       : 使用者中斷旗標 {stop, reason}。

LoopController 每一輪前讀取所有對話 (含使用者新訊息) 交給 Advisor 決策, 並把 LLM 的
檢視說明與決定 append 回對話; 使用者可隨時 append 訊息或設定 stop。
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


def _conv_path(run_dir: str) -> str:
    return os.path.join(run_dir, "conversation.jsonl")


def _ctrl_path(run_dir: str) -> str:
    return os.path.join(run_dir, "control.json")


def append(run_dir: str, role: str, text: str, kind: str = "msg",
           **extra) -> dict:
    """新增一則對話。role ∈ llm/user/system; kind ∈ review/decision/user_msg/status/final。"""
    os.makedirs(run_dir, exist_ok=True)
    entry = {"ts": time.time(), "role": role, "kind": kind, "text": text}
    entry.update(extra)
    with open(_conv_path(run_dir), "a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read(run_dir: str) -> list[dict]:
    p = _conv_path(run_dir)
    if not os.path.isfile(p):
        return []
    out = []
    with open(p, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def user_messages_after(run_dir: str, ts: float) -> list[dict]:
    """取得 ts 之後的使用者訊息 (供 Advisor 納入下一輪決策)。"""
    return [e for e in read(run_dir)
            if e.get("role") == "user" and e.get("ts", 0) > ts]


def set_stop(run_dir: str, reason: str = "使用者中斷") -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(_ctrl_path(run_dir), "w", encoding="utf8") as f:
        json.dump({"stop": True, "reason": reason, "ts": time.time()},
                  f, ensure_ascii=False)


def clear_stop(run_dir: str) -> None:
    """清除中斷旗標 (繼續實驗前呼叫, 否則迴圈第一輪就會再次停止)。"""
    try:
        os.remove(_ctrl_path(run_dir))
    except OSError:
        pass


def stop_requested(run_dir: str) -> tuple[bool, str]:
    p = _ctrl_path(run_dir)
    if not os.path.isfile(p):
        return False, ""
    try:
        with open(p, encoding="utf8") as f:
            d = json.load(f)
        return bool(d.get("stop")), d.get("reason", "")
    except Exception:
        return False, ""
