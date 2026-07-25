"""出口管制 — **所有**送往 Claude API 的 payload 的唯一出口
(docs/data_firewall_design.md §7)。

任何其他模組都不得直接呼叫 `client.messages.create` / `.stream`;
`sentinel.install()` 會在執行期檢查呼叫堆疊, 非本檔發出的呼叫一律拋 EgressViolation。

每次呼叫依序:
  1. guard  — 對 system + prompt 全文做精確比對, 命中即中止 (fail-closed);
  2. audit  — 落地 `runs/<run>/privacy_audit.jsonl` (含 verdict / 違規 / 全文);
  3. 送出;
  4. 對回應再掃一次 (防模型把不該回的東西複述回來而落檔)。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Iterable, Optional

from .errors import EgressViolation
from .redact import Guard, Violation

AUDIT_FILENAME = "privacy_audit.jsonl"

_AUDIT_LOCK = threading.Lock()


@dataclass
class EgressContext:
    """一次 run 的出口設定。由 LoopController / qa_agent 建立並掛到 advisor 上。"""
    guard: Guard
    run_dir: Optional[str] = None
    mode: str = "strict"                  # strict | standard | off
    user_input_policy: str = "warn"       # warn | block | off
    audit: bool = True
    # 本次呼叫之外的額外「使用者親自輸入」片段 (guidance 等)
    extra_user_segments: list[str] = field(default_factory=list)

    @property
    def audit_path(self) -> Optional[str]:
        return (os.path.join(self.run_dir, AUDIT_FILENAME)
                if self.run_dir and self.audit else None)


# ---------------------------------------------------------------------------
# 掃描與稽核
# ---------------------------------------------------------------------------
def check(ctx: EgressContext, text: str,
          user_segments: Iterable[str] = ()) -> tuple[str, list[Violation]]:
    """回傳 (verdict, violations)。verdict ∈ ok / warn / blocked。

    命中若完全落在「使用者親自輸入」的片段內, 依 user_input_policy 降級為警示 —
    使用者有權自願揭露, 但一律留下稽核紀錄。
    """
    vios = ctx.guard.scan(text or "")
    if not vios:
        return "ok", []
    segs = [s for s in list(user_segments) + list(ctx.extra_user_segments) if s]

    def _from_user(v: Violation) -> bool:
        matched = (text or "")[v.start:v.end]
        return bool(matched) and any(matched in s for s in segs)

    if ctx.mode == "off":
        return "warn", vios

    hard = [v for v in vios if not _from_user(v)]
    soft = [v for v in vios if _from_user(v)]
    if ctx.user_input_policy == "block":
        hard, soft = vios, []
    elif ctx.user_input_policy == "off":
        soft = []
    if hard:
        return "blocked", hard + soft
    return ("warn", soft) if soft else ("ok", [])


def audit(ctx: EgressContext, record: dict) -> None:
    path = ctx.audit_path
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _AUDIT_LOCK, open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _enforce(ctx: EgressContext, label: str, text: str,
             user_segments: Iterable[str], kind: str, extra: dict) -> str:
    verdict, vios = check(ctx, text, user_segments)
    rec = {
        "ts": time.time(), "label": label, "kind": kind, "verdict": verdict,
        "mode": ctx.mode, "n_chars": len(text or ""),
        "sha256": sha256((text or "").encode("utf8")).hexdigest(),
        "violations": [v.as_dict() for v in vios],
    }
    rec.update(extra)
    audit(ctx, rec)
    if verdict == "blocked":
        rules = sorted({v.rule for v in vios})
        raise EgressViolation(
            f"資料圍欄攔截了一次外送（{label}）：payload 含疑似資料識別內容 "
            f"（規則: {', '.join(rules)}）。呼叫已中止, 未送出任何資料。"
            f"詳見 {AUDIT_FILENAME}。", [v.as_dict() for v in vios], label)
    return verdict


def guard_payload(ctx: EgressContext, label: str, text: str,
                  user_segments: Iterable[str] = (), kind: str = "payload") -> str:
    """非 API 的出口 (例: SkillAdvisor 寫給 Claude Code 的交握 JSON)。"""
    return _enforce(ctx, label, text, user_segments, kind, {})


# ---------------------------------------------------------------------------
# API 呼叫 (唯一出口)
# ---------------------------------------------------------------------------
def _payload_text(system, messages) -> str:
    parts = []
    if isinstance(system, str):
        parts.append(system)
    elif system:
        parts += [b.get("text", "") for b in system if isinstance(b, dict)]
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts += [b.get("text", "") for b in c if isinstance(b, dict)]
    return "\n\n".join(p for p in parts if p)


def create(client, *, ctx: EgressContext, label: str, model: str,
           system, messages, user_segments: Iterable[str] = (), **kw):
    """經圍欄送出一次 messages.create。"""
    text = _payload_text(system, messages)
    verdict = _enforce(ctx, label, text, user_segments, "request",
                       {"model": model, "payload": text})
    resp = client.messages.create(model=model, system=system,
                                  messages=messages, **kw)
    _audit_response(ctx, label, model, verdict, resp)
    return resp


def stream_text(client, *, ctx: EgressContext, label: str, model: str,
                system, messages, on_delta: Optional[Callable[[str], None]] = None,
                user_segments: Iterable[str] = (), **kw):
    """經圍欄送出一次串流請求; 回傳 (完整文字, 最終 message)。

    整個 `with` 區塊留在本檔內, 讓 sentinel 的堆疊檢查對串流同樣成立。
    """
    text = _payload_text(system, messages)
    verdict = _enforce(ctx, label, text, user_segments, "request",
                       {"model": model, "payload": text})
    parts: list[str] = []
    with client.messages.stream(model=model, system=system,
                                messages=messages, **kw) as stream:
        for t in stream.text_stream:
            parts.append(t)
            if on_delta:
                try:
                    on_delta("".join(parts))
                except Exception:
                    pass
        resp = stream.get_final_message()
    out = "".join(parts)
    _audit_response(ctx, label, model, verdict, resp, out)
    return out, resp


def _audit_response(ctx: EgressContext, label: str, model: str,
                    req_verdict: str, resp, text: Optional[str] = None) -> None:
    """回應也掃一次 — 主要是防模型複述資料後落檔到 runs/。"""
    if text is None:
        try:
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text")
        except Exception:
            text = ""
    verdict, vios = check(ctx, text or "")
    audit(ctx, {
        "ts": time.time(), "label": label, "kind": "response", "model": model,
        "request_verdict": req_verdict, "verdict": verdict,
        "n_chars": len(text or ""),
        "violations": [v.as_dict() for v in vios],
        "stop_reason": getattr(resp, "stop_reason", None),
    })


def read_audit(run_dir: str) -> list[dict]:
    """讀回稽核紀錄 (供 web 的隱私分頁)。"""
    path = os.path.join(run_dir, AUDIT_FILENAME)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
