"""PrivacyContext — 把假名表、出口設定、使用者提供的事實綁成一個物件, 交給決策層。

決策層 (LLMAdvisor / SkillAdvisor) 不自己決定隱私政策, 只透過本物件取得
「可以送出去的東西」。未設定時一律以 strict 建構 (fail-closed)。

本模組刻意**不** import `agent.config` — PrivacyConfig 以 duck typing 傳入, 避免
`agent/__init__` → privacy → config 的循環相依。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from . import alias as alias_mod
from .alias import AliasMap
from .egress import EgressContext
from .facts import DatasetFacts, UserFacts
from .redact import Guard, to_facts

USER_FACTS_FILENAME = "user_facts.json"


# ---------------------------------------------------------------------------
# 使用者提供的事實 (管道 B 的落地)
# ---------------------------------------------------------------------------
def load_user_facts(run_dir: Optional[str]) -> UserFacts:
    if not run_dir:
        return UserFacts()
    path = os.path.join(run_dir, USER_FACTS_FILENAME)
    try:
        with open(path, encoding="utf8") as f:
            return UserFacts.model_validate_json(f.read())
    except Exception:
        return UserFacts()


def save_user_facts(run_dir: str, uf: UserFacts) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, USER_FACTS_FILENAME), "w",
              encoding="utf8") as f:
        f.write(uf.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
@dataclass
class PrivacyContext:
    alias: AliasMap
    egress: EgressContext
    user_facts: UserFacts = field(default_factory=UserFacts)
    mode: str = "strict"
    log_feedback: str = "structured"          # none | structured | raw
    allow_free_text_questions: bool = False
    expensive_analyzers_need_consent: bool = True
    run_dir: Optional[str] = None
    n_folds: Optional[int] = None

    # ---- 建立 ------------------------------------------------------
    @classmethod
    def build(cls, priv: Any, profile, *, run_dir: Optional[str] = None,
              data_root: Optional[str] = None, exempt_text: str = "",
              guidance: str = "", n_folds: Optional[int] = None,
              ) -> "PrivacyContext":
        """由 PrivacyConfig + DatasetProfile 建立。priv 為 duck-typed 設定物件。"""
        mode = _attr(priv, "mode", "strict")
        salt = alias_mod.load_salt(_attr(priv, "salt_file", None))
        names = list(getattr(profile, "class_names", []) or [])
        policy = _attr(priv, "class_names", "hashed")
        if policy == "plain":
            revealed = list(names)
        elif policy == "user_approved":
            revealed = [n for n in _attr(priv, "revealed_classes", []) or []
                        if n in names]
        else:
            revealed = []

        root = getattr(profile, "root", "") or ""
        am = None
        if run_dir:
            am = AliasMap.load(run_dir)
            # 政策或資料集換了就重建 (避免沿用舊的授權狀態)
            if am is not None and (am.dataset_root != os.path.realpath(root)
                                   or sorted(am.revealed) != sorted(revealed)):
                am = None
        if am is None:
            am = AliasMap.build(root, names, salt, revealed=revealed)
            if run_dir:
                am.save(run_dir)

        guard = Guard.build(
            data_root=data_root or _default_data_root(),
            dataset_root=root, class_names=names, revealed=revealed,
            exempt_text=exempt_text,
            scan_filenames=_attr(priv, "scan_filenames", True),
        )
        ectx = EgressContext(
            guard=guard, run_dir=run_dir, mode=mode,
            user_input_policy=_attr(priv, "guard_on_user_input", "warn"),
            audit=bool(_attr(priv, "egress_audit", True)),
            extra_user_segments=[guidance] if guidance else [],
        )
        return cls(
            alias=am, egress=ectx, user_facts=load_user_facts(run_dir),
            mode=mode,
            log_feedback=_attr(priv, "log_feedback", "structured"),
            allow_free_text_questions=bool(
                _attr(priv, "allow_free_text_questions", False)),
            expensive_analyzers_need_consent=bool(
                _attr(priv, "expensive_analyzers_need_consent", True)),
            run_dir=run_dir, n_folds=n_folds,
        )

    @classmethod
    def strict_for(cls, profile, run_dir: Optional[str] = None,
                   exempt_text: str = "") -> "PrivacyContext":
        """未取得設定時的 fail-closed 退路 — 一律用最嚴格的政策。"""
        return cls.build(_StrictDefaults(), profile, run_dir=run_dir,
                         exempt_text=exempt_text)

    # ---- 使用 ------------------------------------------------------
    def facts(self, profile) -> DatasetFacts:
        return to_facts(profile, self.alias, self.user_facts, self.n_folds)

    def refresh_user_facts(self) -> UserFacts:
        self.user_facts = load_user_facts(self.run_dir)
        return self.user_facts

    def note_user_text(self, text: str) -> None:
        """登記一段「使用者親自輸入」的文字 (guidance / 問卷答案)。"""
        if text and text not in self.egress.extra_user_segments:
            self.egress.extra_user_segments.append(self.alias.substitute(text))


class _StrictDefaults:
    mode = "strict"
    class_names = "hashed"
    log_feedback = "structured"
    allow_free_text_questions = False
    egress_audit = True
    guard_on_user_input = "warn"
    salt_file = None
    revealed_classes: list = []
    expensive_analyzers_need_consent = True
    scan_filenames = True


def _attr(obj: Any, name: str, default):
    v = getattr(obj, name, None)
    return default if v is None else v


def _default_data_root() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data")
