"""SkillAdvisor — 決策層委派給 Claude Code skill (設計文件 §5.2 / §12 P7).

需求原文:「將來我們會提供一個 skill 來做這個選擇的工作」「將來在 skill 要根據資料
來選不同的 encoder」。SkillAdvisor 與 LLMAdvisor / HeuristicAdvisor 介面完全相同,
故決策層可整組抽換 (只換 advisor.type=skill)。

委派機制 (file-based handoff):
Agent 主程式跑在 python subprocess, skill 跑在 Claude Code。兩者以 JSON 檔交握:
  1. SkillAdvisor 把「決策請求」(method + DatasetProfile + registry 快照) 寫到
     requests 目錄 (`<handoff_dir>/req_<n>.json`)。
  2. 由 Claude Code 執行 `finetune-advisor` skill 讀取請求、產生對應契約物件
     (EncoderChoice[] / Recipe / NextAction), 寫回 `<handoff_dir>/resp_<n>.json`。
  3. SkillAdvisor 讀回並驗證。

因『skill 由 Claude Code 觸發』無法在純 subprocess 內同步完成, 本實作提供:
  - 契約與交握協定 (寫請求 / 讀回應 / schema 驗證);
  - 若對應 resp 檔已存在則直接採用 (離線/預先產生模式);
  - 否則退回 `fallback` (預設 HeuristicAdvisor), 讓迴圈仍可推進。
未來把交握接上實際 skill 呼叫時, 只需替換 `_invoke_skill`。
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from .advisor import HeuristicAdvisor
from .privacy import egress as egress_mod
from .privacy import redact
from .privacy.context import PrivacyContext
from .privacy.facts import ErrorFacts
from .schemas import (
    DatasetProfile, EncoderChoice, EnsembleSpec, InfoRequest, NextAction,
    Recipe, SearchOverride, TrialResult,
)

_DEFAULT_HANDOFF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "_skill_handoff")


class SkillAdvisor:
    def __init__(self, handoff_dir: str = _DEFAULT_HANDOFF,
                 fallback: Optional[object] = None, wait_seconds: float = 0.0,
                 privacy: Optional[PrivacyContext] = None):
        self.handoff_dir = handoff_dir
        self.fallback = fallback or HeuristicAdvisor()
        self.wait_seconds = wait_seconds
        os.makedirs(self.handoff_dir, exist_ok=True)
        self._n = 0
        # 資料圍欄: 交握檔是**同等級的出口** (要交給 Claude Code 讀), 一樣要消毒 + 掃描
        self.privacy = privacy
        self.analyses: list = []
        self.analyzer_catalog: list[dict] = []

    # ---- 資料圍欄 ------------------------------------------------------
    def _ctx(self, profile: Optional[DatasetProfile] = None) -> PrivacyContext:
        if self.privacy is None:
            self.privacy = PrivacyContext.strict_for(profile)
        return self.privacy

    def _facts_payload(self, profile: Optional[DatasetProfile]) -> dict:
        """交握 payload 的資料段 — 與 LLMAdvisor 共用同一組消毒函式。"""
        ctx = self._ctx(profile)
        out = {"dataset_facts": (ctx.facts(profile).model_dump()
                                 if profile is not None else None),
               "user_facts": ctx.refresh_user_facts().model_dump()}
        if self.analyses:
            out["analyses"] = [a.model_dump() for a in self.analyses]
        return out

    # ---- 交握 ----------------------------------------------------------
    def _write_request(self, method: str, payload: dict,
                       profile: Optional[DatasetProfile] = None,
                       user_segments: tuple = ()) -> str:
        """寫出決策請求。寫檔前先過出口掃描 — 命中即中止, 不留下含資料的檔案。"""
        self._n += 1
        req = {"method": method, "skill": "finetune-advisor", "payload": payload}
        text = json.dumps(req, ensure_ascii=False, indent=2)
        egress_mod.guard_payload(self._ctx(profile).egress,
                                 f"skill:{method}", text,
                                 user_segments=user_segments,
                                 kind="skill_handoff")
        path = os.path.join(self.handoff_dir, f"req_{self._n:03d}_{method}.json")
        with open(path, "w", encoding="utf8") as f:
            f.write(text)
        return path

    def _resp_path(self, req_path: str) -> str:
        return req_path.replace("req_", "resp_", 1)

    def _invoke_skill(self, req_path: str) -> Optional[dict]:
        """讀取由 skill 產生的回應。未來可在此觸發實際 Claude Code skill 呼叫。

        目前: 若 resp 檔已存在 (預先產生 / 外部流程寫入) 則讀回; 可選擇性等待。
        """
        resp_path = self._resp_path(req_path)
        deadline = time.monotonic() + self.wait_seconds
        while True:
            if os.path.isfile(resp_path):
                try:
                    with open(resp_path, encoding="utf8") as f:
                        return json.load(f)
                except Exception:
                    return None
            if self.wait_seconds <= 0 or time.monotonic() >= deadline:
                return None
            # 短暫輪詢 (僅在明確設定 wait_seconds 時)
            _sleep(0.5)

    # ---- 決策方法 (介面同 HeuristicAdvisor/LLMAdvisor) ------------------
    def suggest_task_templates(self, profile: DatasetProfile) -> list:
        from . import task_template
        return task_template.suggest(profile)

    def plan_information(self, profile: DatasetProfile) -> InfoRequest:
        req = self._write_request("plan_information", {
            **self._facts_payload(profile),
            "analyzer_catalog": self.analyzer_catalog,
        }, profile)
        resp = self._invoke_skill(req)
        if resp and "info" in resp:
            try:
                return InfoRequest.model_validate(resp["info"])
            except Exception:
                pass
        return InfoRequest()

    def propose_ensemble(self, profile: DatasetProfile,
                         history: list[TrialResult], ensemble_cfg):
        """委派 skill 選 ensemble 成員; 無回應則退回 heuristic 規則式選擇。

        skill 回傳的成員為假名 trial_id, 以反查表還原成真實 id (白名單)。
        """
        alias = self._ctx(profile).alias
        done = [t for t in history if t.status == "done"
                and t.recipe.encoder.model_key != "ensemble"]
        req = self._write_request("propose_ensemble", {
            **self._facts_payload(profile),
            "history": [redact.trial_facts(t, alias).model_dump() for t in history],
            "ensemble_cfg": {"min_members": ensemble_cfg.min_members,
                             "max_members": ensemble_cfg.max_members,
                             "method": ensemble_cfg.method},
        }, profile)
        resp = self._invoke_skill(req)
        if resp and "ensemble" in resp:
            try:
                spec = EnsembleSpec.model_validate(resp["ensemble"])
                real = redact.resolve_trial_ids(
                    spec.member_trial_ids, done, alias)[: ensemble_cfg.max_members]
                if len(real) >= ensemble_cfg.min_members:
                    spec.member_trial_ids = real
                    return spec
            except Exception:
                pass
        return self.fallback.propose_ensemble(profile, history, ensemble_cfg)

    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]:
        from . import encoder_registry as reg
        req = self._write_request("select_encoders", {
            **self._facts_payload(profile),
            "available_encoders": [c.model_dump() for c in reg.available_cards()],
        }, profile)
        resp = self._invoke_skill(req)
        if resp and "encoders" in resp:
            try:
                avail = {c.model_key for c in reg.available_cards()}
                out = [EncoderChoice.model_validate(e) for e in resp["encoders"]]
                out = [c for c in out if c.model_key in avail]
                if out:
                    return out
            except Exception:
                pass
        return self.fallback.select_encoders(profile)

    def compose_recipe(self, profile: DatasetProfile, encoder: EncoderChoice,
                       preset: str = "default") -> Recipe:
        req = self._write_request("compose_recipe", {
            **self._facts_payload(profile),
            "encoder": encoder.model_dump(), "preset": preset,
        }, profile)
        resp = self._invoke_skill(req)
        if resp and "recipe" in resp:
            try:
                return Recipe.model_validate(resp["recipe"])
            except Exception:
                pass
        return self.fallback.compose_recipe(profile, encoder, preset)

    def propose_next(self, profile: DatasetProfile, encoder: EncoderChoice,
                     history: list[TrialResult],
                     base: Optional[TrialResult] = None) -> NextAction:
        alias = self._ctx(profile).alias
        req = self._write_request("propose_next", {
            **self._facts_payload(profile), "encoder": encoder.model_dump(),
            "history": [redact.trial_facts(t, alias).model_dump() for t in history],
            "base_trial_id": alias.substitute(base.trial_id) if base else None,
        }, profile)
        resp = self._invoke_skill(req)
        if resp and "next_action" in resp:
            try:
                return NextAction.model_validate(resp["next_action"])
            except Exception:
                pass
        return self.fallback.propose_next(profile, encoder, history, base=base)

    def propose_debug(self, profile: DatasetProfile, encoder: EncoderChoice,
                      trial: TrialResult, error_facts: ErrorFacts,
                      workspace_dir: Optional[str] = None,
                      log_tail: Optional[str] = None) -> Optional[Recipe]:
        alias = self._ctx(profile).alias
        payload = {
            **self._facts_payload(profile), "encoder": encoder.model_dump(),
            "trial": redact.trial_facts(trial, alias).model_dump(),
            "error_facts": (error_facts or ErrorFacts()).model_dump(),
            "workspace_dir": workspace_dir,
        }
        if log_tail:      # 只有 privacy.log_feedback="raw" (mode=off) 才會有
            payload["log_tail"] = log_tail
        req = self._write_request("propose_debug", payload, profile)
        resp = self._invoke_skill(req)
        if resp and "recipe" in resp:
            try:
                if resp["recipe"] is None:   # skill 明確表示放棄該分支
                    return None
                return Recipe.model_validate(resp["recipe"])
            except Exception:
                pass
        return self.fallback.propose_debug(profile, encoder, trial, error_facts,
                                           workspace_dir, log_tail)

    def review_and_decide(self, profile: DatasetProfile, encoder: EncoderChoice,
                          history: list[TrialResult], discussion: list[dict],
                          base: Optional[TrialResult] = None,
                          workspace_dir: Optional[str] = None) -> NextAction:
        alias = self._ctx(profile).alias
        req = self._write_request("review_and_decide", {
            **self._facts_payload(profile), "encoder": encoder.model_dump(),
            "history": [redact.trial_facts(t, alias).model_dump() for t in history],
            "discussion": redact.discussion_facts(discussion, alias),
            "base_trial_id": alias.substitute(base.trial_id) if base else None,
            "workspace_dir": workspace_dir,
        }, profile, tuple(redact.user_segments(discussion, alias)))
        resp = self._invoke_skill(req)
        if resp and "next_action" in resp:
            try:
                return NextAction.model_validate(resp["next_action"])
            except Exception:
                pass
        return self.fallback.review_and_decide(profile, encoder, history,
                                               discussion, base=base,
                                               workspace_dir=workspace_dir)

    def review_search_choice(self, profile: DatasetProfile, tree: list[dict],
                             proposal: dict,
                             discussion: list[dict]) -> SearchOverride:
        alias = self._ctx(profile).alias
        req = self._write_request("review_search_choice", {
            **self._facts_payload(profile),
            "tree": redact.scrub(tree, alias),
            "proposal": redact.scrub(proposal, alias),
            "discussion": redact.discussion_facts(discussion, alias),
        }, profile, tuple(redact.user_segments(discussion, alias)))
        resp = self._invoke_skill(req)
        if resp and "search_override" in resp:
            try:
                return SearchOverride.model_validate(resp["search_override"])
            except Exception:
                pass
        return self.fallback.review_search_choice(profile, tree, proposal,
                                                  discussion)


def _sleep(s: float) -> None:  # 隔離以便測試時 monkeypatch
    time.sleep(s)
