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
from .schemas import (
    DatasetProfile, EncoderChoice, NextAction, Recipe, TrialResult,
)

_DEFAULT_HANDOFF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "_skill_handoff")


class SkillAdvisor:
    def __init__(self, handoff_dir: str = _DEFAULT_HANDOFF,
                 fallback: Optional[object] = None, wait_seconds: float = 0.0):
        self.handoff_dir = handoff_dir
        self.fallback = fallback or HeuristicAdvisor()
        self.wait_seconds = wait_seconds
        os.makedirs(self.handoff_dir, exist_ok=True)
        self._n = 0

    # ---- 交握 ----------------------------------------------------------
    def _write_request(self, method: str, payload: dict) -> str:
        self._n += 1
        req = {"method": method, "skill": "finetune-advisor", "payload": payload}
        path = os.path.join(self.handoff_dir, f"req_{self._n:03d}_{method}.json")
        with open(path, "w", encoding="utf8") as f:
            json.dump(req, f, ensure_ascii=False, indent=2)
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

    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]:
        from . import encoder_registry as reg
        req = self._write_request("select_encoders", {
            "profile": profile.model_dump(),
            "available_encoders": [c.model_dump() for c in reg.available_cards()],
        })
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
            "profile": profile.model_dump(),
            "encoder": encoder.model_dump(), "preset": preset,
        })
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
        req = self._write_request("propose_next", {
            "profile": profile.model_dump(), "encoder": encoder.model_dump(),
            "history": [t.model_dump() for t in history],
            "base": base.model_dump() if base else None,
        })
        resp = self._invoke_skill(req)
        if resp and "next_action" in resp:
            try:
                return NextAction.model_validate(resp["next_action"])
            except Exception:
                pass
        return self.fallback.propose_next(profile, encoder, history, base=base)

    def propose_debug(self, profile: DatasetProfile, encoder: EncoderChoice,
                      trial: TrialResult, log_tail: str,
                      workspace_dir: Optional[str] = None) -> Optional[Recipe]:
        req = self._write_request("propose_debug", {
            "profile": profile.model_dump(), "encoder": encoder.model_dump(),
            "trial": trial.model_dump(), "log_tail": log_tail,
            "workspace_dir": workspace_dir,
        })
        resp = self._invoke_skill(req)
        if resp and "recipe" in resp:
            try:
                if resp["recipe"] is None:   # skill 明確表示放棄該分支
                    return None
                return Recipe.model_validate(resp["recipe"])
            except Exception:
                pass
        return self.fallback.propose_debug(profile, encoder, trial, log_tail,
                                           workspace_dir)

    def review_and_decide(self, profile: DatasetProfile, encoder: EncoderChoice,
                          history: list[TrialResult], discussion: list[dict],
                          base: Optional[TrialResult] = None,
                          workspace_dir: Optional[str] = None) -> NextAction:
        req = self._write_request("review_and_decide", {
            "profile": profile.model_dump(), "encoder": encoder.model_dump(),
            "history": [t.model_dump() for t in history],
            "discussion": discussion,
            "base": base.model_dump() if base else None,
            "workspace_dir": workspace_dir,
        })
        resp = self._invoke_skill(req)
        if resp and "next_action" in resp:
            try:
                return NextAction.model_validate(resp["next_action"])
            except Exception:
                pass
        return self.fallback.review_and_decide(profile, encoder, history,
                                               discussion, base=base,
                                               workspace_dir=workspace_dir)


def _sleep(s: float) -> None:  # 隔離以便測試時 monkeypatch
    time.sleep(s)
