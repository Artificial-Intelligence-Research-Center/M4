"""LLMAdvisor — 以 Claude API 實作決策層 (設計文件 §5.2, P2).

依 `claude-api` 參考:
- 官方 `anthropic` SDK; 每個決策方法綁一個 Pydantic schema, 輸出即驗證。
  (本環境 structured-outputs 的 grammar 首次編譯常逾時且每次要等 ~60s, 故改以
   JSON 模式: 請模型直接輸出符合 schema 的 JSON, 再用 Pydantic 驗證/重試 — 見 _messages_parse。)
- 模型 `claude-opus-4-8`; `thinking={"type": "adaptive"}`。
- 穩定內容 (encoder registry 目錄、TaskTemplate、決策規則) 放前面加 `cache_control`;
  volatile (本次 facts、history) 放後面, 以命中 prompt cache。
- LLM 只吐「決策」(選哪些 encoder、用哪些元件、超參), 實體 Recipe 由本模組 (純程式)
  依 registry 白名單建構, 避免 LLM 直接產碼 / 產出不存在的元件。

**資料圍欄 (docs/data_firewall_design.md)**:
- 本模組屬於**決策平面**, 不得觸碰資料集檔案, 也不得直接呼叫 anthropic SDK。
  所有 API 呼叫經 `privacy.egress`; `privacy.sentinel` 會在執行期擋下繞道。
- 進 prompt 的資料集描述一律是 `DatasetFacts` (假名化、無路徑、無真實類別名),
  **不是** `DatasetProfile`。失敗診斷用 `ErrorFacts`, 不是原始 log。
- LLM 需要更多資料特性時只有兩條路: 點名執行已註冊的分析器 (`request_analysis`),
  或提出問題請使用者親自回答 (`questions`)。

環境需求: `pip install anthropic` + 設定 ANTHROPIC_API_KEY (或 `ant auth login`)。
若不可用, `build_advisor(type="llm")` 會在呼叫時拋出清楚錯誤; 可用 fallback=HeuristicAdvisor()
讓迴圈退回規則式決策。
"""
from __future__ import annotations

import json
import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import encoder_registry as reg
from . import presets
from .privacy import egress as egress_mod
from .privacy import redact
from .privacy.context import PrivacyContext
from .privacy.facts import AnalysisFacts, ErrorFacts, UserFacts
from .privacy.facts import UserQuestion as _UQ
from .schemas import (
    ComponentRef, DatasetProfile, EncoderChoice, HeadSpec, HyperParams,
    InfoRequest, NextAction, Recipe, SearchOverride, TrialResult,
)

_MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# LLM 決策的回應 schema (扁平、可驗證; 之後由純程式映射成 Recipe / EncoderChoice)
# ---------------------------------------------------------------------------
class _UserQuestionOut(BaseModel):
    """LLM 想問使用者的一個問題 (管道 B)。"""
    key: str
    question: str
    kind: Literal["choice", "multi", "number", "bool", "short_text"] = "choice"
    options: list[str] = Field(default_factory=list)
    why: str = ""
    blocking: bool = False


class _InfoMixin(BaseModel):
    """所有決策共用: 需要更多資料特性時的兩條合法管道。"""
    # 要執行的分析器 key (只接受「可用資源目錄 > analyzers」列出的); 不需要就留空
    request_analysis: list[str] = Field(default_factory=list)
    # 要請使用者親自回答的問題; 不需要就留空
    questions: list[_UserQuestionOut] = Field(default_factory=list)


class _EncoderPick(BaseModel):
    model_key: str
    adaptation: Literal["finetune", "lp"] = "finetune"
    rationale: str = ""


class _EncoderSelection(_InfoMixin):
    """select_encoders 的回應。"""
    encoders: list[_EncoderPick] = Field(default_factory=list)


class _InfoPlan(_InfoMixin):
    """plan_information 的回應 — 實驗開跑前的資訊蒐集規劃。"""
    reason: str = ""


class _RecipeDecision(_InfoMixin):
    """compose_recipe 的回應 — 只吐元件選擇與超參, 不吐 Recipe 全物件。"""
    preset: Literal["default", "paper", "mae"] = "default"
    head_type: Literal["linear", "mlp"] = "linear"
    head_hidden_dims: list[int] = Field(default_factory=list)
    head_dropout: float = 0.0
    pooling: Literal["global_pool", "cls_token"] = "global_pool"
    loss_name: Literal["cross_entropy", "weighted_ce", "focal"] = "cross_entropy"
    regularizers: list[str] = Field(default_factory=list)  # registry keys
    augmentation: str = "timm_randaug"
    # 超參覆寫 (相對 preset); None = 沿用 preset 起始值 (使用者可在「超參起點」頁編輯)。
    # 只有在有明確理由要調整時才填, 否則留 null 沿用起始值。
    blr: Optional[float] = None
    layer_decay: Optional[float] = None
    drop_path: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    # 若調整了任何上述超參數, 必須在此逐項說明原因; 沒有原因的調整不會被套用。
    hparam_reason: str = ""
    rationale: str = ""


class _CodeEdit(BaseModel):
    """一個對訓練程式的 exact find/replace 編輯 (套用在 run 目錄的副本上)。

    ⚠ 資料圍欄: 允許 LLM 改程式 + 把 log 回饋給 LLM = 完整的資料讀取通道。
    strict 模式強制 allow_code_edit=False (見 config._apply_privacy_mode);
    其他模式下原始 log 也不會回饋 (只回 ErrorFacts)。
    """
    file: str        # 相對 repo 根的 .py, 例: "main_finetune.py", "util/datasets.py"
    find: str        # 必須逐字存在於檔案中的原片段
    replace: str     # 取代後的新片段


class _NextDecision(_InfoMixin):
    """propose_next 的回應 — 變異動作 + 對應元件/超參變更。"""
    stop: bool = False
    reason: str = ""
    mutation: Literal[
        "add_regularizer", "swap_head", "change_augmentation",
        "add_auxiliary_task", "adjust_hparams", "edit_code", "none",
    ] = "none"
    # 變異內容 (依 mutation 使用其一)
    add_regularizer: Optional[str] = None
    new_head_type: Optional[Literal["linear", "mlp"]] = None
    new_augmentation: Optional[str] = None
    # 超參覆寫以 JSON 字串表示 (避免開放 dict 讓 structured-output grammar 過重),
    # 例: '{"blr": 0.001, "drop_path": 0.3}'
    hparam_overrides_json: str = ""
    # mutation="edit_code" 時使用 (improve 階段限 main_finetune.py; 見 review 的提示)
    code_edits: list[_CodeEdit] = Field(default_factory=list)


class _ReviewDecision(_NextDecision):
    """review_and_decide 的回應 — 在 _NextDecision 上加一段給使用者看的檢視說明。"""
    narrative: str = ""


class _SearchChoice(BaseModel):
    """review_search_choice 的回應 — 是否覆寫樹搜尋 policy 選出的節點。"""
    override: bool = False
    stage: Literal["draft", "improve", "debug", "resume"] = "improve"
    parent_id: Optional[str] = None       # 目標節點的 trial_id; stage=draft 時留空
    # 只在 stage=draft 時使用 (指定新起點的 encoder / adaptation / 超參起點)
    encoder: Optional[str] = None
    adaptation: Optional[Literal["finetune", "lp"]] = None
    preset: Optional[str] = None
    reason: str = ""


class _DebugDecision(_InfoMixin):
    """propose_debug 的回應 — 診斷失敗原因並修正 Recipe (aideml debug 階段)。"""
    give_up: bool = False          # 判定無從修起 (如資料/環境問題) → 放棄該分支
    diagnosis: str = ""            # 依 ErrorFacts 判斷的失敗原因
    # 修正內容 (只改必要的最小面向; 未填 = 不動)
    hparam_overrides_json: str = ""            # 例: '{"batch_size": 12, "accum_iter": 2}'
    loss_name: Optional[Literal["cross_entropy", "weighted_ce", "focal"]] = None
    remove_regularizer: Optional[str] = None
    new_augmentation: Optional[str] = None
    # 允許修改程式時 (allow_code_edit) 才會被套用; 修改版放 run 目錄, 原始程式不動
    code_edits: list[_CodeEdit] = Field(default_factory=list)


# ---------------------------------------------------------------------------
class LLMAdvisor:
    """呼叫 Claude API 做決策; 介面與 HeuristicAdvisor 相同, 可被 LoopController 使用。

    環境不對 (無 anthropic SDK / 無金鑰 / API 失敗) 時**直接拋錯**, 不會靜默退回 heuristic;
    錯誤會顯示於 CLI / web 背景工作與對話框。若要規則式決策請明確 advisor.type=heuristic。

    `privacy` (PrivacyContext) 由 LoopController / qa_agent 掛上; 未掛上時以 strict
    自動建構 (fail-closed) — 決策層在任何情況下都不會拿到未消毒的資料。
    """

    def __init__(self, model: str = _MODEL, guidance: str = "",
                 allow_code_edit: bool = False,
                 privacy: Optional[PrivacyContext] = None):
        self.model = model
        self.guidance = guidance.strip()
        self.allow_code_edit = allow_code_edit  # 允許 LLM 修改程式 (副本, 見 code_workspace)
        self._client = None
        self.log_path = None   # 設定後, 每次 LLM 呼叫的完整 prompt/回應會落地到此 (jsonl)
        self.privacy = privacy
        # 已執行的分析器結果 (由 LoopController 填入, 隨 prompt 一起帶給 LLM)
        self.analyses: list[AnalysisFacts] = []
        # 分析器目錄 (key/說明/成本), 供 LLM 知道有哪些程式可以點名執行
        self.analyzer_catalog: list[dict] = []
        # 本次決策附帶的資訊需求 (由呼叫端讀取後處理)
        self.last_info: InfoRequest = InfoRequest()

    def _log_call(self, record: dict) -> None:
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- 環境檢查: 在實驗開始前呼叫, 環境不對即以清楚訊息拋錯 -----------
    def check_environment(self) -> None:
        """驗證 SDK + 金鑰 + 連線。環境不對即以清楚訊息拋 RuntimeError。"""
        client = self._get_client()  # 無 anthropic 時已拋清楚錯誤
        try:
            # 輕量請求驗證金鑰與連線 (快速失敗, 不重試); models.list 不帶任何 payload
            client.with_options(timeout=20.0, max_retries=0).models.list(limit=1)
        except Exception as e:
            raise RuntimeError(
                "advisor=llm 無法連上 Claude API：請確認已設定 ANTHROPIC_API_KEY "
                "（或已 `ant auth login`）且網路可連線。"
                f"（{type(e).__name__}: {e}）。或改用 advisor.type=heuristic。") from e

    # ---- SDK client (延後建立) ----------------------------------------
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "advisor=llm 需要 anthropic SDK，但未安裝。請執行 "
                    "`pip install anthropic`，或改用 advisor.type=heuristic。") from e
            try:
                self._client = anthropic.Anthropic()  # 自環境/ant profile 解析金鑰
            except Exception as e:
                raise RuntimeError(
                    "advisor=llm 無法建立 Claude client（可能缺少 ANTHROPIC_API_KEY，"
                    f"或尚未 `ant auth login`）：{e}。或改用 advisor.type=heuristic。") from e
        return self._client

    # ---- 資料圍欄 ------------------------------------------------------
    def _ctx(self, profile: Optional[DatasetProfile] = None) -> PrivacyContext:
        """取得 PrivacyContext; 未掛上時以 strict 建構 (fail-closed)。"""
        if self.privacy is None:
            self.privacy = PrivacyContext.strict_for(
                profile, exempt_text=self._registry_context())
        if self.guidance:
            self.privacy.note_user_text(self.guidance)
        return self.privacy

    def _facts_block(self, profile: Optional[DatasetProfile]) -> str:
        """prompt 的資料段: DatasetFacts + 分析器結果 + 使用者提供的事實。

        這是決策層唯一能看到的「資料」— 全部是本地程式產生的統計量或使用者自己填的。
        """
        ctx = self._ctx(profile)
        parts = []
        if profile is not None:
            parts.append(
                "【資料集事實 DatasetFacts】(由本地程式分析後消毒; 你看不到也拿不到原始影像、"
                "檔名或路徑。class_labels 是假名, 真實名稱不會提供)\n"
                + ctx.facts(profile).model_dump_json(indent=2))
        else:
            parts.append("【資料集事實】（尚無分析結果）")
        if self.analyses:
            parts.append(
                "【追加分析結果】(你先前點名、由本地已註冊分析器執行後的輸出)\n"
                + json.dumps([a.model_dump() for a in self.analyses],
                             ensure_ascii=False, indent=2))
        uf = ctx.refresh_user_facts()
        for a in uf.answers:
            ctx.note_user_text(a.value)   # 使用者自願提供 → 出口掃描只警示不中止
        if _has_user_facts(uf):
            parts.append(
                "【使用者親自提供的事實】(使用者在 UI 上填寫; 可信度高於任何推測)\n"
                + uf.model_dump_json(indent=2))
        return "\n\n".join(parts)

    def _info_note(self, ctx: PrivacyContext) -> str:
        """告訴 LLM 兩條合法的補充資訊管道 (以及目前有哪些分析器可用)。"""
        cat = ""
        if self.analyzer_catalog:
            cat = ("\n可點名的分析器 (request_analysis 只接受這些 key):\n"
                   + json.dumps(self.analyzer_catalog, ensure_ascii=False, indent=2))
        q = ("在 questions 提出問題請**使用者親自回答** (例: 影像模態、拍攝部位、"
             "類別是否有序、標註品質)。")
        if not ctx.allow_free_text_questions:
            q += "（本模式只接受 kind=choice/multi/number/bool，不接受 short_text。）"
        return (
            "\n\n【需要更多資料特性時】你看不到原始資料, 也不要猜。只有兩條路: "
            "(1) 在 request_analysis 點名要執行的分析器, 結果會在下一輪提供給你; "
            f"(2) {q}"
            "兩者都不需要時一律留空 — 不要為了保險而每輪都要求。" + cat)

    _RULES = (
        "你是 MedClaw 的決策層。依資料集特性從『可用資源目錄』(白名單) 中選擇, "
        "不得使用目錄外的 encoder 或元件。鬆散原則: 小資料/高不平衡→傾向 lp 或較小 lr、"
        "較強 regularization / weighted 或 focal loss; 醫療影像→優先醫療 DAP encoder (若可取得); "
        "多 encoder 時求來源多樣 (自然 vs 醫療、MAE vs Dino) 以利比較; 改良動作要有限、正交、"
        "可驗證, 一次只變異一個面向。**調整任何超參數 (blr/layer_decay/drop_path/weight_decay/"
        "epochs/batch_size) 時, 必須明確說明原因; 沒有充分理由就沿用起始值, 不要直接改動。**"
        "只輸出結構化決策, 不要產生訓練程式碼。\n\n"
        "【資料圍欄】使用者的原始資料 (影像、檔名、路徑、真實類別名) **絕對不會**提供給你, "
        "這是刻意的設計。資料集以假名 dataset_ref 表示, 類別以 C0..Cn 表示。"
        "不要要求、猜測或推論這些識別資訊, 也不要在輸出中提及路徑或檔名。"
        "需要知道資料特性時, 只能用 request_analysis (執行已註冊的本地分析程式) 或 "
        "questions (請使用者親自回答) 這兩條管道。"
    )

    _RETRYABLE = ("overloaded", "rate_limit", "429", "500", "502", "503", "529",
                  "timeout", "timed out", "connection")

    def _registry_context(self) -> str:
        """穩定的可用資源目錄 (放前面, 加 cache_control)。"""
        cards = [c.model_dump() for c in reg.all_cards(include_unavailable=True)]
        return json.dumps({
            "encoders": cards,
            "component_registry": {
                "head": ["linear", "mlp"],
                "pooling": ["global_pool", "cls_token"],
                "regularizer": ["drop_path", "weight_decay", "layer_decay",
                                "label_smoothing", "mixup", "cutmix"],
                "augmentation": ["timm_randaug", "resize_crop_normalize"],
                "loss": ["cross_entropy", "weighted_ce", "focal"],
            },
            "hparam_presets": presets.as_dict(),
            "task_template": "fundus_classification",
            "analyzers": self.analyzer_catalog,
        }, ensure_ascii=False, indent=2)

    # ---- 唯一出口 ------------------------------------------------------
    def _messages_parse(self, volatile_text: str, schema, retries: int = 3,
                        label: str = "", profile: Optional[DatasetProfile] = None,
                        user_segments: tuple = ()):
        """以 JSON 模式取得結構化決策 (刻意不用 messages.parse 的 grammar,
        因本環境 grammar 首次編譯常逾時且每次要等 ~60s; JSON 模式快且穩)。

        所有呼叫經 `privacy.egress` — payload 先過出口掃描, 命中即中止並落地稽核。
        """
        ctx = self._ctx(profile)
        client = self._get_client().with_options(timeout=240.0, max_retries=1)
        if self.guidance:
            volatile_text = (f"【使用者引導方向 (請優先納入考量)】\n{self.guidance}\n\n"
                             + volatile_text)
        system = [
            {"type": "text", "text": self._RULES},
            {"type": "text", "text": "可用資源目錄:\n" + self._registry_context(),
             "cache_control": {"type": "ephemeral"}},  # 穩定→快取
        ]
        system_text = "\n\n".join(b["text"] for b in system)
        sch = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        base = (volatile_text + "\n\n請只輸出一個符合下列 JSON schema 的 JSON 物件, "
                "不要任何多餘文字、不要 markdown code fence：\n" + sch)
        prompt, last = base, None
        for attempt in range(retries):
            rec = {"ts": time.time(), "label": label, "attempt": attempt,
                   "model": self.model, "system": system_text, "prompt": prompt}
            try:
                # 注意: adaptive thinking 的思考 token 也計入 max_tokens。4096 曾多次
                # 被思考吃光導致正文空白或 JSON 截斷在字串中間 (pydantic EOF error),
                # 故給足額度; display=summarized 讓思考摘要可落地到 log 以利除錯。
                resp = egress_mod.create(
                    client, ctx=ctx.egress, label=label,
                    model=self.model, system=system,
                    messages=[{"role": "user", "content": prompt}],
                    user_segments=user_segments,
                    max_tokens=16000,
                    thinking={"type": "adaptive", "display": "summarized"})
                text = "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", None) == "text").strip()
                think = "".join(getattr(b, "thinking", "") for b in resp.content
                                if getattr(b, "type", None) == "thinking")
                rec["response"] = text
                rec["stop_reason"] = resp.stop_reason
                if think:
                    rec["thinking"] = think
                if resp.stop_reason == "max_tokens":
                    raise RuntimeError(
                        f"輸出達到 max_tokens 被截斷 (正文 {len(text)} chars), "
                        f"請精簡 reason/narrative 後重試")
                raw = text
                if "{" in raw and "}" in raw:  # 取最外層 JSON, 去掉 fence/多餘文字
                    raw = raw[raw.find("{"): raw.rfind("}") + 1]
                out = schema.model_validate_json(raw)
                rec["ok"] = True
                self._log_call(rec)
                self._capture_info(out)
                return out
            except Exception as e:
                last = e
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"
                self._log_call(rec)
                from .privacy.errors import EgressViolation
                if isinstance(e, EgressViolation):
                    raise            # 圍欄攔截: 不重試, 直接往上拋
                if attempt >= retries - 1:
                    raise
                msg = str(e).lower()
                if any(k in msg for k in self._RETRYABLE):
                    time.sleep(2 * (attempt + 1))          # 暫時性錯誤: 退避重試
                else:
                    prompt = base + f"\n\n(上次輸出無法解析為合法 JSON: {e}. 請只輸出合法 JSON。)"
        raise last  # pragma: no cover

    def _capture_info(self, decision) -> None:
        """把決策附帶的資訊需求收下來, 供呼叫端 (LoopController) 處理。"""
        analyses = list(getattr(decision, "request_analysis", []) or [])
        qs = []
        allow_text = self._ctx().allow_free_text_questions
        for q in getattr(decision, "questions", []) or []:
            kind = q.kind
            if kind == "short_text" and not allow_text:
                kind = "choice" if q.options else "bool"
            qs.append(_UQ(key=q.key, question=q.question, kind=kind,
                          options=q.options, why=q.why, blocking=q.blocking))
        self.last_info = InfoRequest(analyses=analyses, questions=qs,
                                     reason=getattr(decision, "reason", "") or "")

    # ---- 下游任務起點 (資料驅動, 不需 LLM) ----------------------------
    def suggest_task_templates(self, profile: DatasetProfile) -> list:
        from . import task_template
        return task_template.suggest(profile)

    # ---- (0) 實驗開跑前的資訊蒐集規劃 ---------------------------------
    def plan_information(self, profile: DatasetProfile) -> InfoRequest:
        """看過 DatasetFacts 後, 決定還需要哪些分析 / 要問使用者什麼。

        這是「LLM 想知道資料特性」的唯一入口 — 它不能自己去看資料。
        """
        ctx = self._ctx(profile)
        d: _InfoPlan = self._messages_parse(
            self._facts_block(profile)
            + "\n\n實驗即將開始。在挑選 encoder 與組 Recipe 之前, 請判斷你還缺哪些"
              "會**實質改變決策**的資料特性。\n"
              "- 能由本地程式算出來的 → 放進 request_analysis;\n"
              "- 只有使用者知道的 (影像模態、拍攝部位、類別是否有序、標註可信度、"
              "有無臨床上不可接受的錯誤型態) → 放進 questions;\n"
              "- 都不缺就兩個都留空 (這是完全合理的答案)。\n"
              "問題請精簡, 一次最多 3 題, 且要能用選項回答。"
            + self._info_note(ctx),
            _InfoPlan, label="plan_information", profile=profile)
        info = self.last_info
        info.reason = d.reason or info.reason
        return info

    # ---- (1) 多 encoder ------------------------------------------------
    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]:
        n = 3
        ctx = self._ctx(profile)
        out: _EncoderSelection = self._messages_parse(
            self._facts_block(profile)
            + f"\n\n請從可用 encoder 中選最多 {n} 個作為比較 (依資料特性排序, 最合適在前)。"
            + self._info_note(ctx),
            _EncoderSelection, label="select_encoders", profile=profile)
        avail = {c.model_key for c in reg.available_cards()}
        choices = [
            EncoderChoice(model_key=p.model_key, adaptation=p.adaptation,
                          rationale=p.rationale)
            for p in out.encoders if p.model_key in avail
        ]
        if not choices:
            raise RuntimeError("LLM 未回傳任何可用的 encoder（不在白名單內）。")
        return choices

    # ---- (2)(3) 組 Recipe ---------------------------------------------
    def compose_recipe(self, profile: DatasetProfile, encoder: EncoderChoice,
                       preset: str = "default") -> Recipe:
        start_hp = presets.get(preset).model_dump()
        ctx = self._ctx(profile)
        d: _RecipeDecision = self._messages_parse(
            self._facts_block(profile)
            + f"\n\n已選 encoder: {encoder.model_key} (adaptation={encoder.adaptation})。\n"
              f"起始超參數 (preset={preset}, 由使用者設定, 你可沿用或調整):\n"
              f"{json.dumps(start_hp, ensure_ascii=False)}\n\n"
              f"請為此 encoder 組出訓練 Recipe (選 head/pooling/loss/regularizer/augmentation)。"
              f"超參欄位 (blr/layer_decay/drop_path/weight_decay/epochs/batch_size)："
              f"**只有在有明確理由要調整時才填該欄位；否則留 null 以沿用上面的起始值。"
              f"任何超參數的改變都必須在 hparam_reason 逐項說明原因 (為何調整、依據什麼)；"
              f"不要無理由直接改動——沒有原因的調整將被忽略、沿用起始值。**"
            + self._info_note(ctx),
            _RecipeDecision, label="compose_recipe", profile=profile)
        return self._build_recipe(profile, encoder, d)

    def _build_recipe(self, profile: DatasetProfile, encoder: EncoderChoice,
                      d: _RecipeDecision) -> Recipe:
        hp = presets.get(d.preset)  # 起始超參數 (使用者可在「超參起點」頁編輯)
        reason = (d.hparam_reason or "").strip()
        changed = {}
        for f in ("blr", "layer_decay", "drop_path", "weight_decay", "epochs", "batch_size"):
            v = getattr(d, f)
            if v is None or v == getattr(hp, f):
                continue
            if not reason:
                # 沒有原因的超參調整一律不套用 (要求: 任何改變都要附原因)
                continue
            changed[f] = {"from": getattr(hp, f), "to": v}
            setattr(hp, f, v)
        prov = {"template": "fundus_classification", "preset": d.preset,
                "advisor": "llm", "rationale": d.rationale}
        if changed:
            prov["hparam_changes"] = changed
            prov["hparam_reason"] = reason
        return Recipe(
            encoder=encoder,
            task={"type": profile.task_type},
            heads=[HeadSpec(type=d.head_type, output_dim=profile.num_classes,
                            hidden_dims=d.head_hidden_dims, dropout=d.head_dropout)],
            pooling=d.pooling,
            regularizers=[ComponentRef(name=r) for r in d.regularizers],
            augmentation=ComponentRef(name=d.augmentation),
            losses=[ComponentRef(name=d.loss_name)],
            hparams=hp,
            provenance=prov,
        )

    def _hist(self, history: list[TrialResult]) -> list[dict]:
        """把歷史整理給 LLM: 一律轉成 TrialFacts (trial_id 假名化、provenance 白名單)。

        含目前 epochs 與逐 epoch train/val loss 曲線 (供判斷 epochs)。
        """
        alias = self._ctx().alias
        return [redact.trial_facts(t, alias).model_dump() for t in history]

    @staticmethod
    def _pick_base(history: list[TrialResult],
                   base: Optional[TrialResult]) -> Optional[TrialResult]:
        """變異基準: policy 選出的 base 優先; 無則退回歷史最佳。"""
        if base is not None and base.status == "done":
            return base
        return max((t for t in history if t.status == "done"),
                   key=lambda t: t.primary_score,
                   default=history[-1] if history else None)

    def _base_id(self, trial: Optional[TrialResult]) -> str:
        if trial is None:
            return "無"
        return self._ctx().alias.substitute(trial.trial_id or "")

    # ---- (5) 改良 / 停止 ----------------------------------------------
    def propose_next(self, profile: DatasetProfile, encoder: EncoderChoice,
                     history: list[TrialResult],
                     base: Optional[TrialResult] = None) -> NextAction:
        hist = self._hist(history)
        bt = self._pick_base(history, base)
        ctx = self._ctx(profile)
        d: _NextDecision = self._messages_parse(
            self._facts_block(profile)
            + f"\n\nencoder: {encoder.model_key}。已完成 trials (由舊到新):\n"
              f"{json.dumps(hist, ensure_ascii=False, indent=2)}\n\n"
              f"本輪的變異基準 (樹搜尋 policy 依指標機率選出): {self._base_id(bt)}。\n"
              f"請決定: 對此基準 Recipe 做『一個』正交變異再試, 或停止。"
            + self._info_note(ctx),
            _NextDecision, label="propose_next", profile=profile)
        next_recipe = None
        if not d.stop and bt is not None:
            next_recipe = self._apply_mutation(bt.recipe, d)
            if next_recipe is not None:
                next_recipe.provenance["reason"] = d.reason
                next_recipe.provenance["advisor"] = "llm"
        return NextAction(stop=d.stop or next_recipe is None, reason=d.reason,
                          mutation=d.mutation, next_recipe=next_recipe,
                          info=self.last_info)

    # ---- 樹搜尋節點選擇的覆寫機會 (policy 選完 → 決策層過目) ----------
    def review_search_choice(self, profile: DatasetProfile, tree: list[dict],
                             proposal: dict,
                             discussion: list[dict]) -> SearchOverride:
        """policy 已選好節點, 這裡讓 LLM 有改選的機會。

        傳入整棵解答樹 (每個節點附「目前允許哪些階段」) 與這次的選擇;
        LLM 可維持原議或改選。回傳的選擇由 LoopController 再驗證一次。
        """
        ctx = self._ctx(profile)
        alias = ctx.alias
        talk = redact.discussion_facts(discussion, alias)
        segs = tuple(redact.user_segments(discussion, alias))
        tree = redact.scrub(tree, alias)
        proposal = redact.scrub(proposal, alias)
        try:
            d: _SearchChoice = self._messages_parse(
                self._facts_block(profile)
                + f"\n\n目前的解答樹 (每個節點的 allowed 欄位 = 這個節點現在允許的階段):\n"
                  f"{json.dumps(tree, ensure_ascii=False, indent=2)}\n\n"
                  f"樹搜尋 policy 這一輪的選擇:\n"
                  f"{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
                  f"與使用者的討論記錄 (由舊到新):\n"
                  f"{json.dumps(talk, ensure_ascii=False, indent=2)}\n\n"
                  f"policy 是規則 + 依指標抽樣, 看不懂討論內容。請你過目這個選擇:\n"
                  f"- 維持原議 → override=false（多數情況；policy 的抽樣本身有探索價值，"
                  f"不要只因為『分數不是最高』就改選）。\n"
                  f"- 改選 → override=true, 並給 stage 與 parent_id。可用的 stage 只有\n"
                  f"  該節點 allowed 裡列出的; stage=\"draft\" 表示開一條全新起點 "
                  f"(parent_id 留空, 可用 encoder/adaptation/preset 指定新起點的配置)。\n"
                  f"**特別注意討論中標有【QA 結論】的訊息**: 那是問答 agent 依實驗資料"
                  f"對使用者提問得出的結論, 若其中要求換 encoder / 開新 draft / 回頭改良"
                  f"某個特定節點, 這裡就是唯一能落實的地方 (改良階段只能在已選定的節點上"
                  f"做變異, 換不了節點也開不了 draft)。\n"
                  f"reason 請用一句話說明維持或改選的依據 (會顯示給使用者)。",
                _SearchChoice, label="review_search_choice", profile=profile,
                user_segments=segs)
        except Exception:
            return SearchOverride(override=False)   # 失敗不阻斷迴圈, 沿用 policy
        return SearchOverride(
            override=d.override, stage=d.stage, parent_id=d.parent_id,
            encoder=d.encoder, adaptation=d.adaptation, preset=d.preset,
            reason=d.reason,
        )

    # ---- 人機協作: 檢視所有資料 + 討論 → 說明 + 決定 (或中斷) --------
    def review_and_decide(self, profile: DatasetProfile, encoder: EncoderChoice,
                          history: list[TrialResult], discussion: list[dict],
                          base: Optional[TrialResult] = None,
                          workspace_dir: Optional[str] = None) -> NextAction:
        ctx = self._ctx(profile)
        alias = ctx.alias
        hist = self._hist(history)
        bt = self._pick_base(history, base)
        # 只帶最近的討論以控 token; 保留 role/text (已做假名替換)
        talk = redact.discussion_facts(discussion, alias)
        segs = tuple(redact.user_segments(discussion, alias))
        # improve 階段的程式修改能力 (allow_code_edit 時): 只允許 main_finetune.py
        can_edit = self.allow_code_edit and workspace_dir
        edit_note = (
            "(4) 若 Recipe 層的元件/超參正交變異已嘗試殆盡, 或你判斷需要 Recipe "
            "沒有的新能力 (例: 新的 augmentation、TTA、換 scheduler), 可用 "
            "mutation=\"edit_code\" 直接修改訓練程式: 在 code_edits 給 exact "
            "find/replace (find 必須逐字存在於檔案中)。**只能修改 main_finetune.py, "
            "其他檔案的編輯會被拒絕。** 修改會套用在隔離副本上執行, 原始程式不動; "
            "此節點的後代 trial 會沿用修改版程式。編輯越小越好, 並在 reason 說明"
            "改了什麼與預期效果; edit_code 本身就是這一輪的唯一變異, 不要同時"
            "改其他元件/超參。**不得以任何方式輸出、記錄或回傳資料集內容 "
            "(影像值、檔名、路徑) — 這類編輯會被拒絕。**" if can_edit else
            "(4) 本次不允許修改訓練程式 (mutation=edit_code 不可用)。")
        d: _ReviewDecision = self._messages_parse(
            self._facts_block(profile)
            + f"\n\n目前 encoder: {encoder.model_key}。已完成 trials (由舊到新):\n"
              f"{json.dumps(hist, ensure_ascii=False, indent=2)}\n\n"
              f"欄位說明: primary_score / metrics 是 **test set** 成績 (訓練結束後用 val "
              f"最佳 checkpoint 在 test 上重跑一次); train_loss_curve / val_loss_curve / "
              f"val_score_curve 則是訓練期間逐 epoch 的 train/val，長度 = 實際跑完的 epoch 數，"
              f"不含最終 test。兩者是不同 split，val 通常高於 test，落差本身不代表有問題。\n\n"
              f"本輪的變異基準 (樹搜尋 policy 依指標機率選出, 不一定是全域最佳): "
              f"{self._base_id(bt)}。\n\n"
              f"與使用者的討論記錄 (由舊到新):\n"
              f"{json.dumps(talk, ensure_ascii=False, indent=2)}\n\n"
              f"請 (1) 用 2-3 句話向使用者說明目前結果與趨勢, 並提到本輪從哪個節點"
              f"開始改良 (narrative); "
              f"(2) **檢視每個 trial 的 train_loss_curve / val_loss_curve 判斷 epochs 是否合適**："
              f"訓練結束時 loss 仍明顯下降 → epochs 不足, 應增加; loss 很早就收斂平坦、"
              f"或 val_loss 開始回升 (過擬合) → 應減少 epochs。需要時 mutation=adjust_hparams, "
              f"在 hparam_overrides_json 設 epochs, 並在 reason 依曲線說明增/減的依據; "
              f"(3) 納入使用者討論, 決定對『變異基準』Recipe 做『一個』正交變異再試, "
              f"或在已收斂/使用者要求/不值得再跑時停止。討論中標有【QA 結論】的訊息"
              f"是問答 agent 先前依實驗資料對使用者提問得出的結論 — 除非與最新數據"
              f"矛盾, 本輪決策應優先採納最近的一則; "
            + edit_note + self._info_note(ctx),
            _ReviewDecision, label="review_and_decide", profile=profile,
            user_segments=segs)
        next_recipe = None
        if not d.stop and bt is not None:
            next_recipe = self._apply_mutation(
                bt.recipe, d, workspace_dir=workspace_dir,
                tag=f"improve_t{len(history)}")
            if next_recipe is not None:
                next_recipe.provenance["reason"] = d.reason
                next_recipe.provenance["advisor"] = "llm"
        return NextAction(stop=d.stop or next_recipe is None, reason=d.reason,
                          narrative=d.narrative, mutation=d.mutation,
                          next_recipe=next_recipe, info=self.last_info)

    # ---- 問答 agent: 使用者提問時立即以實驗資料回答 (獨立於決策迴圈) ----
    def answer_question(self, profile: Optional[DatasetProfile],
                        history: list[TrialResult], tree: Optional[dict],
                        discussion: list[dict], question: str,
                        on_delta=None) -> tuple[str, str]:
        """回答使用者在討論頻道的提問/方向; 回傳 (answer, conclusion)。

        以 **streaming** 方式生成 (純文字, 非 JSON — 才能逐段即時顯示):
        每收到新片段就呼叫 on_delta(累積全文), 供 web 即時渲染打字效果。
        要求模型在最後獨立一行以「結論：」開頭給一句結論, 完成後拆出來
        供決策層下一輪採用; 純知識性問題可無結論行。"""
        ctx = self._ctx(profile)
        alias = ctx.alias
        hist = self._hist(history)
        talk = redact.discussion_facts(discussion, alias, limit=12)
        segs = tuple(redact.user_segments(discussion, alias, limit=12)) + (question,)
        tree_txt = (json.dumps(redact.scrub(tree, alias), ensure_ascii=False)
                    if tree else "（尚無解答樹）")
        volatile = (
            f"你現在的角色是**實驗問答助理** (獨立於決策層): 使用者剛在討論頻道"
            f"提出問題或方向, 請立即依下面的實驗資料回答, 不要做任何 Recipe 決策。\n\n"
            + self._facts_block(profile)
            + f"\n\n已完成 trials (由舊到新):\n"
              f"{json.dumps(hist, ensure_ascii=False, indent=2)}\n\n"
              f"解答樹 (AIDE 式搜尋; stage/parent/metric):\n{tree_txt}\n\n"
              f"近期討論 (由舊到新):\n{json.dumps(talk, ensure_ascii=False, indent=2)}\n\n"
              f"使用者剛提出：「{question}」\n\n"
              f"請直接以**純文字**回答使用者 (不要 JSON、不要 markdown 標題): "
              f"具體引用上面 trial 的數據/曲線/樹結構佐證, 3-6 句, 白話。"
              f"若使用者問的是你看不到的東西 (原始影像、檔名、路徑、真實類別名), "
              f"請直接說明資料圍欄的緣故你看不到, 並建議他改用「執行分析器」或"
              f"「直接告訴你」的方式提供。"
              f"回答結束後**另起最後獨立一行**, 以「結論：」開頭, 給一句『下一輪決策"
              f"可直接採用的結論或方向』(例: 結論：優先對 t4 加 mixup)。"
              f"若問題與實驗方向無關 (純知識性提問), 則省略結論行。")
        if self.guidance:
            volatile = (f"【使用者引導方向 (請優先納入考量)】\n{self.guidance}\n\n"
                        + volatile)
        client = self._get_client().with_options(timeout=240.0, max_retries=1)
        system = [
            {"type": "text", "text": self._RULES},
            {"type": "text", "text": "可用資源目錄:\n" + self._registry_context(),
             "cache_control": {"type": "ephemeral"}},  # 與決策層同前綴 → 命中快取
        ]
        rec = {"ts": time.time(), "label": "qa_answer", "model": self.model,
               "system": "\n\n".join(b["text"] for b in system), "prompt": volatile}
        try:
            text, resp = egress_mod.stream_text(
                client, ctx=ctx.egress, label="qa_answer", model=self.model,
                system=system, messages=[{"role": "user", "content": volatile}],
                on_delta=on_delta, user_segments=segs,
                max_tokens=16000,
                thinking={"type": "adaptive", "display": "summarized"})
            text = (text or "").strip()
            rec["response"] = text
            rec["stop_reason"] = resp.stop_reason
            think = "".join(getattr(b, "thinking", "") for b in resp.content
                            if getattr(b, "type", None) == "thinking")
            if think:
                rec["thinking"] = think
            rec["ok"] = True
            self._log_call(rec)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            self._log_call(rec)
            raise
        # 拆出最後的「結論：」行
        answer, conclusion = text, ""
        lines = text.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i].strip()
            if not ln:
                continue
            if ln.startswith("結論：") or ln.startswith("結論:"):
                conclusion = ln[3:].strip()
                answer = "\n".join(lines[:i]).strip()
            break  # 只看最後一個非空行
        return answer, conclusion

    # ---- 樹搜尋 debug 階段 (aideml): 依 ErrorFacts 修正 Recipe / 程式 ----
    def propose_debug(self, profile: DatasetProfile, encoder: EncoderChoice,
                      trial: TrialResult, error_facts: ErrorFacts,
                      workspace_dir: Optional[str] = None) -> Optional[Recipe]:
        """依失敗 trial 的 **ErrorFacts** 提出修正; None = 放棄該分支。

        ⚠ 資料圍欄: 這裡收到的是由 `analyzers.error_extract` 以允許清單抽取出的
        結構化事實, **不是原始 log** — 原始 log 含 `data_path`、traceback 中的影像
        路徑, 一律不外送 (docs/data_firewall_design.md §8.1)。

        allow_code_edit=True 且提供 workspace_dir 時, LLM 可另提 code_edits
        (exact find/replace); 由 code_workspace 套用在 run 目錄的副本上,
        該 trial 改跑副本 — **原始程式不會被修改**。
        """
        ctx = self._ctx(profile)
        alias = ctx.alias
        can_edit = self.allow_code_edit and workspace_dir
        edit_note = (
            "你也可以用 code_edits 修改訓練程式 (main_finetune.py / engine_finetune.py / "
            "models_vit.py / util/*.py): 每個編輯給 file + find (逐字存在的原片段) + replace。"
            "修改會套用在隔離副本上執行, 原始程式不動。只在 Recipe 層修不了時才改程式, "
            "編輯越小越好。**不得加入任何會輸出資料集內容 (影像值、檔名、路徑) 的程式碼。**"
            if can_edit else
            "本次不允許修改程式 (code_edits 會被忽略), 只能調整 Recipe/超參。")
        recipe_view = redact.scrub(trial.recipe.model_dump(), alias)
        d: _DebugDecision = self._messages_parse(
            self._facts_block(profile)
            + f"\n\nencoder: {encoder.model_key}。以下 trial 訓練失敗:\n"
              f"Recipe:\n{json.dumps(recipe_view, ensure_ascii=False, indent=2)}\n\n"
              f"失敗事實 (ErrorFacts — 由本地程式從訓練 log 以允許清單抽取; "
              f"原始 log 含路徑與檔名, 依資料圍欄不提供給你):\n"
              f"{error_facts.model_dump_json(indent=2)}\n\n"
              f"請診斷失敗原因 (diagnosis), 並提出『最小』修正後重試: 超參覆寫用 "
              f"hparam_overrides_json, 元件變更用 loss_name/remove_regularizer/"
              f"new_augmentation。{edit_note} "
              f"若 error_class=unknown 且沒有足以判斷的線索, 寧可保守調整 (例: 降 lr) "
              f"或 give_up=true; 不要臆測資料內容。"
              f"若判斷是資料或環境問題、重試無意義, 則 give_up=true。"
            + self._info_note(ctx),
            _DebugDecision, label="propose_debug", profile=profile)
        if d.give_up:
            return None

        r = trial.recipe.model_copy(deep=True)
        changes: list[str] = []
        try:
            overrides = json.loads(d.hparam_overrides_json) if d.hparam_overrides_json else {}
        except Exception:
            overrides = {}
        for k, v in overrides.items():
            if hasattr(r.hparams, k):
                changes.append(f"{k}={v}")
                setattr(r.hparams, k, v)
        if d.loss_name and (not r.losses or r.losses[0].name != d.loss_name):
            r.losses = [ComponentRef(name=d.loss_name)]
            changes.append(f"loss={d.loss_name}")
        if d.remove_regularizer:
            before = len(r.regularizers)
            r.regularizers = [c for c in r.regularizers
                              if c.name != d.remove_regularizer]
            if len(r.regularizers) != before:
                changes.append(f"-{d.remove_regularizer}")
        if d.new_augmentation:
            r.augmentation = ComponentRef(name=d.new_augmentation)
            changes.append(f"aug={d.new_augmentation}")

        # 程式修改: 套用在 run 目錄的副本 (code_workspace), 原始程式不動
        if can_edit and d.code_edits:
            from . import code_workspace
            ws = code_workspace.create(
                workspace_dir, f"debug_{trial.trial_id}",
                [e.model_dump() for e in d.code_edits],
                parent_code_dir=trial.recipe.code_dir)
            if ws:
                r.code_dir = ws["code_dir"]
                changes.append(f"code_edits×{len(ws['applied'])}")

        if not changes:
            return None  # 沒有任何有效修正 → 重跑同樣配方無意義

        prov = dict(r.provenance)
        prov["mutated_from"] = {"trial": trial.trial_id,
                                "mutation": trial.recipe.provenance.get("mutation")}
        prov["mutation"] = f"debug:{', '.join(changes)}"
        prov["reason"] = f"診斷: {d.diagnosis}"
        prov["advisor"] = "llm"
        r.provenance = prov
        return r

    def _apply_mutation(self, base: Recipe, d: _NextDecision,
                        workspace_dir: Optional[str] = None,
                        tag: str = "") -> Recipe:
        r = base.model_copy(deep=True)
        prov = dict(r.provenance)
        prov["mutated_from"] = base.provenance
        if d.mutation == "edit_code" and d.code_edits:
            # improve 階段的程式修改: 只允許 main_finetune.py (系統其他部份不可改),
            # 套用在 run 目錄的隔離副本 (code_workspace), 原始程式不動。
            ws = None
            if self.allow_code_edit and workspace_dir:
                from . import code_workspace
                ws = code_workspace.create(
                    workspace_dir, tag or "improve_edit",
                    [e.model_dump() for e in d.code_edits],
                    parent_code_dir=base.code_dir,
                    allowed_files=("main_finetune.py",))
            if ws:
                r.code_dir = ws["code_dir"]
                prov["mutation"] = (f"edit_code:main_finetune.py"
                                    f"×{len(ws['applied'])}")
                prov["code_edits"] = {"workspace": ws["code_dir"],
                                      "applied": len(ws["applied"]),
                                      "failed": len(ws["failed"])}
            else:
                # 編輯全數無效 (find 不符 / 檔案不在白名單) 或未允許修改程式
                # → 不動 recipe, 只記錄; 該輪等同重跑基準 (不讓整個實驗停掉)
                prov["mutation"] = "edit_code:未套用(編輯無效或未允許)"
        elif d.mutation == "add_regularizer" and d.add_regularizer:
            if d.add_regularizer not in [c.name for c in r.regularizers]:
                r.regularizers.append(ComponentRef(name=d.add_regularizer))
            prov["mutation"] = f"add_regularizer:{d.add_regularizer}"
        elif d.mutation == "swap_head" and d.new_head_type:
            r.heads[0].type = d.new_head_type
            prov["mutation"] = f"swap_head:{d.new_head_type}"
        elif d.mutation == "change_augmentation" and d.new_augmentation:
            r.augmentation = ComponentRef(name=d.new_augmentation)
            prov["mutation"] = f"change_augmentation:{d.new_augmentation}"
        elif d.mutation == "adjust_hparams" and d.hparam_overrides_json:
            try:
                overrides = json.loads(d.hparam_overrides_json)
            except Exception:
                overrides = {}
            for k, v in overrides.items():
                if hasattr(r.hparams, k):
                    setattr(r.hparams, k, v)
            prov["mutation"] = f"adjust_hparams:{overrides}"
        r.provenance = prov
        return r


def _has_user_facts(uf: UserFacts) -> bool:
    return bool(uf.answers or uf.modality or uf.anatomy
                or uf.class_ordinal is not None)
