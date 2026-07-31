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

**端點可切換 (Anthropic 直連 / OpenRouter)**: 見 `_get_client`。設了
`ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` 就改走該端點 (例: OpenRouter 的
Anthropic-compatible 端點 `https://openrouter.ai/api`, 模型名如
`anthropic/claude-opus-4.5`, 以 `MEDCLAW_LLM_MODEL` 指定); 兩者都沒設則行為與
原本完全相同。⚠ 指到代理端點等於把 (已消毒的) payload 交給第三方轉送 —
見 docs/data_firewall_design.md §7.5。
"""
from __future__ import annotations

import json
import os
import time
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from . import encoder_registry as reg
from . import presets
from .privacy import egress as egress_mod
from .privacy import redact
from .privacy.context import PrivacyContext
from .privacy.facts import AnalysisFacts, ErrorFacts, UserFacts
from .privacy.facts import UserQuestion as _UQ
from .schemas import (
    ComponentRef, DatasetProfile, EncoderChoice, EnsembleSpec, HeadSpec,
    HyperParams, InfoRequest, NextAction, Recipe, SearchOverride, TrialResult,
)

_MODEL = "claude-opus-4-8"


def default_model() -> str:
    """預設模型; `MEDCLAW_LLM_MODEL` 可覆寫。

    換端點 (見 `_get_client`) 通常要連模型名一起換 (OpenRouter 用
    `anthropic/claude-opus-4.5` 這種帶 provider 前綴的 id), 走 env 就不必去改
    `~/.medclaw/settings.json` 或每張表單的預設值。
    """
    return (os.getenv("MEDCLAW_LLM_MODEL") or "").strip() or _MODEL


def _endpoint_override() -> tuple[str, str]:
    """(base_url, auth_token) — 兩者皆空 = 走 Anthropic 官方端點 (原本的行為)。

    對齊 Claude Code / Anthropic SDK 的既有慣例: `ANTHROPIC_BASE_URL` 指端點,
    `ANTHROPIC_AUTH_TOKEN` 是 bearer token (OpenRouter 的 `sk-or-...`);
    `OPENROUTER_API_KEY` 只是常見的別名, 一併接受。
    """
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    token = ((os.getenv("ANTHROPIC_AUTH_TOKEN")
              or os.getenv("OPENROUTER_API_KEY") or "").strip())
    return base_url, token


def _endpoint_name() -> str:
    """給錯誤訊息用的端點稱呼。"""
    base_url = _endpoint_override()[0]
    if not base_url:
        return "Claude API"
    if "openrouter" in base_url:
        return f"OpenRouter ({base_url})"
    return f"自訂端點 ({base_url})"


def _credential_hint() -> str:
    """給錯誤訊息用: 這個端點該設哪個環境變數。"""
    if any(_endpoint_override()):
        return "ANTHROPIC_BASE_URL 與 ANTHROPIC_AUTH_TOKEN"
    return "ANTHROPIC_API_KEY（或已 `ant auth login`）"


def _extra_body() -> dict:
    """自訂端點要額外帶的 body 欄位 (官方端點回空 dict → 呼叫端不帶 extra_body)。

    OpenRouter 會把同一個 `anthropic/*` model 路由到不同 provider (實測會落到
    Amazon Bedrock 或 Anthropic first-party)。兩者都支援 thinking 與
    `cache_control`(含 1h ttl), 但 **prompt cache 是各 provider 各自持有的** ——
    一個 run 在兩者之間跳動就等於每次都 cache miss, 而本模組整個 prompt 分段設計
    (見 _data_block) 就是為了命中那筆快取。故預設把 provider 釘在 anthropic
    first-party (也是 OpenRouter 文件唯一保證 Anthropic-compatible 的路徑)。

    `MEDCLAW_LLM_PROVIDER=off` 可關掉釘選 (讓 OpenRouter 自由路由); 給別的值則
    釘到該 provider。
    """
    base_url = _endpoint_override()[0]
    if "openrouter" not in base_url:
        return {}
    p = (os.getenv("MEDCLAW_LLM_PROVIDER") or "anthropic").strip()
    if p.lower() in ("off", "any", ""):
        return {}
    return {"provider": {"only": [p]}}


def normalize_model_key(model: str) -> str:
    """把各端點的模型 id 正規化成 Anthropic 官方寫法, 供查表用 (見 _cache_min_chars)。

    OpenRouter: `~anthropic/claude-opus-4.5:beta` → `claude-opus-4-5`
    (去 `~` 別名記號、去 provider 前綴、截掉 `:` 變體後綴、`.` 版號改 `-`)。
    """
    m = (model or "").strip().lstrip("~")
    m = m.split(":", 1)[0]
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.replace(".", "-")


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
    stop: bool = Field(
        default=False,
        description="結束『整個實驗』的搜尋 — 只在所有分支都已收斂、使用者要求停止, "
                    "或再跑任何 trial 都不值得時才設 true。只是本輪這個基準節點沒得改, "
                    "請改用 prune_branch, 不要用 stop。")
    prune_branch: bool = Field(
        default=False,
        description="只放棄『本輪的變異基準節點』這條分支 (例如該節點特徵無訊號、"
                    "已知走不通), 實驗會自動改從樹上其他節點繼續。這不會結束實驗。")
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


class _EnsemblePick(_InfoMixin):
    """propose_ensemble 的回應 — 選成員 (aliased trial_id) + 集成方法。"""
    do_ensemble: bool = False               # 是否值得集成 (成員 <2 或不值得 → false)
    member_trial_ids: list[str] = Field(default_factory=list)  # 成員的 (假名) trial_id
    method: Literal["equal", "val_weighted", "stacking"] = "val_weighted"
    rationale: str = ""


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

    def __init__(self, model: str = "", guidance: str = "",
                 allow_code_edit: bool = False,
                 privacy: Optional[PrivacyContext] = None):
        self.model = model or default_model()
        self.guidance = guidance.strip()
        self.allow_code_edit = allow_code_edit  # 允許 LLM 修改程式 (副本, 見 code_workspace)
        self._client = None
        self._log_path: Optional[str] = None
        self.privacy = privacy
        # 已執行的分析器結果 (由 LoopController 填入, 隨 prompt 一起帶給 LLM)
        self.analyses: list[AnalysisFacts] = []
        # 分析器目錄 (key/說明/成本), 供 LLM 知道有哪些程式可以點名執行
        self.analyzer_catalog: list[dict] = []
        # 本次決策附帶的資訊需求 (由呼叫端讀取後處理)
        self.last_info: InfoRequest = InfoRequest()
        # 本 run 內觀察到的**最長**呼叫間隔 — 決定 cache TTL 用 5m 還是 1h
        # (見 _next_cache_ttl)
        self._last_call_ts: Optional[float] = None
        self._max_gap = 0.0
        # 自訂端點未透傳 thinking 參數時, 本實例自動降級 (見 _thinking)
        self._thinking_off = False

    # ---- 呼叫紀錄落地; 設定路徑時順便接續上一段的 cache 間隔統計 ---------
    @property
    def log_path(self) -> Optional[str]:
        """設定後, 每次 LLM 呼叫的完整 prompt/回應會落地到此 (jsonl)。"""
        return self._log_path

    @log_path.setter
    def log_path(self, path: Optional[str]) -> None:
        self._log_path = path
        if path:
            self._seed_gap_from_log(path)

    def _seed_gap_from_log(self, path: str) -> None:
        """resume 時從既有 llm_calls.jsonl 接續呼叫間隔統計。

        中斷後 resume 會建立新的 LLMAdvisor, `_max_gap` 從 0 開始, 於是又要等到
        第一次訓練空檔過完才知道該開 1h — 那一輪的 cache write 必然白付。既有
        紀錄的 ts 就是上一段的呼叫節奏, 直接拿來接續, resume 後第一輪就選得對。

        壞檔/缺欄位一律略過 (統計只是 ttl 的啟發式, 不值得為它擋住整個 run)。
        """
        ts: list[float] = []
        try:
            with open(path, encoding="utf8") as f:
                for line in f:
                    try:
                        t = json.loads(line).get("ts")
                    except (ValueError, AttributeError):
                        continue
                    if isinstance(t, (int, float)):
                        ts.append(float(t))
        except OSError:
            return
        for prev, cur in zip(ts, ts[1:]):
            self._note_gap(cur - prev)

    @staticmethod
    def _usage_fields(resp) -> dict:
        """回應的 token 用量 — 落地是為了**事後驗證 prompt cache 有沒有命中**。

        `cache_read_input_tokens` 長期為 0 就代表 _data_block 的分段設計沒發揮
        (換端點後尤其要看: 快取是各 provider 各自持有的, 見 _extra_body)。
        """
        u = getattr(resp, "usage", None)
        if u is None:
            return {}
        out = {k: getattr(u, k, None) for k in
               ("input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")}
        return {"usage": {k: v for k, v in out.items() if v is not None}}

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
            if _endpoint_override()[0]:
                # 自訂端點 (OpenRouter 等) 沒有 Anthropic 格式的 GET /v1/models
                # (回傳欄位不合 SDK 的 ModelInfo → 會驗證失敗, 被誤報成金鑰錯誤),
                # 故改以一次最小 messages 呼叫當 ping。順帶驗證 model id 存在。
                # 必須經 egress: privacy.sentinel 會擋掉不是從 egress.py 發出的呼叫。
                egress_mod.create(
                    client.with_options(timeout=30.0, max_retries=0),
                    ctx=self._ctx().egress, label="check_environment",
                    model=self.model,
                    system=[{"type": "text", "text": "ping"}],
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1, **self._extra())
            else:
                # 官方端點: models.list 不帶任何 payload 也不計費 (快速失敗, 不重試)
                client.with_options(timeout=20.0, max_retries=0).models.list(limit=1)
        except Exception as e:
            raise RuntimeError(
                f"advisor=llm 無法連上 {_endpoint_name()}：請確認已設定"
                f" {_credential_hint()} 且網路可連線, 且 model『{self.model}』"
                f"在該端點存在。"
                f"（{type(e).__name__}: {e}）。或改用 advisor.type=heuristic。") from e

    # ---- SDK client (延後建立) ----------------------------------------
    def _get_client(self):
        """建立 SDK client; 依 env 決定打官方端點還是自訂 (OpenRouter) 端點。

        自訂端點以 `Authorization: Bearer` 認證, 所以要 **顯式**把 x-api-key 關掉:
        `web/settings.apply_env()` 會把 `~/.medclaw/settings.json` 裡的舊 Anthropic
        key 灌進 os.environ, SDK 見到 api_key 就會優先送 x-api-key, 在 OpenRouter
        上直接 401。api_key="" 讓 auth_headers 走 bearer, Omit() 再把空 header 拿掉。
        """
        if self._client is None:
            try:
                import anthropic
                from anthropic._types import Omit
            except ImportError as e:
                raise RuntimeError(
                    "advisor=llm 需要 anthropic SDK，但未安裝。請執行 "
                    "`pip install anthropic`，或改用 advisor.type=heuristic。") from e
            base_url, token = _endpoint_override()
            try:
                if base_url or token:
                    kw = {"api_key": "", "auth_token": token,
                          "default_headers": {"X-Api-Key": Omit()}}
                    if base_url:
                        kw["base_url"] = base_url
                    self._client = anthropic.Anthropic(**kw)
                else:
                    self._client = anthropic.Anthropic()  # 自環境/ant profile 解析金鑰
            except Exception as e:
                raise RuntimeError(
                    f"advisor=llm 無法建立 {_endpoint_name()} client（可能缺少"
                    f" {_credential_hint()}）：{e}。或改用 advisor.type=heuristic。") from e
        return self._client

    # ---- thinking 參數 (自訂端點可能未透傳 → 可關) ----------------------
    def _thinking(self) -> Optional[dict]:
        """回傳要送的 thinking 參數; None = 整個參數不送。

        `MEDCLAW_LLM_THINKING=off` 可預先關掉 (代理端點未支援 adaptive thinking 時);
        另外 `_messages_parse` 遇到明確指向 thinking 的錯誤會自動降級一次
        (見 self._thinking_off)。
        """
        if self._thinking_off:
            return None
        if (os.getenv("MEDCLAW_LLM_THINKING") or "").strip().lower() == "off":
            return None
        return {"type": "adaptive", "display": "summarized"}

    def _extra(self) -> dict:
        """要傳給 SDK 的 extra_body (自訂端點的 provider 路由; 見 _extra_body)。"""
        body = _extra_body()
        return {"extra_body": body} if body else {}

    def _maybe_disable_thinking(self, err: Exception) -> bool:
        """錯誤指向 thinking 參數 → 關掉它 (本實例) 並回 True 表示值得重試一次。"""
        if self._thinking_off or self._thinking() is None:
            return False
        if "thinking" not in str(err).lower():
            return False
        self._thinking_off = True
        return True

    # ---- 資料圍欄 ------------------------------------------------------
    def _ctx(self, profile: Optional[DatasetProfile] = None) -> PrivacyContext:
        """取得 PrivacyContext; 未掛上時以 strict 建構 (fail-closed)。"""
        if self.privacy is None:
            self.privacy = PrivacyContext.strict_for(
                profile, exempt_text=self._registry_context())
        if self.guidance:
            self.privacy.note_user_text(self.guidance)
        return self.privacy

    _ANALYSES_HEADER = ("\n\n【追加分析結果】(你先前點名、由本地已註冊分析器執行後的"
                        "輸出; 每行一筆 JSON)\n")

    def _facts_segments(self, profile: Optional[DatasetProfile]) -> list[str]:
        """DatasetFacts + 分析器結果, 切成「寫出後就不再變」的段 (見 _data_block)。

        這是決策層唯一能看到的「資料」— 全部是本地程式產生的統計量。
        使用者事實見 _user_facts_block: 它會變, 所以排在 trials 之後。
        """
        ctx = self._ctx(profile)
        if profile is not None:
            segs = ["【資料集事實 DatasetFacts】(由本地程式分析後消毒; 你看不到也拿不到"
                    "原始影像、檔名或路徑。class_labels 是假名, 真實名稱不會提供)\n"
                    + ctx.facts(profile).model_dump_json(indent=2)]
        else:
            segs = ["【資料集事實】（尚無分析結果）"]
        if self.analyses:
            # 依 key 排序 + 每筆自成一段。排序讓內容與「LLM 點名的順序」脫鉤
            # (resume 後順序不保證相同); 分段則讓新增分析器時既有的段仍逐字不變
            # —— 兩者都是 cache 命中的前提, 理由見 _data_block。
            segs.append(self._ANALYSES_HEADER)
            segs += [json.dumps(a.model_dump(), ensure_ascii=False) + "\n"
                     for a in sorted(self.analyses, key=lambda x: x.key)]
        return segs

    def _user_facts_block(self, profile: Optional[DatasetProfile]) -> str:
        """使用者在 UI 上填寫的事實; 沒有時回空字串。

        內容會隨使用者回答問題而**改寫** (不是 append), 所以它是穩定前綴裡唯一
        會變的一段 — 位置與 breakpoint 的安排見 _data_block。
        """
        ctx = self._ctx(profile)
        uf = ctx.refresh_user_facts()
        for a in uf.answers:
            ctx.note_user_text(a.value)   # 使用者自願提供 → 出口掃描只警示不中止
        if not _has_user_facts(uf):
            return ""
        # 開頭自帶分隔: content block 之間 API 不會補任何字元, 而分隔字元留在
        # 本段開頭才不會動到前一段的結尾 (前一段必須維持逐字穩定)。
        return ("\n\n【使用者親自提供的事實】(使用者在 UI 上填寫; 可信度高於任何推測)\n"
                + uf.model_dump_json(indent=2))

    # prompt cache 的可快取前綴下限 (tokens) — **依模型不同, 且不隨世代單調遞減**。
    # 低於下限即使下了 breakpoint 也只會付 cache write 卻永遠讀不到 (API 不報錯,
    # 只是 cache_creation_input_tokens=0)。
    _CACHE_MIN_TOKENS = {
        "claude-opus-5": 512,
        "claude-fable-5": 512,
        "claude-mythos-5": 512,
        "claude-opus-4-8": 1024,
        "claude-sonnet-5": 1024,
        "claude-sonnet-4-6": 1024,
        "claude-sonnet-4-5": 1024,
        "claude-opus-4-7": 2048,
        "claude-opus-4-6": 4096,
        "claude-opus-4-5": 4096,
        "claude-haiku-4-5": 4096,
    }
    _CACHE_MIN_TOKENS_DEFAULT = 4096   # 不認得的模型取最保守值
    # 換算 token → 字元。這段 prompt 是中文敘述混 ASCII 的 JSON: 中文約 1 char≈1
    # token, 而 JSON 的 key/數字可到 1 token≈4 chars。取 4 是保守側 —— 寧可少下
    # 一個 breakpoint, 也不要下了卻讀不到。
    _CHARS_PER_TOKEN = 4

    @property
    def _cache_min_chars(self) -> int:
        """本模型下值得下 breakpoint 的最短前綴 (字元)。

        先正規化模型名 — 經 OpenRouter 時 id 是 `anthropic/claude-opus-4.5` 這種
        寫法, 直接 startswith 比對會全部 miss 而落到最保守的 4096, 白白少下
        breakpoint (見 normalize_model_key)。
        """
        key = normalize_model_key(self.model)
        tok = next((v for k, v in self._CACHE_MIN_TOKENS.items()
                    if key.startswith(k)), self._CACHE_MIN_TOKENS_DEFAULT)
        return tok * self._CHARS_PER_TOKEN

    # 同一次請求裡的所有 breakpoint 必須用同一個 ttl: API 規定 render 順序
    # (tools → system → messages) 上, ttl 長的 block 不得排在 ttl 短的之後,
    # system 用 5m 而 messages 用 1h 會直接 400。
    _CACHE_TTL_THRESHOLD_S = 240.0     # 曾出現超過 4 分鐘的間隔才值得開 1h
    # 超過 1h 的間隔連 1h cache 都活不過 → 不論選哪個 ttl 都必然 miss, 這時反而
    # 該用比較便宜的 5m 寫入。所以這種間隔不列入「值得開 1h」的證據, 否則一次
    # 隔夜中斷 (fold4 那次隔了 8 小時) 就會讓 resume 後每輪都多付 2x。
    _CACHE_GAP_CEILING_S = 3600.0

    def _note_gap(self, gap: float) -> None:
        if 0.0 < gap <= self._CACHE_GAP_CEILING_S:
            self._max_gap = max(self._max_gap, gap)

    def _next_cache_ttl(self) -> str:
        """決定這次請求要用的 cache TTL, 並把本次呼叫時間記進統計。

        1h 的 cache write 要 2x (5m 只要 1.25x), 只有在「下次呼叫時 5m 早就過期」
        的情況下才划算。判斷依據是**本 run 內觀察到的最長呼叫間隔**: 只要曾經
        出現過超過門檻的間隔, 就代表這個 run 的節奏會讓 5m entry 死在半路 ——
        決策迴圈的兩次 review_and_decide 之間隔著一次完整訓練, 小資料集可能只要
        3 分鐘 (5m 撐得住, 用 5m 省下那 0.75x), 大資料集動輒 10~20 分鐘 (5m 必死,
        每輪都在付寫入費卻零命中, 改 1h 只要有第二次讀就回本)。

        **刻意用 max 而不是平均**: 平均會被連發的呼叫稀釋 —— 決策前的
        plan_information → review_search_choice → review_and_decide 是 30 秒內連發,
        再加上重試會連三發, 這些接近 0 的間隔會把平均壓到門檻以下, 結果訓練空檔
        再長也永遠選 5m。一次長間隔就足以讓 entry 過期, 所以該看的是最大值。

        全新 run 的第一次呼叫沒有間隔可看, 保守用 5m, 要到第一次長間隔之後才切到
        1h — 那一輪的浪費無法避免 (事前不知道訓練要跑多久)。但 **resume 不必再付
        一次**: 設定 log_path 時已從既有紀錄接續統計 (見 _seed_gap_from_log)。
        每次請求只呼叫一次。
        """
        now = time.time()
        if self._last_call_ts is not None:
            self._note_gap(now - self._last_call_ts)
        self._last_call_ts = now
        return "1h" if self._max_gap > self._CACHE_TTL_THRESHOLD_S else "5m"

    _HIST_HEADER = "\n\n已完成 trials (由舊到新, 每行一筆 JSON; trial_id 為假名):\n"

    def _data_block(self, profile: Optional[DatasetProfile],
                    history: Optional[list[TrialResult]] = None) -> list[str]:
        """user message 的**穩定前綴**, 切成一串「寫出後就不再變動」的段。

        每一段各成一個 content block, 段的切法就是 cache 的命脈:

          [0]   DatasetFacts                      — 一個 run 內固定
          [1..] 追加分析標頭 + 每個分析器一段      — 只會 append
          [k]   trials 標頭 + **每個 trial 一段**  — 只會 append
          [-1]  使用者親自提供的事實               — 會**改寫**, 所以排最後

        **關鍵: 快取比對是以 content block 邊界為單位, 不是任意位元組前綴。**
        早期版本把整段 trials 塞在一個會長大的 block 裡, 內容確實是乾淨的
        append-only 前綴鏈, 但上一輪的斷點 (例: 66,512 字元處) 在這一輪落到那個
        block (74,325 字元) 的**中間** —— 沒有邊界可對, 於是幾萬 token 每輪重寫
        卻一次都沒讀到。實測佐證: 同一批位元組放在固定不變的 block 裡 (system、
        不帶 history 的小呼叫) 每次都命中, 只有會長大的那塊從來沒中過。

        所以每個 trial 各自成段: 下一輪多一筆時, 前面所有段仍**逐字相同且邊界
        不變**, 上一輪在最後一段結尾寫下的 entry 這一輪還找得到 (breakpoint 只
        往回找 20 個 block, 而每輪只多 1 段)。這就是官方多輪對話的作法。

        使用者事實排最後同理: 它會改寫, 放前面的話使用者一回答問題就把後面那
        50k+ 的 trials 全部推掉。

        所有 label (決策/QA) 都用完全相同的段開頭 → 共用同一筆 cache。
        **任何隨輪次變動的東西** (目前 encoder、本輪變異基準、解答樹、討論、
        本輪指令/schema) 一律放在這之後, 否則整段前綴會失效。

        trials 用 JSONL (每筆一行) 而不是 `json.dumps(list, indent=2)`: 陣列寫法
        每多一筆就會改寫上一筆收尾的 "}\\n" → "},\\n", 連內容都不是前綴了
        (順帶省下 indent 的 token)。
        """
        segs = self._facts_segments(profile)
        if history:
            segs.append(self._HIST_HEADER)
            segs += [json.dumps(t, ensure_ascii=False) + "\n"
                     for t in self._hist(history)]
        return [s for s in segs + [self._user_facts_block(profile)] if s]

    # 一次請求最多 4 個 breakpoint; system 佔掉 1, 其餘留給 user message。
    _MAX_USER_BREAKPOINTS = 3

    def _breakpoint_indices(self, stable: Sequence[str]) -> list[int]:
        """哪幾段要下 cache_control。

        額度有限 (最多 3), 所以只下在**最後**幾段: 這一輪的最後一段是下一輪的
        讀取點, 而倒數第二、三段是上一輪/上上輪寫下的 entry —— 保留它們等於多留
        兩個退路, 中間漏掉一輪 (例如某次呼叫失敗) 也還接得回去。

        門檻用**累計**長度而非單段長度: cache 的最小前綴是從 prompt 開頭算起的,
        所以接在 50k 之後的一小段照樣值得下 breakpoint。
        """
        eligible, total = [], 0
        for i, seg in enumerate(stable):
            total += len(seg)
            if total >= self._cache_min_chars:
                eligible.append(i)
        return eligible[-self._MAX_USER_BREAKPOINTS:]

    def _cache_log_fields(self, stable: Sequence[str]) -> dict:
        """落地用: 段數、總長、各 breakpoint 的累計位移 (供事後比對命中率)。"""
        marked = set(self._breakpoint_indices(stable))
        offsets, total = [], 0
        for i, seg in enumerate(stable):
            total += len(seg)
            if i in marked:
                offsets.append(total)
        return {"cached_prefix_chars": total, "cache_segments": len(stable),
                "cache_breakpoints": offsets}

    def _user_content(self, stable: Sequence[str], volatile_text: str,
                      ttl: str) -> list[dict]:
        """把 user message 拆成 [穩定段 × N, 本輪 volatile] 的 content blocks。

        一段一個 block 是刻意的 —— 快取比對以 block 邊界為單位, 段的切法見
        _data_block。breakpoint 只下在 _breakpoint_indices() 選出的那幾段。
        """
        marked = set(self._breakpoint_indices(stable))
        blocks: list[dict] = []
        for i, seg in enumerate(stable):
            b = {"type": "text", "text": seg}
            if i in marked:
                b["cache_control"] = {"type": "ephemeral", "ttl": ttl}
            blocks.append(b)
        return blocks + [{"type": "text", "text": volatile_text}]

    def _system(self, ttl: str) -> list[dict]:
        """system: 決策規則 + 可用資源目錄 (跨 run 都不變) → 第一個 cache breakpoint。

        ttl 必須與 user message 的 breakpoint 相同 — 見 _next_cache_ttl 的說明。
        """
        return [
            {"type": "text", "text": self._RULES},
            {"type": "text", "text": "可用資源目錄:\n" + self._registry_context(),
             "cache_control": {"type": "ephemeral", "ttl": ttl}},
        ]

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
                        user_segments: tuple = (), stable: Sequence[str] = ()):
        """以 JSON 模式取得結構化決策 (刻意不用 messages.parse 的 grammar,
        因本環境 grammar 首次編譯常逾時且每次要等 ~60s; JSON 模式快且穩)。

        `stable` 是 `_data_block()` 產生的穩定前綴段落, 每段各成一個帶 cache_control
        的 content block; `volatile_text` (本輪指令 + schema + 重試提示) 放在其後,
        所以重試不會讓前綴失效。

        所有呼叫經 `privacy.egress` — payload 先過出口掃描, 命中即中止並落地稽核。
        """
        ctx = self._ctx(profile)
        client = self._get_client().with_options(timeout=240.0, max_retries=1)
        if self.guidance:
            volatile_text = (f"【使用者引導方向 (請優先納入考量)】\n{self.guidance}\n\n"
                             + volatile_text)
        ttl = self._next_cache_ttl()     # 每次請求算一次; 重試沿用同一個值
        system = self._system(ttl)
        system_text = "\n\n".join(b["text"] for b in system)
        sch = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        base = (volatile_text + "\n\n請只輸出一個符合下列 JSON schema 的 JSON 物件, "
                "不要任何多餘文字、不要 markdown code fence：\n" + sch)
        prompt, last = base, None
        for attempt in range(retries):
            rec = {"ts": time.time(), "label": label, "attempt": attempt,
                   "model": self.model, "system": system_text,
                   "prompt": "".join(stable) + prompt, "cache_ttl": ttl,
                   **self._cache_log_fields(stable)}
            try:
                # 注意: adaptive thinking 的思考 token 也計入 max_tokens。4096 曾多次
                # 被思考吃光導致正文空白或 JSON 截斷在字串中間 (pydantic EOF error),
                # 故給足額度; display=summarized 讓思考摘要可落地到 log 以利除錯。
                thinking = self._thinking()
                if thinking is None:
                    rec["thinking_disabled"] = True
                resp = egress_mod.create(
                    client, ctx=ctx.egress, label=label,
                    model=self.model, system=system,
                    messages=[{"role": "user",
                               "content": self._user_content(
                                   list(stable), prompt, ttl)}],
                    user_segments=user_segments,
                    max_tokens=16000, **self._extra(),
                    **({"thinking": thinking} if thinking else {}))
                text = "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", None) == "text").strip()
                think = "".join(getattr(b, "thinking", "") for b in resp.content
                                if getattr(b, "type", None) == "thinking")
                rec["response"] = text
                rec["stop_reason"] = resp.stop_reason
                rec.update(self._usage_fields(resp))
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
                if self._maybe_disable_thinking(e):
                    # 端點不吃 thinking 參數 (代理未透傳): 關掉後重試, prompt 不變。
                    # 這種錯誤是參數層的, 一定在第一次嘗試就發生, 所以重試額度夠。
                    rec["thinking_disabled"] = True
                    self._log_call({**rec, "note": "thinking 參數被端點拒絕, 已關閉重試"})
                    continue
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
            "\n\n實驗即將開始。在挑選 encoder 與組 Recipe 之前, 請判斷你還缺哪些"
              "會**實質改變決策**的資料特性。\n"
              "- 能由本地程式算出來的 → 放進 request_analysis;\n"
              "- 只有使用者知道的 (影像模態、拍攝部位、類別是否有序、標註可信度、"
              "有無臨床上不可接受的錯誤型態) → 放進 questions;\n"
              "- 都不缺就兩個都留空 (這是完全合理的答案)。\n"
              "問題請精簡, 一次最多 3 題, 且要能用選項回答。"
            + self._info_note(ctx),
            _InfoPlan, label="plan_information", profile=profile,
            stable=self._data_block(profile))
        info = self.last_info
        info.reason = d.reason or info.reason
        return info

    # ---- (1) 多 encoder ------------------------------------------------
    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]:
        n = 3
        ctx = self._ctx(profile)
        out: _EncoderSelection = self._messages_parse(
            f"\n\n請從可用 encoder 中選最多 {n} 個作為比較 (依資料特性排序, 最合適在前)。"
            + self._info_note(ctx),
            _EncoderSelection, label="select_encoders", profile=profile,
            stable=self._data_block(profile))
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
            f"\n\n已選 encoder: {encoder.model_key} (adaptation={encoder.adaptation})。\n"
              f"起始超參數 (preset={preset}, 由使用者設定, 你可沿用或調整):\n"
              f"{json.dumps(start_hp, ensure_ascii=False)}\n\n"
              f"請為此 encoder 組出訓練 Recipe (選 head/pooling/loss/regularizer/augmentation)。"
              f"超參欄位 (blr/layer_decay/drop_path/weight_decay/epochs/batch_size)："
              f"**只有在有明確理由要調整時才填該欄位；否則留 null 以沿用上面的起始值。"
              f"任何超參數的改變都必須在 hparam_reason 逐項說明原因 (為何調整、依據什麼)；"
              f"不要無理由直接改動——沒有原因的調整將被忽略、沿用起始值。**"
            + self._info_note(ctx),
            _RecipeDecision, label="compose_recipe", profile=profile,
            stable=self._data_block(profile))
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
        bt = self._pick_base(history, base)
        ctx = self._ctx(profile)
        d: _NextDecision = self._messages_parse(
            f"\n\nencoder: {encoder.model_key}。\n"
            f"本輪的變異基準 (樹搜尋 policy 依指標機率選出): {self._base_id(bt)}。\n"
            f"請決定: 對此基準 Recipe 做『一個』正交變異再試, 或放棄此分支 "
            f"(prune_branch), 或結束整個實驗 (stop)。"
            + self._info_note(ctx),
            _NextDecision, label="propose_next", profile=profile,
            stable=self._data_block(profile, history))
        next_recipe = None
        if not d.stop and not d.prune_branch and bt is not None:
            next_recipe = self._apply_mutation(bt.recipe, d)
            if next_recipe is not None:
                next_recipe.provenance["reason"] = d.reason
                next_recipe.provenance["advisor"] = "llm"
        return NextAction(stop=d.stop, prune_branch=d.prune_branch or
                          (not d.stop and next_recipe is None),
                          reason=d.reason,
                          mutation=d.mutation, next_recipe=next_recipe,
                          info=self.last_info)

    # ---- 收尾/搜尋中集成: 選成員 (docs/ensemble_design.md) ------------
    def propose_ensemble(self, profile: DatasetProfile,
                         history: list[TrialResult],
                         ensemble_cfg) -> Optional[EnsembleSpec]:
        """看 TrialFacts 選 ≥2 個夠強且多樣的成員組 ensemble; 不值得則回 None。

        ⚠ 資料圍欄: LLM 只看到假名 trial_id, 回傳假名; 本地以反查表還原成真實 id
        (白名單, 防幻覺)。實際機率平均在 agent/ensembler (資料平面) 執行。
        """
        done = [t for t in history if t.status == "done"
                and t.recipe.encoder.model_key != "ensemble"]
        if len(done) < ensemble_cfg.min_members:
            return None
        ctx = self._ctx(profile)
        d: _EnsemblePick = self._messages_parse(
            f"\n\n請判斷是否值得把多個模型組成 ensemble (機率軟投票) 以提升下游效能。\n"
              f"- 選 {ensemble_cfg.min_members}–{ensemble_cfg.max_members} 個**夠強且多樣**"
              f"的成員 (不同 encoder / 不同 domain 的預測較不相關, 集成增益更大; "
              f"明顯落後的弱模型會拖累, 不要納入)。\n"
              f"- method: equal(等權) / val_weighted(依 val 求權重) / stacking(val 上訓練"
              f" meta-learner); 成員多且各有所長時 val_weighted 或 stacking 通常較好。\n"
              f"- member_trial_ids 請填上面清單中的**假名 trial_id**; 只有 status=done 的可選。\n"
              f"- 若成員不足 2 個, 或彼此高度相似 (集成沒意義), do_ensemble=false。"
            + self._info_note(ctx),
            _EnsemblePick, label="propose_ensemble", profile=profile,
            stable=self._data_block(profile, history))
        if not d.do_ensemble:
            return None
        real_ids = redact.resolve_trial_ids(
            d.member_trial_ids, done, ctx.alias)[: ensemble_cfg.max_members]
        if len(real_ids) < ensemble_cfg.min_members:
            return None
        return EnsembleSpec(member_trial_ids=real_ids, method=d.method,
                            rationale=d.rationale)

    # ---- 報告用: 把最佳 recipe 的譜系寫成一段白話說明 --------------------
    def narrate_best(self, profile: DatasetProfile, trial: TrialResult) -> str:
        """用 2–3 句白話說明最佳 recipe 的組成與勝出原因 (供 report 顯示)。失敗回空。"""
        try:
            ctx = self._ctx(profile)
            facts = redact.trial_facts(trial, ctx.alias).model_dump()
            recipe = redact.scrub(trial.recipe.model_dump(), ctx.alias)
            prompt = (
                "以下是本次搜尋選出的最佳 trial（已假名化）。請用 2–3 句**白話中文**說明："
                "這個 recipe 由哪個 encoder + 什麼 head／loss／regularizer／關鍵超參組成，"
                "以及它為何是最佳（若有變異或超參調整，帶到原因）。只輸出這段說明，"
                "不要列點、不要 JSON、不要提假名 ID。\n\n"
                f"最佳 trial 事實：\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
                f"recipe：\n{json.dumps(recipe, ensure_ascii=False, indent=2)}")
            client = self._get_client().with_options(timeout=120.0, max_retries=1)
            resp = egress_mod.create(
                client, ctx=ctx.egress, label="narrate_best", model=self.model,
                system=[{"type": "text", "text": self._RULES}],
                messages=[{"role": "user", "content": prompt}],
                user_segments=(), max_tokens=800, **self._extra(),
                **({"thinking": t} if (t := self._thinking()) else {}))
            return "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text").strip()
        except Exception:
            return ""

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
                f"\n\n目前的解答樹 (每個節點的 allowed 欄位 = 這個節點現在允許的階段):\n"
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
                user_segments=segs, stable=self._data_block(profile))
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
            # 注意順序: 隨輪次變動的東西 (目前 encoder、變異基準、討論) 一律放在
            # stable (facts + trials + 使用者事實) **之後**, 否則整段快取前綴會失效。
            f"\n\n欄位說明: primary_score / metrics 是 **test set** 成績 (訓練結束後用 val "
              f"最佳 checkpoint 在 test 上重跑一次); train_loss_curve / val_loss_curve / "
              f"val_score_curve 則是訓練期間逐 epoch 的 train/val，長度 = 實際跑完的 epoch 數，"
              f"不含最終 test。兩者是不同 split，val 通常高於 test，落差本身不代表有問題。\n\n"
              f"目前 encoder: {encoder.model_key}。\n"
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
              f"(3) 納入使用者討論, 決定對『變異基準』Recipe 做『一個』正交變異再試; "
              f"若這個基準節點本身走不通 (例如凍結特徵無訊號、權重載入不完整), "
              f"設 prune_branch=true 放棄『這條分支』即可 — 實驗會自動改從樹上其他節點"
              f"繼續, 不要因此設 stop; stop=true 專門保留給『所有分支都已收斂 / 使用者"
              f"要求停止 / 再跑任何 trial 都不值得』的情況, 它會結束整個實驗。"
              f"討論中標有【QA 結論】的訊息"
              f"是問答 agent 先前依實驗資料對使用者提問得出的結論 — 除非與最新數據"
              f"矛盾, 本輪決策應優先採納最近的一則; "
            + edit_note + self._info_note(ctx),
            _ReviewDecision, label="review_and_decide", profile=profile,
            user_segments=segs, stable=self._data_block(profile, history))
        next_recipe = None
        if not d.stop and not d.prune_branch and bt is not None:
            next_recipe = self._apply_mutation(
                bt.recipe, d, workspace_dir=workspace_dir,
                tag=f"improve_t{len(history)}")
            if next_recipe is not None:
                next_recipe.provenance["reason"] = d.reason
                next_recipe.provenance["advisor"] = "llm"
        return NextAction(stop=d.stop, prune_branch=d.prune_branch or
                          (not d.stop and next_recipe is None),
                          reason=d.reason,
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
        talk = redact.discussion_facts(discussion, alias, limit=12)
        segs = tuple(redact.user_segments(discussion, alias, limit=12)) + (question,)
        tree_txt = (json.dumps(redact.scrub(tree, alias), ensure_ascii=False)
                    if tree else "（尚無解答樹）")
        # 穩定前綴 (facts + trials) 與決策層逐字相同 → 共用同一筆 prompt cache;
        # 角色說明與本輪問題一律放在其後。
        stable = self._data_block(profile, history)
        volatile = (
            f"\n\n你現在的角色是**實驗問答助理** (獨立於決策層): 使用者剛在討論頻道"
            f"提出問題或方向, 請立即依上面的實驗資料回答, 不要做任何 Recipe 決策。\n\n"
            + f"解答樹 (AIDE 式搜尋; stage/parent/metric):\n{tree_txt}\n\n"
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
        ttl = self._next_cache_ttl()
        system = self._system(ttl)       # 與決策層同前綴 → 命中同一筆快取
        rec = {"ts": time.time(), "label": "qa_answer", "model": self.model,
               "system": "\n\n".join(b["text"] for b in system),
               "prompt": "".join(stable) + volatile, "cache_ttl": ttl,
               **self._cache_log_fields(stable)}
        try:
            text, resp = egress_mod.stream_text(
                client, ctx=ctx.egress, label="qa_answer", model=self.model,
                system=system,
                messages=[{"role": "user",
                           "content": self._user_content(stable, volatile, ttl)}],
                on_delta=on_delta, user_segments=segs,
                max_tokens=16000, **self._extra(),
                **({"thinking": t} if (t := self._thinking()) else {}))
            text = (text or "").strip()
            rec["response"] = text
            rec["stop_reason"] = resp.stop_reason
            rec.update(self._usage_fields(resp))
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
            f"\n\nencoder: {encoder.model_key}。以下 trial 訓練失敗:\n"
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
            _DebugDecision, label="propose_debug", profile=profile,
            stable=self._data_block(profile))
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
