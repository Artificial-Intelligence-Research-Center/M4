"""AgentConfig — 設定檔契約與載入 (設計文件 §7).

對應 YAML:

    data_path: ./data/5_fold_PAPILA/PAPILA_seed42_fold0
    task: { type: classification }
    task_template: fundus_classification
    eval:
      primary_metric: auroc
      report_metrics: [accuracy, f1, auroc, kappa]
      aggregation: mean_std
    loop:
      encoders_per_run: 3
      trials_per_encoder: 4
      max_trials: 12
      patience: 2
      num_drafts: 2
      debug_prob: 0.5
      max_debug_depth: 2
    advisor: { type: llm, model: claude-opus-4-8, allow_code_edit: false }
    budget: { max_wall_clock_min: null }
    privacy: { mode: strict }     # 資料圍欄; 見 docs/data_firewall_design.md

CLI: python -m agent.auto_finetune --config config.yaml
"""
from __future__ import annotations

import os
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .schemas import EvalConfig


class LoopConfig(BaseModel):
    """迴圈與停止條件 (設計文件 §5.7 StopPolicy 聯集)。

    StopPolicy = max_trials 硬上限 ∪ 時間預算 ∪ 使用者/Advisor 停止 ∪
    「已完成 ≥ min_trials 且連續 patience 輪『改良嘗試』(improve/resume) 無提升」。
    draft/debug 是探索/修復, 不計入 patience (但有提升照樣歸零)。"""
    encoders_per_run: int = 3      # (已停用) draft 輪替的 encoder 上限改由 num_drafts 決定
    trials_per_encoder: int = 1    # (已停用) 舊的每 encoder 輪數; 改由全域樹搜尋控制
    max_trials: int = 12           # 全域 trial 總數上限 (= 樹搜尋總輪數)
    patience: int = 4              # 連續 patience 輪 improve/resume 無提升才停
    min_trials: int = 6            # 至少完成這麼多輪才允許 patience 停止 (保護探索期)
    min_delta: float = 0.0         # 視為「有提升」的最小 primary 增幅
    # 全域樹搜尋 (參考 aideml agent.search.*): 先開滿 draft → 機率性 debug →
    # 依指標機率抽節點 improve (單一解答樹, draft 跨 encoder)
    num_drafts: int = 3            # 起手 draft 數 (輪流各 encoder; 同 encoder 換 preset)
    debug_prob: float = 0.5        # 每輪以此機率優先除錯 buggy leaf
    max_debug_depth: int = 2       # 連續除錯鏈上限 (超過就放棄該分支)
    improve_temperature: float = 0.2  # 依指標抽樣的 softmax 溫度; <=0 = greedy 只選最佳
    # policy 選完節點後, 讓決策層 (Advisor) 有改選的機會 — 討論/【QA 結論】要求
    # 換 encoder、開新 draft、回頭改某個節點時, 這是唯一能落實的地方。
    # heuristic advisor 一律沿用 policy; 不合法的改選也會被拒絕並沿用 policy。
    select_override: bool = True
    # 繼續訓練策略: 選中節點的 curve 未收斂時, 從其 checkpoint 續訓再多跑幾個 epoch
    resume_unconverged: bool = True  # 開/關此策略
    resume_epochs: int = 20          # 每次續訓多跑的 epoch 數
    max_resumes: int = 2             # 同一分支連續續訓上限 (仍未收斂就放棄續訓)
    resume_prob: float = 0.5         # 符合續訓條件時以此機率選 resume (其餘落到 improve)
    max_failed_resumes: int = 2      # 全 run 連續這麼多次 resume 無提升 → 本 run 停用 resume


class AdvisorConfig(BaseModel):
    type: Literal["heuristic", "llm", "skill"] = "llm"   # 預設用 LLM 決策 (無 SDK/金鑰時各方法自動退回 heuristic)
    model: str = "claude-opus-4-8"
    preset: str = "default"        # HeuristicAdvisor 冷啟動超參起點
    guidance: str = ""             # 使用者引導方向 (LLMAdvisor 會納入決策 prompt)
    # 允許 LLM 修改訓練程式 (debug 階段): 修改版放 run 目錄下 (code_workspace),
    # 原始程式不動。生成工作時可勾選。
    allow_code_edit: bool = False


class BudgetConfig(BaseModel):
    max_wall_clock_min: Optional[float] = None
    max_llm_tokens: Optional[int] = None


class GpuOptConfig(BaseModel):
    """GPU 利用率最佳化: 每個 trial 背景取樣 nvidia-smi, 利用率太低時對後續
    衍生的 recipe 採取措施 (記憶體有餘裕→加大 batch; 否則→增加 dataloader worker)。"""
    optimize: bool = True          # 開/關 (取樣照做, 只影響是否自動調整)
    util_target: float = 60.0      # 平均利用率低於此 (%) 視為太低
    sample_interval_s: float = 5.0  # 取樣間隔 (秒)
    mem_target: float = 0.85       # 加大 batch 後預估記憶體峰值不可超過總量的此比例
    max_batch_size: int = 256      # batch 上限
    max_num_workers: int = 16      # dataloader worker 上限
    # 預縮圖磁碟快取 (高解析度資料集的 decode 瓶頸): run 起手依 image_size_stats 決定
    cache_auto: bool = True        # 原圖中位短邊 ≥ 1.5×cache_short_side 時自動啟用
    cache_short_side: int = 512    # 快取短邊 (~2×input_size, 留 RandomResizedCrop 餘裕)


class EnsembleConfig(BaseModel):
    """收尾集成 (docs/ensemble_design.md 方案 A)。

    樹搜尋停止後, 把多個已訓練 trial 以『機率層級軟投票』組成 ensemble。
    重用各 trial 已落地的 predictions_*.csv, 不重訓、不需 GPU。勝過最佳單模型
    才採用為交付。權重最佳化一律在 val 上進行 (test 只報告), 杜絕洩漏。"""
    enabled: bool = True
    # equal=等權軟投票; val_weighted=依 val 求權重; stacking=val 上訓練 meta-learner
    method: Literal["equal", "val_weighted", "stacking"] = "val_weighted"
    # 成員選擇是否用 LLM (獨立於 advisor.type): False=一律規則式選擇 (heuristic);
    # True=用 LLM 依 TrialFacts 選成員 (主 advisor 非 LLM 時會另建專用 LLMAdvisor;
    # 環境不可用則自動退回 heuristic)。見 docs/ensemble_design.md。
    llm_select: bool = False
    min_members: int = 2           # 至少幾個成員才組 ensemble
    max_members: int = 4           # 最多納入幾個成員 (過多會攤薄強成員)
    member_delta: float = 0.05     # 成員門檻: primary ≥ 最佳單模型 − 此值
    require_diverse_encoders: bool = True  # 優先涵蓋不同 encoder (去相關)
    # 方案 B: 搜尋『中』的 ensemble stage — 單模型改良進入平坦期時, 中途即組 ensemble,
    # 並隨模型池成長重複嘗試 (結果獨立追蹤, 不干擾單模型樹搜尋的 best/parent 機制)。
    in_search: bool = False        # 開/關方案 B (預設只在收尾集成 = 方案 A)
    search_patience: int = 2       # 連續無提升的改良輪數達此值 → 觸發一次搜尋中集成
    max_search_ensembles: int = 3  # 全 run 搜尋中集成次數上限 (避免每輪重跑)


class PrivacyConfig(BaseModel):
    """資料圍欄 (docs/data_firewall_design.md §9)。

    `mode` 是巨集: 載入時會覆寫個別欄位, 讓「寫了 strict 卻同時開 code_edit」
    這種組合在型別層就不可能成立。
    """
    mode: Literal["strict", "standard", "off"] = "strict"
    dataset_alias: bool = True              # 資料集路徑/名稱以假名替代
    class_names: Literal["hashed", "user_approved", "plain"] = "hashed"
    revealed_classes: list[str] = Field(default_factory=list)  # user_approved 時使用者勾選的真實類別名
    log_feedback: Literal["none", "structured", "raw"] = "structured"
    allow_free_text_questions: bool = False  # 是否允許 LLM 向使用者要自由文字
    ask_user_when_unsure: bool = True        # 允許 LLM 提出問題請使用者回答
    egress_audit: bool = True
    guard_on_user_input: Literal["warn", "block", "off"] = "warn"
    salt_file: Optional[str] = None          # 預設 ~/.medclaw/privacy_salt
    expensive_analyzers_need_consent: bool = True
    scan_filenames: bool = True              # 出口掃描是否納入抽樣檔名主幹


class AgentConfig(BaseModel):
    data_path: str
    task: dict = Field(default_factory=lambda: {"type": "classification"})
    task_template: str = "fundus_classification"
    experiment_name: Optional[str] = None
    eval: EvalConfig = Field(default_factory=EvalConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    gpu: GpuOptConfig = Field(default_factory=GpuOptConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    device: int = 0
    dry_run: bool = False
    stream_logs: bool = True   # True: 子程序輸出即時顯示於 console + 寫檔; False: 只寫檔

    @model_validator(mode="after")
    def _apply_privacy_mode(self) -> "AgentConfig":
        """mode 巨集: strict 強制最嚴; standard 只夾住會造成洩漏的組合; off 不干預。

        關鍵在 `advisor.allow_code_edit` — 允許 LLM 改訓練程式 + 把 log 回饋給 LLM,
        兩者合起來就是一條完整的資料讀取通道 (設計文件 §8)。strict 直接拿掉寫入能力;
        standard 保留寫入但強制切斷 raw log 回讀。
        """
        p = self.privacy
        if p.mode == "strict":
            p.dataset_alias = True
            p.class_names = "hashed"
            p.log_feedback = "structured" if p.log_feedback != "none" else "none"
            p.allow_free_text_questions = False
            self.advisor.allow_code_edit = False
        elif p.mode == "standard":
            p.dataset_alias = True
            if p.class_names == "plain":
                p.class_names = "user_approved"
            if p.log_feedback == "raw":
                p.log_feedback = "structured"
        if p.class_names != "user_approved":
            p.revealed_classes = []
        return self

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        with open(path, encoding="utf8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def dump_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f,
                           allow_unicode=True, sort_keys=False)
