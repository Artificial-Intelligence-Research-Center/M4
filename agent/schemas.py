"""模組間的資料契約 (Pydantic v2).

對應設計文件 §6。P0 只落地分類任務所需欄位, 但 Recipe/Head 等結構已為
未來 (segmentation/regression/multi-task) 預留擴充點。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .privacy.facts import UserQuestion


# ---------------------------------------------------------------------------
# 資料集分析
#
# ⚠ DatasetProfile 屬於**資料平面**, 內含絕對路徑與真實類別名 —
#   **不得直接進入 prompt**。決策層一律使用 privacy.facts.DatasetFacts
#   (由 privacy.redact.to_facts 產生)。見 docs/data_firewall_design.md §4。
# ---------------------------------------------------------------------------
class DatasetProfile(BaseModel):
    root: str
    task_type: Literal["classification"] = "classification"
    num_classes: int
    class_names: list[str]
    class_counts: dict[str, int]
    n_train: int
    n_val: int
    n_test: int
    has_kfold: bool = False
    image_size_stats: dict = Field(default_factory=dict)  # min/max/median H,W
    is_grayscale: bool = False
    imbalance_ratio: float = 1.0  # 最多類 / 最少類
    # (已移除 modality_hint — 舊版由路徑關鍵字猜測, 既不準又等於把路徑內容送進決策。
    #  影像模態改由使用者在 UI 指定, 存於 runs/<run>/user_facts.json)


# ---------------------------------------------------------------------------
# Encoder 目錄
# ---------------------------------------------------------------------------
class EncoderCard(BaseModel):
    """一張 encoder 卡 = baseline_models/<model_key>/model.yaml 的內容。

    見 docs/model_registry_design.md §3。model_key 省略時由目錄名補上。
    """
    model_key: str = ""               # registry key (= 目錄名), 也用於 task_id
    model: str                        # main_finetune --model (架構家族)
    model_arch: str                   # main_finetune --model_arch
    weight: str                       # 目錄內權重檔名 / 路徑 / HF id
    embed_dim: int = 1024
    patch_size: int = 16
    input_size: int = 224
    domain: Literal["natural", "medical_dap"] = "natural"
    available: bool = True            # 權重是否可在本機取得 (gated -> False)
    notes: str = ""
    # 來源目錄 (由 registry 掃描時填入)。exclude=True → 不進 model_dump(),
    # 因此不會混進送給 LLM 的 registry context, 也不會被寫回 model.yaml。
    model_dir: str = Field(default="", exclude=True)


class EncoderChoice(BaseModel):
    model_key: str
    adaptation: Literal["finetune", "lp"] = "finetune"
    rationale: str = ""


# ---------------------------------------------------------------------------
# Recipe (可組合的訓練配方) — P0 為最小組合
# ---------------------------------------------------------------------------
class ComponentRef(BaseModel):
    """可組合元件的引用 + 參數 (regularizer / augmentation / loss ...)。"""
    name: str
    params: dict = Field(default_factory=dict)


class HeadSpec(BaseModel):
    type: Literal["linear", "mlp"] = "linear"
    output_dim: int
    hidden_dims: list[int] = Field(default_factory=list)
    dropout: float = 0.0
    target: str = "main"  # 多任務時對應的標的


class HyperParams(BaseModel):
    batch_size: int = 24
    epochs: int = 50
    blr: float = 5e-3
    layer_decay: float = 0.65
    drop_path: float = 0.2
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    input_size: int = 224
    accum_iter: int = 1
    num_workers: Optional[int] = None  # dataloader worker 數; None = main_finetune 預設 (10)
    # 預縮圖磁碟快取短邊 (高解析度資料集降 decode 成本); None = 不用快取
    cache_resized: Optional[int] = None


class Recipe(BaseModel):
    encoder: EncoderChoice
    task: dict = Field(default_factory=lambda: {"type": "classification"})
    heads: list[HeadSpec] = Field(default_factory=list)
    pooling: Literal["global_pool", "cls_token"] = "global_pool"
    regularizers: list[ComponentRef] = Field(default_factory=list)
    augmentation: Optional[ComponentRef] = None
    losses: list[ComponentRef] = Field(default_factory=list)
    hparams: HyperParams = Field(default_factory=HyperParams)
    provenance: dict = Field(default_factory=dict)  # 起點 template / 上一 trial 變異
    # LLM 修改程式時的隔離副本目錄 (run 目錄下, 見 code_workspace); None = 原始程式
    code_dir: Optional[str] = None
    # 繼續訓練策略 (樹搜尋 resume 階段): 從此 checkpoint 續訓 resume_epochs 個 epoch
    resume_from: Optional[str] = None
    resume_epochs: int = 0


# ---------------------------------------------------------------------------
# 評估設定 & Trial 結果
# ---------------------------------------------------------------------------
class EvalConfig(BaseModel):
    primary_metric: str = "score"  # 迴圈最佳化目標
    report_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "f1", "roc_auc", "kappa"]
    )
    # single: 只在單一 fold 搜尋; mean_std: 單一最佳 recipe 套到各 fold 重跑彙整;
    # per_fold: 每個 fold 各自跑一次完整搜尋, 獨立找出該 fold 的最佳 recipe。
    aggregation: Literal["single", "mean_std", "per_fold"] = "single"


class TrialResult(BaseModel):
    trial_id: str
    recipe: Recipe
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_score: float = 0.0
    ckpt_path: Optional[str] = None
    log_path: Optional[str] = None
    status: Literal["pending", "running", "done", "failed"] = "pending"
    message: str = ""
    epoch_curve: dict = Field(default_factory=dict)  # 逐 epoch train/val loss + score (供判斷 epochs)
    # 訓練期間的 GPU 取樣 (util_avg/util_median/mem_peak_mb/mem_total_mb/n_samples)
    gpu_stats: dict = Field(default_factory=dict)


class EnsembleSpec(BaseModel):
    """一次集成的『策略』— 由決策層產生 (metadata, 可進 LLM prompt)。

    ⚠ 資料圍欄: 決策層只選 trial_id 與方法, 不碰 predictions。實際的機率平均
    在資料平面 (agent/ensembler.py) 執行, 只有彙整指標回流。見 docs/ensemble_design.md。
    """
    member_trial_ids: list[str] = Field(default_factory=list)  # 選中的成員 (≥2)
    method: Literal["equal", "val_weighted", "stacking"] = "equal"
    weights: Optional[list[float]] = None    # method=val_weighted 時由本地在 val 上求得
    rationale: str = ""


class EnsembleResult(BaseModel):
    """集成結果 — 與 TrialResult 同構 (有 metrics/primary_score), 可同口徑比較。"""
    ensemble_id: str
    spec: EnsembleSpec
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_score: float = 0.0
    member_ckpts: list[str] = Field(default_factory=list)  # 各成員 checkpoint-best.pth
    pred_path: Optional[str] = None          # 落地的集成 predictions_test.csv
    n_samples: int = 0                        # 對齊後參與集成的樣本數
    status: Literal["done", "failed"] = "done"
    message: str = ""


class SearchOverride(BaseModel):
    """決策層對「樹搜尋 policy 已選出的節點」的覆寫 (§5.7)。

    policy 先依規則/機率選好 (stage, parent)，再把選擇連同整棵樹交給 Advisor 過目；
    Advisor 可維持原選擇 (override=False)，或改選別的節點/階段。LoopController 會
    驗證合法性 (節點存在、狀態允許該階段)，不合法就沿用 policy 的選擇。
    """
    override: bool = False
    stage: Literal["draft", "improve", "debug", "resume"] = "improve"
    parent_id: Optional[str] = None      # 目標節點 trial_id; stage=draft 時忽略
    # 以下只在 stage=draft 時有意義 (指定新起點要用哪個 encoder / 超參起點)
    encoder: Optional[str] = None
    adaptation: Optional[Literal["finetune", "lp"]] = None
    preset: Optional[str] = None
    reason: str = ""                     # 覆寫或維持原議的理由 (寫進討論給使用者看)


class InfoRequest(BaseModel):
    """決策層要求補充資料特性的兩條合法管道 (docs/data_firewall_design.md §5/§6)。

    LLM 看不到原始資料; 需要更多特性時只能 (a) 點名執行**已註冊**的分析器, 或
    (b) 提出問題請使用者親自回答。兩者以外沒有第三條路。
    """
    analyses: list[str] = Field(default_factory=list)     # analyzers 註冊表的 key
    questions: list[UserQuestion] = Field(default_factory=list)
    reason: str = ""


class NextAction(BaseModel):
    stop: bool
    reason: str = ""
    narrative: str = ""  # LLM 對目前結果的簡短檢視說明 (給使用者看的對話內容)
    mutation: Literal[
        "add_regularizer", "swap_head", "change_augmentation",
        "add_auxiliary_task", "adjust_hparams", "edit_code", "none",
    ] = "none"
    next_recipe: Optional[Recipe] = None
    # 決策同時要求補充資訊 (下一輪生效); 空值 = 不需要
    info: InfoRequest = Field(default_factory=InfoRequest)
