"""模組間的資料契約 (Pydantic v2).

對應設計文件 §6。P0 只落地分類任務所需欄位, 但 Recipe/Head 等結構已為
未來 (segmentation/regression/multi-task) 預留擴充點。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 資料集分析
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
    modality_hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Encoder 目錄
# ---------------------------------------------------------------------------
class EncoderCard(BaseModel):
    model_key: str                    # registry key, 也用於 task_id
    model: str                        # main_finetune --model
    model_arch: str                   # main_finetune --model_arch
    weight: str                       # --finetune 檔案路徑 (本地) 或 HF id
    embed_dim: int = 1024
    patch_size: int = 16
    input_size: int = 224
    domain: Literal["natural", "medical_dap"] = "natural"
    available: bool = True            # 權重是否可在本機取得 (gated -> False)
    notes: str = ""


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
    aggregation: Literal["single", "mean_std"] = "single"


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


class NextAction(BaseModel):
    stop: bool
    reason: str = ""
    narrative: str = ""  # LLM 對目前結果的簡短檢視說明 (給使用者看的對話內容)
    mutation: Literal[
        "add_regularizer", "swap_head", "change_augmentation",
        "add_auxiliary_task", "adjust_hparams", "edit_code", "none",
    ] = "none"
    next_recipe: Optional[Recipe] = None
