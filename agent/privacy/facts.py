"""對 LLM 開放的資料契約 (docs/data_firewall_design.md §4)。

**只有本檔定義的型別可以進入 prompt。** `DatasetProfile` / 原始 log / 原始
`provenance` 一律不得直接外送 — 必須先經 `redact` 轉成這裡的 facts。

設計原則: 欄位型別限定為數值 / 列舉 / 布林 / 上述型別的容器。自由字串只出現在
「由我們自己的程式產生」或「使用者親自輸入」的欄位, 且都會經過出口掃描。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 影像模態與部位: 由使用者在 UI 選擇 (不從路徑猜)
Modality = Literal[
    "fundus", "oct", "xray", "ct", "mri", "ultrasound",
    "dermoscopy", "pathology", "endoscopy", "other",
]
Anatomy = Literal[
    "eye", "chest", "brain", "skin", "gi", "breast", "bone",
    "abdomen", "other",
]

ErrorClass = Literal[
    "oom", "shape_mismatch", "nan_loss", "dataloader", "checkpoint",
    "cuda", "config", "unknown",
]


# ---------------------------------------------------------------------------
# 資料集
# ---------------------------------------------------------------------------
class ImageSizeFacts(BaseModel):
    n_sampled: int = 0
    width_min: Optional[int] = None
    width_median: Optional[int] = None
    width_max: Optional[int] = None
    height_min: Optional[int] = None
    height_median: Optional[int] = None
    height_max: Optional[int] = None


class DatasetFacts(BaseModel):
    """唯一允許進入 prompt 的資料集描述 (取代 DatasetProfile)。"""
    dataset_ref: str                      # 假名, 例 "DS-a1b2c3#f0"
    task_type: Literal["classification"] = "classification"
    num_classes: int
    class_labels: list[str] = Field(default_factory=list)   # 預設 C0..Cn
    class_counts: list[int] = Field(default_factory=list)   # 與 class_labels 同序
    class_ordinal: Optional[bool] = None   # 由使用者回答, 不猜
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    has_kfold: bool = False
    n_folds: Optional[int] = None
    imbalance_ratio: float = 1.0
    is_grayscale: bool = False
    image_size: ImageSizeFacts = Field(default_factory=ImageSizeFacts)
    modality: Optional[Modality] = None
    anatomy: Optional[Anatomy] = None


# ---------------------------------------------------------------------------
# Trial 歷史 (取代 llm_advisor._hist 的原始 dict)
# ---------------------------------------------------------------------------
class ProvenanceFacts(BaseModel):
    """Recipe 來歷的白名單視圖 — 原始 provenance 是自由 dict, 不得直接外送。"""
    template: Optional[str] = None
    preset: Optional[str] = None
    advisor: Optional[str] = None
    mutation: Optional[str] = None
    reason: str = ""
    hparam_reason: str = ""
    hparam_changes: dict = Field(default_factory=dict)
    mutated_from: dict = Field(default_factory=dict)
    gpu_opt: dict = Field(default_factory=dict)
    search: dict = Field(default_factory=dict)
    cache_resized: Optional[int] = None
    code_edits: dict = Field(default_factory=dict)


class TrialFacts(BaseModel):
    trial_id: str                          # 已假名化 (不含資料集名)
    status: str
    primary_score: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    epochs: int = 0
    provenance: ProvenanceFacts = Field(default_factory=ProvenanceFacts)
    train_loss_curve: list[float] = Field(default_factory=list)
    val_loss_curve: list[float] = Field(default_factory=list)
    val_score_curve: list[float] = Field(default_factory=list)
    gpu_stats: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 失敗診斷 (取代 raw log tail)
# ---------------------------------------------------------------------------
class ErrorFacts(BaseModel):
    """由 analyzers.error_extract 以『允許清單抽取』產生; 不含任何路徑或檔名。"""
    error_class: ErrorClass = "unknown"
    exc_type: Optional[str] = None        # 已知例外型別名; 未知一律 None
    message_template: str = ""            # 比對到的已知樣板, 非原文
    tensor_shapes: list[list[int]] = Field(default_factory=list)
    requested_mb: Optional[float] = None  # OOM: 嘗試配置量
    free_mb: Optional[float] = None       # OOM: 剩餘量
    at_epoch: Optional[int] = None
    at_step: Optional[int] = None
    n_lines_scanned: int = 0
    n_epochs_completed: int = 0
    last_train_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    truncated: bool = False               # 有無法歸類的內容被丟棄


# ---------------------------------------------------------------------------
# 分析器 (管道 A) — 詳細契約見 agent/analyzers/
# ---------------------------------------------------------------------------
class AnalysisFacts(BaseModel):
    """一個分析器的輸出 (已通過其 output_schema 驗證)。"""
    key: str
    ok: bool = True
    error: str = ""                       # 執行失敗時的簡短原因 (我們自己的訊息)
    result: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 詢問使用者 (管道 B)
# ---------------------------------------------------------------------------
class UserQuestion(BaseModel):
    key: str
    question: str
    kind: Literal["choice", "multi", "number", "bool", "short_text"] = "choice"
    options: list[str] = Field(default_factory=list)
    why: str = ""
    blocking: bool = False


class UserAnswer(BaseModel):
    key: str
    question: str = ""
    value: str = ""                       # 一律以字串保存 (multi 以 ", " 串接)
    ts: float = 0.0


class UserFacts(BaseModel):
    """使用者親自提供的資料特性 (runs/<run>/user_facts.json)。"""
    modality: Optional[Modality] = None
    anatomy: Optional[Anatomy] = None
    class_ordinal: Optional[bool] = None
    answers: list[UserAnswer] = Field(default_factory=list)

    def answer_for(self, key: str) -> Optional[str]:
        for a in reversed(self.answers):
            if a.key == key:
                return a.value
        return None

    def answered_keys(self) -> set[str]:
        return {a.key for a in self.answers}
