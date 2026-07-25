"""TaskTemplate 目錄 — 下游任務起點 (設計文件 §5.8, P6).

每個 TaskTemplate = 針對某類下游情境的:
  - 預設 Recipe 骨架 (head/pooling/loss/aug + 超參 preset)
  - 相容元件白名單 (Advisor 只在此範圍內選/變異)
  - 預設 EvalConfig (primary_metric / report_metrics / aggregation)

v1 落地 `fundus_classification` (現有分類流程)。另附 `regression` 骨架以走通
「不只分類」的抽象 (head/loss/metric 隨 Task 切換); 其實體訓練待 main_finetune
的 regression head/loss 接上 (元件目錄已預留 supported=False)。

Advisor.suggest_task_templates() 依 DatasetProfile 提示候選起點供選擇。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import DatasetProfile, EvalConfig


class TaskTemplate(BaseModel):
    key: str
    task_type: str                              # classification / regression / segmentation
    description: str = ""
    default_preset: str = "default"
    default_head: str = "linear"
    default_pooling: str = "global_pool"
    default_loss: str = "cross_entropy"
    default_augmentation: str = "timm_randaug"
    # 相容元件白名單 (registry keys); Advisor/RecipeBuilder 據此限制選擇
    allowed_heads: list[str] = Field(default_factory=lambda: ["linear", "mlp"])
    allowed_losses: list[str] = Field(default_factory=lambda: ["cross_entropy"])
    allowed_regularizers: list[str] = Field(
        default_factory=lambda: ["drop_path", "weight_decay", "layer_decay",
                                 "label_smoothing", "mixup", "cutmix"])
    eval: EvalConfig = Field(default_factory=EvalConfig)
    ready: bool = True                          # 是否已可端到端訓練 (regression 待 main_finetune)


_CATALOG: dict[str, TaskTemplate] = {
    "fundus_classification": TaskTemplate(
        key="fundus_classification",
        task_type="classification",
        description="眼底影像單標籤多類分類 (現有 repo 流程落地)。",
        default_loss="cross_entropy",
        allowed_heads=["linear", "mlp"],
        allowed_losses=["cross_entropy", "weighted_ce", "focal"],
        eval=EvalConfig(primary_metric="score",
                        report_metrics=["accuracy", "f1", "roc_auc", "kappa"]),
        ready=True,
    ),
    "regression": TaskTemplate(
        key="regression",
        task_type="regression",
        description="連續值回歸 (如疾病嚴重度分數)。走通『不只分類』抽象; "
                    "端到端訓練待 main_finetune 接上 regression head/mse。",
        default_head="mlp",
        default_loss="mse",
        allowed_heads=["mlp", "regression"],
        allowed_losses=["mse"],
        allowed_regularizers=["drop_path", "weight_decay", "layer_decay"],
        eval=EvalConfig(primary_metric="r2", report_metrics=["mae", "r2"],
                        aggregation="mean_std"),
        ready=False,
    ),
}


def get(key: str) -> TaskTemplate:
    if key not in _CATALOG:
        raise KeyError(f"未知 TaskTemplate: {key}. 可用: {list(_CATALOG)}")
    return _CATALOG[key]


def all_templates() -> list[TaskTemplate]:
    return list(_CATALOG.values())


def suggest(profile: DatasetProfile) -> list[TaskTemplate]:
    """依 DatasetProfile 提示候選下游任務起點 (最合適在前)。"""
    out: list[TaskTemplate] = []
    if profile.task_type == "classification":
        out.append(_CATALOG["fundus_classification"])
    elif profile.task_type == "regression":
        out.append(_CATALOG["regression"])
    else:
        # 未知型態: 全部列出供選擇
        out = all_templates()
    # 其餘作為次要候選 (供「提示不同起點」)
    for t in all_templates():
        if t not in out:
            out.append(t)
    return out
