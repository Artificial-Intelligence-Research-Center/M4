"""ComponentRegistry — 可註冊、可組合的訓練元件目錄 (設計文件 §5.4, P3).

把 Recipe 的可組合元件 (head / regularizer / augmentation / loss) 集中登錄, 並描述
每個元件如何映射到 main_finetune.py 的 CLI 參數 (或標記為 in-process hook / 尚未接上)。
新增能力 = 新增一筆註冊, 不動核心 (開放擴充第一原則)。

RecipeBuilder 依此目錄:
  (a) 把 Recipe 元件轉成 main_finetune 參數;
  (b) 驗證元件是否在白名單內、是否已接上 (相容性檢查)。
"""
from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel


class ComponentCard(BaseModel):
    name: str
    category: str                 # head / regularizer / augmentation / loss / pooling
    supported: bool = True        # 是否已接上 main_finetune (False = schema 有、待接)
    notes: str = ""


# ---- Head -----------------------------------------------------------------
_HEADS = [
    ComponentCard(name="linear", category="head", notes="原本單層 Linear head。"),
    ComponentCard(name="mlp", category="head",
                  notes="多層 head; main_finetune --head_type mlp (P3)。"),
    ComponentCard(name="segmentation", category="head", supported=False,
                  notes="預留 (P6); 需 seg decoder + dice/ce。"),
    ComponentCard(name="regression", category="head", supported=False,
                  notes="預留 (P6); 需 mse + reg metric。"),
]

# ---- Pooling --------------------------------------------------------------
_POOLING = [
    ComponentCard(name="global_pool", category="pooling"),
    ComponentCard(name="cls_token", category="pooling"),
]

# ---- Regularizer (多數已是 main_finetune 旗標) ------------------------------
_REGULARIZERS = [
    ComponentCard(name="drop_path", category="regularizer", notes="--drop_path (走 hparams)。"),
    ComponentCard(name="weight_decay", category="regularizer", notes="--weight_decay (走 hparams)。"),
    ComponentCard(name="layer_decay", category="regularizer", notes="--layer_decay (走 hparams)。"),
    ComponentCard(name="label_smoothing", category="regularizer", notes="--smoothing。"),
    ComponentCard(name="mixup", category="regularizer", notes="--mixup (需 alpha, 預設 0.8)。"),
    ComponentCard(name="cutmix", category="regularizer", notes="--cutmix (需 alpha, 預設 1.0)。"),
    ComponentCard(name="ema", category="regularizer", supported=False, notes="預留。"),
    ComponentCard(name="r_drop", category="regularizer", supported=False, notes="預留。"),
]

# ---- Augmentation ---------------------------------------------------------
_AUGS = [
    ComponentCard(name="timm_randaug", category="augmentation",
                  notes="現況 build_transform 的 RandAug (--aa)。"),
    ComponentCard(name="resize_crop_normalize", category="augmentation", supported=False,
                  notes="預留: 最小 aug; 待 build_transform 組態化。"),
]

# ---- Loss -----------------------------------------------------------------
_LOSSES = [
    ComponentCard(name="cross_entropy", category="loss", notes="--loss cross_entropy (預設)。"),
    ComponentCard(name="weighted_ce", category="loss", notes="--loss weighted_ce (P3)。"),
    ComponentCard(name="focal", category="loss", notes="--loss focal --focal_gamma (P3)。"),
    ComponentCard(name="dice_ce", category="loss", supported=False, notes="預留 (seg)。"),
    ComponentCard(name="mse", category="loss", supported=False, notes="預留 (reg)。"),
]

_ALL: list[ComponentCard] = _HEADS + _POOLING + _REGULARIZERS + _AUGS + _LOSSES
_BY_NAME = {c.name: c for c in _ALL}


def get(name: str) -> Optional[ComponentCard]:
    return _BY_NAME.get(name)


def is_supported(name: str) -> bool:
    c = _BY_NAME.get(name)
    return bool(c and c.supported)


def by_category(category: str) -> list[ComponentCard]:
    return [c for c in _ALL if c.category == category]


# ---- Recipe 元件 → main_finetune CLI 映射 ----------------------------------
# 回傳 (額外 CLI 參數 list, 未接元件警告 list)。走 hparams 的 regularizer 不在此處理。
_REG_DEFAULT_ALPHA = {"mixup": 0.8, "cutmix": 1.0}


def regularizer_cli(refs) -> tuple[list[str], list[str]]:
    """把 recipe.regularizers 轉成 main_finetune 旗標 (label_smoothing/mixup/cutmix)。"""
    cli, unsupported = [], []
    for r in refs:
        card = _BY_NAME.get(r.name)
        if card is None or not card.supported:
            unsupported.append(f"regularizer={r.name}")
            continue
        if r.name == "label_smoothing":
            cli += ["--smoothing", str(r.params.get("smoothing", 0.1))]
        elif r.name in ("mixup", "cutmix"):
            alpha = r.params.get("alpha", _REG_DEFAULT_ALPHA[r.name])
            cli += [f"--{r.name}", str(alpha)]
        # drop_path/weight_decay/layer_decay 由 hparams 處理, 此處略過
    return cli, unsupported


def loss_cli(refs) -> tuple[list[str], list[str]]:
    """把 recipe.losses[0] 轉成 --loss (+ --focal_gamma)。"""
    if not refs:
        return [], []
    lo = refs[0]
    card = _BY_NAME.get(lo.name)
    if card is None or not card.supported:
        return [], [f"loss={lo.name}"]
    cli = ["--loss", lo.name]
    if lo.name == "focal":
        cli += ["--focal_gamma", str(lo.params.get("gamma", 2.0))]
    return cli, []


def head_cli(head) -> tuple[list[str], list[str]]:
    """把 recipe.heads[0] 轉成 --head_type (+ mlp 參數)。"""
    card = _BY_NAME.get(head.type)
    if card is None or not card.supported:
        return [], [f"head={head.type}"]
    if head.type == "linear":
        return ["--head_type", "linear"], []
    cli = ["--head_type", "mlp"]
    if head.hidden_dims:
        cli += ["--head_hidden_dims", *[str(d) for d in head.hidden_dims]]
    if head.dropout:
        cli += ["--head_dropout", str(head.dropout)]
    return cli, []
