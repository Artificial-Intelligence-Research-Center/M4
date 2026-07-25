"""EncoderRegistry — 可用 encoder 目錄 (設計文件 §5.3).

新增 encoder = 新增一筆 EncoderCard。model/model_arch 與 main_finetune.py
的 get_model_info / models_vit 對應。available=False 表示權重 gated / 尚未取得。
"""
from __future__ import annotations

import os

from .schemas import EncoderCard

# 專案根目錄 (agent/ 的上一層)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


# P0: 先登錄本機可取得的免費 encoder + 標記 gated 的 RETFound (available=False)
_CARDS: list[EncoderCard] = [
    EncoderCard(
        model_key="dinov2_vitl14",
        model="Dinov2", model_arch="dinov2_vitl14",
        weight="baseline_models/dinov2_vitl14_pretrain.pth",
        embed_dim=1024, patch_size=14, domain="natural",
        notes="實際權重由 timm pretrained=True 於執行時載入; 本地檔僅供探索。",
    ),
    EncoderCard(
        model_key="mae_pretrain_vit_large",
        model="MAE", model_arch="MAE",
        weight="baseline_models/mae_pretrain_vit_large.pth",
        embed_dim=1024, patch_size=16, domain="natural",
    ),
    EncoderCard(
        model_key="vit_large_patch16_224",
        model="SL_VIT", model_arch="SL_VIT",
        weight="baseline_models/vit_large_patch16_224.pth",
        embed_dim=1024, patch_size=16, domain="natural",
        notes="HF ViT 權重 key 與 timm ViT 不完全對齊, 可能載入不完整 (見 README 註記)。",
    ),
    # gated DAP 模型 (需 HF token / 存取核准) — 目前不可用, 列出供 Advisor 參考
    EncoderCard(
        model_key="RETFound_dinov2_meh",
        model="RETFound_dinov2", model_arch="retfound_dinov2",
        weight="RETFound_dinov2_meh", embed_dim=1024, patch_size=14,
        domain="medical_dap", available=False,
        notes="HF YukunZhou/* gated; 取得後 main_finetune 會自動下載。",
    ),
]

_BY_KEY = {c.model_key: c for c in _CARDS}


def all_cards(include_unavailable: bool = False) -> list[EncoderCard]:
    return [c for c in _CARDS if include_unavailable or c.available]


def available_cards() -> list[EncoderCard]:
    """僅回傳本機權重確實存在的 encoder。"""
    out = []
    for c in _CARDS:
        if not c.available:
            continue
        # HF id (非本地路徑) 無法在此檢查, 直接視為可用
        if os.sep in c.weight or c.weight.endswith((".pth", ".pt")):
            if not os.path.exists(_abs(c.weight)):
                continue
        out.append(c)
    return out


def get(model_key: str) -> EncoderCard:
    if model_key not in _BY_KEY:
        raise KeyError(f"未知 encoder: {model_key}. 可用: {list(_BY_KEY)}")
    return _BY_KEY[model_key]


def weight_path(card: EncoderCard) -> str:
    """回傳給 main_finetune --finetune 的值 (本地檔轉絕對路徑, HF id 原樣)。"""
    if card.weight.endswith((".pth", ".pt")) or os.sep in card.weight:
        return _abs(card.weight)
    return card.weight
