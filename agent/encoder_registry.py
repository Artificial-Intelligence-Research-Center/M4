"""EncoderRegistry — 可用 encoder 目錄 (設計文件 §5.3)。

實作 docs/model_registry_design.md：**一個 model = 一個自描述目錄**

    baseline_models/
      <model_key>/            # 目錄名即 model_key
        model.yaml            # manifest (EncoderCard 欄位)
        weights.pth           # 權重 (本地權重時)

新增 encoder = 新增一個目錄, 不必改任何 Python。registry 由掃描目錄產生,
對外介面 (all_cards / available_cards / get / weight_path) 維持不變。

載入採 fail-soft: 個別 manifest 壞掉只跳過該 model 並記 warning (見 warnings()),
不讓單一壞檔擋掉整個 registry。baseline_models/ 不存在或沒有任何 model.yaml
時, 回退到內建 _BUILTIN_CARDS, 確保未遷移的環境照常運作。
"""
from __future__ import annotations

import os
import re

import yaml
from pydantic import ValidationError

from .schemas import EncoderCard

# 專案根目錄 (agent/ 的上一層)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(_ROOT, "baseline_models")
MANIFEST = "model.yaml"

# 架構家族白名單 (§5.2) — 對照 main_finetune.py 的建模與載權重分支。
# registry 驗證與 UI 表單下拉共用這一份, 避免兩處不同步。
ARCH_WHITELIST: tuple[str, ...] = (
    "Dinov2", "Dinov3", "MAE", "SL_VIT",
    "RETFound_mae", "RETFound_dinov2", "GastroNet", "Pixio",
)

_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# 沒有任何 model.yaml 目錄時的回退卡片 (weight 仍為 repo 根相對路徑, §9)
_BUILTIN_CARDS: list[EncoderCard] = [
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

# 掃描結果快取: (cards, warnings)。None = 尚未載入; reload() 清掉。
_CACHE: tuple[list[EncoderCard], list[str]] | None = None


# ---------------------------------------------------------------------------
# 權重路徑解析 (§3.2)
# ---------------------------------------------------------------------------
def _is_local_weight(weight: str) -> bool:
    """本地檔案 (而非 HF id)? — 含路徑分隔符或以 .pth/.pt 結尾。"""
    return weight.endswith((".pth", ".pt")) or "/" in weight or os.sep in weight


def weight_path(card: EncoderCard) -> str:
    """回傳給 main_finetune --finetune 的值。

    1. HF id (無路徑分隔符且非 .pth/.pt)  → 原樣回傳, 交給 HF 下載分支
    2. 本目錄相對檔名 (weights.pth)       → <model_dir>/weights.pth  (新制)
    3. 絕對路徑 / repo 根相對路徑          → 原樣 / 接 repo 根       (舊制相容)
    """
    w = card.weight
    if not _is_local_weight(w):
        return w
    if os.path.isabs(w):
        return w
    if "/" in w or os.sep in w:                 # 舊制: 相對 repo 根
        return os.path.join(_ROOT, w)
    base = card.model_dir or _ROOT              # 新制: 相對 model 目錄
    return os.path.join(base, w)


def weight_exists(card: EncoderCard) -> bool:
    """本地權重是否存在 (HF id 無法在此檢查, 視為存在)。"""
    return not _is_local_weight(card.weight) or os.path.exists(weight_path(card))


# ---------------------------------------------------------------------------
# 掃描載入 (§4.1)
# ---------------------------------------------------------------------------
def _load_manifest(model_dir: str) -> tuple[EncoderCard | None, list[str]]:
    """讀一個目錄的 model.yaml → (card, warnings)。壞檔回 (None, warnings)。"""
    key = os.path.basename(model_dir)
    path = os.path.join(model_dir, MANIFEST)
    warn: list[str] = []

    if not _KEY_RE.match(key):
        return None, [f"{key}/: 目錄名含不合法字元 (只允許 [A-Za-z0-9_.-])"]

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        return None, [f"{key}/{MANIFEST}: 讀取失敗 — {e}"]

    if not isinstance(raw, dict):
        return None, [f"{key}/{MANIFEST}: 內容不是 YAML 物件"]

    declared = raw.get("model_key")
    if declared and declared != key:
        return None, [f"{key}/{MANIFEST}: model_key '{declared}' 與目錄名不符"]
    raw = {**raw, "model_key": key, "model_dir": model_dir}

    try:
        card = EncoderCard.model_validate(raw)
    except ValidationError as e:
        return None, [f"{key}/{MANIFEST}: 欄位驗證失敗 — {e.error_count()} 個問題: "
                      + "; ".join(f"{'.'.join(str(x) for x in d['loc'])}: {d['msg']}"
                                  for d in e.errors()[:3])]

    if card.model not in ARCH_WHITELIST:
        return None, [f"{key}/{MANIFEST}: 未知架構家族 '{card.model}' "
                      f"(可用: {', '.join(ARCH_WHITELIST)}) — "
                      "新架構需先在 models_vit.py / main_finetune.py 註冊"]

    # 權重缺失不擋載入, 只警告; available_cards() 會自動排除 (§5.3)
    if card.available and not weight_exists(card):
        warn.append(f"{key}/: 權重缺失 — 找不到 {weight_path(card)}")

    return card, warn


def _scan() -> tuple[list[EncoderCard], list[str]]:
    cards: list[EncoderCard] = []
    warnings: list[str] = []
    seen: set[str] = set()

    try:
        names = sorted(os.listdir(MODELS_DIR))
    except OSError:
        names = []

    for n in names:
        # 以 . 或 _ 開頭的目錄 (_archive / .trash) 一律略過, 供 UI 做暫存/軟刪除
        if n.startswith((".", "_")):
            continue
        d = os.path.join(MODELS_DIR, n)
        if not os.path.isfile(os.path.join(d, MANIFEST)):
            continue                      # 只有含 model.yaml 的目錄才算一個 model
        card, warn = _load_manifest(d)
        warnings += warn
        if card is None:
            continue
        if card.model_key in seen:        # 理論上目錄名唯一; 防大小寫/連結重複
            warnings.append(f"{card.model_key}: 重複的 model_key, 以先掃到者為準")
            continue
        seen.add(card.model_key)
        cards.append(card)

    if not cards:
        warnings.append(
            f"{MODELS_DIR} 下沒有任何 {MANIFEST} 目錄, 回退內建 encoder 目錄 "
            "(遷移: python -m agent.migrate_model_registry)")
        cards = list(_BUILTIN_CARDS)
    return cards, warnings


def _loaded() -> tuple[list[EncoderCard], list[str]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _scan()
    return _CACHE


def reload() -> list[EncoderCard]:
    """重新掃描 baseline_models/ (供 UI 增刪改後刷新, 不必重啟程序)。"""
    global _CACHE
    _CACHE = None
    return _loaded()[0]


def warnings() -> list[str]:
    """本次載入收集到的問題 (壞掉的 manifest / 缺失的權重), 供 UI 顯示。"""
    return list(_loaded()[1])


# ---------------------------------------------------------------------------
# 對外介面 (§4.2 — 簽章不變)
# ---------------------------------------------------------------------------
def all_cards(include_unavailable: bool = False) -> list[EncoderCard]:
    return [c for c in _loaded()[0] if include_unavailable or c.available]


def available_cards() -> list[EncoderCard]:
    """僅回傳本機權重確實存在的 encoder (HF id 視為可用)。"""
    return [c for c in _loaded()[0] if c.available and weight_exists(c)]


def get(model_key: str) -> EncoderCard:
    by_key = {c.model_key: c for c in _loaded()[0]}
    if model_key not in by_key:
        raise KeyError(f"未知 encoder: {model_key}. 可用: {list(by_key)}")
    return by_key[model_key]


def model_dir(model_key: str) -> str:
    """該 model 的目錄 (UI 定位檔案用); 內建回退卡片沒有目錄, 回空字串。"""
    return get(model_key).model_dir
