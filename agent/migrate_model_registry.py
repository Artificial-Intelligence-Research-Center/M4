"""一次性 migration: baseline_models/*.pth (平鋪) → baseline_models/<key>/ (目錄)。

見 docs/model_registry_design.md §8。權重用 os.rename 搬移 (同檔案系統零成本,
不複製 1.2 GB), manifest 內容取自 encoder_registry._BUILTIN_CARDS。

    python -m agent.migrate_model_registry --dry-run   # 只列出將要做的事
    python -m agent.migrate_model_registry             # 實際執行

可重複執行: 已經是目錄形式 (model.yaml 存在) 的 model 會被跳過。
"""
from __future__ import annotations

import argparse
import os

import yaml

from . import encoder_registry as reg

# 舊卡片 model_key → 目錄名。刻意沿用原 model_key 當目錄名 (目錄名即 key),
# 讓既有 runs/ 歷史與 --encoder 參數繼續有效。
_TARGET_DIR = {c.model_key: c.model_key for c in reg._BUILTIN_CARDS}


def _manifest(card) -> dict:
    """EncoderCard → model.yaml 內容 (model_key 省略, 以目錄名為準)。"""
    d = card.model_dump()
    d.pop("model_key", None)
    if reg._is_local_weight(card.weight):
        d["weight"] = "weights.pth"          # 新制: 相對 model 目錄
    return {k: v for k, v in d.items() if v != "" or k == "weight"}


def migrate(dry_run: bool = False) -> list[str]:
    actions: list[str] = []
    os.makedirs(reg.MODELS_DIR, exist_ok=True)

    for card in reg._BUILTIN_CARDS:
        dst_dir = os.path.join(reg.MODELS_DIR, _TARGET_DIR[card.model_key])
        manifest_path = os.path.join(dst_dir, reg.MANIFEST)
        if os.path.exists(manifest_path):
            actions.append(f"skip   {card.model_key}: 已存在 {reg.MANIFEST}")
            continue

        # 1) 搬權重 (本地權重且來源檔存在時)
        src = os.path.join(reg._ROOT, card.weight)
        moved = None
        if reg._is_local_weight(card.weight) and os.path.isfile(src):
            moved = os.path.join(dst_dir, "weights.pth")
            actions.append(f"move   {os.path.relpath(src, reg._ROOT)} → "
                           f"{os.path.relpath(moved, reg._ROOT)}")
        elif reg._is_local_weight(card.weight):
            actions.append(f"warn   {card.model_key}: 找不到權重 {card.weight}"
                           f" (仍會建立 {reg.MANIFEST})")
        actions.append(f"write  {os.path.relpath(manifest_path, reg._ROOT)}")

        if dry_run:
            continue
        os.makedirs(dst_dir, exist_ok=True)
        if moved:
            os.rename(src, moved)
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_manifest(card), f, allow_unicode=True,
                           sort_keys=False, default_flow_style=False)

    if not dry_run:
        reg.reload()
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="只列出將要做的事")
    args = ap.parse_args()

    for line in migrate(dry_run=args.dry_run):
        print(line)
    if args.dry_run:
        print("\n(dry-run: 沒有任何檔案被改動)")
    else:
        cards = reg.all_cards(include_unavailable=True)
        print(f"\n完成。registry 現有 {len(cards)} 個 model: "
              + ", ".join(c.model_key for c in cards))
        for w in reg.warnings():
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
