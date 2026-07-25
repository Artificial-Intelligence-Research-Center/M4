"""報告產生 (設計文件 §9) — 各 encoder 最佳 recipe 比較 + 全域最佳。

輸出 runs/<experiment>/report.md。純程式, 讀 ledger / TrialResult。
"""
from __future__ import annotations

import os
from typing import Optional

from .config import AgentConfig
from .schemas import DatasetProfile, TrialResult


def _best_of(trials: list[TrialResult]) -> Optional[TrialResult]:
    done = [t for t in trials if t.status == "done"]
    return max(done, key=lambda t: t.primary_score) if done else None


def _fmt_metrics(m: dict, keys: list[str]) -> str:
    from .evaluator import _METRIC_ALIASES
    cells = []
    for k in keys:
        real = _METRIC_ALIASES.get(k, k)
        v = m.get(real, m.get(k))
        cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
    return " | ".join(cells)


def write_report(run_dir: str, profile: DatasetProfile, cfg: AgentConfig,
                 choices, per_encoder: dict, best: Optional[TrialResult],
                 dry_run: bool = False, fold_summary: Optional[dict] = None) -> str:
    keys = cfg.eval.report_metrics
    lines: list[str] = []
    lines.append(f"# 自動微調報告 — {os.path.basename(run_dir)}\n")
    if dry_run:
        lines.append("> ⚠️ dry_run：僅組指令未實際訓練，指標為空。\n")

    lines.append("## 資料集\n")
    lines.append(f"- root: `{profile.root}`")
    lines.append(f"- 任務: {profile.task_type}，{profile.num_classes} 類 "
                 f"{profile.class_names}")
    lines.append(f"- 樣本數: train={profile.n_train} / val={profile.n_val} "
                 f"/ test={profile.n_test}")
    lines.append(f"- 不平衡比: {profile.imbalance_ratio}，"
                 f"modality: {profile.modality_hint}\n")

    lines.append("## 各 encoder 最佳 Recipe 比較\n")
    lines.append(f"| encoder | adaptation | trials | primary({cfg.eval.primary_metric}) "
                 f"| {' | '.join(keys)} |")
    lines.append("| --- | --- | --- | --- | " + " | ".join("---" for _ in keys) + " |")
    for enc in choices:
        trials = per_encoder.get(enc.model_key, [])
        b = _best_of(trials)
        if b is not None:
            lines.append(
                f"| {enc.model_key} | {enc.adaptation} | {len(trials)} | "
                f"{b.primary_score:.4f} | {_fmt_metrics(b.metrics, keys)} |")
        else:
            status = trials[-1].status if trials else "—"
            lines.append(f"| {enc.model_key} | {enc.adaptation} | {len(trials)} | "
                         f"— ({status}) | " + " | ".join("—" for _ in keys) + " |")

    lines.append("\n## 全域最佳\n")
    if best is not None:
        lines.append(f"- **encoder**: `{best.recipe.encoder.model_key}` "
                     f"({best.recipe.encoder.adaptation})")
        lines.append(f"- **primary_score**: {best.primary_score:.4f}")
        lines.append(f"- **trial_id**: `{best.trial_id}`")
        lines.append(f"- **checkpoint**: `{best.ckpt_path}`")
        lines.append(f"- **recipe.provenance**: {best.recipe.provenance}")
    else:
        lines.append("- （無成功 trial）")

    if fold_summary:
        p = fold_summary["primary"]
        lines.append("\n## 多 fold 彙整 (全域最佳 Recipe)\n")
        lines.append(f"- encoder: `{fold_summary['encoder']}`，folds: {fold_summary['folds']}")
        lines.append(f"- **primary {cfg.eval.primary_metric}**: "
                     f"{p['mean']:.4f} ± {p['std']:.4f} (n={p['n']})")
        for k, mv in sorted(fold_summary["metrics"].items()):
            lines.append(f"  - {k}: {mv['mean']:.4f} ± {mv['std']:.4f}")

    lines.append(f"\n_共執行 {sum(len(v) for v in per_encoder.values())} trials，"
                 f"ledger: `{os.path.join(run_dir, 'ledger.jsonl')}`_\n")

    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    return path
