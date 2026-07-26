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
                 dry_run: bool = False, fold_summary: Optional[dict] = None,
                 ensemble=None, per_fold_summary: Optional[dict] = None) -> str:
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
    # 影像模態由使用者提供 (不從路徑猜); 沒填就不顯示
    from .privacy import load_user_facts
    uf = load_user_facts(run_dir)
    modality = "、".join(x for x in (uf.modality, uf.anatomy) if x) or "未指定"
    lines.append(f"- 不平衡比: {profile.imbalance_ratio}，modality: {modality}\n")

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

    if per_fold_summary is not None:
        p = per_fold_summary["primary"]
        pm = cfg.eval.primary_metric
        lines.append("\n## 逐 fold 獨立搜尋（每個 fold 各自找最佳 recipe）\n")
        lines.append(f"- **各 fold 最佳 primary {pm}**: "
                     f"{p['mean']:.4f} ± {p['std']:.4f}（n={p['n']}）")
        lines.append(f"\n| fold | encoder | adaptation | trials | primary({pm}) "
                     f"| {' | '.join(keys)} | 關鍵超參 |")
        lines.append("| --- | --- | --- | --- | --- | "
                     + " | ".join("---" for _ in keys) + " | --- |")
        for r in per_fold_summary["per_fold"]:
            if "encoder" not in r:
                lines.append(f"| {r['fold']} | — | — | {r.get('n_trials', 0)} | "
                             "— | " + " | ".join("—" for _ in keys) + " | — |")
                continue
            hp = r.get("hparams", {})
            hp_s = (f"blr={hp.get('blr')} · epochs={hp.get('epochs')} · "
                    f"layer_decay={hp.get('layer_decay')} · drop_path={hp.get('drop_path')}")
            lines.append(
                f"| {r['fold']} | {r['encoder']} | {r.get('adaptation', '-')} | "
                f"{r.get('n_trials', 0)} | {r['primary_score']:.4f} | "
                f"{_fmt_metrics(r.get('metrics', {}), keys)} | {hp_s} |")
        lines.append("\n> 各 fold 從頭獨立搜尋，最佳 encoder／超參可能不同；"
                     "解答樹分別存於 `search_tree_fold0.json …`。\n")

    if ensemble is not None and ensemble.status == "done":
        best_s = best.primary_score if best is not None else None
        won = best_s is not None and ensemble.primary_score > best_s
        lines.append("\n## 集成 (Ensemble)\n")
        lines.append(f"- **成員** ({len(ensemble.spec.member_trial_ids)}): "
                     + "、".join(f"`{t}`" for t in ensemble.spec.member_trial_ids))
        lines.append(f"- **方法**: {ensemble.spec.method}"
                     + (f"，權重 {ensemble.spec.weights}" if ensemble.spec.weights else ""))
        lines.append(f"- **primary({cfg.eval.primary_metric})**: "
                     f"{ensemble.primary_score:.4f}"
                     + (f"（vs 最佳單模型 {best_s:.4f}，"
                        f"{'勝出 +' + format(ensemble.primary_score - best_s, '.4f') if won else '未勝出'}）"
                        if best_s is not None else ""))
        lines.append(f"- **指標**: {_fmt_metrics(ensemble.metrics, keys)}")
        lines.append(f"- **對齊樣本數**: {ensemble.n_samples}")
        lines.append(f"- **集成 predictions**: `{ensemble.pred_path}`")
        if ensemble.member_ckpts:
            lines.append(f"- **成員 checkpoints**（推論需同時載入）:")
            for c in ensemble.member_ckpts:
                lines.append(f"  - `{c}`")
        lines.append(f"- **交付**: {'集成勝出，建議以集成交付' if won else '維持最佳單模型交付'}")

    lines.append(f"\n_共執行 {sum(len(v) for v in per_encoder.values())} trials，"
                 f"ledger: `{os.path.join(run_dir, 'ledger.jsonl')}`_\n")

    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    return path
