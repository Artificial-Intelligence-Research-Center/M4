"""M4 自動微調 Agent — 設定檔驅動入口 (設計文件 §7).

用法:
    python -m agent.auto_finetune --config config.yaml
    python -m agent.auto_finetune --config config.yaml --dry_run
    # 或直接以旗標覆寫 (無 config 時):
    python -m agent.auto_finetune --data_path ./data/5_fold_PAPILA/PAPILA_seed42_fold0
    # 繼續之前停止的實驗 (從 run 目錄載回設定 + 解答樹, 接續下一輪);
    # 可同時用旗標放寬停止條件, 例如 --max_trials 20:
    python -m agent.auto_finetune --resume run_20260725_005626

外層多 encoder 廣度比較 (P1) 由 LoopController 執行; 決策層由 config.advisor.type
決定 (heuristic / llm / skill)。
"""
from __future__ import annotations

import argparse
import os
import time

from .advisor import build_advisor
from .config import AgentConfig
from .loop_controller import LoopController

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cfg: AgentConfig, run_dir: str | None = None, resume: bool = False) -> dict:
    name = cfg.experiment_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_dir or os.path.join(_ROOT, "runs", name)
    advisor = build_advisor(cfg.advisor)
    controller = LoopController(cfg, advisor, run_dir)
    return controller.run(resume=resume)


def resolve_run_dir(name_or_path: str) -> str:
    """--resume 參數 → run 目錄絕對路徑 (接受 runs/ 下的名稱或任意路徑)。"""
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    return os.path.join(_ROOT, "runs", name_or_path)


def main():
    ap = argparse.ArgumentParser("M4 auto-finetune agent (config-driven, P1 multi-encoder)")
    ap.add_argument("--config", default=None, help="YAML 設定檔 (§7)")
    ap.add_argument("--data_path", default=None, help="無 config 時直接指定資料路徑")
    ap.add_argument("--resume", default=None, metavar="RUN",
                    help="繼續之前停止的實驗: runs/ 下的 run 名稱或 run 目錄路徑 "
                         "(自其 config.yaml 載回設定, 可再用其他旗標覆寫)")
    ap.add_argument("--encoders_per_run", type=int, default=None,
                    help="(已停用) draft 輪替 encoder 上限改由 --num_drafts 決定")
    ap.add_argument("--num_drafts", type=int, default=None)
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--min_trials", type=int, default=None,
                    help="至少完成這麼多輪才允許 patience 提早停")
    ap.add_argument("--patience", type=int, default=None,
                    help="連續這麼多輪 improve/resume 無提升才停 (draft/debug 不計)")
    ap.add_argument("--primary_metric", default=None)
    ap.add_argument("--advisor", default=None, choices=["heuristic", "llm", "skill"])
    ap.add_argument("--preset", default=None, choices=["default", "paper", "mae"])
    ap.add_argument("--allow_code_edit", action="store_true",
                    help="允許 LLM 修改訓練程式 (副本放 run 目錄下, 原始程式不動)")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_stream", action="store_true",
                    help="不即時顯示子程序輸出 (只寫 log 檔)")
    args = ap.parse_args()

    run_dir = None
    if args.resume:
        run_dir = resolve_run_dir(args.resume)
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            ap.error(f"找不到 {cfg_path}，無法繼續此 run")
        cfg = AgentConfig.load(cfg_path)
    elif args.config:
        cfg = AgentConfig.load(args.config)
    elif args.data_path:
        cfg = AgentConfig(data_path=args.data_path)
    else:
        ap.error("需提供 --config、--data_path 或 --resume")

    # 旗標覆寫 (方便快速實驗)
    if args.encoders_per_run is not None:
        cfg.loop.encoders_per_run = args.encoders_per_run
    if args.num_drafts is not None:
        cfg.loop.num_drafts = args.num_drafts
    if args.max_trials is not None:
        cfg.loop.max_trials = args.max_trials
    if args.min_trials is not None:
        cfg.loop.min_trials = args.min_trials
    if args.patience is not None:
        cfg.loop.patience = args.patience
    if args.primary_metric is not None:
        cfg.eval.primary_metric = args.primary_metric
    if args.advisor is not None:
        cfg.advisor.type = args.advisor
    if args.preset is not None:
        cfg.advisor.preset = args.preset
    if args.allow_code_edit:
        cfg.advisor.allow_code_edit = True
    if args.device is not None:
        cfg.device = args.device
    if args.dry_run:
        cfg.dry_run = True
    if args.no_stream:
        cfg.stream_logs = False

    out = run(cfg, run_dir=run_dir, resume=bool(args.resume))
    print(f"\n=== 完成: {out['run_dir']} ===")
    print(f"encoder 數: {len(out['choices'])}，總 trials: {out['n_trials']}")
    for enc in out["choices"]:
        trials = out["per_encoder"].get(enc.model_key, [])
        done = [t for t in trials if t.status == "done"]
        if done:
            b = max(done, key=lambda t: t.primary_score)
            print(f"  {enc.model_key:28s} best primary={b.primary_score:.4f} "
                  f"({len(trials)} trials)")
        else:
            st = trials[-1].status if trials else "—"
            print(f"  {enc.model_key:28s} — ({st})")
    if out["best"] is not None:
        b = out["best"]
        print(f"\n🏆 全域最佳: {b.recipe.encoder.model_key} "
              f"primary={b.primary_score:.4f}  trial={b.trial_id}")
    print(f"📄 報告: {out['report_path']}")


if __name__ == "__main__":
    main()
