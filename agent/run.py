"""單一 trial 編排 (P0) — 串起 Analyzer → Advisor → Trainer → Evaluator → Ledger.

用法:
    python -m agent.run --data_path ./data/5_fold_PAPILA/PAPILA_seed42_fold0
    python -m agent.run --data_path ... --dry_run          # 只組指令不訓練
    python -m agent.run --data_path ... --encoder mae_pretrain_vit_large

未來 (P1+) LoopController 會把「單一 trial」擴成多 encoder × 多 recipe 迴圈。
"""
from __future__ import annotations

import argparse
import os
import time

from . import dataset_analyzer, evaluator as ev
from .advisor import HeuristicAdvisor
from .ledger import Ledger
from .schemas import EvalConfig, TrialResult
from .trainer import run_trial, gpu_allocatable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def single_trial(data_path: str, *, run_dir: str, preset: str = "default",
                 encoder_key: str | None = None, eval_cfg: EvalConfig | None = None,
                 dry_run: bool = False, advisor=None, device_index: int = 0) -> dict:
    """執行一個 trial, 回傳摘要 dict。web/CLI 共用此函式。"""
    advisor = advisor or HeuristicAdvisor()
    eval_cfg = eval_cfg or EvalConfig()

    # 1. 分析資料集
    profile = dataset_analyzer.analyze(data_path)

    # 2. 選 encoder (多個) — P0 單一 trial 取指定或第一個
    choices = advisor.select_encoders(profile)
    if not choices:
        raise RuntimeError("沒有可用的 encoder (檢查 baseline_models/ 權重是否存在)")
    if encoder_key:
        choices = [c for c in choices if c.model_key == encoder_key] or choices
    encoder = choices[0]

    # 3. 組 Recipe
    recipe = advisor.compose_recipe(profile, encoder, preset=preset)

    # 4. 準備輸出 & Ledger
    ledger = Ledger(run_dir)
    ledger.write_profile(profile.model_dump_json(indent=2))
    ds_name = os.path.basename(os.path.normpath(data_path))
    task_id = f"{encoder.model_key}_{ds_name}_{preset}"
    trials_dir = os.path.join(run_dir, "trials")
    log_path = os.path.join(run_dir, "logs", f"log_{task_id}.txt")

    trial = TrialResult(trial_id=task_id, recipe=recipe, status="running")

    # 5. 訓練 (subprocess 包裝 main_finetune)
    tr = run_trial(
        recipe, data_path=data_path, num_classes=profile.num_classes,
        output_dir=trials_dir, task_id=task_id, device_index=device_index,
        log_path=log_path, dry_run=dry_run,
    )
    trial.log_path = tr.get("log_path")

    if dry_run:
        trial.status = "pending"
        trial.message = "dry_run: 已組指令未執行。" + (
            f" 未接元件: {tr['unsupported']}" if tr["unsupported"] else "")
        return {"profile": profile, "recipe": recipe, "trial": trial,
                "command": tr["command_str"], "unsupported": tr["unsupported"]}

    # 6. 評估 + 落地
    if tr["returncode"] == 0:
        res = ev.evaluate(tr["task_dir"], eval_cfg, mode="test")
        trial.metrics = res["metrics"]
        trial.primary_score = res["primary_score"]
        trial.ckpt_path = res["ckpt_path"]
        trial.status = "done"
    else:
        trial.status = "failed"
        trial.message = f"main_finetune 回傳碼 {tr['returncode']}, 見 log。"

    ledger.append(trial)
    return {"profile": profile, "recipe": recipe, "trial": trial,
            "command": tr["command_str"], "unsupported": tr["unsupported"]}


def main():
    ap = argparse.ArgumentParser("M4 auto-finetune agent (P0 single trial)")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--run_dir", default=os.path.join(_ROOT, "runs",
                    time.strftime("run_%Y%m%d_%H%M%S")))
    ap.add_argument("--preset", default="default", choices=["default", "paper", "mae"])
    ap.add_argument("--encoder", default=None, help="指定 encoder model_key")
    ap.add_argument("--primary_metric", default="score")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not args.dry_run and not gpu_allocatable(args.device):
        print(f"⚠️  GPU {args.device} 目前無法配置 CUDA (可能被外部佔用 / "
              f"Exclusive_Process)。可加 --dry_run 只組指令。")

    out = single_trial(
        args.data_path, run_dir=args.run_dir, preset=args.preset,
        encoder_key=args.encoder, eval_cfg=EvalConfig(primary_metric=args.primary_metric),
        dry_run=args.dry_run, device_index=args.device,
    )
    t = out["trial"]
    print(f"\n=== Trial: {t.trial_id} ===")
    print(f"encoder : {out['recipe'].encoder.model_key} "
          f"({out['recipe'].encoder.adaptation})")
    print(f"command : {out['command']}")
    if out["unsupported"]:
        print(f"未接元件 (待 P3): {out['unsupported']}")
    print(f"status  : {t.status}")
    if t.status == "done":
        print(f"primary : {t.primary_score:.4f}  metrics={t.metrics}")
    print(f"run_dir : {args.run_dir}")


if __name__ == "__main__":
    main()
