"""RecipeBuilder — 把 Recipe 轉成 main_finetune.py 的執行指令 (設計文件 §5.4).

P3: 透過 ComponentRegistry 把可組合元件 (head / loss / regularizer) 映射成
main_finetune 的 CLI 參數。仍未接上的元件 (seg/reg head、預留 loss/aug…) 記錄在
unsupported 清單 (僅提示, 不阻擋)。
"""
from __future__ import annotations

import os
import sys

from . import component_registry as comp
from . import encoder_registry as reg
from .schemas import Recipe

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAIN = os.path.join(_ROOT, "main_finetune.py")


def unsupported_components(recipe: Recipe) -> list[str]:
    """列出尚未接上 main_finetune 的 Recipe 元件 (僅提示, 不阻擋)。"""
    out: list[str] = []
    for h in recipe.heads:
        if not comp.is_supported(h.type):
            out.append(f"head={h.type}")
    _, l_un = comp.loss_cli(recipe.losses)
    out += l_un
    _, r_un = comp.regularizer_cli(recipe.regularizers)
    out += r_un
    if recipe.augmentation and not comp.is_supported(recipe.augmentation.name):
        out.append(f"augmentation={recipe.augmentation.name}")
    return out


def build_command(recipe: Recipe, *, data_path: str, num_classes: int,
                  output_dir: str, task_id: str) -> list[str]:
    card = reg.get(recipe.encoder.model_key)
    hp = recipe.hparams

    # LLM 修改程式時 (code_workspace): 跑 run 目錄下的副本, 原始程式不動
    main_py = _MAIN
    if recipe.code_dir:
        cand = os.path.join(recipe.code_dir, "main_finetune.py")
        if os.path.isfile(cand):
            main_py = cand

    cmd = [
        sys.executable, main_py,
        "--model", card.model,
        "--model_arch", card.model_arch,
        "--finetune", reg.weight_path(card),
        "--savemodel",
        "--batch_size", str(hp.batch_size),
        "--epochs", str(hp.epochs),
        "--accum_iter", str(hp.accum_iter),
        "--blr", str(hp.blr),
        "--layer_decay", str(hp.layer_decay),
        "--drop_path", str(hp.drop_path),
        "--weight_decay", str(hp.weight_decay),
        "--warmup_epochs", str(hp.warmup_epochs),
        "--nb_classes", str(num_classes),
        "--data_path", os.path.abspath(data_path),
        "--output_dir", output_dir,
        "--input_size", str(hp.input_size),
        "--task", task_id,
        "--adaptation", recipe.encoder.adaptation,
    ]
    # GPU 利用率最佳化: agent 跑的 trial 一律開 persistent_workers (統計中性,
    # 純省 worker 重啟時間; baseline 手動跑不帶旗標時行為不變)
    cmd += ["--persistent_workers"]
    if hp.num_workers is not None:
        cmd += ["--num_workers", str(hp.num_workers)]
    if hp.cache_resized:
        cmd += ["--cache_resized", str(hp.cache_resized)]

    # 繼續訓練 (resume 階段): 從既有 checkpoint 續訓 --more_epochs 個 epoch。
    # main_finetune 會沿用 checkpoint 內的 args (超參與資料), 只覆寫 task/output_dir/epochs。
    if recipe.resume_from and recipe.resume_epochs > 0:
        cmd += ["--resume", recipe.resume_from,
                "--more_epochs", str(recipe.resume_epochs)]

    # pooling: main_finetune 預設 global_pool=True; cls_token 用 --cls_token 關閉
    if recipe.pooling == "global_pool":
        cmd.append("--global_pool")
    else:
        cmd.append("--cls_token")

    # Recipe 元件 (P3): head / loss / regularizer → CLI (未接元件靜默略過, 見 unsupported)
    if recipe.heads:
        head_c, _ = comp.head_cli(recipe.heads[0])
        cmd += head_c
    loss_c, _ = comp.loss_cli(recipe.losses)
    cmd += loss_c
    reg_c, _ = comp.regularizer_cli(recipe.regularizers)
    cmd += reg_c
    return cmd


def build_command_str(recipe: Recipe, **kw) -> str:
    return " ".join(build_command(recipe, **kw))
