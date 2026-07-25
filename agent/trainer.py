"""Trainer — 以 subprocess 包裝 main_finetune.py 執行一次訓練 (設計文件 §5.5).

GPU 可用性判斷: 修正 script.py 以 memoryUtil<0.1 誤判的問題 (Exclusive_Process
模式下外部佔用的卡會被誤判可用), 改以「實際嘗試配置 CUDA」判斷 (設計文件 §11-3)。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Callable, Optional

from . import recipe_builder
from .gpu_monitor import GpuSampler
from .schemas import Recipe


def gpu_allocatable(device_index: int = 0, timeout: int = 30) -> bool:
    """實際嘗試在該 GPU 配置一小塊 CUDA 記憶體, 成功才算可用。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    code = "import torch; torch.randn(8).cuda(); print('ok')"
    try:
        r = subprocess.run(
            ["python", "-c", code], env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return r.returncode == 0 and b"ok" in r.stdout
    except Exception:
        return False


def run_trial(recipe: Recipe, *, data_path: str, num_classes: int,
              output_dir: str, task_id: str, device_index: int = 0,
              log_path: Optional[str] = None, dry_run: bool = False,
              stream: bool = True, gpu_sample_interval_s: float = 5.0,
              low_util_target: Optional[float] = None,
              low_util_after_s: float = 180.0,
              on_low_util: Optional[Callable[[dict], None]] = None) -> dict:
    """執行一次訓練。回傳 dict(command, task_dir, log_path, returncode, dry_run,
    gpu_stats)。訓練期間背景取樣 GPU 利用率/記憶體 (GPU 利用率最佳化用)。

    low_util_target + on_low_util: 訓練開始 low_util_after_s 秒後檢查一次取樣
    平均, 低於目標就呼叫 on_low_util(snapshot) — 讓使用者在訓練中就看到警示,
    而不是等 trial 結束。

    dry_run=True 時只組指令不執行 (供無 GPU 環境驗證)。
    stream=True 時把子程序輸出即時 tee 到 console 與 log 檔 (預設, 便於看進度);
    stream=False 只寫 log 檔 (console 靜默)。
    """
    cmd = recipe_builder.build_command(
        recipe, data_path=data_path, num_classes=num_classes,
        output_dir=output_dir, task_id=task_id,
    )
    # main_finetune 把結果寫到 output_dir/<task_id>/
    task_dir = os.path.join(output_dir, task_id)
    result = {
        "command": cmd,
        "command_str": " ".join(cmd),
        "task_dir": task_dir,
        "log_path": log_path,
        "returncode": None,
        "dry_run": dry_run,
        "unsupported": recipe_builder.unsupported_components(recipe),
    }
    if dry_run:
        return result

    os.makedirs(output_dir, exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    # 子行程 stdout 不緩衝: 否則寫檔是 8KB 區塊緩衝, 快速 trial (資料管線加速後
    # 一個 trial 可能 <2 分鐘) 整場只 flush 一兩次, web UI 看不到逐 epoch 進度
    env["PYTHONUNBUFFERED"] = "1"

    sampler = GpuSampler(device_index, sample_interval_s=gpu_sample_interval_s).start()
    watch_stop = threading.Event()
    if low_util_target is not None and on_low_util is not None:
        def _watch():
            if watch_stop.wait(low_util_after_s):
                return  # 訓練先結束
            snap = sampler.snapshot()
            if snap and snap.get("util_avg", 100.0) < low_util_target:
                try:
                    on_low_util(snap)
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
    try:
        if log_path and stream:
            # tee: 逐行同時寫 console + log 檔
            with open(log_path, "w", encoding="utf8") as f:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    f.write(line)
                proc.wait()
        elif log_path:
            with open(log_path, "w") as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                proc.wait()
        else:
            proc = subprocess.Popen(cmd, env=env)
            proc.wait()
    finally:
        watch_stop.set()
        result["gpu_stats"] = sampler.stop()
    result["returncode"] = proc.returncode
    return result
