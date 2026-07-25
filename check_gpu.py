#!/usr/bin/env python
"""GPU / CUDA 配置診斷程式.

用途: 判斷「torch 能否真正在 GPU 上配置記憶體並運算」, 而不是只看 is_available()。
在 Exclusive_Process 模式下, is_available() 會回 True, 但被別的行程佔用時實際配置會失敗。

用法:
    conda activate M4
    python check_gpu.py                 # 檢查 device 0
    python check_gpu.py --device 1      # 檢查指定 device
"""
import argparse
import subprocess
import sys


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"(執行失敗: {e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    print("=" * 60)
    print(" GPU / CUDA 配置診斷")
    print("=" * 60)

    # 1. driver / nvidia-smi 層級
    print("\n[1] nvidia-smi 概況")
    print(sh(["nvidia-smi", "--query-gpu=index,name,compute_mode,memory.used,"
              "memory.total,utilization.gpu", "--format=csv,noheader"]) or "(無輸出)")
    print("    佔用行程 (pid, used_memory):")
    apps = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    print("    " + (apps.replace("\n", "\n    ") if apps else "(無)"))

    # 2. torch 層級
    print("\n[2] torch 層級")
    try:
        import torch
    except Exception as e:
        print(f"    ✗ import torch 失敗: {e}")
        return 2
    print(f"    torch 版本            : {torch.__version__}")
    print(f"    torch.cuda.is_available(): {torch.cuda.is_available()}  "
          f"(注意: 這只是查裝置, 不代表能配置)")
    if not torch.cuda.is_available():
        print("    ✗ 沒有可見的 CUDA 裝置。")
        return 2
    print(f"    可見 GPU 數           : {torch.cuda.device_count()}")

    # 3. 真正的配置 + 運算測試
    print(f"\n[3] 實際配置測試 (device {args.device})")
    try:
        dev = torch.device(f"cuda:{args.device}")
        x = torch.randn(2048, 2048, device=dev)   # ~16 MB
        y = x @ x                                   # 真正跑一次 kernel
        torch.cuda.synchronize(dev)
        name = torch.cuda.get_device_name(args.device)
        used = torch.cuda.memory_allocated(dev) / 1024**2
        del x, y
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"    ✗ 配置/運算失敗: {str(e).splitlines()[0]}")
        print("\n" + "=" * 60)
        print(" 結論: ✗ 無法配置 CUDA — GPU 被佔用或不可用")
        print("   多半是 Exclusive_Process 模式下已被其他行程 (見上方佔用清單) 持有 context。")
        print("=" * 60)
        return 1

    print(f"    ✓ 配置成功       : {name}")
    print(f"    ✓ 矩陣乘法完成   : 已配置 {used:.1f} MB, kernel 正常")
    print("\n" + "=" * 60)
    print(f" 結論: ✓ GPU (device {args.device}) 可用 — torch 能配置並運算")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
