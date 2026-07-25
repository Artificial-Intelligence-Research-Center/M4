"""GpuSampler — 訓練期間背景取樣 GPU 利用率/記憶體 (GPU 利用率最佳化用).

trainer.run_trial 在訓練 subprocess 執行期間啟動一條 daemon thread, 每隔
sample_interval 秒呼叫 nvidia-smi 取樣; 結束後彙整成 stats dict 存進
TrialResult.gpu_stats, 供 LoopController 判斷是否要提高利用率 (加大 batch /
增加 dataloader worker)。

注意: 取樣的是「整張卡」的利用率 (Exclusive_Process 模式下即本訓練行程);
前 warmup_skip 個樣本 (模型載入/首個 epoch 暖身) 不計入平均。
nvidia-smi 不存在或失敗時安靜回傳 {} (不影響訓練)。
"""
from __future__ import annotations

import statistics
import subprocess
import threading


def _query(device_index: int) -> tuple[float, float, float] | None:
    """回傳 (util%, mem_used_MB, mem_total_MB); 失敗回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device_index),
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        if out.returncode != 0:
            return None
        u, m, t = out.stdout.decode().strip().splitlines()[0].split(",")
        return float(u), float(m), float(t)
    except Exception:
        return None


class GpuSampler:
    def __init__(self, device_index: int = 0, sample_interval_s: float = 5.0,
                 warmup_skip: int = 3):
        self.device_index = device_index
        self.interval = max(sample_interval_s, 0.1)
        self.warmup_skip = warmup_skip
        self._samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            s = _query(self.device_index)
            if s is not None:
                self._samples.append(s)
            self._stop.wait(self.interval)

    def _stats(self) -> dict:
        body = self._samples[self.warmup_skip:]
        if len(body) < 3:
            return {}
        utils = [s[0] for s in body]
        mems = [s[1] for s in body]
        return {
            "util_avg": round(sum(utils) / len(utils), 1),
            "util_median": round(statistics.median(utils), 1),
            "mem_peak_mb": round(max(mems), 0),
            "mem_total_mb": round(body[0][2], 0),
            "n_samples": len(body),
            "sample_interval_s": self.interval,
        }

    def snapshot(self) -> dict:
        """取樣進行中讀取目前彙整 (mid-trial 低利用率警示用); 樣本不足回 {}。"""
        return self._stats()

    def stop(self) -> dict:
        """停止取樣並彙整。樣本不足 (扣掉暖身後 < 3) 回 {} (資訊不足不決策)。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 11)
        return self._stats()
