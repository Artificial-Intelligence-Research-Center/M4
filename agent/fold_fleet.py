"""逐 fold 平行搜尋艦隊 (parallel per-fold search) — aggregation=per_fold 的執行層。

每個 fold 是一個**完全獨立的子執行** (`runs/<parent>/foldN/`): 自己的對話、ledger、
解答樹、LLM 記錄、隱私稽核與 report。父 run 只保存一份 `folds.json` 索引 + 彙整報告，
並把各 fold **平均分配到所有 GPU 上平行執行**。Web UI 以 `run=<parent>/foldN` 位址
切換到任一 fold 檢視其進程 (對話與結果徹底分開)。

執行器抽象 (未來擴充):
- 目前 `_LocalLauncher`: 在本機以執行緒 + GPU 輪替 (每顆 GPU 同時最多一個 fold) 跑。
  訓練本身是子程序 (trainer 設 CUDA_VISIBLE_DEVICES)，故執行緒不受 GIL 影響。
- 未來 `PyTorchJobLauncher` (見檔尾 stub): 每個 fold 提交一個 PyTorchJob，在 K8s 不同
  節點執行；只需替換 launcher，run_fleet 的協調/彙整/UI 契約不變。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import conversation as convo
from . import dataset_analyzer
from . import metric_registry as mreg
from . import report as report_mod
from . import trainer
from .config import AgentConfig
from .loop_controller import discover_folds
from .schemas import EncoderChoice

# 資源等待策略 (fold 層級): GPU 記憶體被占用時, 等待其釋出再啟動 / 重試。
_GPU_POLL_S = 20          # 輪詢 GPU 可配置性的間隔 (秒)
_GPU_WAIT_MAX_S = 1800    # 單次等待上限 (秒); 逾時仍嘗試啟動
_OOM_RETRIES = 2          # fold 因資源不足失敗時, 等待後重試的次數上限
_OOM_HINTS = ("out of memory", "oom", "cuda error", "cublas", "cudnn",
              "no kernel image", "device-side assert")
_PACK_POLL_S = 15         # 監看各 GPU util/mem 的間隔 (秒)
_PACK_LOW_STREAK = 2      # 連續這麼多次量到「低 util + 低 mem」才加開共卡 (避免瞬間抖動)


def _gpu_util_mem(devices) -> dict:
    """即時讀各 GPU 的 (utilization%, used/total mem) — nvidia-smi 單次快照。"""
    out: dict = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 4:
                continue
            idx = int(float(p[0]))
            if idx in devices:
                total = float(p[3]) or 1.0
                out[idx] = (float(p[1]), float(p[2]) / total)
    except Exception:
        pass
    return out


class _Scheduler:
    """負載感知 + 自適應共卡排程器。

    起始每卡容量 cap=1 (一 fold 一卡)。監看執行緒發現某卡『只跑一個 fold 且 util 低、
    mem 也低』並持續 `_PACK_LOW_STREAK` 次時, 把該卡容量提到 2 (至多 max_per), 讓佇列中
    的下一個 fold 立即加開到同一張卡, 用閒置週期提高利用率 —— 取代『加大 batch』。
    acquire 一律優先選 active 最少的卡 (先把每張卡各鋪一個 fold, 再視情況共用)。
    """

    def __init__(self, devices, max_per, util_below, mem_below, parent):
        self.devices = list(devices)
        self.max_per = max(1, int(max_per))
        self.util_below = util_below
        self.mem_below = mem_below
        self.parent = parent
        self.active = {d: 0 for d in self.devices}
        self.cap = {d: 1 for d in self.devices}
        self._low = {d: 0 for d in self.devices}
        self.cond = threading.Condition()
        self._stop = False

    def acquire(self) -> int:
        with self.cond:
            while True:
                cands = [d for d in self.devices if self.active[d] < self.cap[d]]
                if cands:
                    d = min(cands, key=lambda x: self.active[x])
                    self.active[d] += 1
                    return d
                self.cond.wait(timeout=5)

    def release(self, d: int) -> None:
        with self.cond:
            self.active[d] = max(0, self.active[d] - 1)
            if self.active[d] <= 1:          # 不再共用 → 收回額外容量, 重新評估
                self.cap[d] = 1
                self._low[d] = 0
            self.cond.notify_all()

    def stop(self) -> None:
        with self.cond:
            self._stop = True
            self.cond.notify_all()

    def monitor(self) -> None:
        if self.max_per <= 1:
            return
        while True:
            with self.cond:
                if self._stop:
                    return
            stats = _gpu_util_mem(self.devices)
            bumped = []
            with self.cond:
                for d in self.devices:
                    if self.active[d] == 1 and self.cap[d] < self.max_per:
                        um = stats.get(d)
                        if (um and um[0] < self.util_below and um[1] < self.mem_below):
                            self._low[d] += 1
                            if self._low[d] >= _PACK_LOW_STREAK:
                                self.cap[d] += 1
                                self._low[d] = 0
                                bumped.append((d, um))
                        else:
                            self._low[d] = 0
                if bumped:
                    self.cond.notify_all()
            for d, um in bumped:
                convo.append(self.parent, "system",
                             f"GPU {d} 使用率偏低（util {um[0]:.0f}% · mem "
                             f"{um[1] * 100:.0f}%），加開一個 fold 共用此卡以提高利用率。",
                             kind="status")
            time.sleep(_PACK_POLL_S)


# ---------------------------------------------------------------------------
# GPU 偵測
# ---------------------------------------------------------------------------
def detect_gpus() -> list[int]:
    """回傳可用 GPU index 清單。nvidia-smi 優先 (不必載入 torch); 皆失敗回 [0]。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        idx = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        if idx:
            return idx
    except Exception:
        pass
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return list(range(n))
    except Exception:
        pass
    return [0]


# ---------------------------------------------------------------------------
# 每個 fold 的啟動 (可替換為 PyTorchJobLauncher)
# ---------------------------------------------------------------------------
def _launch_fold(child_cfg: AgentConfig, sub_dir: str, resume: bool = False) -> dict:
    """本機啟動器: 在本行程內直接跑一個 fold 的完整搜尋 (訓練仍走子程序)。

    resume=True: 從該 fold 既有 ledger 重建解答樹後繼續 (跳過 draft, 只跑
    improve/debug/resume), 依新的 max_trials/patience 追加步數並重產報告。
    未來多節點: 換成提交 PyTorchJob 並輪詢, 回傳同形狀的 dict (best / n_trials / report_path)。
    """
    from . import auto_finetune
    return auto_finetune.run(child_cfg, run_dir=sub_dir, resume=resume)


def _is_resource_error(err: Exception) -> bool:
    return any(h in str(err).lower() for h in _OOM_HINTS)


def _wait_allocatable(device: int, parent: str, label: str) -> None:
    """在此 GPU 上啟動前, 等待它能實際配置記憶體 (被占用/OOM 後釋出)。逾時仍嘗試。"""
    waited, warned = 0, False
    while not trainer.gpu_allocatable(device, timeout=30):
        if not warned:
            convo.append(parent, "system",
                         f"{label} 等待 GPU {device} 記憶體釋出中…（每 {_GPU_POLL_S}s 檢查）",
                         kind="status")
            warned = True
        if waited >= _GPU_WAIT_MAX_S:
            convo.append(parent, "system",
                         f"{label} 等待 GPU {device} 逾時（{_GPU_WAIT_MAX_S}s），仍嘗試啟動。",
                         kind="status")
            return
        time.sleep(_GPU_POLL_S)
        waited += _GPU_POLL_S
    if warned:
        convo.append(parent, "system", f"{label} GPU {device} 已可用，開始執行。",
                     kind="status")


def _copy_user_facts(parent_dir: str, sub_dir: str) -> None:
    """把父 run 的使用者事實 (模態/部位/類別序數) 複製給子 fold，讓其 LLM 也看得到。"""
    src = os.path.join(parent_dir, "user_facts.json")
    if os.path.isfile(src):
        try:
            shutil.copy2(src, os.path.join(sub_dir, "user_facts.json"))
        except Exception:
            pass


def _write_manifest(parent_dir: str, entries: list, devices: list, done: bool) -> None:
    try:
        with open(os.path.join(parent_dir, "folds.json"), "w", encoding="utf8") as f:
            json.dump({"mode": "per_fold_parallel", "devices": devices,
                       "done": done, "folds": entries}, f,
                      ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 艦隊主流程
# ---------------------------------------------------------------------------
def run_fleet(cfg: AgentConfig, run_dir: str,
              devices: Optional[list] = None, resume: bool = False) -> dict:
    """對每個 sibling fold 平行跑一次完整搜尋 (各自獨立子執行), 最後彙整。

    resume=True: 每個 fold 從既有樹後繼續 (跳過 draft, 依新的 max_trials/patience
    追加 improve/debug/resume 步數), 最後重產父層彙整報告。
    """
    folds = discover_folds(cfg.data_path)
    parent = os.path.abspath(run_dir)
    os.makedirs(parent, exist_ok=True)

    # 單 fold: 沒得平行, 直接當一般 run 跑 (子執行=父 run)
    if len(folds) <= 1:
        c = cfg.model_copy(deep=True)
        c.eval.aggregation = "single"
        return _launch_fold(c, parent, resume=resume)

    devices = devices or detect_gpus()
    ttype = cfg.task.get("type", "classification")
    entries = [{"index": i, "name": os.path.basename(os.path.normpath(fp)),
                "subdir": f"fold{i}", "device": None, "status": "queued",
                "primary": None, "best": None} for i, fp in enumerate(folds)]
    lock = threading.Lock()

    def flush(done=False):
        with lock:
            _write_manifest(parent, entries, devices, done)

    g = cfg.gpu
    packing = bool(getattr(g, "pack_low_util", False)
                   and getattr(g, "max_folds_per_gpu", 1) > 1)
    flush()
    convo.append(parent, "system",
                 (f"繼續逐 fold 平行搜尋（max_trials={cfg.loop.max_trials}, "
                  f"patience={cfg.loop.patience}；每個 fold 在既有樹後繼續，跳過 draft）："
                  if resume else "逐 fold 平行搜尋：")
                 + f"{len(folds)} 個 fold 平均分配到 {len(devices)} 顆 "
                 f"GPU {devices} 上獨立執行。每個 fold 有自己的對話與結果，"
                 f"可在左側「fold」選擇器切換查看。"
                 + (f"（低利用率時最多 {g.max_folds_per_gpu} 個 fold 共用一卡，"
                    f"取代加大 batch）" if packing else ""), kind="status")

    sched = _Scheduler(devices, g.max_folds_per_gpu if packing else 1,
                       getattr(g, "pack_util_below", 35.0),
                       getattr(g, "pack_mem_below", 0.5), parent)
    monitor = threading.Thread(target=sched.monitor, daemon=True)
    monitor.start()
    results: dict[int, dict] = {}

    def run_one(i: int, fold_path: str) -> None:
        device = sched.acquire()                  # 阻塞直到某卡有空位 (可能與他人共用)
        try:
            with lock:
                entries[i]["device"] = device
                entries[i]["status"] = "running"
                entries[i]["shared"] = sched.active[device] > 1
            flush()
            shared = entries[i].get("shared")
            convo.append(parent, "system",
                         f"Fold {i + 1}（{entries[i]['name']}）開始，GPU {device}"
                         + ("（與另一個 fold 共用）。" if shared else "。"),
                         kind="status")
            sub = os.path.join(parent, f"fold{i}")
            os.makedirs(sub, exist_ok=True)
            _copy_user_facts(parent, sub)
            child = cfg.model_copy(deep=True)
            child.data_path = fold_path
            child.device = device
            child.eval.aggregation = "single"     # 子執行不再遞迴分 fold
            child.experiment_name = None
            if packing:
                child.gpu.grow_batch = False       # 共卡模式: 不加大 batch (改以共用提高利用率)
            label = f"Fold {i + 1}（{entries[i]['name']}）"
            # 啟動前先等 GPU 可配置 (被占用/前一工作 OOM 未釋出時等待)
            _wait_allocatable(device, parent, label)
            # 資源不足失敗 → 等待釋出後重試 (fold 層級; 單一 trial 的 OOM 由
            # LoopController debug 階段以降 batch_size 處理, 見設計文件)
            attempt = 0
            while True:
                try:
                    out = _launch_fold(child, sub, resume=resume)
                    break
                except Exception as e:            # noqa: BLE001
                    if _is_resource_error(e) and attempt < _OOM_RETRIES:
                        attempt += 1
                        convo.append(parent, "system",
                                     f"{label} 疑似資源不足（{e}）；等待 GPU {device} "
                                     f"釋出後重試（第 {attempt}/{_OOM_RETRIES} 次）…",
                                     kind="status")
                        _wait_allocatable(device, parent, label)
                        continue
                    raise
            results[i] = out
            best = out.get("best")
            ens = out.get("ensemble")
            with lock:
                entries[i]["status"] = "finished"
                entries[i]["primary"] = best.primary_score if best else None
                entries[i]["best"] = best.trial_id if best else None
                entries[i]["n_done"] = out.get("n_trials", 0)
                if ens is not None and getattr(ens, "status", "") == "done" and best:
                    entries[i]["ens_primary"] = ens.primary_score
                    entries[i]["ens_method"] = ens.spec.method
                    entries[i]["ens_won"] = ens.primary_score > best.primary_score
            flush()
            convo.append(
                parent, "system",
                f"Fold {i + 1}（{entries[i]['name']}）完成"
                + (f"，最佳 primary={best.primary_score:.4f}"
                   f"（{best.recipe.encoder.model_key}）。" if best
                   else "，無成功 trial。"), kind="decision")
        except Exception as e:                    # noqa: BLE001
            with lock:
                entries[i]["status"] = "failed"
                entries[i]["error"] = str(e)
            flush()
            convo.append(parent, "system",
                         f"Fold {i + 1} 失敗：{e}", kind="status")
        finally:
            sched.release(device)

    # max_workers = fold 數: 每個 fold 都有執行緒, 在 sched.acquire() 阻塞等卡位;
    # 共卡條件成立時, 等待中的 fold 立即遞補到該卡 (不需等別的 fold 全部跑完)。
    with ThreadPoolExecutor(max_workers=len(folds)) as ex:
        futures = [ex.submit(run_one, i, fp) for i, fp in enumerate(folds)]
        for fu in futures:
            fu.result()
    sched.stop()

    # ---- 彙整 ----------------------------------------------------------
    per_fold_rows, best_trials, scores = [], [], []
    overall = None
    for i in range(len(folds)):
        out = results.get(i) or {}
        best = out.get("best")
        ens = out.get("ensemble")             # 該 fold 子執行的收尾集成結果
        row = {"fold": entries[i]["name"], "n_trials": out.get("n_trials", 0),
               "device": entries[i]["device"], "subdir": f"fold{i}",
               "status": entries[i]["status"]}
        if best is not None:
            row.update(encoder=best.recipe.encoder.model_key,
                       adaptation=best.recipe.encoder.adaptation,
                       primary_score=best.primary_score, metrics=best.metrics,
                       hparams=best.recipe.hparams.model_dump(),
                       trial_id=best.trial_id)
            if ens is not None and getattr(ens, "status", "") == "done":
                row["ens_primary"] = ens.primary_score
                row["ens_method"] = ens.spec.method
                row["ens_won"] = ens.primary_score > best.primary_score
            best_trials.append(best)
            scores.append(best.primary_score)
            if overall is None or best.primary_score > overall.primary_score:
                overall = best
        per_fold_rows.append(row)

    pfs = {"folds": [r["fold"] for r in per_fold_rows], "per_fold": per_fold_rows,
           "best_trials": best_trials, "primary": mreg.aggregate(scores),
           "metrics": mreg.aggregate_metrics([t.metrics for t in best_trials])}

    # ---- 跨 fold 集成: 用各 fold 最佳模型在共同 test set 上找最佳組合 ----
    cross = None
    cf_members = []
    for i in range(len(folds)):
        out = results.get(i) or {}
        b = out.get("best")
        if b is not None and b.trial_id:
            td = os.path.join(parent, f"fold{i}", "trials", b.trial_id)
            if os.path.isdir(td):
                cf_members.append({"label": f"fold{i}:{b.recipe.encoder.model_key}",
                                   "task_dir": td, "ckpt": b.ckpt_path,
                                   "primary": b.primary_score})
    if len(cf_members) >= 2:
        try:
            from . import ensembler
            cross = ensembler.cross_fold_combine(
                cf_members, cfg.eval, os.path.join(parent, "ensembles", "cross_fold"))
            if cross.status == "done":
                convo.append(parent, "system",
                             f"跨 fold 集成：用 {len(cf_members)} 個 fold 的最佳模型，"
                             f"選出 {len(cross.spec.member_trial_ids)} 個組合，"
                             f"primary={cross.primary_score:.4f}（{cross.message}）。",
                             kind="decision")
            else:
                convo.append(parent, "system", f"跨 fold 集成未完成：{cross.message}",
                             kind="status")
        except Exception as e:                        # noqa: BLE001
            convo.append(parent, "system", f"跨 fold 集成失敗（{e}）。", kind="status")
            cross = None

    per_encoder: dict = {}
    for t in best_trials:
        per_encoder.setdefault(t.recipe.encoder.model_key, []).append(t)
    choices = [EncoderChoice(model_key=k, adaptation="finetune") for k in per_encoder]
    try:
        profile = dataset_analyzer.analyze(cfg.data_path, task_type=ttype)
        report_path = report_mod.write_report(
            parent, profile, cfg, choices, per_encoder, overall,
            per_fold_summary=pfs, cross_fold=cross)
    except Exception:
        report_path = None

    p = pfs["primary"]
    cf_note = (f"；跨 fold 集成 primary={cross.primary_score:.4f}"
               if cross is not None and cross.status == "done" else "")
    convo.append(parent, "system",
                 f"全部 fold 完成。整體最佳單模型 primary="
                 + (f"{overall.primary_score:.4f}" if overall else "—")
                 + f"；各 fold 最佳 mean±std = {p['mean']:.4f}±{p['std']:.4f}"
                   f"（n={p['n']}）" + cf_note + "。", kind="final")
    flush(done=True)
    return {"best": overall,
            "n_trials": sum(r.get("n_trials", 0) for r in per_fold_rows),
            "report_path": report_path, "run_dir": parent, "cross_fold": cross,
            "per_fold_summary": pfs, "ensemble": None, "fold_summary": None}


def reaggregate_parent(parent_dir: str):
    """重讀各 fold 目前狀態, 重產父層彙整報告 (含跨 fold 集成) + 更新 folds.json。

    用於『單獨繼續某個 fold』之後 — 該 fold 自己的 report 由其子執行更新, 但父層彙整
    需要在這裡一併刷新, 否則全域報告會過時。從各 fold 的 ledger 重讀最佳 trial。
    回傳新的 report 路徑 (失敗回 None)。
    """
    parent = os.path.abspath(parent_dir)
    try:
        with open(os.path.join(parent, "folds.json"), encoding="utf8") as f:
            pj = json.load(f)
    except Exception:
        return None
    if pj.get("mode") != "per_fold_parallel":
        return None
    try:
        cfg = AgentConfig.load(os.path.join(parent, "config.yaml"))
    except Exception:
        return None
    from .ledger import Ledger

    entries = pj.get("folds", [])
    per_fold_rows, best_trials, scores, cf_members = [], [], [], []
    overall = None
    for ent in entries:
        sub = os.path.join(parent, ent.get("subdir", ""))
        try:
            hist = Ledger(sub).history()
        except Exception:
            hist = []
        done = [t for t in hist if t.status == "done"
                and t.recipe.encoder.model_key != "ensemble"]
        best = max(done, key=lambda t: t.primary_score, default=None)
        row = {"fold": ent.get("name"), "n_trials": len(done),
               "device": ent.get("device"), "subdir": ent.get("subdir"),
               "status": ent.get("status")}
        ent["n_done"] = len(done)                            # folds.json 帶上 trial 數
        for k in ("ens_primary", "ens_method", "ens_won"):   # 沿用完成時存下的集成資訊
            if ent.get(k) is not None:
                row[k] = ent[k]
        if best is not None:
            row.update(encoder=best.recipe.encoder.model_key,
                       adaptation=best.recipe.encoder.adaptation,
                       primary_score=best.primary_score, metrics=best.metrics,
                       hparams=best.recipe.hparams.model_dump(),
                       trial_id=best.trial_id)
            ent["primary"], ent["best"] = best.primary_score, best.trial_id
            best_trials.append(best)
            scores.append(best.primary_score)
            if overall is None or best.primary_score > overall.primary_score:
                overall = best
            td = os.path.join(sub, "trials", best.trial_id)
            if os.path.isdir(td):
                cf_members.append({"label": f"{ent.get('subdir')}:{best.recipe.encoder.model_key}",
                                   "task_dir": td, "ckpt": best.ckpt_path,
                                   "primary": best.primary_score})
        per_fold_rows.append(row)

    pfs = {"folds": [r["fold"] for r in per_fold_rows], "per_fold": per_fold_rows,
           "best_trials": best_trials, "primary": mreg.aggregate(scores),
           "metrics": mreg.aggregate_metrics([t.metrics for t in best_trials])}
    cross = None
    if len(cf_members) >= 2:
        try:
            from . import ensembler
            cross = ensembler.cross_fold_combine(
                cf_members, cfg.eval, os.path.join(parent, "ensembles", "cross_fold"))
        except Exception:
            cross = None
    per_encoder: dict = {}
    for t in best_trials:
        per_encoder.setdefault(t.recipe.encoder.model_key, []).append(t)
    choices = [EncoderChoice(model_key=k, adaptation="finetune") for k in per_encoder]
    rp = None
    try:
        profile = dataset_analyzer.analyze(
            cfg.data_path, task_type=cfg.task.get("type", "classification"))
        rp = report_mod.write_report(parent, profile, cfg, choices, per_encoder,
                                     overall, per_fold_summary=pfs, cross_fold=cross)
    except Exception:
        rp = None
    _write_manifest(parent, entries, pj.get("devices") or [], True)
    return rp


# ---------------------------------------------------------------------------
# 未來: 多節點執行 (PyTorchJob) — 預留介面
# ---------------------------------------------------------------------------
class PyTorchJobLauncher:
    """未來以 Kubernetes PyTorchJob 在不同節點跑每個 fold 的啟動器 (尚未實作)。

    實作方向 (替換 `_launch_fold`):
      1. 把 child_cfg dump 到 `sub_dir/config.yaml`（sub_dir 須位於共享儲存 PVC，
         Web UI 才讀得到各 fold 的對話/ledger/解答樹）。
      2. 依樣板產生 PyTorchJob CRD：image、command=`python -m agent.auto_finetune
         --config <sub>/config.yaml`、resources.limits.nvidia.com/gpu=1、掛載 PVC。
      3. `kubectl apply -f` 提交，輪詢 job 狀態直到完成，再從 `sub_dir` 讀回 best。
      4. run_fleet 的協調、folds.json 契約、彙整與 UI 皆不需改動 —— 只換這個 launcher。
    """

    def launch(self, child_cfg: AgentConfig, sub_dir: str) -> dict:  # pragma: no cover
        raise NotImplementedError(
            "PyTorchJobLauncher 尚未實作；目前以本機多 GPU (_launch_fold) 執行。")
