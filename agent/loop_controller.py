"""LoopController — 步驟 5 的兩層迴圈編排 (設計文件 §5.7).

外層 (廣度, P1 重點): 對 Advisor 建議的多個 encoder 逐一比較。
內層 (深度, 改良): 每個 encoder 內做 AIDE 式解答樹搜尋 (參考 WecoAI/aideml):
    每個 trial 是樹上一個 Node, 每輪由 _search_policy 選下一步 —
      1. draft   — 起手不足 num_drafts 時, 以不同 preset 再開一條新起點 (無 parent);
      2. debug   — 以 debug_prob 機率挑一個 buggy leaf (debug_depth 未超限) 修正重跑;
      3. improve — greedy 選 metric 最佳節點, 交給 Advisor.review_and_decide 變異。
    metric 回饋自然修剪: 差的分支不再被 greedy 選中, 樹落地 search_tree.json。

停止條件 StopPolicy (聯集): max_trials 硬上限、每 encoder patience 無提升、
    時間預算 (budget.max_wall_clock_min)、Advisor 判定收斂 (review 的 stop)。
維護跨 encoder 全域 best_trial, 產出 report.md。
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from typing import Optional

from . import (conversation, dataset_analyzer, evaluator as ev, log_curves,
               metric_registry as mreg, presets, report as report_mod)
from .config import AgentConfig
from .journal import Journal, Node
from .ledger import Ledger
from .schemas import DatasetProfile, EvalConfig, Recipe, TrialResult
from .trainer import run_trial


def _better(a: float, b: Optional[float]) -> bool:
    return b is None or a > b


def discover_folds(data_path: str) -> list[str]:
    """由單一 fold 路徑推出所有 sibling fold (…_foldN)。找不到則回傳 [data_path]。"""
    data_path = os.path.abspath(os.path.normpath(data_path))
    parent, name = os.path.dirname(data_path), os.path.basename(data_path)
    m = re.search(r"(.*_fold)(\d+)$", name)
    if not m or not os.path.isdir(parent):
        return [data_path]
    prefix = m.group(1)
    folds = sorted(
        os.path.join(parent, d) for d in os.listdir(parent)
        if re.fullmatch(re.escape(prefix) + r"\d+", d)
        and os.path.isdir(os.path.join(parent, d))
    )
    return folds or [data_path]


class LoopController:
    def __init__(self, cfg: AgentConfig, advisor, run_dir: str):
        self.cfg = cfg
        self.advisor = advisor
        self.run_dir = os.path.abspath(run_dir)
        self.ledger = Ledger(self.run_dir)
        self.trials_dir = os.path.join(self.run_dir, "trials")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.eval_cfg: EvalConfig = cfg.eval
        self._t0 = 0.0
        self.n_trials = 0
        self.best: Optional[TrialResult] = None
        self._stopped = False   # 使用者/LLM 中斷
        self.journal: Optional[Journal] = None  # 全域解答樹 (落地 search_tree.json)
        self._last_gpu_stats: dict = {}  # 最近一個 trial 的 GPU 取樣 (draft 的 boost 依據)
        self._resume_fails = 0           # 連續無提升的 resume 次數 (自適應停用)
        self._resume_disabled = False    # 連續失敗達上限後本 run 停用 resume
        # LLM 完整 prompt/回應落地 (LLMAdvisor 支援時)
        if hasattr(advisor, "log_path"):
            advisor.log_path = os.path.join(self.run_dir, "llm_calls.jsonl")

    # ---- 護欄 ----------------------------------------------------------
    def _budget_exhausted(self) -> bool:
        if self.n_trials >= self.cfg.loop.max_trials:
            return True
        max_min = self.cfg.budget.max_wall_clock_min
        if max_min is not None and (time.time() - self._t0) / 60.0 >= max_min:
            return True
        return False

    def _write_current(self, task_id: str, recipe: Recipe, log_path: str) -> None:
        try:
            with open(os.path.join(self.run_dir, "current_trial.json"),
                      "w", encoding="utf8") as f:
                json.dump({"trial_id": task_id, "recipe": recipe.model_dump(),
                           "log": os.path.basename(log_path)}, f, ensure_ascii=False)
        except Exception:
            pass

    def _clear_current(self) -> None:
        try:
            os.remove(os.path.join(self.run_dir, "current_trial.json"))
        except OSError:
            pass

    def _execute(self, recipe: Recipe, profile: DatasetProfile,
                 task_id: str) -> TrialResult:
        """組指令 -> 訓練 -> 評估 -> 落地 ledger, 回傳 TrialResult。"""
        log_path = os.path.join(self.logs_dir, f"log_{task_id}.txt")
        trial = TrialResult(trial_id=task_id, recipe=recipe, status="running")

        head = recipe.heads[0].type if recipe.heads else "-"
        loss = recipe.losses[0].name if recipe.losses else "-"
        mut = recipe.provenance.get("mutation", "-")
        if not self.cfg.dry_run:
            print(f"\n{'=' * 72}\n"
                  f"▶ trial {self.n_trials + 1}/{self.cfg.loop.max_trials}: {task_id}\n"
                  f"  encoder={recipe.encoder.model_key} ({recipe.encoder.adaptation})  "
                  f"head={head}  loss={loss}  mutation={mut}\n"
                  f"  fold={os.path.basename(os.path.normpath(self.cfg.data_path))}  "
                  f"epochs={recipe.hparams.epochs}  blr={recipe.hparams.blr}\n"
                  f"  log: {log_path}\n"
                  f"{'=' * 72}", flush=True)

        # 供 web 即時顯示「進行中 trial」的完整參數 (ledger 只在完成後才有)
        if not self.cfg.dry_run:
            self._write_current(task_id, recipe, log_path)

        def _on_low_util(snap: dict) -> None:
            # 訓練中警示 (一次): 使用者當下就看得到 agent 已注意到低利用率
            self._say("system",
                      f"⚠ 進行中 trial {task_id} 的 GPU 利用率偏低"
                      f"（目前平均 {snap.get('util_avg', 0):.0f}%，"
                      f"目標 ≥{self.cfg.gpu.util_target:.0f}%）。"
                      f"訓練中無法調整 batch，此 trial 完成後會自動對後續 recipe "
                      f"採取措施（加大 batch / 增加 dataloader worker）。", kind="status")

        tr = run_trial(
            recipe, data_path=self.cfg.data_path, num_classes=profile.num_classes,
            output_dir=self.trials_dir, task_id=task_id, device_index=self.cfg.device,
            log_path=log_path, dry_run=self.cfg.dry_run,
            stream=self.cfg.stream_logs,
            gpu_sample_interval_s=self.cfg.gpu.sample_interval_s,
            low_util_target=(self.cfg.gpu.util_target
                             if self.cfg.gpu.optimize else None),
            on_low_util=_on_low_util,
        )
        trial.log_path = tr.get("log_path")
        trial.gpu_stats = tr.get("gpu_stats") or {}
        if trial.gpu_stats:
            frac = log_curves.data_fraction(log_path)
            if frac is not None:
                trial.gpu_stats["data_frac"] = frac  # 資料載入佔比 (dataloader 瓶頸判斷)
            self._last_gpu_stats = {**trial.gpu_stats, "_from": task_id}

        if self.cfg.dry_run:
            trial.status = "pending"
            trial.message = "dry_run: 已組指令未執行。" + (
                f" 未接元件: {tr['unsupported']}" if tr["unsupported"] else "")
            trial.metrics = {"_command": 0.0}  # 佔位, 供 report 顯示
            return trial

        if tr["returncode"] == 0:
            res = ev.evaluate(tr["task_dir"], self.eval_cfg, mode="test")
            trial.metrics = res["metrics"]
            trial.primary_score = res["primary_score"]
            trial.ckpt_path = res["ckpt_path"]
            trial.epoch_curve = log_curves.parse(log_path)  # 逐 epoch 曲線, 供 epochs 判斷
            trial.status = "done"
        else:
            trial.status = "failed"
            trial.message = f"main_finetune 回傳碼 {tr['returncode']}, 見 log。"
        self.ledger.append(trial)
        self.n_trials += 1
        gb = f"{self.best.primary_score:.4f}" if self.best else "—"
        gpu = (f"  gpu_util={trial.gpu_stats['util_avg']:.0f}%"
               if trial.gpu_stats.get("util_avg") is not None else "")
        print(f"✔ {task_id}: status={trial.status} "
              f"primary({self.eval_cfg.primary_metric})="
              f"{trial.primary_score:.4f}  [全域最佳={gb}]{gpu}", flush=True)
        return trial

    # ---- 全域 AIDE 式樹搜尋 (單一解答樹, 跨 encoder) ---------------------
    def _select_improve_node(self, journal: Journal) -> tuple[Node, float]:
        """依指標機率選要改善的節點: 對所有成功節點的 primary 分數做 softmax
        加權抽樣 (improve_temperature 控制; <=0 = greedy 只選最佳)。
        回傳 (節點, 被選中的機率) — 機率會顯示給使用者。"""
        good = journal.good_nodes
        T = self.cfg.loop.improve_temperature
        if T <= 0 or len(good) == 1:
            return journal.get_best_node(), 1.0
        mx = max(n.metric for n in good)
        ws = [math.exp((n.metric - mx) / T) for n in good]
        tot = sum(ws)
        r = random.random() * tot
        acc = 0.0
        for n, w in zip(good, ws):
            acc += w
            if r <= acc:
                return n, w / tot
        return good[-1], ws[-1] / tot

    def _search_policy(self, journal: Journal) -> tuple[Optional[Node], float]:
        """選下一輪要長出 child 的節點 (None = 開新 draft)。對應 aideml
        Agent.search_policy, 但 improve 改為依指標機率抽樣 (使用者可看到機率):
        1. draft 不足 num_drafts → 繼續 draft;
        2. 以 debug_prob 機率挑失敗 leaf 除錯;
        3. 依 primary 分數 softmax 抽一個成功節點改善; 無成功節點 → 回頭 draft。"""
        scfg = self.cfg.loop
        if len(journal.draft_nodes) < scfg.num_drafts:
            return None, 1.0

        if random.random() < scfg.debug_prob:
            debuggable = [
                n for n in journal.buggy_nodes
                if n.is_leaf and n.debug_depth <= scfg.max_debug_depth
                and not n.debug_exhausted
            ]
            if debuggable:
                return random.choice(debuggable), 1.0 / len(debuggable)

        if not journal.good_nodes:
            return None, 1.0
        return self._select_improve_node(journal)

    def _draft_choice(self, journal: Journal, choices) -> tuple:
        """第 n 個 draft 的 (encoder, preset): 先輪流各 encoder (多樣性),
        同一 encoder 再次 draft 時換下一組超參 preset。"""
        n = len(journal.draft_nodes)
        enc = choices[n % len(choices)]
        base = self.cfg.advisor.preset
        order = [base] + [p for p in presets.names() if p != base]
        preset = order[(n // len(choices)) % len(order)]
        return enc, preset

    def _debug_recipe(self, profile: DatasetProfile, encoder,
                      node: Node) -> Optional[Recipe]:
        """除錯 (對應 aideml debug 階段): 讀失敗 trial 的 log 尾端, 交給 Advisor
        提出修正 Recipe。advisor=llm 且 allow_code_edit=True 時, LLM 可一併修改
        訓練程式 — 修改版由 code_workspace 放在 <run_dir>/src/ 之下執行,
        原始程式不會被修改。回 None = 無從修起, 放棄該分支。"""
        tail = ""
        lp = node.trial.log_path if node.trial else None
        if lp and os.path.isfile(lp):
            try:
                with open(lp, errors="ignore") as f:
                    tail = "".join(f.readlines()[-120:])
            except OSError:
                pass
        return self.advisor.propose_debug(
            profile, encoder, node.trial, tail,
            workspace_dir=os.path.join(self.run_dir, "src"))

    def _maybe_resume(self, node: Node) -> Optional[Recipe]:
        """繼續訓練策略: 節點訓練完但 curve 未收斂 → 從其 checkpoint 續訓
        resume_epochs 個 epoch (新 trial 節點, stage=resume)。不適用回 None。

        避免 resume 壟斷/浪費的護欄:
          - 已有 resume child 的節點不再續訓 (重複續訓 = 完全相同的計算);
          - resume 節點自己沒賺 (≤ parent + min_delta) 不再往下追;
          - 符合條件也只以 resume_prob 機率選擇, 讓 improve 有機會;
          - 全 run 連續 max_failed_resumes 次 resume 無提升 → 停用 (self._resume_disabled)。"""
        scfg = self.cfg.loop
        if not scfg.resume_unconverged or scfg.resume_epochs <= 0:
            return None
        if self._resume_disabled:
            return None
        if node.resume_depth >= scfg.max_resumes:
            return None
        if any(c.stage == "resume" for c in node.children):
            return None  # 這個節點的續訓已存在 (child), 再做一次是重複計算
        if (node.stage == "resume" and node.parent is not None
                and node.parent.metric is not None and node.metric is not None
                and node.metric <= node.parent.metric + scfg.min_delta):
            return None  # 上一段續訓沒帶來提升, 不再往下追
        t = node.trial
        if (t is None or t.status != "done" or not t.ckpt_path
                or not os.path.isfile(t.ckpt_path)):
            return None
        unc, why = log_curves.unconverged(t.epoch_curve or {})
        if not unc:
            return None
        if random.random() >= scfg.resume_prob:
            return None  # 機率性選擇: 其餘機率落到 improve, 避免 resume 壟斷

        r = t.recipe.model_copy(deep=True)
        r.resume_from = t.ckpt_path
        r.resume_epochs = scfg.resume_epochs
        r.hparams.epochs = r.hparams.epochs + scfg.resume_epochs  # 顯示用; 實際由 --more_epochs 控制
        prov = dict(r.provenance)
        prov["mutated_from"] = {"trial": t.trial_id,
                                "mutation": t.recipe.provenance.get("mutation")}
        prov["mutation"] = f"continue_training:+{scfg.resume_epochs}ep"
        prov["reason"] = f"未收斂 ({why}), 從 {os.path.basename(t.ckpt_path)} 續訓"
        r.provenance = prov
        return r

    def _cache_short_side(self, profile: DatasetProfile) -> Optional[int]:
        """run 起手決定是否啟用預縮圖快取 (整個 run 的所有 trial 一致, 保可比性):
        原圖中位短邊 ≥ 1.5×cache_short_side (高解析度、decode 昂貴) 才啟用。"""
        g = self.cfg.gpu
        if not g.cache_auto or g.cache_short_side <= 0:
            return None
        stats = profile.image_size_stats or {}
        try:
            short = min(stats["width"]["median"], stats["height"]["median"])
        except (KeyError, TypeError):
            return None
        return g.cache_short_side if short >= 1.5 * g.cache_short_side else None

    def _apply_gpu_boost(self, recipe: Recipe, parent: Optional[Node],
                         n_train: Optional[int] = None) -> None:
        """GPU 利用率最佳化: 依據節點的 GPU 取樣 (draft 無 parent 時退回「最近
        一個 trial」的取樣), 平均利用率低於 gpu.util_target 時對本輪 recipe 採取措施:
          - 記憶體有餘裕 → batch_size 加倍 (利用率 < 目標一半且記憶體夠時 ×4);
            accum_iter>1 時折算維持有效 batch (accum=1 時 blr 依 MAE 慣例
            按有效 batch 自動縮放);
          - 資料載入佔比 data_frac 高 (dataloader 瓶頸) → 增加 num_workers
            (上限 clamp 到每 epoch iteration 數 — worker 多於 iteration 無意義)。
        兩者可同時採用。resume 節點不適用 (main_finetune 沿用 checkpoint 內的
        args)。調整記錄在 provenance.gpu_opt 並告知使用者。"""
        g = self.cfg.gpu
        stats = (parent.trial.gpu_stats or {}) if parent and parent.trial else {}
        src = parent.id if parent and parent.trial else None
        if not stats and self._last_gpu_stats:
            stats = self._last_gpu_stats           # draft: 用最近 trial 的量測
            src = stats.get("_from")
        util = stats.get("util_avg")
        if not g.optimize or util is None or util >= g.util_target:
            return
        hp = recipe.hparams
        peak, total = stats.get("mem_peak_mb"), stats.get("mem_total_mb")
        data_frac = stats.get("data_frac")
        changes = {}

        # 1) 記憶體有餘裕 → 加大 batch (利用率極低且記憶體夠時 ×4)
        factor = 4 if util < g.util_target / 2 else 2
        while factor > 1:
            if (peak and total and peak * factor <= total * g.mem_target
                    and hp.batch_size * factor <= g.max_batch_size):
                break
            factor //= 2
        if factor > 1:
            changes["batch_size"] = {"from": hp.batch_size,
                                     "to": hp.batch_size * factor}
            hp.batch_size *= factor
            if hp.accum_iter > 1:
                new_ai = max(1, hp.accum_iter // factor)
                changes["accum_iter"] = {"from": hp.accum_iter, "to": new_ai}
                hp.accum_iter = new_ai

        # 2) dataloader 瓶頸 (資料載入佔比高) → 增加 worker (以調整後 batch 計算
        #    每 epoch iteration 數作上限; 小資料集 worker 多於 iteration 無意義)
        cur_workers = hp.num_workers if hp.num_workers is not None else 10
        target_workers = g.max_num_workers
        if n_train:
            iters = max(1, n_train // max(1, hp.batch_size))
            target_workers = min(target_workers, max(2, iters))
        if (data_frac is not None and data_frac >= 0.3
                and cur_workers < target_workers):
            changes["num_workers"] = {"from": cur_workers, "to": target_workers}
            hp.num_workers = target_workers

        if not changes:
            # 利用率低但無計可施 (batch/worker 都到頂) → 至少告知
            self._say("system",
                      f"⚠ GPU 利用率 {util:.0f}% 偏低，但 batch/num_workers 已達上限，"
                      f"無法再自動提高。", kind="status")
            return
        prov = dict(recipe.provenance)
        prov["gpu_opt"] = {"from_trial": src, "util_avg": util,
                           "data_frac": data_frac,
                           "target": g.util_target, "changes": changes}
        recipe.provenance = prov
        desc = "、".join(f"{k} {v['from']}→{v['to']}" for k, v in changes.items())
        why = []
        if data_frac is not None and data_frac >= 0.3:
            why.append(f"資料載入佔 {data_frac:.0%} 步時（dataloader 瓶頸）")
        if peak and total:
            why.append(f"記憶體峰值 {peak / 1024:.1f}/{total / 1024:.1f} GB")
        self._say("system",
                  f"⚡ GPU 最佳化：{src} 訓練時平均 GPU 利用率 {util:.0f}%，"
                  f"低於目標 {g.util_target:.0f}%（{'；'.join(why)}），"
                  f"本輪 recipe 調整：{desc}", kind="status")

    def _write_trees(self) -> None:
        if self.journal is None:
            return
        try:
            with open(os.path.join(self.run_dir, "search_tree.json"),
                      "w", encoding="utf8") as f:
                json.dump(self.journal.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _tree_search(self, profile: DatasetProfile, choices,
                     journal: Optional[Journal] = None,
                     history: Optional[list[TrialResult]] = None) -> list[TrialResult]:
        """單一全域解答樹 (AIDE 式): 先開滿 num_drafts 個 draft (輪流各 encoder),
        之後每輪由 _search_policy 選節點 (debug / resume / 依指標機率 improve)。
        每輪都會在對話中說明本輪從哪個節點開始。
        resume 時傳入重建好的 journal + 既有 history, 從下一輪接續。"""
        preloaded = bool(history)
        journal = self.journal = journal if journal is not None else Journal()
        history = history if history is not None else []
        ds_name = os.path.basename(os.path.normpath(self.cfg.data_path))
        stale = 0
        max_rounds = self.cfg.loop.max_trials
        max_attempts = max_rounds * 3  # 護欄: 選點連續失敗時不空轉

        # run 起手決定預縮圖快取 (高解析度資料集的 decode 瓶頸; 全 run 一致)
        # (依 profile+config 決定, 具決定性 → resume 時結果與先前一致, 保可比性)
        cache_n = self._cache_short_side(profile)
        if cache_n and not self.cfg.dry_run and not preloaded:
            st = profile.image_size_stats or {}
            wh = (f"{st.get('width', {}).get('median', '?')}×"
                  f"{st.get('height', {}).get('median', '?')}")
            self._say("system",
                      f"⚡ GPU 最佳化：原圖解析度高（中位 {wh}），本 run 所有 trial "
                      f"啟用預縮圖快取（短邊 {cache_n}，首個 epoch 建立快取後 "
                      f"decode 成本大幅下降），並開啟 persistent_workers。", kind="status")

        for attempt in range(max_attempts):
            if len(history) >= max_rounds:
                break
            # 使用者中斷?
            stop, why = conversation.stop_requested(self.run_dir)
            if stop:
                self._say("system", f"實驗中斷（{why}）。", kind="final")
                self._stopped = True
                break
            if self._budget_exhausted():
                self._say("system", "達預算上限，停止。", kind="final")
                break

            k = len(history)
            # ---- policy: 決定這一輪長在哪個節點下 ------------------------
            parent, prob = self._search_policy(journal)
            if parent is None:
                stage = "draft"
                encoder, preset = self._draft_choice(journal, choices)
                self._say("system",
                          f"第 {k} 輪（draft）：開新起點 draft "
                          f"{len(journal.draft_nodes) + 1}/{self.cfg.loop.num_drafts} — "
                          f"encoder={encoder.model_key}（{encoder.adaptation}）, "
                          f"preset={preset}（{self.cfg.advisor.type} 決策中，請稍候…）",
                          kind="status", round=k)
                recipe = self.advisor.compose_recipe(profile, encoder, preset=preset)
                self._say("llm", f"draft [{encoder.model_key}]（preset={preset}）："
                          f"head={recipe.heads[0].type if recipe.heads else '-'}, "
                          f"loss={recipe.losses[0].name if recipe.losses else '-'}。"
                          + (f" {recipe.encoder.rationale}" if recipe.encoder.rationale else ""),
                          kind="decision", round=k)
            elif parent.is_buggy:
                stage = "debug"
                encoder = parent.recipe.encoder
                self._say("system", f"第 {k} 輪（debug）：從失敗節點 {parent.id} "
                          f"開始除錯（debug_prob 抽中）。", kind="status", round=k)
                recipe = self._debug_recipe(profile, encoder, parent)
                if recipe is None:
                    parent.debug_exhausted = True
                    self._say("system", f"{parent.id} 無從除錯，放棄該分支。",
                              kind="status", round=k)
                    continue
                self._say("llm", f"除錯 {parent.id}："
                          f"{recipe.provenance.get('mutation', '')}",
                          kind="decision", round=k)
            elif (resumed := self._maybe_resume(parent)) is not None:
                # 繼續訓練策略: 選中節點 curve 未收斂 → 先從其 checkpoint 續訓, 不變異
                stage = "resume"
                encoder = parent.recipe.encoder
                self._say("system",
                          f"第 {k} 輪（resume）：從節點 {parent.id}"
                          f"（score={parent.metric:.4f}，被選機率 {prob:.0%}）繼續訓練。",
                          kind="status", round=k)
                recipe = resumed
                self._say("llm", f"繼續訓練 {parent.id}："
                          f"{recipe.provenance.get('reason', '')}"
                          f"（+{recipe.resume_epochs} epochs）", kind="decision", round=k)
            else:
                stage = "improve"
                encoder = parent.recipe.encoder
                # 依指標機率選中的節點 → 交給 Advisor (以它為變異基準) 檢視 + 討論
                self._say("system",
                          f"第 {k} 輪（improve）：本輪從節點 {parent.id}"
                          f"（encoder={encoder.model_key}, score={parent.metric:.4f}，"
                          f"依指標抽樣機率 {prob:.0%}）開始改善；"
                          f"{self.cfg.advisor.type} 決策中，請稍候…",
                          kind="status", round=k)
                discussion = conversation.read(self.run_dir)
                # workspace_dir: 允許修改程式時 (allow_code_edit), improve 階段可
                # 對 main_finetune.py 提 code_edits (副本隔離, 原始程式不動)
                action = self.advisor.review_and_decide(
                    profile, encoder, history, discussion, base=parent.trial,
                    workspace_dir=os.path.join(self.run_dir, "src"))
                if action.narrative:
                    self._say("llm", action.narrative, kind="review", round=k)
                if action.stop or action.next_recipe is None:
                    self._say("llm", f"決定停止：{action.reason or '已收斂'}",
                              kind="decision", round=k)
                    break
                self._say("llm", f"對節點 {parent.id} 的變異：{action.mutation}"
                          f"（{action.reason}）", kind="decision", round=k)
                recipe = action.next_recipe

            if stage != "resume":
                # resume 欄位只屬於 resume 節點; 從 resume 節點變異出的 child 要重新從頭訓練
                recipe.resume_from, recipe.resume_epochs = None, 0
                # 預縮圖快取: 全 run 一致 (resume 沿用 checkpoint args, 天然一致)
                if cache_n:
                    recipe.hparams.cache_resized = cache_n
                # GPU 利用率最佳化: 參考節點 (draft 用最近一個 trial) 利用率太低
                # → 加大 batch / 增加 dataloader worker
                self._apply_gpu_boost(recipe, parent, n_train=profile.n_train)
            prov = dict(recipe.provenance)
            prov["search"] = {"stage": stage,
                              "parent": parent.id if parent is not None else None,
                              "select_prob": round(prob, 4)}
            if cache_n and stage != "resume":
                prov["cache_resized"] = cache_n
            recipe.provenance = prov
            node = journal.append(Node(recipe, parent=parent, stage=stage))

            # ---- 執行 ---------------------------------------------------
            task_id = f"{recipe.encoder.model_key}_{ds_name}_t{k}"
            self._say("system", f"開始第 {k} 輪 trial：{task_id}（{stage}）",
                      kind="status", round=k, trial_id=task_id)
            trial = self._execute(recipe, profile, task_id)
            node.trial = trial
            history.append(trial)
            self._write_trees()

            scfg = self.cfg.loop
            if trial.status == "done":
                prev = self.best.primary_score if self.best else None
                if _better(trial.primary_score, prev):
                    self.best = trial
                if prev is None or trial.primary_score > prev + scfg.min_delta:
                    stale = 0          # 有實質提升 (任何 stage) → 歸零
                elif stage in ("improve", "resume"):
                    stale += 1         # 只有「改良嘗試」無提升才累計; draft/debug 不計

            # resume 成效追蹤: 連續 max_failed_resumes 次沒賺 → 本 run 停用 resume
            if stage == "resume" and trial.status == "done" and parent is not None:
                pm = parent.metric
                if pm is not None and trial.primary_score > pm + scfg.min_delta:
                    self._resume_fails = 0
                else:
                    self._resume_fails += 1
                    if (not self._resume_disabled
                            and self._resume_fails >= scfg.max_failed_resumes):
                        self._resume_disabled = True
                        self._say("system",
                                  f"連續 {self._resume_fails} 次繼續訓練皆無提升，"
                                  f"本 run 停用 resume 策略（改良輪次全數交給 improve）。",
                                  kind="status", round=k)
            done_msg = (f"完成 {task_id}：status={trial.status}，"
                        f"primary({self.eval_cfg.primary_metric})="
                        f"{trial.primary_score:.4f}"
                        f"（全域最佳={self.best.primary_score:.4f}）"
                        if self.best else f"完成 {task_id}：status={trial.status}")
            u = trial.gpu_stats.get("util_avg")
            if u is not None:
                done_msg += f"，GPU 利用率 {u:.0f}%"
                if u < self.cfg.gpu.util_target:
                    done_msg += "（偏低，下一輪將自動調整）"
            self._say("system", done_msg, kind="status", round=k, trial_id=task_id)

            if len(history) >= scfg.min_trials and stale >= scfg.patience:
                msg = (f"已完成 {len(history)} 輪（≥ min_trials={scfg.min_trials}），"
                       f"且連續 {stale} 輪改良/續訓無提升"
                       f"（patience={scfg.patience}，draft/debug 不計），停止。")
                if self.best is not None:
                    msg += f"全域最佳 primary={self.best.primary_score:.4f}。"
                self._say("system", msg, kind="final")
                break
        return history

    def _say(self, role: str, text: str, kind: str = "msg", **extra) -> None:
        conversation.append(self.run_dir, role, text, kind=kind, **extra)

    # ---- 繼續實驗: 從 run 目錄載回狀態 ---------------------------------
    def _load_profile(self) -> Optional[DatasetProfile]:
        p = os.path.join(self.run_dir, "dataset_profile.json")
        try:
            with open(p, encoding="utf8") as f:
                return DatasetProfile.model_validate_json(f.read())
        except Exception:
            return None

    @staticmethod
    def _resume_choices(journal: Journal) -> list:
        """從既有 draft 節點還原 encoder 輪替順序 (保持與先前一致)。"""
        out, seen = [], set()
        for n in journal.draft_nodes:
            k = n.recipe.encoder.model_key
            if k not in seen:
                seen.add(k)
                out.append(n.recipe.encoder)
        return out

    # ---- 外層: 多 encoder 廣度比較 -------------------------------------
    def run(self, resume: bool = False) -> dict:
        self._t0 = time.time()
        profile = self._load_profile() if resume else None
        if profile is None:
            profile = dataset_analyzer.analyze(
                self.cfg.data_path, task_type=self.cfg.task.get("type", "classification"))
        self.ledger.write_profile(profile.model_dump_json(indent=2))
        self.cfg.dump_yaml(os.path.join(self.run_dir, "config.yaml"))

        journal: Optional[Journal] = None
        prior_history: Optional[list[TrialResult]] = None
        choices: list = []
        if resume:
            # 上次的中斷旗標要先清掉, 否則第一輪就會再次停止
            conversation.clear_stop(self.run_dir)
            prior = self.ledger.history()
            # _bestfold 是多 fold 彙整的重跑, 不屬於解答樹
            prior_history = [t for t in prior
                             if not (t.trial_id or "").endswith("_bestfold")]
            journal = Journal.rebuild(prior_history)
            self.n_trials = len(prior)
            done = [t for t in prior_history if t.status == "done"]
            self.best = max(done, key=lambda t: t.primary_score, default=None)
            choices = self._resume_choices(journal)
            if not self.cfg.dry_run:
                b = (f"目前最佳 {self.best.recipe.encoder.model_key} "
                     f"primary={self.best.primary_score:.4f}。" if self.best else "")
                self._say("system",
                          f"▶ 繼續實驗：已載入先前 {len(prior_history)} 個 trial，"
                          f"解答樹已重建。{b}從第 {len(prior_history)} 輪接續"
                          f"（max_trials={self.cfg.loop.max_trials}）。", kind="status")
        elif not self.cfg.dry_run:
            self._say("system", f"實驗開始：資料 {os.path.basename(os.path.normpath(self.cfg.data_path))}"
                      f"（{profile.num_classes} 類, 不平衡比 {profile.imbalance_ratio}）。"
                      f"advisor={self.cfg.advisor.type}。你可以隨時在下方留言引導方向，"
                      f"或按「中斷實驗」停止。", kind="status")
            self._say("system", f"{self.cfg.advisor.type} 正在依資料挑選 encoder（請稍候…）",
                      kind="status")

        if not choices:   # 全新 run, 或 resume 時連一個 draft 都還沒跑
            choices = self.advisor.select_encoders(profile)
            if not choices:
                raise RuntimeError("沒有可用的 encoder (檢查 baseline_models/ 權重是否存在)")
            # draft 輪替的 encoder 上限 = num_drafts (encoders_per_run 已停用)
            choices = choices[: max(1, self.cfg.loop.num_drafts)]
            if not self.cfg.dry_run:
                self._say("llm", "依資料選出要比較的 encoder："
                          + "、".join(f"{c.model_key}({c.adaptation})" for c in choices),
                          kind="decision")

        # 單一全域解答樹搜尋 (draft 跨 encoder; per_encoder 僅供 report 分組)
        history = self._tree_search(profile, choices,
                                    journal=journal, history=prior_history)
        per_encoder: dict[str, list[TrialResult]] = {}
        for t in history:
            per_encoder.setdefault(t.recipe.encoder.model_key, []).append(t)

        # P5: 多 fold 彙整 — 先在單 fold 篩選, 再把全域最佳 Recipe 擴到所有 fold (§5.7)
        fold_summary = self._maybe_multifold(profile)

        self._clear_current()  # 全部完成, 移除「進行中」標記

        if not self.cfg.dry_run:
            if self.best is not None:
                self._say("llm", f"實驗結束。全域最佳：{self.best.recipe.encoder.model_key} "
                          f"primary={self.best.primary_score:.4f}（trial={self.best.trial_id}）。"
                          + ("（已中斷）" if self._stopped else ""), kind="final")
            else:
                self._say("system", "實驗結束，無成功 trial。", kind="final")

        report_path = report_mod.write_report(
            self.run_dir, profile, self.cfg, choices, per_encoder,
            self.best, dry_run=self.cfg.dry_run, fold_summary=fold_summary)

        return {
            "profile": profile,
            "choices": choices,
            "per_encoder": per_encoder,
            "best": self.best,
            "n_trials": self.n_trials,
            "fold_summary": fold_summary,
            "report_path": report_path,
            "run_dir": self.run_dir,
        }

    def _maybe_multifold(self, profile: DatasetProfile) -> Optional[dict]:
        """aggregation=mean_std 時, 把全域最佳 Recipe 在所有 sibling fold 重跑並彙整。"""
        if self.cfg.eval.aggregation != "mean_std" or self.best is None:
            return None
        folds = discover_folds(self.cfg.data_path)
        if len(folds) <= 1:
            return None
        base_fold = os.path.abspath(os.path.normpath(self.cfg.data_path))
        scores, metrics_list, done_folds = [], [], []
        # 已在 base_fold 跑過的最佳成績直接沿用
        scores.append(self.best.primary_score)
        metrics_list.append(self.best.metrics)
        done_folds.append(os.path.basename(base_fold))

        # 繼續實驗時: 先前已完成的 _bestfold trial 若 recipe 相同, 直接沿用不重跑
        prior_folds = {t.trial_id: t for t in self.ledger.history()
                       if t.status == "done"
                       and (t.trial_id or "").endswith("_bestfold")}

        for fold in folds:
            if os.path.abspath(fold) == base_fold:
                continue
            if self._budget_exhausted():
                break
            fname = os.path.basename(fold)
            task_id = f"{self.best.recipe.encoder.model_key}_{fname}_bestfold"
            # 用最佳 recipe 在此 fold 重跑; 若最佳來自 resume 節點, 其他 fold 要從頭訓練
            # (不能沿用原 fold 的 checkpoint), epochs 已含續訓延長量
            fold_recipe = self.best.recipe.model_copy(deep=True)
            fold_recipe.resume_from, fold_recipe.resume_epochs = None, 0
            cached = prior_folds.get(task_id)
            if cached is not None and cached.recipe.model_dump() == fold_recipe.model_dump():
                scores.append(cached.primary_score)
                metrics_list.append(cached.metrics)
                done_folds.append(fname)
                continue
            saved = self.cfg.data_path
            self.cfg.data_path = fold
            try:
                trial = self._execute(fold_recipe, profile, task_id)
            finally:
                self.cfg.data_path = saved
            if trial.status == "done":
                scores.append(trial.primary_score)
                metrics_list.append(trial.metrics)
                done_folds.append(fname)

        agg = mreg.aggregate(scores)
        return {
            "encoder": self.best.recipe.encoder.model_key,
            "folds": done_folds,
            "primary": agg,
            "metrics": mreg.aggregate_metrics(metrics_list),
        }
