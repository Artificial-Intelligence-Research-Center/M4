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

from . import (analyzers, conversation, dataset_analyzer,
               encoder_registry as reg, ensembler, evaluator as ev, log_curves,
               metric_registry as mreg, presets, report as report_mod)
from .analyzers import error_extract
from .config import AgentConfig
from .journal import Journal, Node
from .ledger import Ledger
from .privacy import PrivacyContext, load_user_facts
from .privacy.facts import ErrorFacts, UserQuestion
from .schemas import (DatasetProfile, EncoderChoice, EvalConfig, InfoRequest,
                      Recipe, TrialResult)
from .trainer import run_trial


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _better(a: float, b: Optional[float]) -> bool:
    return b is None or a > b


def _data_root_of(data_path: str) -> str:
    """出口掃描要用的資料根目錄 (用來推出「哪些目錄名屬於資料集識別」)。"""
    repo_data = os.path.realpath(os.path.join(_REPO_ROOT, "data"))
    real = os.path.realpath(data_path)
    if real == repo_data or real.startswith(repo_data + os.sep):
        return repo_data
    parent = os.path.dirname(real)
    return os.path.dirname(parent) or parent


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
        self.ensemble = None    # 最佳集成結果 (EnsembleResult) — 跨所有集成回合
        self.ensembles: list = []          # 全部集成回合 (方案A收尾 + 方案B搜尋中)
        self._n_search_ens = 0             # 已執行的搜尋中集成次數 (方案B 上限用)
        self._last_ens_pool = 0            # 上次搜尋中集成時的模型池大小 (避免每輪重跑)
        self._ens_llm = None               # 專用 LLM 成員選擇器 (llm_select 且主 advisor 非 LLM 時)
        self._cur_fold = None              # per_fold 模式: 目前搜尋中的 fold {index,name}
        self._stopped = False   # 使用者/LLM 中斷
        self.journal: Optional[Journal] = None  # 全域解答樹 (落地 search_tree.json)
        self._last_gpu_stats: dict = {}  # 最近一個 trial 的 GPU 取樣 (draft 的 boost 依據)
        self._resume_fails = 0           # 連續無提升的 resume 次數 (自適應停用)
        self._resume_disabled = False    # 連續失敗達上限後本 run 停用 resume
        # 資料圍欄 (docs/data_firewall_design.md): 需要 profile 才建得起來, 見 _setup_privacy
        self.privacy: Optional[PrivacyContext] = None
        self._ds_tag = "ds"              # trial_id 用的資料集假名 (不含真實名稱)
        self._asked: set[str] = set()    # 已經問過使用者的問題 key (不重複問)
        # LLM 完整 prompt/回應落地 (LLMAdvisor 支援時)
        if hasattr(advisor, "log_path"):
            advisor.log_path = os.path.join(self.run_dir, "llm_calls.jsonl")

    # ---- 資料圍欄 -------------------------------------------------------
    def _setup_privacy(self, profile: DatasetProfile) -> PrivacyContext:
        """建立 PrivacyContext 並掛到決策層上。

        決策層從這一刻起才拿得到資料集描述 — 而且只拿得到 DatasetFacts (假名化、
        無路徑、無真實類別名)。分析器目錄一併交給它, 讓它知道可以點名哪些程式。
        """
        exempt = ""
        if hasattr(self.advisor, "_registry_context"):
            try:
                exempt = self.advisor._registry_context()
            except Exception:
                exempt = ""
        n_folds = len(discover_folds(self.cfg.data_path))
        ctx = PrivacyContext.build(
            self.cfg.privacy, profile, run_dir=self.run_dir,
            data_root=_data_root_of(self.cfg.data_path),
            exempt_text=exempt, guidance=self.cfg.advisor.guidance,
            n_folds=n_folds if n_folds > 1 else None)
        self.privacy = ctx
        self._ds_tag = re.sub(r"[^\w.-]+", "_", ctx.alias.dataset_ref)
        if hasattr(self.advisor, "privacy"):
            self.advisor.privacy = ctx
        if hasattr(self.advisor, "analyzer_catalog"):
            self.advisor.analyzer_catalog = analyzers.catalog()
        return ctx

    def _fold_tag(self, fold: str) -> str:
        """某個 fold 的假名 (多 fold 彙整時的 trial_id 用)。"""
        from .privacy import alias as alias_mod
        salt = alias_mod.load_salt(self.cfg.privacy.salt_file)
        return re.sub(r"[^\w.-]+", "_", alias_mod.dataset_ref(fold, salt))

    def _analysis_dir(self) -> str:
        return os.path.join(self.run_dir, "analysis")

    # ---- 管道 A: 執行 LLM 點名的分析器 ---------------------------------
    def _run_analyses(self, keys: list[str], k: Optional[int] = None) -> None:
        """執行註冊表內的分析器; 目錄外的 key 一律拒絕並告知使用者。"""
        adv_list = getattr(self.advisor, "analyses", None)
        if adv_list is None:
            return
        done = {a.key for a in adv_list}
        for key in dict.fromkeys(keys):       # 去重且保序
            if key in done:
                continue
            a = analyzers.REGISTRY.get(key)
            if a is None:
                self._say("system", f"決策層要求執行分析器「{key}」，但它不在註冊表中，"
                          f"已拒絕（可用: {'、'.join(analyzers.REGISTRY)}）。",
                          kind="status", round=k)
                continue
            if (a.needs_consent
                    and self.cfg.privacy.expensive_analyzers_need_consent
                    and not self._has_consent(key)):
                self._ask_user([UserQuestion(
                    key=f"consent:{key}", kind="bool", blocking=False,
                    question=f"是否同意執行分析器「{key}」？（{a.description}）",
                    why=f"這是高成本分析（cost={a.cost}），需要你同意才會跑。")], k)
                continue
            self._say("system", f"執行分析器 {key}（{a.cost}）…", kind="status", round=k)
            facts = analyzers.run_cached(key, self.cfg.data_path,
                                         cache_dir=self._analysis_dir())
            adv_list.append(facts)
            if facts.ok:
                self._say("system", f"分析器 {key} 完成，結果已提供給決策層："
                          f"{json.dumps(facts.result, ensure_ascii=False)[:400]}",
                          kind="status", round=k)
            else:
                self._say("system", f"分析器 {key} 執行失敗：{facts.error}",
                          kind="status", round=k)

    def _has_consent(self, key: str) -> bool:
        uf = load_user_facts(self.run_dir)
        v = (uf.answer_for(f"consent:{key}") or "").strip().lower()
        return v in ("true", "yes", "是", "1", "同意")

    # ---- 管道 B: 請使用者親自回答 --------------------------------------
    def _ask_user(self, questions: list[UserQuestion],
                  k: Optional[int] = None) -> None:
        """把問題貼進討論頻道 (kind=question); web UI 會渲染成表單卡。"""
        if not self.cfg.privacy.ask_user_when_unsure:
            return
        fresh = [q for q in questions if q.key and q.key not in self._asked]
        if not fresh:
            return
        answered = load_user_facts(self.run_dir).answered_keys()
        fresh = [q for q in fresh if q.key not in answered]
        if not fresh:
            return
        for q in fresh:
            self._asked.add(q.key)
        conversation.append(
            self.run_dir, "llm",
            "我需要一些只有你知道的資料特性（我看不到原始資料）：\n"
            + "\n".join(f"・{q.question}" for q in fresh),
            kind="question", round=k,
            questions=[q.model_dump() for q in fresh])

    def _await_answers(self, keys: set[str], timeout_s: float = 1800.0,
                       poll_s: float = 5.0) -> bool:
        """等待使用者回答 (blocking 問題)。逾時或使用者中斷即放棄, 不卡死實驗。"""
        if not keys:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if keys <= load_user_facts(self.run_dir).answered_keys():
                return True
            stop, _ = conversation.stop_requested(self.run_dir)
            if stop:
                return False
            time.sleep(poll_s)
        return False

    def _handle_info(self, info: Optional[InfoRequest],
                     k: Optional[int] = None) -> None:
        """處理決策層的資訊需求 — 兩條合法管道 (分析器 / 問使用者)。"""
        if not info:
            return
        if info.analyses:
            self._run_analyses(info.analyses, k)
        if info.questions:
            self._ask_user(info.questions, k)
            blocking = {q.key for q in info.questions if q.blocking}
            if blocking:
                self._say("system",
                          f"等待你回答 {len(blocking)} 個問題後再繼續"
                          f"（最多等 30 分鐘，逾時會用保守預設繼續）。",
                          kind="status", round=k)
                if not self._await_answers(blocking):
                    self._say("system", "未取得回覆，改用保守預設繼續。",
                              kind="status", round=k)
        if self.privacy is not None:
            self.privacy.refresh_user_facts()

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
        # per_fold 模式: 蓋上 fold index (供 UI 分 fold 顯示; 非白名單欄位, 不進 LLM facts)
        if self._cur_fold is not None:
            recipe.provenance["fold"] = self._cur_fold["index"]
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
        回傳 (節點, 被選中的機率) — 機率會顯示給使用者。
        決策層已 prune 的分支 (improve_exhausted) 不再入選; 全被 prune 時回 (None, 1.0)。"""
        good = journal.improvable_nodes
        if not good:
            return None, 1.0
        T = self.cfg.loop.improve_temperature
        if T <= 0 or len(good) == 1:
            return max(good, key=lambda n: n.metric), 1.0
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
        3. 依 primary 分數 softmax 抽一個成功節點改善; 無可改善節點 → 回頭 draft。"""
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

        if not journal.improvable_nodes:
            return None, 1.0
        return self._select_improve_node(journal)

    def _resume_ready(self, node: Node) -> bool:
        """這個節點現在允許 resume 嗎 (只看護欄, 不擲骰子)。"""
        return self._maybe_resume(node, force=True) is not None

    def _debuggable(self, node: Node) -> bool:
        return (node.is_buggy and node.is_leaf and not node.debug_exhausted
                and node.debug_depth <= self.cfg.loop.max_debug_depth)

    def _debuggable_any(self, journal: Journal) -> bool:
        return any(self._debuggable(n) for n in journal.nodes)

    def _stop_looks_premature(self, journal: Journal, parent: Node,
                              k: int, max_rounds: int) -> bool:
        """決策層在 improve 輪回的 stop, 是否更像「只想放棄這條分支」?

        判定為 True 的條件 (兩者皆須成立):
          1. 輪數還沒過半 — 過半後的 stop 多半是真的收斂了, 尊重它;
          2. 除了本輪這個基準節點, 樹上還有其他沒被放棄的成功節點可以繼續改良。
        背景: stop 與 prune_branch 語意相近, 舊版 schema 甚至沒有 prune_branch,
        決策層在明顯走不通的節點上回 stop, 整個 fold 就跟著提早結束 (實例:
        run_20260729_151635/fold2 只跑 3 個 trial 就收工)。
        """
        if not getattr(self.cfg.loop, "stop_fuse", True):
            return False
        if k * 2 >= max_rounds:
            return False
        return any(n is not parent for n in journal.improvable_nodes)

    def _tree_view(self, journal: Journal) -> list[dict]:
        """把解答樹整理成可讀清單給決策層; allowed = 該節點現在允許的階段。"""
        out = []
        for n in journal.nodes:
            if not n.evaluated:
                continue
            allowed = []
            # 決策層先前 prune 掉的分支不再列為可長 child 的節點 (仍留在樹上供比較)
            if n.metric is not None and not n.improve_exhausted:
                allowed.append("improve")
                if self._resume_ready(n):
                    allowed.append("resume")
            if self._debuggable(n):
                allowed.append("debug")
            prov = n.recipe.provenance or {}
            out.append({
                "trial_id": n.id,
                "stage": n.stage,
                "parent": n.parent.id if n.parent is not None else None,
                "encoder": n.recipe.encoder.model_key,
                "adaptation": n.recipe.encoder.adaptation,
                "status": n.trial.status,
                "primary_score": n.metric,
                "epochs": n.recipe.hparams.epochs,
                "mutation": prov.get("mutation"),
                "is_leaf": n.is_leaf,
                "pruned": n.improve_exhausted,
                "allowed": allowed,
            })
        return out

    def _apply_search_override(self, journal: Journal, profile: DatasetProfile,
                               parent: Optional[Node], prob: float, k: int,
                               ) -> tuple[Optional[Node], float, Optional[str], dict]:
        """policy 選完 → 交給決策層過目, 讓討論/QA 結論真的能改變選到哪個節點。

        回傳 (parent, prob, forced_stage, draft_pref);
        forced_stage=None 表示沿用 policy (階段仍由節點狀態推導)。
        決策層的選擇會在這裡驗證, 不合法就沿用 policy 並把原因說給使用者聽。
        """
        if not self.cfg.loop.select_override:
            return parent, prob, None, {}
        # policy 這輪的提案 (與下方分支的推導規則一致)
        if parent is None:
            prop_stage = "draft"
        elif parent.is_buggy:
            prop_stage = "debug"
        else:
            prop_stage = "improve"   # resume 之後才擲骰; 對決策層一律呈現為 improve
        proposal = {"stage": prop_stage,
                    "parent_id": parent.id if parent is not None else None,
                    "select_prob": round(prob, 4),
                    "note": ("draft 名額未滿" if prop_stage == "draft" and
                             len(journal.draft_nodes) < self.cfg.loop.num_drafts
                             else "依指標 softmax 抽樣" if prop_stage == "improve"
                             else "debug_prob 抽中")}
        tree = self._tree_view(journal)
        try:
            ov = self.advisor.review_search_choice(
                profile, tree, proposal, conversation.read(self.run_dir))
        except Exception as e:                       # noqa: BLE001
            self._say("system", f"決策層檢視節點選擇失敗（{e}），沿用 policy 的選擇。",
                      kind="status", round=k)
            return parent, prob, None, {}
        if not ov or not ov.override:
            if ov is not None and (ov.reason or "").strip():
                self._say("llm", f"維持 policy 的選擇：{ov.reason.strip()}",
                          kind="decision", round=k)
            return parent, prob, None, {}

        # ---- 驗證: 不合法就沿用 policy ----------------------------------
        def _reject(why: str):
            self._say("system", f"決策層想改選（{ov.stage}"
                      f"{'/' + ov.parent_id if ov.parent_id else ''}），"
                      f"但{why}，沿用 policy 的選擇。", kind="status", round=k)
            return parent, prob, None, {}

        if ov.stage == "draft":
            pref = {}
            if ov.encoder:
                try:
                    card = reg.get(ov.encoder)
                except KeyError:
                    card = None
                if card is None or not card.available:
                    return _reject(f"指定的 encoder {ov.encoder} 不在可用目錄中（或權重未取得）")
                pref["encoder"] = EncoderChoice(
                    model_key=ov.encoder,
                    adaptation=ov.adaptation or "finetune",
                    rationale=ov.reason)
            if ov.preset:
                if ov.preset not in presets.names():
                    return _reject(f"指定的 preset {ov.preset} 不存在")
                pref["preset"] = ov.preset
            self._say("llm", f"改選：開一條新 draft"
                      + (f"（encoder={ov.encoder}"
                         + (f"/{ov.adaptation}" if ov.adaptation else "")
                         + (f", preset={ov.preset}" if ov.preset else "") + "）"
                         if pref else "")
                      + f" — {ov.reason}", kind="decision", round=k)
            return None, 1.0, "draft", pref

        target = next((n for n in journal.nodes
                       if n.id and n.id == ov.parent_id), None)
        if target is None:
            return _reject(f"找不到節點 {ov.parent_id}")
        if ov.stage == "improve" and target.metric is None:
            return _reject("該節點沒有成功的分數，無法作為改良基準")
        if ov.stage in ("improve", "resume") and target.improve_exhausted:
            return _reject("該分支先前已被決策層放棄（pruned）")
        if ov.stage == "debug" and not self._debuggable(target):
            return _reject("該節點不是可除錯的失敗葉節點（或除錯鏈已達上限）")
        if ov.stage == "resume" and not self._resume_ready(target):
            return _reject("該節點不符合續訓條件（已收斂／已有續訓分支／無 checkpoint）")

        self._say("llm", f"改選：從節點 {target.id} 做 {ov.stage}"
                  f"（policy 原本選 {proposal['stage']}"
                  f"{'/' + proposal['parent_id'] if proposal['parent_id'] else ''}）"
                  f" — {ov.reason}", kind="decision", round=k)
        return target, 1.0, ov.stage, {}

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
        """除錯 (對應 aideml debug 階段): 把失敗 trial 的 log 交給 **本地** 抽取器
        產生 ErrorFacts, 再交給 Advisor 提出修正 Recipe。

        ⚠ 資料圍欄: 原始 log 含 `Namespace(... data_path='/…')` 與 traceback 中的
        影像路徑, **不會**交給 LLM (docs/data_firewall_design.md §8.1)。只有在
        privacy.log_feedback="raw" (僅 mode=off 可設) 時才會附上原文。

        advisor=llm 且 allow_code_edit=True 時, LLM 可一併修改訓練程式 — 修改版由
        code_workspace 放在 <run_dir>/src/ 之下執行, 原始程式不會被修改。
        回 None = 無從修起, 放棄該分支。"""
        lp = node.trial.log_path if node.trial else None
        mode = self.cfg.privacy.log_feedback
        if mode == "none":
            facts = ErrorFacts()
        else:
            facts = error_extract.extract(
                lp, epoch_curve=(node.trial.epoch_curve if node.trial else None))
        kwargs = {"workspace_dir": os.path.join(self.run_dir, "src")}
        if mode == "raw" and lp and os.path.isfile(lp):
            try:
                with open(lp, errors="ignore") as f:
                    kwargs["log_tail"] = "".join(f.readlines()[-120:])
            except OSError:
                pass
        try:
            return self.advisor.propose_debug(
                profile, encoder, node.trial, facts, **kwargs)
        except TypeError:
            # 舊介面 (不吃 log_tail) 的 Advisor
            kwargs.pop("log_tail", None)
            return self.advisor.propose_debug(
                profile, encoder, node.trial, facts, **kwargs)

    def _maybe_resume(self, node: Node, force: bool = False) -> Optional[Recipe]:
        """繼續訓練策略: 節點訓練完但 curve 未收斂 → 從其 checkpoint 續訓
        resume_epochs 個 epoch (新 trial 節點, stage=resume)。不適用回 None。

        避免 resume 壟斷/浪費的護欄:
          - 已有 resume child 的節點不再續訓 (重複續訓 = 完全相同的計算);
          - resume 節點自己沒賺 (≤ parent + min_delta) 不再往下追;
          - 符合條件也只以 resume_prob 機率選擇, 讓 improve 有機會;
          - 全 run 連續 max_failed_resumes 次 resume 無提升 → 停用 (self._resume_disabled)。

        force=True (決策層明確指定 resume): 只跳過「機率性選擇」那一關, 其餘護欄照舊
        — 那些是正確性/重複計算的限制, 不是探索與否的取捨。"""
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
        if not force and random.random() >= scfg.resume_prob:
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
        #    grow_batch=False 時跳過 (逐 fold 平行共卡模式改以多 fold 共用提高利用率,
        #    不加大 batch 以免拖慢收斂 / 與同卡的另一個 fold 搶記憶體)。
        factor = 4 if util < g.util_target / 2 else 2
        while getattr(g, "grow_batch", True) and factor > 1:
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

    def _snapshot_tree(self, name: str) -> None:
        """把目前 journal 另存一份 (per_fold 模式: 每個 fold 的解答樹存成一檔)。"""
        if self.journal is None:
            return
        try:
            with open(os.path.join(self.run_dir, name), "w", encoding="utf8") as f:
                json.dump(self.journal.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _write_folds_json(self, current: int, entries: list) -> None:
        """per_fold 索引 (供 UI 分 fold 顯示): 目前 fold + 各 fold 的名稱/解答樹檔/最佳。"""
        try:
            with open(os.path.join(self.run_dir, "folds.json"), "w",
                      encoding="utf8") as f:
                json.dump({"mode": "per_fold", "current": current,
                           "folds": entries}, f, ensure_ascii=False, indent=2)
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
        # trial_id 用資料集**假名** — 這些 id 會出現在送往 LLM 的歷史與解答樹中,
        # 用真實目錄名等於每一輪都把資料集名稱送出去 (docs/data_firewall_design.md §4.3)
        ds_name = self._ds_tag
        # patience 計數 (stale): resume 時**由既有樹重算並沿用**, 不從 0 重來 —— 讓
        # patience 成為整個搜尋 (含歷次 resume) 的累計「連續無提升」預算, 與 UI 顯示一致。
        stale = self._replay_stale(history) if preloaded else 0
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
            # 決策層 override: policy 是規則+抽樣, 看不懂討論/QA 結論 —
            # 選完後交給 Advisor 過目, 讓它有改選節點/階段的機會 (會驗證合法性)。
            forced, draft_pref = None, {}
            if any(n.evaluated for n in journal.nodes):   # 樹上有結果才有得選
                parent, prob, forced, draft_pref = self._apply_search_override(
                    journal, profile, parent, prob, k)
            # resume 判定要在分支前算好 (forced 時跳過機率關卡, 其餘護欄照舊)
            resumed = None
            if (parent is not None and not parent.is_buggy
                    and forced in (None, "resume")):
                resumed = self._maybe_resume(parent, force=(forced == "resume"))

            if forced == "draft" or (forced is None and parent is None):
                stage = "draft"
                encoder, preset = self._draft_choice(journal, choices)
                encoder = draft_pref.get("encoder") or encoder
                preset = draft_pref.get("preset") or preset
                which = ("額外起點（決策層指定）" if forced == "draft" else
                         f"draft {len(journal.draft_nodes) + 1}/{self.cfg.loop.num_drafts}")
                self._say("system",
                          f"第 {k} 輪（draft）：開新起點 {which} — "
                          f"encoder={encoder.model_key}（{encoder.adaptation}）, "
                          f"preset={preset}（{self.cfg.advisor.type} 決策中，請稍候…）",
                          kind="status", round=k)
                recipe = self.advisor.compose_recipe(profile, encoder, preset=preset)
                self._say("llm", f"draft [{encoder.model_key}]（preset={preset}）："
                          f"head={recipe.heads[0].type if recipe.heads else '-'}, "
                          f"loss={recipe.losses[0].name if recipe.losses else '-'}。"
                          + (f" {recipe.encoder.rationale}" if recipe.encoder.rationale else ""),
                          kind="decision", round=k)
            elif forced == "debug" or (forced is None and parent.is_buggy):
                stage = "debug"
                encoder = parent.recipe.encoder
                self._say("system", f"第 {k} 輪（debug）：從失敗節點 {parent.id} 開始除錯"
                          f"（{'決策層指定' if forced == 'debug' else 'debug_prob 抽中'}）。",
                          kind="status", round=k)
                recipe = self._debug_recipe(profile, encoder, parent)
                if recipe is None:
                    parent.debug_exhausted = True
                    self._say("system", f"{parent.id} 無從除錯，放棄該分支。",
                              kind="status", round=k)
                    continue
                self._say("llm", f"除錯 {parent.id}："
                          f"{recipe.provenance.get('mutation', '')}",
                          kind="decision", round=k)
            elif resumed is not None:
                # 繼續訓練策略: 選中節點 curve 未收斂 → 先從其 checkpoint 續訓, 不變異
                stage = "resume"
                encoder = parent.recipe.encoder
                self._say("system",
                          f"第 {k} 輪（resume）：從節點 {parent.id}"
                          f"（score={parent.metric:.4f}，"
                          f"{'決策層指定' if forced == 'resume' else f'被選機率 {prob:.0%}'}）"
                          f"繼續訓練。", kind="status", round=k)
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
                          f"{'決策層指定' if forced == 'improve' else f'依指標抽樣機率 {prob:.0%}'}）"
                          f"開始改善；{self.cfg.advisor.type} 決策中，請稍候…",
                          kind="status", round=k)
                discussion = conversation.read(self.run_dir)
                # workspace_dir: 允許修改程式時 (allow_code_edit), improve 階段可
                # 對 main_finetune.py 提 code_edits (副本隔離, 原始程式不動)
                action = self.advisor.review_and_decide(
                    profile, encoder, history, discussion, base=parent.trial,
                    workspace_dir=os.path.join(self.run_dir, "src"))
                if action.narrative:
                    self._say("llm", action.narrative, kind="review", round=k)
                # 決策層順帶要求的資訊 (分析器 / 問使用者) — 下一輪就會看到結果
                self._handle_info(getattr(action, "info", None), k)
                prune = getattr(action, "prune_branch", False) or action.next_recipe is None
                # 保險絲: 決策層說 stop, 但樹上還有沒被放棄的節點、且才跑不到一半的
                # 輪數 → 多半是把「放棄這條分支」誤寫成「結束整個實驗」(歷史 bug:
                # stop 欄位沒有說明, LLM 在死節點上回 stop 導致整場提早收工)。
                # 降級成 prune, 搜尋改從別的節點繼續。
                if action.stop and self._stop_looks_premature(
                        journal, parent, k, max_rounds):
                    self._say("system",
                              f"決策層在第 {k} 輪要求結束實驗，但樹上仍有未放棄的節點"
                              f"且輪數未過半 — 視為放棄 {parent.id} 這條分支，改從其他"
                              f"節點繼續（要真的結束請用「中斷實驗」）。",
                              kind="status", round=k)
                    prune = True
                elif action.stop:
                    self._say("llm", f"決定停止：{action.reason or '已收斂'}",
                              kind="decision", round=k)
                    break
                if prune:
                    parent.improve_exhausted = True
                    self._say("llm", f"放棄分支 {parent.id}："
                              f"{action.reason or '此節點無值得再試的變異'}",
                              kind="decision", round=k)
                    if not journal.improvable_nodes and not self._debuggable_any(journal):
                        self._say("llm", "樹上所有分支都已放棄，結束搜尋。",
                                  kind="decision", round=k)
                        break
                    continue
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
                              "select_prob": round(prob, 4),
                              # 節點是 policy 抽的還是決策層改選的 (前端會標示)
                              "overridden": forced is not None}
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

            # 方案B: 搜尋『中』集成 — 改良進入平坦期 + 模型池較上次成長 → 中途組 ensemble
            ec = self.cfg.ensemble
            if (getattr(ec, "enabled", False) and getattr(ec, "in_search", False)
                    and not self.cfg.dry_run and stale >= ec.search_patience
                    and self._n_search_ens < ec.max_search_ensembles):
                pool = sum(1 for t in history if t.status == "done"
                           and t.recipe.encoder.model_key != "ensemble")
                if pool >= ec.min_members and pool > self._last_ens_pool:
                    self._last_ens_pool = pool
                    self._n_search_ens += 1
                    try:
                        self._run_ensemble_round(profile, history, f"t{k}")
                    except Exception as e:               # noqa: BLE001
                        self._say("system", f"搜尋中集成失敗（{e}）。", kind="status")

            if len(history) >= scfg.min_trials and stale >= scfg.patience:
                msg = (f"已完成 {len(history)} 輪（≥ min_trials={scfg.min_trials}），"
                       f"且連續 {stale} 輪改良/續訓無提升"
                       f"（patience={scfg.patience}，draft/debug 不計），停止。")
                if self.best is not None:
                    msg += f"全域最佳 primary={self.best.primary_score:.4f}。"
                self._say("system", msg, kind="final")
                break
        return history

    # ---- 集成 (docs/ensemble_design.md) --------------------------------
    def _select_ensemble_members(self, profile: DatasetProfile,
                                 history: list[TrialResult], ec):
        """選集成成員 — 由 `ensemble.llm_select` 決定, **獨立於 advisor.type**。

        llm_select=False → 一律規則式選擇 (即使主 advisor 是 LLM)。
        llm_select=True  → 用 LLM 依 TrialFacts 選 (主 advisor 非 LLM 時另建專用
        LLMAdvisor); LLM 不可用或回 None 時自動退回規則式。
        """
        if getattr(ec, "llm_select", False):
            adv = self._ensemble_llm_advisor()
            if adv is not None:
                try:
                    spec = adv.propose_ensemble(profile, history, ec)
                    if spec is not None:
                        return spec
                except Exception as e:                   # noqa: BLE001
                    self._say("system", f"LLM 選集成成員失敗（{e}），改用規則式選擇。",
                              kind="status")
        return ensembler.select_members_heuristic(history, ec)

    def _ensemble_llm_advisor(self):
        """回傳能做 LLM 成員選擇的 advisor (獨立於主 advisor.type); 不可用回 None。"""
        from .advisor import HeuristicAdvisor
        from .llm_advisor import LLMAdvisor
        # 主 advisor 本身就是 LLM/Skill → 直接用其 propose_ensemble
        if (not isinstance(self.advisor, HeuristicAdvisor)
                and hasattr(self.advisor, "propose_ensemble")):
            return self.advisor
        # 主 advisor 是 heuristic → 視需要建一個專用 LLMAdvisor (共用 privacy context)
        if self._ens_llm is None:
            try:
                adv = LLMAdvisor(model=self.cfg.advisor.model, privacy=self.privacy)
                adv.check_environment()
                self._ens_llm = adv
            except Exception as e:                       # noqa: BLE001
                self._say("system", f"LLM 選成員不可用（{e}），改用規則式選擇。",
                          kind="status")
                self._ens_llm = False                    # 快取失敗, 不重試
        return self._ens_llm or None

    def _run_ensemble_round(self, profile: DatasetProfile,
                            history: list[TrialResult], tag: str):
        """執行一次集成 (方案A收尾 / 方案B搜尋中共用)。回傳 EnsembleResult 或 None。

        重用各 trial 已落地的 predictions_*.csv (不重訓)。資料圍欄: 決策層只選成員
        trial_id + 方法; 機率平均在 ensembler (資料平面) 執行, 只有彙整指標回流。
        """
        ec = self.cfg.ensemble
        done = [t for t in history if t.status == "done"
                and t.recipe.encoder.model_key != "ensemble"]
        if len(done) < ec.min_members:
            return None

        # 選成員: 由 ensemble.llm_select 決定 (獨立於 advisor.type), 見 _select_ensemble_members
        spec = self._select_ensemble_members(profile, history, ec)
        if spec is None or len(spec.member_trial_ids) < ec.min_members:
            return None

        by_id = {t.trial_id: t for t in done}
        members = [by_id[t] for t in spec.member_trial_ids if t in by_id]
        if len(members) < ec.min_members:
            return None
        self._say("system",
                  f"集成 {len(members)} 個模型（method={spec.method}，{tag}）："
                  + "、".join(m.trial_id for m in members) + "…", kind="status")

        out_dir = os.path.join(self.run_dir, "ensembles", f"ensemble_{tag}")
        result = ensembler.combine(members, spec, self.eval_cfg,
                                   self.trials_dir, out_dir)
        if result.status != "done":
            self._say("system", f"集成未完成：{result.message}", kind="status")
            return None

        self.ensembles.append(result)
        if self.ensemble is None or result.primary_score > self.ensemble.primary_score:
            self.ensemble = result                       # 追蹤跨回合最佳集成

        best_single = self.best.primary_score if self.best else None
        pm = self.cfg.eval.primary_metric
        if best_single is not None and result.primary_score > best_single:
            self._say("llm",
                      f"集成勝出（{tag}）：{len(members)} 模型 primary({pm})="
                      f"{result.primary_score:.4f}，勝過最佳單模型 {best_single:.4f}"
                      f"（+{result.primary_score - best_single:.4f}）。"
                      f"權重={result.spec.weights}。", kind="decision")
        else:
            self._say("llm",
                      f"集成（{tag}）primary({pm})={result.primary_score:.4f}"
                      + (f"，未勝過最佳單模型 {best_single:.4f}。"
                         if best_single is not None else "。"), kind="decision")
        return result

    def _maybe_ensemble(self, profile: DatasetProfile,
                        history: list[TrialResult]):
        """方案A: 收尾集成 (實驗停止後, 對全部模型池組一次)。回傳最佳集成。"""
        ec = self.cfg.ensemble
        if not getattr(ec, "enabled", False) or self.cfg.dry_run:
            return self.ensemble
        self._run_ensemble_round(profile, history, "final")
        return self.ensemble

    def _replay_stale(self, history: list) -> int:
        """由既有 history 重放算出目前的 stale (連續 improve/resume 無提升輪數)。

        規則與主迴圈一致 (見下方): 有實質提升→歸零; improve/resume 無提升→+1;
        draft/debug 不計。resume 時用此值接續 patience 計數 (與 web 顯示的一致)。
        """
        md = self.cfg.loop.min_delta
        best_s, stale = None, 0
        for t in history:
            if t.status != "done":
                continue
            stage = ((t.recipe.provenance or {}).get("search") or {}).get("stage", "draft")
            if best_s is None or t.primary_score > best_s + md:
                stale = 0
            elif stage in ("improve", "resume"):
                stale += 1
            if best_s is None or t.primary_score > best_s:
                best_s = t.primary_score
        return stale

    def _narrate_best(self, profile: DatasetProfile) -> str:
        """報告用: 讓決策層 (LLM) 把最佳 recipe 寫成一段白話說明; 不支援時回空字串。"""
        if self.best is None:
            return ""
        fn = getattr(self.advisor, "narrate_best", None)
        if fn is None:
            return ""
        try:
            return fn(profile, self.best) or ""
        except Exception:                            # noqa: BLE001
            return ""

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
        # 資料圍欄: 決策層從這裡才拿得到資料集描述, 而且只拿得到 DatasetFacts
        self._setup_privacy(profile)
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

        # 資訊蒐集 (管道 A/B): 決策層看過 DatasetFacts 後, 可以點名要跑哪些分析器、
        # 或提出只有使用者知道的問題。這是它取得資料特性的唯一途徑。
        if not self.cfg.dry_run and hasattr(self.advisor, "plan_information"):
            try:
                info = self.advisor.plan_information(profile)
                if info and (info.analyses or info.questions):
                    self._say("llm", f"開始前我想先補一些資料特性："
                              f"{info.reason or '見下方'}", kind="decision")
                    self._handle_info(info)
            except Exception as e:                   # noqa: BLE001
                self._say("system", f"資訊蒐集階段失敗（{e}），直接以現有事實決策。",
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

        # 逐 fold 獨立搜尋 (aggregation=per_fold): 每個 fold 各自找最佳 recipe
        if (self.cfg.eval.aggregation == "per_fold"
                and len(discover_folds(self.cfg.data_path)) > 1):
            pfs = self._per_fold_search(profile, choices)
            per_encoder = {}
            for t in pfs["best_trials"]:            # 各 fold 最佳依 encoder 分組供報告
                per_encoder.setdefault(t.recipe.encoder.model_key, []).append(t)
            self._clear_current()
            if not self.cfg.dry_run:
                p = pfs["primary"]
                if self.best is not None:
                    self._say("llm",
                              f"逐 fold 搜尋完成（{p['n']} folds）。整體最佳 "
                              f"{self.best.recipe.encoder.model_key} "
                              f"primary={self.best.primary_score:.4f}；各 fold "
                              f"mean±std = {p['mean']:.4f}±{p['std']:.4f}。", kind="final")
                else:
                    self._say("system", "逐 fold 搜尋結束，無成功 trial。", kind="final")
            report_path = report_mod.write_report(
                self.run_dir, profile, self.cfg, choices, per_encoder,
                self.best, dry_run=self.cfg.dry_run, per_fold_summary=pfs)
            return {"profile": profile, "choices": choices,
                    "per_encoder": per_encoder, "best": self.best,
                    "ensemble": None, "n_trials": self.n_trials,
                    "fold_summary": None, "per_fold_summary": pfs,
                    "report_path": report_path, "run_dir": self.run_dir}

        # 單一全域解答樹搜尋 (draft 跨 encoder; per_encoder 僅供 report 分組)
        history = self._tree_search(profile, choices,
                                    journal=journal, history=prior_history)
        per_encoder: dict[str, list[TrialResult]] = {}
        for t in history:
            per_encoder.setdefault(t.recipe.encoder.model_key, []).append(t)

        # 收尾集成 (方案 A): 把多個已訓練 trial 組成軟投票 ensemble (不重訓)
        try:
            self.ensemble = self._maybe_ensemble(profile, history)
        except Exception as e:                           # noqa: BLE001
            self._say("system", f"集成階段失敗（{e}），保留單模型結果。", kind="status")
            self.ensemble = None

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
            self.best, dry_run=self.cfg.dry_run, fold_summary=fold_summary,
            ensemble=self.ensemble, best_narrative=self._narrate_best(profile))

        return {
            "profile": profile,
            "choices": choices,
            "per_encoder": per_encoder,
            "best": self.best,
            "ensemble": self.ensemble,
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
            # trial_id 用假名 (會進 LLM 的歷史); report.md 仍記真實 fold 名 (本機閱讀)
            fname = os.path.basename(fold)
            task_id = (f"{self.best.recipe.encoder.model_key}_"
                       f"{self._fold_tag(fold)}_bestfold")
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

    # ---- 逐 fold 獨立搜尋 (aggregation=per_fold) -----------------------
    def _per_fold_search(self, profile: DatasetProfile, choices) -> dict:
        """每個 sibling fold 各自跑一次完整樹搜尋, 獨立找出該 fold 的最佳 recipe。

        與 mean_std 不同: mean_std 是「單一最佳 recipe 套到各 fold 重跑」; 這裡是
        「每個 fold 從頭獨立搜尋」——各 fold 可能得到不同的最佳 encoder / 超參。
        每個 fold 重建 privacy context (per-fold 假名 / 出口守衛), 解答樹存成獨立檔。
        """
        folds = discover_folds(self.cfg.data_path)
        saved_dp = self.cfg.data_path
        ttype = self.cfg.task.get("type", "classification")
        pm = self.cfg.eval.primary_metric
        per_fold: list[dict] = []
        fold_entries: list[dict] = []        # 供 UI 分 fold 顯示 (folds.json)
        best_trials: list[TrialResult] = []
        scores, metrics_list = [], []
        overall_best: Optional[TrialResult] = None
        total_trials = 0
        try:
            for i, fold in enumerate(folds):
                # 只有『使用者中斷』會結束整個逐 fold 流程; 預算 (max_trials/時間) 每個
                # fold 各自獨立 (見下方重置), 不會因前一個 fold 用完而跳過後面的 fold。
                if self._stopped:
                    self._say("system", "已中斷，結束逐 fold 搜尋。", kind="status")
                    break
                fname = os.path.basename(os.path.normpath(fold))
                self._say("system",
                          f"━━━ Fold {i + 1}/{len(folds)}（{fname}）獨立搜尋開始 ━━━",
                          kind="status")
                # 切到此 fold + 重建 privacy(per-fold 假名/守衛) + 重置搜尋狀態
                self.cfg.data_path = fold
                self._cur_fold = {"index": i, "name": fname}
                fold_entries.append({"index": i, "name": fname,
                                     "tree": f"search_tree_fold{i}.json",
                                     "n_done": 0, "primary": None, "best": None})
                self._write_folds_json(i, fold_entries)   # 標記目前 fold (UI 即時分頁)
                fold_profile = (profile
                                if os.path.abspath(fold) == os.path.abspath(saved_dp)
                                else dataset_analyzer.analyze(fold, task_type=ttype))
                self._setup_privacy(fold_profile)
                if hasattr(self.advisor, "analyses"):
                    self.advisor.analyses = []       # 各 fold 分析結果不互相沿用
                self.journal = None
                self.best = None
                # 每個 fold 的迴圈預算各自獨立: 重置 trial 計數與計時起點 (max_trials /
                # max_wall_clock_min 對每個 fold 分別重新起算), 以及 resume 自適應旗標。
                self.n_trials = 0
                self._t0 = time.time()
                self._resume_fails = 0
                self._resume_disabled = False
                history = self._tree_search(fold_profile, choices)
                total_trials += self.n_trials
                self._snapshot_tree(f"search_tree_fold{i}.json")

                fb = self.best
                n_done = len([t for t in history if t.status == "done"])
                fold_entries[-1]["n_done"] = n_done
                if fb is not None:
                    fold_entries[-1]["primary"] = fb.primary_score
                    fold_entries[-1]["best"] = fb.trial_id
                self._write_folds_json(i, fold_entries)
                row = {"fold": fname, "n_trials": n_done}
                if fb is not None:
                    row.update({
                        "encoder": fb.recipe.encoder.model_key,
                        "adaptation": fb.recipe.encoder.adaptation,
                        "primary_score": fb.primary_score,
                        "metrics": fb.metrics,
                        "hparams": fb.recipe.hparams.model_dump(),
                        "trial_id": fb.trial_id,
                        "mutation": fb.recipe.provenance.get("mutation"),
                    })
                    scores.append(fb.primary_score)
                    metrics_list.append(fb.metrics)
                    best_trials.append(fb)
                    if overall_best is None or fb.primary_score > overall_best.primary_score:
                        overall_best = fb
                    self._say("llm",
                              f"Fold {i + 1}（{fname}）最佳：{fb.recipe.encoder.model_key} "
                              f"primary({pm})={fb.primary_score:.4f}（trial={fb.trial_id}）。",
                              kind="decision")
                else:
                    self._say("system", f"Fold {i + 1}（{fname}）無成功 trial。",
                              kind="status")
                per_fold.append(row)
        finally:
            self.cfg.data_path = saved_dp
            self._cur_fold = None
            self._write_folds_json(-1, fold_entries)   # current=-1: 全部完成
        self.n_trials = total_trials          # 對外顯示全部 fold 的 trial 總數
        self.best = overall_best
        agg = mreg.aggregate(scores)
        return {
            "folds": [r["fold"] for r in per_fold],
            "per_fold": per_fold,
            "best_trials": best_trials,
            "primary": agg,
            "metrics": mreg.aggregate_metrics(metrics_list),
        }
