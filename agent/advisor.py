"""Advisor 決策層 (設計文件 §5.2).

定義可抽換介面 + P0 的 HeuristicAdvisor (純規則, 不呼叫 LLM)。
未來的 LLMAdvisor / SkillAdvisor 只需實作同一組方法。
"""
from __future__ import annotations

import re
from typing import Optional, Protocol

from . import encoder_registry as reg
from . import presets
from .schemas import (
    DatasetProfile, EncoderChoice, HeadSpec, HyperParams, Recipe, TrialResult,
    NextAction, ComponentRef,
)

# 超參起點 (default / paper / mae + 使用者自訂) 現由 presets 模組管理, 可於 web 編輯。
# 保留 HPARAM_PRESETS 名稱作為相容存取 (讀當下值)。
HPARAM_PRESETS = presets.load()


class Advisor(Protocol):
    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]: ...
    def compose_recipe(self, profile: DatasetProfile,
                       encoder: EncoderChoice, preset: str = "default") -> Recipe: ...
    # base = 本輪樹搜尋 policy 依指標機率選出的節點 trial (變異基準); None = 取歷史最佳
    def propose_next(self, profile: DatasetProfile, encoder: EncoderChoice,
                     history: list[TrialResult],
                     base: Optional[TrialResult] = None) -> NextAction: ...
    # 人機協作: 用目前所有資料 + 討論檢視結果, 產生說明 + 下一輪決定 (或中斷)。
    # workspace_dir: improve 階段允許修改程式時的隔離工作區 (限 main_finetune.py)
    def review_and_decide(self, profile: DatasetProfile, encoder: EncoderChoice,
                          history: list[TrialResult], discussion: list[dict],
                          base: Optional[TrialResult] = None,
                          workspace_dir: Optional[str] = None) -> NextAction: ...
    # 樹搜尋 debug 階段 (aideml): 依失敗 trial 的 log 修正 Recipe (允許時可含程式
    # 修改, 副本放 workspace_dir, 原始程式不動); None = 放棄該分支
    def propose_debug(self, profile: DatasetProfile, encoder: EncoderChoice,
                      trial: TrialResult, log_tail: str,
                      workspace_dir: Optional[str] = None) -> Optional[Recipe]: ...
    # 未來: 依資料提示不同下游任務起點供選擇 (§5.8)
    def suggest_task_templates(self, profile: DatasetProfile) -> list: ...


# 使用者訊息中的中斷關鍵字
_STOP_WORDS = ("停止", "停下", "中斷", "別跑", "不要繼續", "stop", "halt", "abort", "終止")


class HeuristicAdvisor:
    """規則式決策 — 離線退路; 也被 LLMAdvisor/SkillAdvisor 於失效時委派。"""

    def suggest_task_templates(self, profile: DatasetProfile) -> list:
        from . import task_template
        return task_template.suggest(profile)

    def review_and_decide(self, profile: DatasetProfile, encoder: EncoderChoice,
                          history: list[TrialResult], discussion: list[dict],
                          base: Optional[TrialResult] = None,
                          workspace_dir: Optional[str] = None) -> NextAction:
        """規則式檢視 + 決定。產生給使用者看的說明, 並尊重使用者的中斷/討論訊息。
        base = policy 選出的變異基準 trial (None 則取歷史最佳)。
        規則式不修改程式, workspace_dir 僅為介面一致而保留。"""
        done = [t for t in history if t.status == "done"]
        best = max(done, key=lambda t: t.primary_score) if done else None

        # 使用者最新訊息 → 若含中斷關鍵字則停止
        user_msgs = [d for d in (discussion or []) if d.get("role") == "user"]
        last_user = user_msgs[-1]["text"] if user_msgs else ""
        if any(w in last_user.lower() for w in _STOP_WORDS):
            return NextAction(stop=True, reason="使用者要求中斷。",
                              narrative=f"收到你的指示「{last_user}」，停止實驗。")

        # 檢視說明
        if best is not None:
            trend = "、".join(f"{t.primary_score:.4f}" for t in done[-4:])
            narrative = (f"已完成 {len(done)} 輪，全域最佳 "
                         f"primary={best.primary_score:.4f}（近況 {trend}）。")
            if base is not None:
                narrative += (f" 本輪以節點 {base.trial_id}"
                              f"（primary={base.primary_score:.4f}）為基準改良。")
            if len(done) >= 2 and done[-1].primary_score <= done[-2].primary_score:
                narrative += " 上一輪未提升，考慮換一個正交變異。"
        else:
            narrative = f"[{encoder.model_key}] 尚無成功 trial，先觀察起手 Recipe 結果。"
        if last_user:
            narrative += f" 已納入你的意見：「{last_user}」。"

        action = self.propose_next(profile, encoder, history, base=base)
        action.narrative = narrative
        return action

    def select_encoders(self, profile: DatasetProfile) -> list[EncoderChoice]:
        cards = reg.available_cards()
        # 小資料 -> linear probe; 否則全微調 (鬆散規則)
        adaptation = "lp" if profile.n_train < 500 else "finetune"
        choices = []
        for c in cards:
            rationale = f"{c.domain} 預訓練; "
            if c.domain == "medical_dap":
                rationale += "醫療 DAP 較貼近眼底影像。"
            else:
                rationale += "自然影像基準線。"
            choices.append(EncoderChoice(
                model_key=c.model_key, adaptation=adaptation, rationale=rationale))
        return choices

    def compose_recipe(self, profile: DatasetProfile,
                       encoder: EncoderChoice, preset: str = "default") -> Recipe:
        hp = presets.get(preset)  # 讀當下值 (web 編輯即生效)
        # 類別不平衡 -> 用 weighted_ce (鬆散規則)
        loss = "weighted_ce" if profile.imbalance_ratio >= 3.0 else "cross_entropy"
        return Recipe(
            encoder=encoder,
            task={"type": profile.task_type},
            heads=[HeadSpec(type="linear", output_dim=profile.num_classes)],
            pooling="global_pool",
            regularizers=[],
            augmentation=ComponentRef(name="timm_randaug"),
            losses=[ComponentRef(name=loss)],
            hparams=hp,
            provenance={"template": "fundus_classification", "preset": preset},
        )

    # P4: 有限、正交、可驗證的變異階梯 (一次只改一個面向)。
    #     每個動作對應 ComponentRegistry 一個可掛載元件, 依序嘗試, 不重複。
    _MUTATION_LADDER = [
        ("add_regularizer", "mixup"),
        ("swap_head", "mlp"),
        ("change_augmentation", "timm_randaug"),  # 佔位: 目前僅一種 aug, 之後擴充
        ("add_regularizer", "label_smoothing"),
    ]

    def propose_next(self, profile: DatasetProfile, encoder: EncoderChoice,
                     history: list[TrialResult],
                     base: Optional[TrialResult] = None) -> NextAction:
        """P4: 對基準 Recipe (policy 選出的 base, 無則取歷史最佳) 施加一個
        尚未用過的變異再試; 用盡則停止。

        provenance 記錄變異譜系 (mutated_from / mutation), 便於分析何種改動有效。
        LoopController 另負責 max_trials / patience / 預算等 StopPolicy 護欄。
        """
        done = [t for t in history if t.status == "done"]
        if base is None or base.status != "done":
            if not done:
                return NextAction(stop=True, reason="無成功 trial 可作為改良起點。")
            base = max(done, key=lambda t: t.primary_score)

        # 已套用過的變異 (從 provenance 蒐集), 避免重複
        used = set()
        for t in history:
            m = t.recipe.provenance.get("mutation")
            if m:
                used.add(m)

        for action, target in self._MUTATION_LADDER:
            tag = f"{action}:{target}"
            if tag in used:
                continue
            recipe = self._mutate(base.recipe, action, target)
            if recipe is None:
                continue
            return NextAction(stop=False, reason=f"改良: {tag}",
                              mutation=action, next_recipe=recipe)

        return NextAction(stop=True, reason="變異階梯已用盡, 停止此 encoder。")

    def propose_debug(self, profile: DatasetProfile, encoder: EncoderChoice,
                      trial: TrialResult, log_tail: str,
                      workspace_dir: Optional[str] = None) -> Optional[Recipe]:
        """規則式除錯 (aideml debug 階段的離線退路): 由 log 尾端猜失敗原因,
        產生修正後 Recipe; 無從修起則回 None (放棄該分支)。
        規則式不修改程式, workspace_dir 僅為介面一致而保留。"""
        tail = (log_tail or "").lower()
        r = trial.recipe.model_copy(deep=True)
        hp = r.hparams
        if "out of memory" in tail or "cuda error" in tail:
            if hp.batch_size <= 1:
                return None
            fix = f"batch_size {hp.batch_size}->{hp.batch_size // 2} (OOM, accum_iter 補償)"
            hp.accum_iter *= 2
            hp.batch_size //= 2
        elif re.search(r"\bnan\b|\binf\b", tail):
            if hp.blr <= 1e-6:
                return None
            fix = f"blr {hp.blr}->{hp.blr / 5:g} (loss 出現 NaN/Inf, 疑似 lr 過大)"
            hp.blr /= 5
        else:
            # 原因不明: 保守降 lr 再試一次 (debug_depth 上限保證不會無限重試)
            if hp.blr <= 1e-6:
                return None
            fix = f"blr {hp.blr}->{hp.blr / 2:g} (失敗原因不明, 保守重試)"
            hp.blr /= 2

        prov = dict(r.provenance)
        prov["mutated_from"] = {"trial": trial.trial_id,
                                "mutation": trial.recipe.provenance.get("mutation")}
        prov["mutation"] = f"debug:{fix}"
        prov["reason"] = f"除錯失敗 trial {trial.trial_id}: {fix}"
        prov["advisor"] = "heuristic"
        r.provenance = prov
        return r

    def _mutate(self, base: Recipe, action: str, target: str) -> Recipe | None:
        r = base.model_copy(deep=True)
        prov = dict(r.provenance)
        prov["mutated_from"] = {k: base.provenance.get(k)
                                for k in ("template", "preset", "mutation")}
        prov["mutation"] = f"{action}:{target}"
        prov["reason"] = f"heuristic 變異階梯: {action}:{target}"
        prov["advisor"] = "heuristic"

        if action == "add_regularizer":
            if target in [c.name for c in r.regularizers]:
                return None
            params = {"alpha": 0.8} if target == "mixup" else {"smoothing": 0.1}
            r.regularizers.append(ComponentRef(name=target, params=params))
        elif action == "swap_head":
            if r.heads and r.heads[0].type == target:
                return None
            if r.heads:
                r.heads[0].type = target
                if target == "mlp" and not r.heads[0].hidden_dims:
                    r.heads[0].hidden_dims = [512]
        elif action == "change_augmentation":
            if r.augmentation and r.augmentation.name == target:
                return None
            r.augmentation = ComponentRef(name=target)
        else:
            return None

        r.provenance = prov
        return r


def build_advisor(advisor_cfg) -> Advisor:
    """依 AdvisorConfig.type 建立決策層 (設計文件 §5.2 可抽換介面)。

    heuristic → HeuristicAdvisor (P0/P1 預設)
    llm       → LLMAdvisor        (P2, 尚未實作)
    skill     → SkillAdvisor      (P7, 尚未實作)
    """
    t = getattr(advisor_cfg, "type", "heuristic")
    guidance = getattr(advisor_cfg, "guidance", "")
    if t == "heuristic":
        return HeuristicAdvisor()
    if t == "llm":
        from .llm_advisor import LLMAdvisor  # 延後匯入, 避免無 anthropic 時 import 失敗
        adv = LLMAdvisor(model=getattr(advisor_cfg, "model", "claude-opus-4-8"),
                         guidance=guidance,
                         allow_code_edit=getattr(advisor_cfg, "allow_code_edit", False))
        adv.check_environment()  # 環境不對即拋清楚錯誤 (不靜默退回 heuristic)
        return adv
    if t == "skill":
        from .skill_advisor import SkillAdvisor
        return SkillAdvisor()
    raise ValueError(f"未知 advisor type: {t}")
