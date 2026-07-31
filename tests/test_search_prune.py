"""prune_branch 與 stop 保險絲 — 迴歸測試。

背景 (run_20260729_151635/fold2): 決策層在一個明顯走不通的節點 (vit_large lp,
權重載入不完整, acc=0.057) 上回了 stop=true, 本意是「放棄這條分支」(它自己的
narrative 寫著「故停止此分支…建議下一步改從 mae 節點延長 epochs」), 但
LoopController 把 stop 當成「結束整個 fold 的搜尋」直接 break, 該 fold 只跑了
3 個 trial 就收工。這裡鎖住修好後的語意。
"""
import os
import tempfile

import pytest

from agent.config import AgentConfig
from agent.journal import Journal, Node
from agent.loop_controller import LoopController
from agent.schemas import (EncoderChoice, InfoRequest, NextAction, Recipe,
                           TrialResult)


def _node(journal: Journal, tid: str, score: float | None,
          status: str = "done", parent: Node | None = None,
          stage: str = "draft") -> Node:
    recipe = Recipe(encoder=EncoderChoice(model_key="enc_" + tid))
    n = Node(recipe, parent=parent, stage=stage)
    n.trial = TrialResult(trial_id=tid, recipe=recipe, status=status,
                          primary_score=score if score is not None else 0.0)
    return journal.append(n)


@pytest.fixture()
def lc():
    with tempfile.TemporaryDirectory() as d:
        yield LoopController(AgentConfig(data_path=d), advisor=None, run_dir=d)


# ---- journal ------------------------------------------------------------
def test_improvable_excludes_pruned():
    j = Journal()
    a = _node(j, "t0", 0.60)
    b = _node(j, "t1", 0.58)
    _node(j, "t2", None, status="failed")
    assert j.improvable_nodes == [a, b]
    b.improve_exhausted = True
    assert j.improvable_nodes == [a]
    # good_nodes 不受影響 — 被 prune 的節點仍留在樹上供比較/集成
    assert len(j.good_nodes) == 2


# ---- 選點 ---------------------------------------------------------------
def test_select_improve_node_skips_pruned(lc):
    j = Journal()
    best = _node(j, "t0", 0.90)
    other = _node(j, "t1", 0.50)
    best.improve_exhausted = True
    node, _ = lc._select_improve_node(j)
    assert node is other, "被 prune 的最佳節點不該再被選為 improve 起點"


def test_select_improve_node_none_when_all_pruned(lc):
    j = Journal()
    for tid, s in (("t0", 0.9), ("t1", 0.5)):
        _node(j, tid, s).improve_exhausted = True
    assert lc._select_improve_node(j) == (None, 1.0)
    # policy 也要跟著回 draft (None) 而不是硬選一個已放棄的節點
    lc.cfg.loop.num_drafts = 0
    lc.cfg.loop.debug_prob = 0.0
    assert lc._search_policy(j)[0] is None


# ---- stop 保險絲 --------------------------------------------------------
def test_stop_fuse_fires_when_other_branches_remain(lc):
    j = Journal()
    dead = _node(j, "t2", 0.057)
    _node(j, "t0", 0.597)
    _node(j, "t1", 0.582)
    # fold2 的情境: 第 3 輪 (max_rounds=12) 在死節點上收到 stop
    assert lc._stop_looks_premature(j, dead, k=3, max_rounds=12) is True


def test_stop_fuse_respects_late_stop(lc):
    j = Journal()
    dead = _node(j, "t2", 0.057)
    _node(j, "t0", 0.597)
    # 輪數過半後的 stop 視為真的收斂, 不降級
    assert lc._stop_looks_premature(j, dead, k=8, max_rounds=12) is False


def test_stop_fuse_respects_stop_when_no_alternative(lc):
    j = Journal()
    only = _node(j, "t0", 0.597)
    assert lc._stop_looks_premature(j, only, k=1, max_rounds=12) is False


def test_stop_fuse_can_be_disabled(lc):
    j = Journal()
    dead = _node(j, "t2", 0.057)
    _node(j, "t0", 0.597)
    lc.cfg.loop.stop_fuse = False
    assert lc._stop_looks_premature(j, dead, k=3, max_rounds=12) is False


# ---- 樹視圖 / 改選驗證 ---------------------------------------------------
def test_tree_view_marks_pruned_and_drops_improve(lc):
    j = Journal()
    n = _node(j, "t0", 0.60)
    n.improve_exhausted = True
    view = lc._tree_view(j)[0]
    assert view["pruned"] is True
    assert "improve" not in view["allowed"] and "resume" not in view["allowed"]


# ---- Advisor 契約 -------------------------------------------------------
def test_next_action_defaults_prune_false():
    assert NextAction(stop=False).prune_branch is False


def test_llm_schema_documents_stop_vs_prune():
    """歷史 bug 的根因: schema 裡 stop 完全沒有 description, system prompt 也沒提,
    決策層無從得知 stop 會結束整場實驗。"""
    from agent.llm_advisor import _ReviewDecision
    props = _ReviewDecision.model_json_schema()["properties"]
    assert "整個實驗" in props["stop"]["description"]
    assert "prune_branch" in props["stop"]["description"]
    assert "不會結束實驗" in props["prune_branch"]["description"]


def test_llm_maps_failed_mutation_to_prune_not_stop(monkeypatch):
    """_apply_mutation 失敗 (next_recipe=None) 以前會被折成 stop → 整場結束;
    現在應該只放棄這條分支。"""
    from agent import llm_advisor as m

    adv = m.LLMAdvisor.__new__(m.LLMAdvisor)
    d = m._ReviewDecision(stop=False, mutation="swap_head", reason="換 head")
    monkeypatch.setattr(m.LLMAdvisor, "_ctx", lambda self, *a, **k: _DummyCtx())
    monkeypatch.setattr(m.LLMAdvisor, "_messages_parse",
                        lambda self, *a, **k: d)
    monkeypatch.setattr(m.LLMAdvisor, "_data_block", lambda self, *a, **k: ())
    monkeypatch.setattr(m.LLMAdvisor, "_apply_mutation",
                        lambda self, *a, **k: None)      # 變異套用失敗
    monkeypatch.setattr(m.LLMAdvisor, "_pick_base",
                        lambda self, h, b: b)
    monkeypatch.setattr(m.LLMAdvisor, "_base_id", lambda self, t: "t0")
    monkeypatch.setattr(m.LLMAdvisor, "_info_note", lambda self, ctx: "")
    adv.last_info = InfoRequest()
    recipe = Recipe(encoder=EncoderChoice(model_key="e"))
    base = TrialResult(trial_id="t0", recipe=recipe, status="done",
                       primary_score=0.5)
    act = adv.propose_next(_DummyProfile(), recipe.encoder, [base], base=base)
    assert act.stop is False and act.prune_branch is True


class _DummyCtx:
    alias = None


class _DummyProfile:
    pass
