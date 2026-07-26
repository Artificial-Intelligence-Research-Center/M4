"""Ensembler 測試 (docs/ensemble_design.md)。

不需 pytest:  /opt/conda/envs/M4/bin/python -m tests.test_ensemble
(函式同時相容 pytest。)

涵蓋: 對齊 (image_path / 缺欄位)、等權與 val 加權集成、防洩漏 (權重只由 val 求)、
成員選擇門檻與 encoder 去重、fail-soft (對齊失敗回 status=failed)、回 None 路徑。
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from agent import ensembler
from agent.config import EnsembleConfig
from agent.privacy import redact
from agent.privacy.alias import AliasMap
from agent.schemas import (EncoderChoice, EnsembleSpec, EvalConfig, Recipe,
                           TrialResult)


# ---- 測試素材 --------------------------------------------------------------
def _write_preds(task_dir: str, mode: str, ids, y_true, probs, classes=("C0", "C1")):
    os.makedirs(task_dir, exist_ok=True)
    df = pd.DataFrame(np.asarray(probs), columns=[f"{c}_score" for c in classes])
    df.insert(0, "image_path", list(ids))
    df["true_label"] = list(y_true)
    df.to_csv(os.path.join(task_dir, f"predictions_{mode}.csv"),
              index=False, encoding="utf-8-sig")


def _trial(tid: str, enc: str, score: float) -> TrialResult:
    return TrialResult(
        trial_id=tid, status="done", primary_score=score,
        recipe=Recipe(encoder=EncoderChoice(model_key=enc)),
        ckpt_path=f"/fake/{tid}/checkpoint-best.pth")


def _cfg() -> EvalConfig:
    return EvalConfig(primary_metric="score",
                      report_metrics=["accuracy", "f1", "roc_auc", "kappa"])


# ---- 對齊 ------------------------------------------------------------------
def test_align_by_image_path_reorders():
    """成員間 image_path 順序不同, 仍以鍵對齊到同一樣本。"""
    with tempfile.TemporaryDirectory() as d:
        _write_preds(os.path.join(d, "a"), "test",
                     ["x", "y", "z"], [0, 1, 0],
                     [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        # b 的列順序打亂
        _write_preds(os.path.join(d, "b"), "test",
                     ["z", "x", "y"], [0, 0, 1],
                     [[0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
        y_true, probs = ensembler._align(d, ["a", "b"], "test")
        assert list(y_true) == [0, 1, 0]           # 依 a (基準) 的順序
        # b 對 x 的機率應被重排到第 0 列 = [0.8, 0.2]
        assert np.allclose(probs[1][0], [0.8, 0.2])


def test_align_mismatched_columns_raises():
    with tempfile.TemporaryDirectory() as d:
        _write_preds(os.path.join(d, "a"), "test", ["x"], [0], [[0.9, 0.1]],
                     classes=("C0", "C1"))
        _write_preds(os.path.join(d, "b"), "test", ["x"], [0], [[0.9, 0.1, 0.0]],
                     classes=("C0", "C1", "C2"))
        try:
            ensembler._align(d, ["a", "b"], "test")
            assert False, "應因類別欄位不一致而 raise"
        except ensembler.AlignError:
            pass


# ---- 集成 (等權 / 加權) ----------------------------------------------------
def test_combine_equal_averages_probs():
    with tempfile.TemporaryDirectory() as d:
        ids, yt = ["x", "y"], [1, 0]
        _write_preds(os.path.join(d, "a"), "test", ids, yt, [[0.4, 0.6], [0.7, 0.3]])
        _write_preds(os.path.join(d, "b"), "test", ids, yt, [[0.2, 0.8], [0.9, 0.1]])
        spec = EnsembleSpec(member_trial_ids=["a", "b"], method="equal")
        members = [_trial("a", "e1", 0.5), _trial("b", "e2", 0.5)]
        out = os.path.join(d, "ens")
        res = ensembler.combine(members, spec, _cfg(), d, out)
        assert res.status == "done" and res.n_samples == 2
        # 落地檔存在, 且第 0 列平均 = [0.3, 0.7]
        got = pd.read_csv(res.pred_path)
        assert np.allclose(got.loc[0, ["C0_score", "C1_score"]].to_numpy(float),
                           [0.3, 0.7])
        assert res.spec.weights == [0.5, 0.5]


def test_val_weighted_prefers_stronger_member_on_val():
    """val 上 a 完美、b 亂猜 → val 加權應把權重壓向 a。"""
    with tempfile.TemporaryDirectory() as d:
        ids = [f"s{i}" for i in range(6)]
        yt = [0, 1, 0, 1, 0, 1]
        a_perfect = [[0.95, 0.05] if y == 0 else [0.05, 0.95] for y in yt]
        b_wrong = [[0.05, 0.95] if y == 0 else [0.95, 0.05] for y in yt]
        for name, p in (("a", a_perfect), ("b", b_wrong)):
            _write_preds(os.path.join(d, name), "val", ids, yt, p)
            _write_preds(os.path.join(d, name), "test", ids, yt, p)
        spec = EnsembleSpec(member_trial_ids=["a", "b"], method="val_weighted")
        members = [_trial("a", "e1", 0.9), _trial("b", "e2", 0.1)]
        res = ensembler.combine(members, spec, _cfg(), d, os.path.join(d, "ens"))
        assert res.status == "done"
        wa, wb = res.spec.weights
        assert wa > wb, f"強成員 a 應獲較大權重, 得 {res.spec.weights}"


def test_combine_missing_predictions_fails_soft():
    with tempfile.TemporaryDirectory() as d:
        _write_preds(os.path.join(d, "a"), "test", ["x"], [0], [[0.9, 0.1]])
        # b 沒有 predictions
        spec = EnsembleSpec(member_trial_ids=["a", "b"], method="equal")
        members = [_trial("a", "e1", 0.5), _trial("b", "e2", 0.5)]
        res = ensembler.combine(members, spec, _cfg(), d, os.path.join(d, "ens"))
        assert res.status == "failed" and "對齊失敗" in res.message


# ---- 成員選擇 --------------------------------------------------------------
def test_select_members_diverse_encoders_and_threshold():
    ec = EnsembleConfig(min_members=2, max_members=3, member_delta=0.05,
                        require_diverse_encoders=True)
    history = [
        _trial("t1", "encA", 0.90),
        _trial("t2", "encA", 0.89),   # 同 encoder 次佳
        _trial("t3", "encB", 0.88),   # 在門檻內 (0.90-0.05=0.85)
        _trial("t4", "encC", 0.70),   # 低於門檻, 應排除
    ]
    spec = ensembler.select_members_heuristic(history, ec)
    assert spec is not None
    picked = spec.member_trial_ids
    assert "t4" not in picked                       # 門檻過濾
    assert picked[:2] == ["t1", "t3"]               # encoder 去重: 各 encoder 先取最佳
    assert len(picked) <= ec.max_members


def test_select_members_returns_none_when_insufficient():
    ec = EnsembleConfig(min_members=2)
    assert ensembler.select_members_heuristic([_trial("t1", "e", 0.9)], ec) is None
    # 全部低於門檻只剩 1 個在 pool 也不足
    ec2 = EnsembleConfig(min_members=2, member_delta=0.01)
    hist = [_trial("t1", "e", 0.9), _trial("t2", "e2", 0.5)]
    assert ensembler.select_members_heuristic(hist, ec2) is None


# ---- stacking (meta-learner) ----------------------------------------------
def test_stacking_combines_via_meta_learner():
    with tempfile.TemporaryDirectory() as d:
        ids = [f"s{i}" for i in range(20)]
        yt = [i % 2 for i in range(20)]
        a = [[0.8, 0.2] if y == 0 else [0.3, 0.7] for y in yt]
        b = [[0.6, 0.4] if y == 0 else [0.45, 0.55] for y in yt]
        for name, p in (("a", a), ("b", b)):
            _write_preds(os.path.join(d, name), "val", ids, yt, p)
            _write_preds(os.path.join(d, name), "test", ids, yt, p)
        spec = EnsembleSpec(member_trial_ids=["a", "b"], method="stacking")
        members = [_trial("a", "e1", 0.7), _trial("b", "e2", 0.6)]
        res = ensembler.combine(members, spec, _cfg(), d, os.path.join(d, "ens"))
        assert res.status == "done"
        assert res.spec.weights is None               # stacking 無單一權重向量
        got = pd.read_csv(res.pred_path)
        s = got[["C0_score", "C1_score"]].to_numpy(float).sum(axis=1)
        assert np.allclose(s, 1.0, atol=1e-6)         # meta-learner 機率行和為 1


def test_stacking_without_val_falls_back_to_equal():
    with tempfile.TemporaryDirectory() as d:
        ids, yt = ["x", "y"], [1, 0]
        for name in ("a", "b"):
            _write_preds(os.path.join(d, name), "test", ids, yt,
                         [[0.4, 0.6], [0.7, 0.3]])           # 只有 test, 無 val
        spec = EnsembleSpec(member_trial_ids=["a", "b"], method="stacking")
        members = [_trial("a", "e1", 0.5), _trial("b", "e2", 0.5)]
        res = ensembler.combine(members, spec, _cfg(), d, os.path.join(d, "ens"))
        assert res.status == "done"
        assert res.spec.weights == [0.5, 0.5]         # 退回等權
        assert "退回" in res.message


# ---- 假名反查還原 (資料圍欄) ----------------------------------------------
def test_resolve_aliased_trial_ids_roundtrip_and_whitelist():
    salt = b"testsalt-abcdef"
    root = "/data/5_fold_PAPILA/PAPILA_seed42_fold0"
    alias = AliasMap.build(root, ["good", "bad"], salt)
    trials = [_trial("encA_PAPILA_seed42_fold0_t1", "encA", 0.9),
              _trial("encB_PAPILA_seed42_fold0_t2", "encB", 0.88)]
    aliased = [alias.substitute(t.trial_id) for t in trials]
    assert aliased[0] != trials[0].trial_id           # 資料集名確實被假名化
    # LLM 回傳: 打亂順序 + 一個幻覺 id
    picked = [aliased[1], aliased[0], "hallucinated_xyz"]
    real = redact.resolve_trial_ids(picked, trials, alias)
    assert real == [trials[1].trial_id, trials[0].trial_id]   # 保序、幻覺被丟棄
    # 直接傳真實 id 也應被接受 (identity)
    assert redact.resolve_trial_ids([trials[0].trial_id], trials, alias) \
        == [trials[0].trial_id]


# ---- pool 排除 ensemble 偽 trial ------------------------------------------
def test_pool_excludes_ensemble_pseudotrials():
    ec = EnsembleConfig(min_members=2)
    hist = [_trial("t1", "encA", 0.90),
            _trial("ens_x", "ensemble", 0.99),   # 偽 trial (encoder=ensemble)
            _trial("t2", "encB", 0.88)]
    spec = ensembler.select_members_heuristic(hist, ec)
    assert spec is not None and "ens_x" not in spec.member_trial_ids


# ---- llm_select 解耦 (獨立於 advisor.type) --------------------------------
def _fake_lc(advisor):
    """建一個最小 LoopController (繞過 __init__) 只為測 _select_ensemble_members。"""
    import types
    from agent.loop_controller import LoopController
    lc = object.__new__(LoopController)
    lc.advisor = advisor
    lc.privacy = None
    lc._ens_llm = None
    lc.cfg = types.SimpleNamespace(advisor=types.SimpleNamespace(model="claude-opus-4-8"))
    lc._say = lambda *a, **k: None
    return lc


class _LLMLike:
    """非 HeuristicAdvisor、帶 propose_ensemble 的假 advisor。"""
    def __init__(self, spec):
        self._spec = spec

    def propose_ensemble(self, profile, history, ec):
        return self._spec


def _two_diverse():
    return [_trial("t1", "encA", 0.90), _trial("t2", "encB", 0.88)]


def test_llm_select_off_forces_heuristic_even_with_llm_advisor():
    # advisor 會選 t2 單一 (不足)，但 llm_select=False 應完全略過它 → 走 heuristic
    boom = _LLMLike(EnsembleSpec(member_trial_ids=["t2"], method="stacking"))
    lc = _fake_lc(boom)
    ec = EnsembleConfig(llm_select=False)
    spec = lc._select_ensemble_members(None, _two_diverse(), ec)
    assert spec is not None
    assert set(spec.member_trial_ids) == {"t1", "t2"}      # heuristic 選出兩個
    assert spec.method == ec.method                         # 非 advisor 的 stacking


def test_llm_select_on_uses_llm_advisor():
    picked = EnsembleSpec(member_trial_ids=["t2", "t1"], method="stacking")
    lc = _fake_lc(_LLMLike(picked))
    ec = EnsembleConfig(llm_select=True)
    spec = lc._select_ensemble_members(None, _two_diverse(), ec)
    assert spec.member_trial_ids == ["t2", "t1"] and spec.method == "stacking"


def test_llm_select_on_none_falls_back_to_heuristic():
    lc = _fake_lc(_LLMLike(None))                           # LLM 說「不值得」→ None
    ec = EnsembleConfig(llm_select=True)
    spec = lc._select_ensemble_members(None, _two_diverse(), ec)
    assert spec is not None and set(spec.member_trial_ids) == {"t1", "t2"}


def test_llm_select_on_heuristic_advisor_builds_dedicated_and_falls_back(monkeypatch=None):
    # 主 advisor 是 heuristic + llm_select=True → 嘗試建專用 LLMAdvisor；
    # 這裡強制建構失敗 → 應退回 heuristic (不崩、不打 API)
    import agent.llm_advisor as lm
    from agent.advisor import HeuristicAdvisor

    class _BoomLLM:
        def __init__(self, **k):
            pass

        def check_environment(self):
            raise RuntimeError("no api (test)")

    orig = lm.LLMAdvisor
    lm.LLMAdvisor = _BoomLLM
    try:
        lc = _fake_lc(HeuristicAdvisor())
        ec = EnsembleConfig(llm_select=True)
        spec = lc._select_ensemble_members(None, _two_diverse(), ec)
        assert spec is not None and set(spec.member_trial_ids) == {"t1", "t2"}
        assert lc._ens_llm is False                        # 失敗已快取, 不重試
    finally:
        lm.LLMAdvisor = orig


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            print(f"✓ {fn.__name__}")
            passed += 1
        except Exception as e:                       # noqa: BLE001
            failed.append(fn.__name__)
            print(f"✗ {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
