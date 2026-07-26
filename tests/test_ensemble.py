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
