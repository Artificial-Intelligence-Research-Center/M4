"""Ensembler — 機率層級軟投票集成 (docs/ensemble_design.md, 方案 A)。

**資料平面模組**: 讀取各 trial 已落地的 predictions_{val,test}.csv, 對齊後加權平均,
用與單模型同口徑的指標評分, 落地集成 predictions。**全程不外送任何逐樣本資料** —
決策層只透過 EnsembleSpec (選 trial_id + 方法) 參與, 只有彙整指標回流。

核心利多: 重用訓練時已存的逐樣本 softmax 機率, 集成不需重訓、不需 GPU。

對齊: predictions_*.csv 已含 `image_path` 欄 (engine_finetune), 以此為鍵 join,
把「各成員列對列對應同一樣本」從假設變成可驗證 (不一致即中止該次集成)。
"""
from __future__ import annotations

import os
from typing import Optional

from . import evaluator as ev
from . import metric_registry as mreg
from .schemas import EnsembleResult, EnsembleSpec, EvalConfig, TrialResult


# ---------------------------------------------------------------------------
# 讀取與對齊
# ---------------------------------------------------------------------------
def _task_dir(trials_dir: str, trial_id: str) -> str:
    return os.path.join(trials_dir, trial_id)


def _read_pred_df(trials_dir: str, trial_id: str, mode: str):
    """讀一個成員的 predictions_{mode}.csv → (DataFrame, score_cols) 或 (None, None)。"""
    import pandas as pd
    path = os.path.join(_task_dir(trials_dir, trial_id), f"predictions_{mode}.csv")
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if "true_label" not in df.columns or not score_cols:
        return None, None
    return df, score_cols


class AlignError(Exception):
    """成員間 predictions 無法安全對齊 (樣本/類別不一致)。"""


def _align(trials_dir: str, trial_ids: list[str], mode: str):
    """把多個成員的 predictions 對齊到同一批樣本、同一類別欄位順序。

    回傳 (y_true, [prob_k ...])，prob_k 為 [N, C] ndarray；失敗時 raise AlignError。
    """
    import numpy as np

    dfs = []
    for tid in trial_ids:
        df, sc = _read_pred_df(trials_dir, tid, mode)
        if df is None:
            raise AlignError(f"成員 {tid} 缺少可用的 predictions_{mode}.csv")
        dfs.append((tid, df, sc))

    base_cols = dfs[0][2]                       # 以第一個成員的類別欄位順序為準
    for tid, df, sc in dfs:
        if set(sc) != set(base_cols):
            raise AlignError(f"成員 {tid} 的類別欄位與基準不一致: {sorted(sc)}")

    use_key = all("image_path" in df.columns for _, df, _ in dfs)
    if use_key:
        # 以 image_path 交集 (保基準順序) 對齊 —— 樣本層級可驗證
        base_ids = list(dict.fromkeys(dfs[0][1]["image_path"].tolist()))
        common = set(base_ids)
        for _, df, _ in dfs[1:]:
            common &= set(df["image_path"].tolist())
        ids = [i for i in base_ids if i in common]
        if not ids:
            raise AlignError("成員間 image_path 無交集, 無法對齊")
        aligned, y_true = [], None
        for tid, df, _ in dfs:
            d = df.drop_duplicates("image_path").set_index("image_path").loc[ids]
            aligned.append(d[base_cols].to_numpy(dtype=float))
            yt = d["true_label"].to_numpy()
            if y_true is None:
                y_true = yt
            elif not np.array_equal(yt, y_true):
                raise AlignError(f"成員 {tid} 對同一樣本的 true_label 與基準不符")
        return np.asarray(y_true), aligned

    # 無 image_path: 退回列順序對齊 (要求等長)
    n = len(dfs[0][1])
    for tid, df, _ in dfs:
        if len(df) != n:
            raise AlignError(f"成員 {tid} 樣本數 {len(df)} 與基準 {n} 不符 (且無 image_path 可對齊)")
    y_true = dfs[0][1]["true_label"].to_numpy()
    aligned = [df[base_cols].to_numpy(dtype=float) for _, df, _ in dfs]
    return y_true, aligned


# ---------------------------------------------------------------------------
# 指標 (比照 engine_finetune 口徑, 從 (y_true, y_prob) 重算)
# ---------------------------------------------------------------------------
def _clf_metrics(y_true, y_prob) -> dict:
    import numpy as np
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                  precision_score, recall_score, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_cls = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    onehot = np.eye(n_cls)[y_true]
    pred_onehot = np.eye(n_cls)[y_pred]
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(onehot, pred_onehot, average="macro", zero_division=0)),
        "precision": float(precision_score(onehot, pred_onehot, average="macro", zero_division=0)),
        "recall": float(recall_score(onehot, pred_onehot, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(onehot, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass  # 單類別/退化情形: 略過 roc_auc
    return m


def _weighted(probs: list, weights) -> "object":
    import numpy as np
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() or 1.0)
    out = np.zeros_like(probs[0], dtype=float)
    for wi, p in zip(w, probs):
        out += wi * p
    return out


def _primary(y_true, y_prob, cfg: EvalConfig) -> float:
    """集成向量的 primary — 沿用 evaluator.compute_primary 的口徑 (score 或內建指標)。

    自訂 (registry) primary 需要 task_dir 才能算, 這裡以內建指標為主 (weight 最佳化用)。
    """
    return ev.compute_primary(_clf_metrics(y_true, y_prob), cfg, task_dir=None, mode="val")


# ---------------------------------------------------------------------------
# 權重最佳化 (在 val 上; test 只報告 — 杜絕洩漏)
# ---------------------------------------------------------------------------
def _simplex_grid(n: int, g: int = 10):
    """列舉長度 n、和為 g 的非負整數組合 → 正規化權重 (含等權)。"""
    def rec(k: int, rem: int):
        if k == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for tail in rec(k - 1, rem - i):
                yield (i,) + tail
    for combo in rec(n, g):
        yield [c / g for c in combo]


def _optimize_weights(val_probs: list, y_true_val, cfg: EvalConfig) -> list:
    """在 val 上以單純形網格搜尋最大化 primary 的權重; 失敗退回等權。"""
    n = len(val_probs)
    equal = [1.0 / n] * n
    try:
        best_w, best_s = equal, _primary(y_true_val, _weighted(val_probs, equal), cfg)
        # n 太大時網格爆炸 → 只用等權 (成員上限由 config.max_members 控管, 一般 ≤4)
        g = 10 if n <= 4 else 0
        for w in (_simplex_grid(n, g) if g else []):
            s = _primary(y_true_val, _weighted(val_probs, w), cfg)
            if s > best_s:
                best_s, best_w = s, w
        return best_w
    except Exception:
        return equal


# ---------------------------------------------------------------------------
# 成員選擇 (heuristic; LLM 版之後再加)
# ---------------------------------------------------------------------------
def select_members_heuristic(history: list[TrialResult], cfg) -> Optional[EnsembleSpec]:
    """規則式選成員: primary 門檻 (best − delta) + 可選的 encoder 去重, 取前 max_members。

    cfg = EnsembleConfig。回傳 None 表示不值得集成 (成員不足)。
    以 primary_score (test) 選成員, 與系統既有 self.best 一致; 權重才在 val 上求。
    """
    done = [t for t in history if t.status == "done"]
    if len(done) < cfg.min_members:
        return None
    ranked = sorted(done, key=lambda t: t.primary_score, reverse=True)
    best = ranked[0].primary_score
    pool = [t for t in ranked if t.primary_score >= best - cfg.member_delta]

    picked: list[TrialResult] = []
    if cfg.require_diverse_encoders:
        seen = set()
        for t in pool:                       # 先每個 encoder 取最佳一個 (去相關)
            enc = t.recipe.encoder.model_key
            if enc not in seen:
                seen.add(enc)
                picked.append(t)
        for t in pool:                       # 名額未滿再補同 encoder 的次佳
            if t not in picked:
                picked.append(t)
    else:
        picked = list(pool)

    picked = picked[: cfg.max_members]
    if len(picked) < cfg.min_members:
        return None
    method = getattr(cfg, "method", "equal")
    return EnsembleSpec(
        member_trial_ids=[t.trial_id for t in picked], method=method,
        rationale=f"heuristic: primary ≥ {best - cfg.member_delta:.4f} 的前 {len(picked)} 個"
                  + ("（encoder 去重）" if cfg.require_diverse_encoders else ""))


# ---------------------------------------------------------------------------
# 集成主流程
# ---------------------------------------------------------------------------
def combine(members: list[TrialResult], spec: EnsembleSpec, cfg: EvalConfig,
            trials_dir: str, out_dir: str) -> EnsembleResult:
    """讀成員 predictions → 對齊校驗 → (val 定權) → test 加權平均 → 評分 → 落地。

    對齊或評分失敗回傳 status='failed' 的 EnsembleResult (fail-soft, 不中斷實驗)。
    """
    import numpy as np
    import pandas as pd

    tids = list(spec.member_trial_ids)
    by_id = {m.trial_id: m for m in members}
    ens_id = spec.member_trial_ids and ("ens_" + "_".join(
        t.split("_")[-1] for t in tids)) or "ensemble"

    def _fail(msg: str) -> EnsembleResult:
        return EnsembleResult(ensemble_id=ens_id, spec=spec, status="failed", message=msg)

    try:
        y_true, test_probs = _align(trials_dir, tids, "test")
    except AlignError as e:
        return _fail(f"對齊失敗: {e}")
    except Exception as e:                        # noqa: BLE001
        return _fail(f"讀取 test predictions 失敗: {e}")

    # 權重: equal 直接均權; val_weighted 在 val 上最佳化 (val 對齊失敗則退等權)
    n = len(test_probs)
    weights = [1.0 / n] * n
    if spec.method == "val_weighted":
        try:
            y_true_val, val_probs = _align(trials_dir, tids, "val")
            weights = _optimize_weights(val_probs, y_true_val, cfg)
        except Exception:
            weights = [1.0 / n] * n               # val 缺失/不對齊 → 退回等權

    ens_prob = _weighted(test_probs, weights)
    metrics = _clf_metrics(y_true, ens_prob)

    # 落地集成 predictions_test.csv (供報告/復現/自訂 primary 計算)
    os.makedirs(out_dir, exist_ok=True)
    # 類別欄位順序取自第一個成員
    _, base_cols = _read_pred_df(trials_dir, tids[0], "test")
    pred_df = pd.DataFrame(ens_prob, columns=base_cols)
    pred_df["true_label"] = np.asarray(y_true).astype(int)
    pred_df["pred_label"] = ens_prob.argmax(axis=1)
    pred_path = os.path.join(out_dir, "predictions_test.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # report_metrics 中的自訂 (registry) 指標: 從落地的 predictions 補算
    for name in cfg.report_metrics:
        real = ev._METRIC_ALIASES.get(name, name)
        if real not in metrics and mreg.get(real) is not None:
            v = ev._custom_metric(real, out_dir, "test")
            if v is not None:
                metrics[real] = v

    primary = ev.compute_primary(metrics, cfg, task_dir=out_dir, mode="test")
    spec.weights = [round(float(w), 4) for w in weights]

    return EnsembleResult(
        ensemble_id=ens_id, spec=spec, metrics=metrics, primary_score=primary,
        member_ckpts=[by_id[t].ckpt_path for t in tids if by_id.get(t) and by_id[t].ckpt_path],
        pred_path=pred_path, n_samples=int(len(y_true)), status="done",
        message=f"{n} 成員 · method={spec.method} · weights={spec.weights}")
