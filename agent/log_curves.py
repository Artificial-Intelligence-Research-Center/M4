"""從 main_finetune 的 stdout log 抽出逐 epoch 曲線 (train/val loss + val score).

供 LoopController 在每個 trial 完成後取得曲線, 交給 Advisor 判斷 epochs 是否合適
(結束時 loss 仍下降→ epochs 不足; 很早收斂 / val loss 回升→ 過擬合, 應減少)。
web 端亦沿用同一組 regex (agent/web/app.py)。
"""
from __future__ import annotations

import os
import re

METRIC_KEYS = ["accuracy", "f1", "roc_auc", "hamming", "jaccard",
               "precision", "recall", "average_precision", "kappa", "score"]
# 數值一律允許負號: kappa 低於隨機水準時為負 (score = (f1+roc_auc+kappa)/3 也跟著變負)。
# 少了 `-?` 整個區塊匹配失敗 → 該 epoch 從 val_score 中消失, 之後每個點的 epoch 位置左移。
_NUM = r"(-?[\d.]+)"
METRIC_RE = re.compile(
    rf"Accuracy:\s*{_NUM},\s*F1 Score:\s*{_NUM},\s*ROC AUC:\s*{_NUM},\s*"
    rf"Hamming Loss:\s*{_NUM},\s*Jaccard Score:\s*{_NUM},\s*Precision:\s*{_NUM},\s*"
    rf"Recall:\s*{_NUM},\s*Average Precision:\s*{_NUM},\s*Kappa:\s*{_NUM},\s*"
    rf"Score:\s*{_NUM}", re.S)

# 訓練跑完後, main_finetune 會用「val 最佳 checkpoint」再跑一次 final_val/test
# (main_finetune.py「Final Test (Best Ckpt)」)。那段輸出與逐 epoch 的 val 區塊完全同格式
# (連 "val loss:" 字樣都一樣) → 抽逐 epoch 曲線前必須先切掉, 否則曲線末端會多出一個
# 其實是 test 的「epoch」: val_loss 憑空跳升、val_score 憑空掉一截。
_FINAL_EVAL_RE = re.compile(
    r"^[^\n]*(?:Val test|Test) with the best model[^\n]*$", re.M)


def split_final_eval(text: str) -> tuple[str, list[dict]]:
    """拆成 (逐 epoch 訓練文字, 最終評估區塊清單)。

    每個最終評估區塊 = 標記行 → 其後第一個指標區塊結束, 解析成
    {mode: "val"|"test", metrics: {...}, loss: float}。
    """
    kept: list[str] = []
    finals: list[dict] = []
    pos = 0
    for m in _FINAL_EVAL_RE.finditer(text):
        if m.start() < pos:            # 已被前一個區塊涵蓋
            continue
        kept.append(text[pos:m.start()])
        mm = METRIC_RE.search(text, m.end())
        stop = mm.end() if mm else len(text)
        block = text[m.start():stop]
        d = {"mode": "val" if "Val test" in m.group(0) else "test"}
        if mm:
            d["metrics"] = dict(zip(METRIC_KEYS,
                                    [float(x) for x in mm.groups()]))
        if lm := re.search(r"val loss:\s*([\d.]+)", block):
            d["loss"] = float(lm.group(1))
        finals.append(d)
        pos = stop
    kept.append(text[pos:])
    return "".join(kept), finals


def parse(log_path: str) -> dict:
    """回傳 {train_loss, val_loss, val_score} (逐 epoch) + final_test (最終評估)。

    三條曲線只含「逐 epoch 的 val」, 長度一致 (= 實際跑完的 epoch 數);
    最佳 checkpoint 的最終 test 成績另外放在 final_test, 不混進曲線。
    """
    if not log_path or not os.path.isfile(log_path):
        return {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return {}
    text, finals = split_final_eval(text)
    train_loss = [round(float(x), 5) for x in re.findall(
        r"Averaged stats:.*?loss:\s*[\d.]+\s*\(([\d.]+)\)", text)]
    val_loss = [round(float(x), 5) for x in re.findall(r"val loss:\s*([\d.]+)", text)]
    val_score = [round(float(m.groups()[-1]), 5) for m in METRIC_RE.finditer(text)]
    final_test = next((f for f in reversed(finals) if f["mode"] == "test"), None)
    return {"train_loss": train_loss, "val_loss": val_loss,
            "val_score": val_score, "final_test": final_test}


_ITER_TIME_RE = re.compile(r"time:\s*([\d.]+)\s+data:\s*([\d.]+)")


def data_fraction(log_path: str) -> float | None:
    """從 log 的逐 iteration `time: X  data: Y` 估計資料載入佔比 (0~1)。

    佔比高 (>~0.3) 表示 dataloader 是瓶頸 — GPU 利用率最佳化會優先增加
    num_workers 而不是只加大 batch。解析不到回 None。
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            pairs = _ITER_TIME_RE.findall(f.read())
    except OSError:
        return None
    if len(pairs) < 5:
        return None
    tot = sum(float(t) for t, _ in pairs)
    dat = sum(float(d) for _, d in pairs)
    return round(dat / tot, 3) if tot > 0 else None


def unconverged(curve: dict, window: int = 3, rel_tol: float = 0.005) -> tuple[bool, str]:
    """判斷曲線『訓練結束時是否還沒收斂』(繼續訓練策略用)。回傳 (未收斂?, 理由)。

    規則 (最後 window 個 epoch 的平均 vs 再前 window 個的平均):
      - val_loss 已回升 (過擬合傾向) → 視為已收斂, 不續訓;
      - train_loss 仍下降超過 rel_tol (相對) 或 val_score 仍在上升 → 未收斂;
      - 曲線太短 (不足 2*window) → 資訊不足, 視為已收斂。
    """
    tl = curve.get("train_loss") or []
    vl = curve.get("val_loss") or []
    vs = curve.get("val_score") or []
    if len(tl) < 2 * window:
        return False, f"曲線太短 ({len(tl)} epochs), 資訊不足視為已收斂"

    def delta(seq):  # 前段平均 - 後段平均 (>0 = 仍在下降)
        prev = sum(seq[-2 * window:-window]) / window
        last = sum(seq[-window:]) / window
        return prev - last, prev

    if len(vl) >= 2 * window:
        d, base = delta(vl)
        if base and d / abs(base) < -rel_tol:
            return False, "val_loss 已回升 (過擬合傾向), 不宜續訓"
    d, base = delta(tl)
    if base and d / abs(base) > rel_tol:
        return True, (f"train_loss 最後 {window} epoch 平均仍下降 "
                      f"{d / abs(base) * 100:.1f}%")
    if len(vs) >= 2 * window:
        d, _ = delta(vs)
        if -d > 0.002:  # score 越大越好: 後段平均仍在上升
            return True, f"val_score 仍在上升 (+{-d:.4f})"
    return False, "loss/score 已趨平"
