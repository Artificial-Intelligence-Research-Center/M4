"""全域設定 (Web「設定」頁) — 持久化 API KEY / model / improve_temperature 等整體參數.

儲存於 `~/.medclaw/settings.json`（與 privacy_salt 同目錄, 刻意放在 repo 之外, 避免
API key 進 git）。API KEY 於載入/儲存時寫入 `os.environ["ANTHROPIC_API_KEY"]`, 讓
LLMAdvisor 直接取用。新實驗 (run_full) 以這裡的值為預設; 表單有填的欄位仍會覆寫。

新增一個整體參數 = 在 FIELDS 加一列, 再到 loop_controller/config 對應套用即可;
模板與解析都由 FIELDS 驅動, 不必改 HTML。
"""
from __future__ import annotations

import json
import os

_DIR = os.path.expanduser(os.path.join("~", ".medclaw"))
_PATH = os.path.join(_DIR, "settings.json")

# 主要指標可選值 (單一來源, 供設定頁與各表單下拉共用)。score=(f1+roc_auc+kappa)/3;
# 其餘為 Evaluator/metric_registry 支援的分類指標 (見 agent/evaluator.py)。
PRIMARY_METRICS = ["score", "accuracy", "f1", "roc_auc", "kappa",
                   "balanced_accuracy", "precision", "recall", "average_precision"]

# (key, label, type, default, choices)  type ∈ {secret, str, int, float, choice}
FIELDS: list[tuple] = [
    ("anthropic_api_key", "ANTHROPIC_API_KEY", "secret", "", None),
    ("advisor_type", "決策層 (advisor)", "choice", "llm", ["llm", "heuristic", "skill"]),
    ("model", "LLM 模型 (model)", "str", "claude-opus-4-8", None),
    ("improve_temperature", "improve_temperature（節點抽樣溫度；≤0=greedy 只選最佳）",
     "float", 0.2, None),
    ("num_drafts", "起手 drafts（跨 encoder）", "int", 3, None),
    ("max_trials", "max_trials（總輪數上限）", "int", 12, None),
    ("min_trials", "min_trials（至少輪數）", "int", 6, None),
    ("patience", "patience（連續無提升容忍）", "int", 4, None),
    ("debug_prob", "debug_prob（優先除錯 buggy leaf 的機率）", "float", 0.5, None),
    ("max_debug_depth", "max_debug_depth（除錯鏈上限）", "int", 2, None),
    ("resume_epochs", "resume_epochs（每次續訓多跑的 epoch）", "int", 20, None),
    ("max_resumes", "max_resumes（同分支續訓上限）", "int", 2, None),
    ("privacy_mode", "資料圍欄 (privacy)", "choice", "strict", ["strict", "standard", "off"]),
    ("primary_metric", "主要指標 (primary_metric)", "choice", "score", PRIMARY_METRICS),
    # ---- 集成 (ensemble; docs/ensemble_design.md) ----
    ("ensemble_enabled", "集成 enabled（收尾把多模型組 ensemble）",
     "choice", "true", ["true", "false"]),
    ("ensemble_method", "集成方法 method",
     "choice", "val_weighted", ["equal", "val_weighted", "stacking"]),
    ("ensemble_llm_select", "集成成員用 LLM 選（關＝heuristic，獨立於 advisor）",
     "choice", "false", ["true", "false"]),
    ("ensemble_min_members", "集成 min_members（至少幾個成員）", "int", 2, None),
    ("ensemble_max_members", "集成 max_members（最多幾個成員）", "int", 4, None),
    ("ensemble_member_delta", "集成 member_delta（成員門檻：best − 此值）",
     "float", 0.05, None),
    ("ensemble_in_search", "搜尋中集成 in_search（方案B：平坦期即中途組）",
     "choice", "false", ["true", "false"]),
    ("ensemble_search_patience", "搜尋中集成 search_patience（觸發的無提升輪數）",
     "int", 2, None),
    ("ensemble_max_search_ensembles", "搜尋中集成次數上限", "int", 3, None),
]

_DEFAULTS = {k: d for k, _l, _t, d, _c in FIELDS}
_TYPES = {k: t for k, _l, t, _d, _c in FIELDS}


def _coerce(key: str, val):
    """把字串/數值轉成該欄位型別; 轉不動就退回內建 default。"""
    t = _TYPES.get(key, "str")
    if t == "int":
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return _DEFAULTS[key]
    if t == "float":
        try:
            return float(val)
        except (TypeError, ValueError):
            return _DEFAULTS[key]
    return "" if val is None else str(val)


def load() -> dict:
    """回傳完整設定 (檔案缺欄位以內建 default 補齊)。"""
    out = dict(_DEFAULTS)
    try:
        with open(_PATH, encoding="utf8") as f:
            raw = json.load(f) or {}
        for k in _DEFAULTS:
            if k in raw:
                out[k] = _coerce(k, raw[k])
    except (OSError, ValueError):
        pass
    return out


def save(values: dict) -> dict:
    """合併並寫入。空字串的 API key 視為「不變更」(避免手滑清掉既有 key)。"""
    cur = load()
    for k in _DEFAULTS:
        if k not in values:
            continue
        v = values[k]
        if k == "anthropic_api_key" and (v is None or str(v).strip() == ""):
            continue
        cur[k] = _coerce(k, v)
        if _TYPES[k] in ("secret", "str", "choice") and isinstance(cur[k], str):
            cur[k] = cur[k].strip()
    os.makedirs(_DIR, exist_ok=True)
    with open(_PATH, "w", encoding="utf8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(_PATH, 0o600)   # 含 API key: 僅本人可讀寫
    except OSError:
        pass
    apply_env(cur)
    return cur


def apply_env(values: dict | None = None) -> None:
    """把設定裡的 API key 灌進 os.environ, 讓 LLMAdvisor 這種讀 env 的元件直接取用。"""
    v = values if values is not None else load()
    key = (v.get("anthropic_api_key") or "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
