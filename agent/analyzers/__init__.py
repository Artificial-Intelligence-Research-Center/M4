"""分析器註冊表 — 「事先寫好的程式」通道 (docs/data_firewall_design.md §5)。

LLM **不能執行任意程式**, 只能從這張表裡**點名** (`request_analysis`) 要跑哪個分析器;
LoopController 在資料平面執行, 輸出經 schema 驗證後才進 prompt。

關鍵約束: 每個分析器的 `output_schema` 欄位型別限定 int / float / bool / Literal /
上述型別的容器 / 巢狀模型 — **不得出現自由字串**。沒有自由字串, 就沒有夾帶通道。
這條由 `schema_violations()` 強制, 註冊時就會檢查, `tests/privacy` 另有一份 CI 測試。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from ..privacy.facts import AnalysisFacts
from . import class_balance, corrupt_files, image_stats, near_dup, split_leakage

Cost = Literal["cheap", "moderate", "expensive"]

_ALLOWED_SCALARS = (int, float, bool, type(None))


@dataclass
class Analyzer:
    key: str
    description: str                       # 給 LLM 看的說明 (會進 system prompt)
    output_schema: type[BaseModel]
    fn: Callable[[str], BaseModel]         # fn(dataset_root) -> 已驗證的輸出
    cost: Cost = "cheap"
    needs_consent: bool = False


# ---------------------------------------------------------------------------
# schema 安全性 — 註冊時強制
# ---------------------------------------------------------------------------
def schema_violations(schema: type[BaseModel], _seen: Optional[set] = None
                      ) -> list[str]:
    """回傳違反「無自由字串」規則的欄位路徑; 空 list = 安全。"""
    _seen = _seen or set()
    if schema in _seen:
        return []
    _seen.add(schema)
    bad: list[str] = []
    for name, field in schema.model_fields.items():
        bad += [f"{schema.__name__}.{name}{s}"
                for s in _type_violations(field.annotation, _seen)]
    return bad


def _type_violations(tp, seen: set) -> list[str]:
    if tp in _ALLOWED_SCALARS:
        return []
    if tp is str:
        return [" (str)"]
    origin = get_origin(tp)
    if origin is Literal:
        # Literal 是列舉, 值域封閉 → 安全 (即使成員是字串)
        return []
    if origin is Union:
        out = []
        for arg in get_args(tp):
            out += _type_violations(arg, seen)
        return out
    if origin in (list, tuple, set, frozenset):
        out = []
        for arg in get_args(tp):
            out += _type_violations(arg, seen)
        return out
    if origin is dict:
        args = get_args(tp)
        # key 允許字串 (是我們自己的欄位名), value 不允許
        return _type_violations(args[1], seen) if len(args) == 2 else [" (dict)"]
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return [f".{v}" for v in schema_violations(tp, seen)]
    return [f" ({getattr(tp, '__name__', tp)})"]


# ---------------------------------------------------------------------------
# 註冊表
# ---------------------------------------------------------------------------
REGISTRY: dict[str, Analyzer] = {}


def register(a: Analyzer) -> Analyzer:
    bad = schema_violations(a.output_schema)
    if bad:
        raise ValueError(
            f"分析器 {a.key} 的 output_schema 含自由字串欄位, 會成為資料夾帶通道: "
            f"{', '.join(bad)}")
    REGISTRY[a.key] = a
    return a


register(Analyzer(
    key="class_balance",
    description="訓練集各類別樣本數、不平衡比、Gini、有效樣本數與建議的 class weight; "
                "並給出是否該用 weighted/focal loss、是否該看 macro-F1 的建議。",
    output_schema=class_balance.ClassBalanceOut, fn=class_balance.run,
    cost="cheap"))

register(Analyzer(
    key="image_stats",
    description="抽樣量測影像尺寸/長寬比百分位、通道模式張數、亮度直方圖與平均飽和度; "
                "用於判斷 input_size、是否需要保留長寬比、是否為灰階資料。",
    output_schema=image_stats.ImageStatsOut, fn=image_stats.run,
    cost="cheap"))

register(Analyzer(
    key="corrupt_files",
    description="抽樣檢查影像能否被 PIL 開啟與解碼, 回報讀不開/截斷的張數與比例 "
                "(不回報是哪些檔案)。訓練一開始就失敗時特別有用。",
    output_schema=corrupt_files.CorruptFilesOut, fn=corrupt_files.run,
    cost="cheap"))

register(Analyzer(
    key="split_leakage",
    description="以檔案雜湊比對 train/val/test 之間有無完全相同的影像 (資料洩漏), "
                "以及重複樣本是否被標成不同類別。只回報數量與比例。",
    output_schema=split_leakage.SplitLeakageOut, fn=split_leakage.run,
    cost="moderate"))

register(Analyzer(
    key="near_dup",
    description="以 perceptual hash 找近重複影像, 回報群數、最大群大小、重複比例, "
                "以及是否跨 split/跨類別。成本較高, 需要使用者同意。",
    output_schema=near_dup.NearDupOut, fn=near_dup.run,
    cost="expensive", needs_consent=True))


# ---------------------------------------------------------------------------
# 執行
# ---------------------------------------------------------------------------
def catalog() -> list[dict]:
    """給 LLM 看的目錄 (放進 system prompt 的可用資源目錄)。"""
    return [{"key": a.key, "description": a.description, "cost": a.cost,
             "needs_consent": a.needs_consent,
             "output_fields": sorted(a.output_schema.model_fields)}
            for a in REGISTRY.values()]


def run(key: str, dataset_root: str) -> AnalysisFacts:
    """執行一個分析器。未註冊的 key 一律拒絕 (LLM 不能執行目錄外的東西)。"""
    a = REGISTRY.get(key)
    if a is None:
        return AnalysisFacts(key=key, ok=False,
                             error=f"未註冊的分析器: {key}")
    try:
        out = a.fn(dataset_root)
        # 再驗一次: 即使實作回傳了別的東西, 也只有 schema 內的欄位出得去
        out = a.output_schema.model_validate(out.model_dump())
        return AnalysisFacts(key=key, ok=True, result=out.model_dump())
    except Exception as e:                       # noqa: BLE001
        return AnalysisFacts(key=key, ok=False,
                             error=f"{type(e).__name__}: {e}")


def run_cached(key: str, dataset_root: str,
               cache_dir: Optional[str] = None) -> AnalysisFacts:
    """同 run(), 但同一個 run 內只算一次 (結果落地 `<run>/analysis/<key>.json`)。"""
    path = os.path.join(cache_dir, f"{key}.json") if cache_dir else None
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf8") as f:
                return AnalysisFacts.model_validate(json.load(f)["facts"])
        except Exception:
            pass
    facts = run(key, dataset_root)
    if path and facts.ok:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf8") as f:
                json.dump({"ts": time.time(), "facts": facts.model_dump()},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return facts


__all__ = ["Analyzer", "REGISTRY", "catalog", "register", "run", "run_cached",
           "schema_violations", "error_extract"]

from . import error_extract  # noqa: E402  (放最後: 避免與 privacy.facts 的循環)
