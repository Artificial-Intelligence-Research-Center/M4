"""超參起點 (HyperParams preset) 儲存 — 可編輯、可持久化.

內建 default/paper/mae 三組起點 (README Exp1)。使用者可在 web「超參起點」頁編輯任一組
或新增自訂 preset; 覆寫值存到 agent/presets.json, 之後每次 load() 疊在內建值上。
Advisor 於 compose_recipe 時以 presets.get(name) 讀取「當下」的值 (不快取), 故編輯即生效。
"""
from __future__ import annotations

import json
import os

from .schemas import HyperParams

_DIR = os.path.dirname(os.path.abspath(__file__))
_FILE = os.path.join(_DIR, "presets.json")

# 內建起點 (與原 advisor.HPARAM_PRESETS 一致)
BUILTIN: dict[str, HyperParams] = {
    "default": HyperParams(batch_size=24, blr=5e-3, layer_decay=0.65, drop_path=0.2),
    "paper":   HyperParams(batch_size=24, blr=5e-4, layer_decay=0.65, drop_path=0.2),
    "mae":     HyperParams(batch_size=32, accum_iter=2, blr=1e-3,
                           layer_decay=0.75, drop_path=0.1),
}


def _read_raw() -> dict:
    if os.path.isfile(_FILE):
        try:
            with open(_FILE, encoding="utf8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_raw(data: dict) -> None:
    with open(_FILE, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load() -> dict[str, HyperParams]:
    """內建 + presets.json 覆寫/新增, 回傳目前所有 preset。"""
    out = {k: v.model_copy() for k, v in BUILTIN.items()}
    for name, fields in _read_raw().items():
        base = out.get(name, HyperParams())
        known = {k: v for k, v in fields.items() if k in HyperParams.model_fields}
        try:
            out[name] = base.model_copy(update=HyperParams.model_validate(
                {**base.model_dump(), **known}).model_dump())
        except Exception:
            pass
    return out


def get(name: str) -> HyperParams:
    d = load()
    return (d.get(name) or d.get("default") or HyperParams()).model_copy()


def names() -> list[str]:
    return list(load())


def as_dict() -> dict[str, dict]:
    return {k: v.model_dump() for k, v in load().items()}


def is_builtin(name: str) -> bool:
    return name in BUILTIN


def save_one(name: str, fields: dict) -> HyperParams:
    """新增/更新一組 preset (fields 值可為字串, 由 pydantic coercion 轉型)。"""
    name = name.strip()
    if not name:
        raise ValueError("preset 名稱不可為空")
    base = BUILTIN.get(name, HyperParams()).model_dump()
    known = {k: fields[k] for k in fields if k in HyperParams.model_fields and fields[k] != ""}
    hp = HyperParams.model_validate({**base, **known})
    data = _read_raw()
    data[name] = hp.model_dump()
    _write_raw(data)
    return hp


def delete(name: str) -> None:
    """刪除自訂 preset (內建的還原為預設值)。"""
    data = _read_raw()
    data.pop(name, None)
    _write_raw(data)


def reset_all() -> None:
    """清除所有覆寫, 全部還原為內建預設。"""
    if os.path.isfile(_FILE):
        os.remove(_FILE)
