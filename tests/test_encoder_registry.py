"""encoder_registry 目錄式載入測試 (docs/model_registry_design.md §11.6)。

涵蓋: 掃描載入 / manifest 驗證 / fallback / available_cards 存在性 /
weight_path 三種解析 / 壞檔 fail-soft。
"""
from __future__ import annotations

import os

import pytest
import yaml

from agent import encoder_registry as reg
from agent.schemas import EncoderCard


def _write_model(root, key: str, **fields) -> str:
    """在 root 下建一個 model 目錄 (預設是一張合法的卡)。"""
    d = os.path.join(root, key)
    os.makedirs(d, exist_ok=True)
    manifest = {"model": "Dinov2", "model_arch": "dinov2_vitl14",
                "weight": "weights.pth", "embed_dim": 1024, "patch_size": 14}
    want_weights = fields.pop("_weights", True)
    manifest.update(fields)
    with open(os.path.join(d, reg.MANIFEST), "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True)
    if manifest.get("weight") == "weights.pth" and want_weights:
        open(os.path.join(d, "weights.pth"), "wb").close()
    return d


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """把 registry 指向臨時目錄, 每個測試各自乾淨的 baseline_models/。"""
    root = tmp_path / "baseline_models"
    root.mkdir()
    monkeypatch.setattr(reg, "MODELS_DIR", str(root))
    monkeypatch.setattr(reg, "_CACHE", None)
    yield str(root)
    reg._CACHE = None          # 別把臨時目錄的結果留給下一個測試


# --------------------------------------------------------------------------
# 掃描載入
# --------------------------------------------------------------------------
def test_scan_loads_manifest_dirs(models_dir):
    _write_model(models_dir, "fundus_dinov2_vitl14", domain="medical_dap",
                 notes="眼底自監督預訓練")
    _write_model(models_dir, "mae_vit_large", model="MAE", model_arch="MAE",
                 patch_size=16)

    cards = reg.all_cards()
    assert sorted(c.model_key for c in cards) == ["fundus_dinov2_vitl14",
                                                  "mae_vit_large"]
    card = reg.get("fundus_dinov2_vitl14")
    assert card.domain == "medical_dap"
    assert card.model_dir == os.path.join(models_dir, "fundus_dinov2_vitl14")
    assert not reg.warnings()


def test_model_key_defaults_to_dir_name(models_dir):
    _write_model(models_dir, "my_encoder")
    assert reg.get("my_encoder").model_key == "my_encoder"


def test_model_dir_not_in_model_dump(models_dir):
    """model_dir 不得進 model_dump() — 它會被送進 LLM prompt / 寫回 YAML。"""
    _write_model(models_dir, "my_encoder")
    assert "model_dir" not in reg.get("my_encoder").model_dump()


def test_ignores_dirs_without_manifest_and_hidden(models_dir):
    _write_model(models_dir, "good")
    os.makedirs(os.path.join(models_dir, "no_manifest"))
    _write_model(models_dir, "_archive")          # 軟刪除
    _write_model(models_dir, ".trash")            # 隱藏
    open(os.path.join(models_dir, "stray.pth"), "wb").close()

    assert [c.model_key for c in reg.all_cards()] == ["good"]


def test_get_unknown_key_raises(models_dir):
    _write_model(models_dir, "good")
    with pytest.raises(KeyError):
        reg.get("nope")


def test_reload_picks_up_new_dir(models_dir):
    _write_model(models_dir, "first")
    assert len(reg.all_cards()) == 1
    _write_model(models_dir, "second")
    assert len(reg.all_cards()) == 1               # 有快取, 尚未看到
    assert len(reg.reload()) == 2


# --------------------------------------------------------------------------
# 驗證 (§5) — 壞檔跳過並記 warning, 不 raise
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key,fields", [
    ("wrong_key", {"model_key": "something_else"}),      # key 與目錄名不符
    ("bad_family", {"model": "NotARealFamily"}),         # 不在白名單
    ("bad_domain", {"domain": "not_a_domain"}),          # enum 不合法
    ("missing_field", {"model_arch": None}),             # 必要欄位缺失
])
def test_invalid_manifest_is_skipped_with_warning(models_dir, key, fields):
    _write_model(models_dir, "good")
    _write_model(models_dir, key, **fields)

    assert [c.model_key for c in reg.all_cards()] == ["good"]
    assert any(key in w for w in reg.warnings())


def test_broken_yaml_is_skipped(models_dir):
    _write_model(models_dir, "good")
    d = os.path.join(models_dir, "broken")
    os.makedirs(d)
    with open(os.path.join(d, reg.MANIFEST), "w", encoding="utf-8") as f:
        f.write("model: [unclosed\n")

    assert [c.model_key for c in reg.all_cards()] == ["good"]
    assert any("broken" in w for w in reg.warnings())


def test_missing_weight_warns_but_card_loads(models_dir):
    _write_model(models_dir, "no_weights", _weights=False)
    assert [c.model_key for c in reg.all_cards()] == ["no_weights"]
    assert any("權重缺失" in w for w in reg.warnings())
    assert reg.available_cards() == []             # 但不算可用


# --------------------------------------------------------------------------
# available_cards
# --------------------------------------------------------------------------
def test_available_cards_filters(models_dir):
    _write_model(models_dir, "ok")
    _write_model(models_dir, "no_file", _weights=False)
    _write_model(models_dir, "gated", model="RETFound_dinov2",
                 model_arch="retfound_dinov2", weight="RETFound_dinov2_meh",
                 available=False)
    _write_model(models_dir, "hf_ok", model="RETFound_mae",
                 model_arch="retfound_mae", weight="RETFound_mae_natureCVPR")

    # HF id 無法在本機檢查 → 視為可用; 權重缺失 / available=False → 排除
    assert sorted(c.model_key for c in reg.available_cards()) == ["hf_ok", "ok"]
    assert sorted(c.model_key for c in reg.all_cards()) == ["hf_ok", "no_file", "ok"]
    assert len(reg.all_cards(include_unavailable=True)) == 4


# --------------------------------------------------------------------------
# weight_path 三種解析 (§3.2 / §9)
# --------------------------------------------------------------------------
def test_weight_path_dir_relative(models_dir):
    _write_model(models_dir, "local")
    card = reg.get("local")
    assert reg.weight_path(card) == os.path.join(models_dir, "local", "weights.pth")


def test_weight_path_hf_id_passthrough():
    card = EncoderCard(model_key="k", model="RETFound_dinov2",
                       model_arch="retfound_dinov2", weight="RETFound_dinov2_meh")
    assert reg.weight_path(card) == "RETFound_dinov2_meh"


def test_weight_path_repo_relative_legacy():
    card = EncoderCard(model_key="k", model="MAE", model_arch="MAE",
                       weight="baseline_models/old.pth")
    assert reg.weight_path(card) == os.path.join(reg._ROOT, "baseline_models/old.pth")


def test_weight_path_absolute_passthrough(tmp_path):
    p = str(tmp_path / "w.pth")
    card = EncoderCard(model_key="k", model="MAE", model_arch="MAE", weight=p)
    assert reg.weight_path(card) == p


# --------------------------------------------------------------------------
# fallback (§9)
# --------------------------------------------------------------------------
def test_fallback_to_builtin_when_empty(models_dir):
    cards = reg.all_cards(include_unavailable=True)
    assert [c.model_key for c in cards] == [c.model_key for c in reg._BUILTIN_CARDS]
    assert any("回退內建" in w for w in reg.warnings())


def test_fallback_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "MODELS_DIR", str(tmp_path / "does_not_exist"))
    monkeypatch.setattr(reg, "_CACHE", None)
    try:
        assert reg.all_cards(include_unavailable=True)
    finally:
        reg._CACHE = None


def test_arch_whitelist_matches_builtin_cards():
    """內建卡片的架構家族必須都在白名單內 (兩者不同步就會炸掉 fallback)。"""
    assert all(c.model in reg.ARCH_WHITELIST for c in reg._BUILTIN_CARDS)
