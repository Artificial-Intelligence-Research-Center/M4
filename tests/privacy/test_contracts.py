"""契約層的圍欄: 分析器 schema 無自由字串、mode 巨集、DatasetFacts 不含識別欄位。"""
from __future__ import annotations

from pydantic import BaseModel

from agent import analyzers
from agent.config import AgentConfig
from agent.privacy.facts import DatasetFacts

from ._helpers import cleanup, make_canary_dataset


def test_analyzer_schemas_have_no_free_text():
    """自由字串 = 夾帶通道。註冊表裡每個 output_schema 都不得有 str 欄位。"""
    assert analyzers.REGISTRY, "註冊表是空的, 測試失效"
    for key, a in analyzers.REGISTRY.items():
        bad = analyzers.schema_violations(a.output_schema)
        assert not bad, f"分析器 {key} 的 schema 含自由字串: {bad}"


def test_register_rejects_free_text_schema():
    class Leaky(BaseModel):
        note: str = ""

    try:
        analyzers.register(analyzers.Analyzer(
            key="_leaky", description="x", output_schema=Leaky,
            fn=lambda root: Leaky()))
    except ValueError:
        assert "_leaky" not in analyzers.REGISTRY
        return
    analyzers.REGISTRY.pop("_leaky", None)
    raise AssertionError("含自由字串的 schema 竟然註冊成功")


def test_unregistered_analyzer_is_refused():
    facts = analyzers.run("rm -rf /", "/tmp")
    assert not facts.ok and "未註冊" in facts.error


def test_strict_mode_macro():
    cfg = AgentConfig(data_path="/tmp/x")
    cfg.advisor.allow_code_edit = True
    cfg.privacy.log_feedback = "raw"
    cfg.privacy.class_names = "plain"
    cfg.privacy.allow_free_text_questions = True
    cfg = AgentConfig.model_validate(cfg.model_dump())      # 重新套用 validator
    assert cfg.advisor.allow_code_edit is False, "strict 必須關掉程式修改權"
    assert cfg.privacy.log_feedback == "structured"
    assert cfg.privacy.class_names == "hashed"
    assert cfg.privacy.allow_free_text_questions is False


def test_standard_mode_macro():
    cfg = AgentConfig(data_path="/tmp/x")
    cfg.privacy.mode = "standard"
    cfg.advisor.allow_code_edit = True
    cfg.privacy.log_feedback = "raw"
    cfg.privacy.class_names = "plain"
    cfg = AgentConfig.model_validate(cfg.model_dump())
    assert cfg.advisor.allow_code_edit is True, "standard 保留程式修改權"
    assert cfg.privacy.log_feedback == "structured", "但仍切斷 raw log 回讀"
    assert cfg.privacy.class_names == "user_approved"


def test_off_mode_is_not_clamped():
    cfg = AgentConfig(data_path="/tmp/x")
    cfg.privacy.mode = "off"
    cfg.advisor.allow_code_edit = True
    cfg.privacy.log_feedback = "raw"
    cfg = AgentConfig.model_validate(cfg.model_dump())
    assert cfg.advisor.allow_code_edit is True
    assert cfg.privacy.log_feedback == "raw"


def test_dataset_facts_has_no_identifying_fields():
    """DatasetFacts 不得有 root / class_names / modality_hint 這類欄位。"""
    fields = set(DatasetFacts.model_fields)
    for banned in ("root", "class_names", "modality_hint", "path", "data_path"):
        assert banned not in fields, f"DatasetFacts 不該有 {banned} 欄位"


def test_to_facts_drops_path_and_class_names():
    from agent import dataset_analyzer
    from agent.privacy import PrivacyContext

    root = make_canary_dataset()
    try:
        profile = dataset_analyzer.analyze(root)
        cfg = AgentConfig(data_path=root)
        ctx = PrivacyContext.build(cfg.privacy, profile)
        facts = ctx.facts(profile)
        blob = facts.model_dump_json()
        assert "PATIENTWANG" not in blob and "PTNSET7788" not in blob
        assert facts.class_labels == ["C0", "C1"]
        assert sum(facts.class_counts) == profile.n_train
        # modality 只能來自使用者, 不從路徑猜
        assert facts.modality is None
    finally:
        cleanup(root)
