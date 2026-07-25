"""主驗收測試 — canary 資料集的任何識別字串都不得出現在送往 API 的 payload 中。

涵蓋決策層的每一條 prompt 建構路徑: plan_information / select_encoders /
compose_recipe / propose_next / review_and_decide / review_search_choice /
propose_debug / answer_question。
"""
from __future__ import annotations

import os

from agent import dataset_analyzer
from agent.config import AgentConfig
from agent.llm_advisor import LLMAdvisor
from agent.privacy import PrivacyContext
from agent.privacy.facts import ErrorFacts
from agent.schemas import (ComponentRef, EncoderChoice, HeadSpec, HyperParams,
                           Recipe, TrialResult)

from ._helpers import (CANARY_DATASET, FakeClient, cleanup, find_tokens,
                       make_canary_dataset)


def _fixture():
    root = make_canary_dataset()
    profile = dataset_analyzer.analyze(root)
    cfg = AgentConfig(data_path=root)
    ctx = PrivacyContext.build(cfg.privacy, profile, run_dir=None,
                              data_root=os.path.dirname(os.path.dirname(root)))
    adv = LLMAdvisor(privacy=ctx)
    adv._client = FakeClient()
    return root, profile, adv, adv._client


def _trial(profile, tid: str, status: str = "done") -> TrialResult:
    """刻意用**舊式**的 trial_id (含真實資料集名) — 圍欄必須把它假名化。"""
    recipe = Recipe(
        encoder=EncoderChoice(model_key="dinov2_vitl14"),
        heads=[HeadSpec(type="linear", output_dim=profile.num_classes)],
        losses=[ComponentRef(name="cross_entropy")],
        hparams=HyperParams(),
        provenance={"template": "fundus_classification", "preset": "default",
                    "mutation": "add_regularizer:mixup",
                    "reason": f"改良 {CANARY_DATASET} 的過擬合",
                    "code_dir": f"/data/{CANARY_DATASET}/src"},
    )
    return TrialResult(trial_id=tid, recipe=recipe, status=status,
                       primary_score=0.71, metrics={"score": 0.71},
                       epoch_curve={"train_loss": [1.0, 0.8],
                                    "val_loss": [1.1, 0.9],
                                    "val_score": [0.6, 0.71]})


def test_no_canary_in_any_payload():
    root, profile, adv, fake = _fixture()
    try:
        hist = [_trial(profile, f"dinov2_vitl14_{CANARY_DATASET}_t0"),
                _trial(profile, f"dinov2_vitl14_{CANARY_DATASET}_t1", "failed")]
        enc = EncoderChoice(model_key="dinov2_vitl14")
        # 討論記錄裡塞滿真實名稱 (模擬 loop_controller 寫的 system 訊息)
        discussion = [
            {"role": "system", "kind": "status",
             "text": f"實驗開始：資料 {CANARY_DATASET}（2 類, 不平衡比 1.0）。"},
            {"role": "user", "kind": "user_msg", "text": "請優先試 focal loss"},
        ]
        tree = [{"trial_id": f"dinov2_vitl14_{CANARY_DATASET}_t0",
                 "stage": "draft", "parent": None, "encoder": "dinov2_vitl14",
                 "allowed": ["improve"]}]

        for call in (
            lambda: adv.plan_information(profile),
            lambda: adv.select_encoders(profile),
            lambda: adv.compose_recipe(profile, enc),
            lambda: adv.propose_next(profile, enc, hist),
            lambda: adv.review_and_decide(profile, enc, hist, discussion),
            lambda: adv.review_search_choice(profile, tree, {"stage": "improve"},
                                             discussion),
            lambda: adv.propose_debug(profile, enc, hist[1],
                                      ErrorFacts(error_class="oom")),
            lambda: adv.answer_question(profile, hist, {"nodes": tree},
                                        discussion, "目前哪個最好？"),
        ):
            try:
                call()
            except Exception:
                pass      # 假 client 回 "{}" → 解析失敗無所謂, 我們要的是 payload

        assert fake.calls, "沒有任何呼叫被記錄, 測試本身失效"
        leaked = find_tokens(fake.all_text())
        assert not leaked, f"canary 字串外洩到 payload: {leaked}"
        # 真實路徑也不得出現
        assert root.lower() not in fake.all_text().lower(), "資料集絕對路徑外洩"
        # 假名機制確實有在運作 (不是因為整個 payload 是空的才通過)
        assert ctx_ref(adv) in fake.all_text(), "payload 中找不到資料集假名"
    finally:
        cleanup(root)


def ctx_ref(adv) -> str:
    return adv.privacy.alias.dataset_ref


def test_raw_log_never_reaches_payload():
    """原始 log (含 data_path 與影像路徑) 不得進 prompt — 只有 ErrorFacts 會。"""
    root, profile, adv, fake = _fixture()
    try:
        from agent.analyzers import error_extract
        log = os.path.join(root, "_fake_log.txt")
        with open(log, "w", encoding="utf8") as f:
            f.write(f"Namespace(batch_size=24, data_path='{root}',\n")
            f.write(f"Traceback: FileNotFoundError: {root}/train/"
                    f"PATIENTWANG_dr3/CANARYTOKEN0001_train_0.jpg\n")
            f.write("torch.cuda.OutOfMemoryError: CUDA out of memory. "
                    "Tried to allocate 2.00 GiB\n")
        facts = error_extract.extract(log)
        assert facts.error_class == "oom"
        # ErrorFacts 本身不含任何路徑
        assert not find_tokens(facts.model_dump_json()), "ErrorFacts 夾帶了識別字串"

        enc = EncoderChoice(model_key="dinov2_vitl14")
        try:
            adv.propose_debug(profile, enc, _trial(profile, "t1", "failed"), facts)
        except Exception:
            pass
        leaked = find_tokens(fake.all_text())
        assert not leaked, f"debug prompt 外洩: {leaked}"
    finally:
        cleanup(root)


def test_egress_blocks_and_audits_a_deliberate_leak():
    """故意讓 prompt 帶上真實路徑 → 呼叫必須中止 (fail-closed) 並留下稽核紀錄。"""
    import tempfile

    from agent import dataset_analyzer
    from agent.privacy import EgressViolation, PrivacyContext, egress

    root = make_canary_dataset()
    run_dir = tempfile.mkdtemp(prefix="medclaw_audit_")
    try:
        profile = dataset_analyzer.analyze(root)
        cfg = AgentConfig(data_path=root)
        ctx = PrivacyContext.build(
            cfg.privacy, profile, run_dir=run_dir,
            data_root=os.path.dirname(os.path.dirname(root)))
        adv = LLMAdvisor(privacy=ctx)
        adv._client = FakeClient()
        # 模擬「有人繞過消毒, 把原始 profile 塞進 prompt」
        adv._facts_block = lambda p: f"DatasetProfile: root={root}"

        try:
            adv.select_encoders(profile)
        except EgressViolation as e:
            assert e.violations, "違規明細沒有被帶出來"
        else:
            raise AssertionError("圍欄沒有攔下含真實路徑的 payload")

        assert not adv._client.calls, "被攔下的呼叫竟然還是送出去了"
        audit = egress.read_audit(run_dir)
        assert any(a.get("verdict") == "blocked" for a in audit), "稽核沒記到攔截"
    finally:
        cleanup(root)
        __import__("shutil").rmtree(run_dir, ignore_errors=True)


def test_trial_id_is_pseudonymised():
    from agent.privacy import redact
    root, profile, adv, _ = _fixture()
    try:
        t = _trial(profile, f"dinov2_vitl14_{CANARY_DATASET}_t3")
        tf = redact.trial_facts(t, adv.privacy.alias)
        assert CANARY_DATASET not in tf.trial_id
        assert adv.privacy.alias.dataset_ref in tf.trial_id
        # provenance 只留白名單欄位, 且路徑類 key 被丟掉
        assert "code_dir" not in tf.provenance.model_dump()
        assert CANARY_DATASET not in tf.provenance.model_dump_json()
    finally:
        cleanup(root)
