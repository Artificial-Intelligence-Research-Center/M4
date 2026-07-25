"""出口掃描器與哨兵。"""
from __future__ import annotations

import os

from agent.privacy import EgressViolation
from agent.privacy.egress import EgressContext, check
from agent.privacy.redact import Guard

from ._helpers import cleanup, make_canary_dataset


def _guard(root: str) -> Guard:
    return Guard.build(data_root=os.path.dirname(os.path.dirname(root)),
                       dataset_root=root,
                       class_names=["PATIENTWANG_dr3", "LINMEIHUA_normal"],
                       exempt_text="fundus classification dinov2 mixup")


def test_guard_blocks_identifiers():
    root = make_canary_dataset()
    try:
        g = _guard(root)
        cases = {
            "dataset_path": f"訓練資料在 {root}",
            "dataset_name": "本輪使用 PTNSET7788_seed42_fold0 這份資料",
            "class_name": "類別 PATIENTWANG_dr3 樣本偏少",
            "image_filename": "讀不到 CANARYTOKEN0001_train_0.jpg",
            "base64_image": "圖片: /9j/4AAQSkZJRgABAQEASABIAAD1234567890",
            "numeric_dump": "tensor: " + "0.13, " * 600,
            "national_id": "病人 A123456789 的影像",
        }
        for expect, text in cases.items():
            rules = {v.rule for v in g.scan(text)}
            assert rules, f"沒攔到 {expect}: {text[:40]}"
        # 正常的決策 prompt 不該被誤攔
        ok = ("資料集 DS-abc123#f0 共 3 類, 不平衡比 4.9; "
              "建議 dinov2_vitl14 finetune + weighted_ce; "
              "train_loss_curve: [1.2, 0.9, 0.7, 0.55]")
        assert not g.scan(ok), f"誤攔正常 prompt: {g.scan(ok)}"
    finally:
        cleanup(root)


def test_user_input_downgraded_to_warning():
    """使用者自願貼上的內容只警示不中止 (guard_on_user_input=warn)。"""
    root = make_canary_dataset()
    try:
        ctx = EgressContext(guard=_guard(root), mode="strict",
                            user_input_policy="warn", audit=False)
        user_text = "我的資料放在 PTNSET7788_seed42_fold0"
        verdict, vios = check(ctx, f"討論記錄:\n{user_text}", [user_text])
        assert verdict == "warn" and vios, "使用者輸入應降級為警示並留下紀錄"

        # 同樣的字串若不是來自使用者 → 一律攔下
        verdict, _ = check(ctx, "系統: PTNSET7788_seed42_fold0", [])
        assert verdict == "blocked"

        # 政策設為 block 時, 連使用者輸入都擋
        ctx.user_input_policy = "block"
        verdict, _ = check(ctx, f"討論:\n{user_text}", [user_text])
        assert verdict == "blocked"
    finally:
        cleanup(root)


def test_mode_off_never_blocks():
    root = make_canary_dataset()
    try:
        ctx = EgressContext(guard=_guard(root), mode="off", audit=False)
        verdict, vios = check(ctx, f"路徑 {root}", [])
        assert verdict == "warn" and vios, "off 模式仍要留下紀錄, 只是不中止"
    finally:
        cleanup(root)


def test_no_sdk_call_outside_egress():
    """CI 檢查: agent/ 底下只有 privacy/egress.py 能出現 messages.create/stream。"""
    import glob
    import re

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    egress = os.path.join(repo, "agent", "privacy", "egress.py")
    pat = re.compile(r"\.messages\.(create|stream|parse|count_tokens)\s*\(")
    offenders = []
    for path in glob.glob(os.path.join(repo, "agent", "**", "*.py"),
                          recursive=True):
        if os.path.abspath(path) == os.path.abspath(egress):
            continue
        with open(path, encoding="utf8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if pat.search(line) and "sentinel" not in path:
                    offenders.append(f"{os.path.relpath(path, repo)}:{i}")
    assert not offenders, f"這些地方繞過了 egress: {offenders}"


def test_sentinel_blocks_direct_sdk_call():
    """未經 egress 的 SDK 呼叫必須拋 EgressViolation。"""
    import agent            # noqa: F401  (import 即安裝哨兵)
    import anthropic
    from agent.privacy import sentinel

    assert sentinel.is_installed()
    client = anthropic.Anthropic(api_key="sk-not-a-real-key")
    try:
        client.messages.create(model="claude-opus-4-8", max_tokens=1,
                               messages=[{"role": "user", "content": "hi"}])
    except EgressViolation:
        return
    raise AssertionError("直接呼叫 SDK 竟然沒有被哨兵擋下")
