"""QA agent — 使用者提問/給方向時, 立刻以既有實驗資料回答 (獨立於決策迴圈).

流程 (web /discuss 觸發, 背景 thread):
  使用者訊息 append 進 conversation 後 → answer(run_dir, question) 讀取 run 目錄的
  dataset_profile / ledger / search_tree / 近期討論 → 開一個獨立 LLM agent 回答 →
  「答案 + 結論」append 回 conversation (role=llm, kind=qa)。

結論以【QA 結論】標記; 主 agent (LLMAdvisor.review_and_decide) 讀討論記錄時,
prompt 已註明此標記 = 問答 agent 依實驗資料得出的結論, 下一輪決策優先納入。
QA 失敗不影響原有流程 — 使用者訊息本來就會在下一輪被決策層讀到。
"""
from __future__ import annotations

import json
import os
import threading
import time

from . import conversation
from .advisor import _STOP_WORDS
from .config import AgentConfig
from .ledger import Ledger
from .schemas import DatasetProfile

# 每個 run 一把鎖: 連續提問時逐一回答, 避免同 run 並發打 API
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _run_lock(run_dir: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(os.path.abspath(run_dir), threading.Lock())


def _load_profile(run_dir: str) -> DatasetProfile | None:
    try:
        with open(os.path.join(run_dir, "dataset_profile.json"), encoding="utf8") as f:
            return DatasetProfile.model_validate_json(f.read())
    except Exception:
        return None


def _load_tree(run_dir: str) -> dict | None:
    try:
        with open(os.path.join(run_dir, "search_tree.json"), encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


# ---- streaming: 生成中的部份文字落地, 供 web 即時顯示打字效果 -------------
def _stream_path(run_dir: str) -> str:
    return os.path.join(run_dir, "qa_stream.json")


def _write_stream(run_dir: str, question: str, text: str) -> None:
    try:
        with open(_stream_path(run_dir), "w", encoding="utf8") as f:
            json.dump({"ts": time.time(), "question": question, "text": text},
                      f, ensure_ascii=False)
    except Exception:
        pass


def _clear_stream(run_dir: str) -> None:
    try:
        os.remove(_stream_path(run_dir))
    except OSError:
        pass


def answer(run_dir: str, question: str) -> dict:
    """以既有實驗資料回答一則使用者訊息; 答案+結論 append 回 conversation。

    回傳 {"ok": bool, "why"/"answer"/"conclusion": ...}。僅 advisor.type=llm 時作用;
    純中斷指令 (停止/stop…) 不回答 — 交給原本的中斷機制處理。
    """
    q = (question or "").strip()
    if not q:
        return {"ok": False, "why": "空訊息"}
    if any(w in q.lower() for w in _STOP_WORDS):
        return {"ok": False, "why": "中斷指令, 交給停止機制"}
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        return {"ok": False, "why": "無 config.yaml"}
    try:
        cfg = AgentConfig.load(cfg_path)
    except Exception as e:
        return {"ok": False, "why": f"config 載入失敗: {e}"}
    if cfg.advisor.type != "llm":
        return {"ok": False, "why": f"advisor={cfg.advisor.type} (QA 需 llm)"}

    with _run_lock(run_dir):
        from .llm_advisor import LLMAdvisor  # 延後匯入 (無 anthropic 時不炸整個 web)
        adv = LLMAdvisor(model=cfg.advisor.model, guidance=cfg.advisor.guidance)
        adv.log_path = os.path.join(run_dir, "llm_calls.jsonl")  # QA 呼叫一併落地

        profile = _load_profile(run_dir)
        history = [t for t in Ledger(run_dir).history()
                   if not (t.trial_id or "").endswith("_bestfold")]
        tree = _load_tree(run_dir)
        talk = conversation.read(run_dir)

        # streaming: 先寫空狀態 (UI 顯示「思考中…」), 之後每 ~0.3s 落地一次部份文字
        _write_stream(run_dir, q, "")
        _last = {"t": 0.0}

        def _on_delta(partial: str) -> None:
            now = time.monotonic()
            if now - _last["t"] >= 0.3:
                _last["t"] = now
                _write_stream(run_dir, q, partial)

        try:
            ans, concl = adv.answer_question(profile, history, tree, talk, q,
                                             on_delta=_on_delta)
        finally:
            _clear_stream(run_dir)   # 先清串流檔再寫正式訊息, 避免前端同時顯示兩份
        text = ans.strip()
        if concl.strip():
            text += f"\n\n【QA 結論（下一輪決策將優先納入）】{concl.strip()}"
        conversation.append(run_dir, "llm", text, kind="qa")
        return {"ok": True, "answer": ans, "conclusion": concl}


def answer_async(run_dir: str, question: str) -> None:
    """背景回答 (web /discuss 用): 失敗時寫一則 system 訊息, 不打斷任何流程。"""
    def _worker():
        try:
            answer(run_dir, question)
        except Exception as e:
            try:
                conversation.append(
                    run_dir, "system",
                    f"問答 agent 失敗：{e}（你的訊息仍會在下一輪決策時被納入）",
                    kind="status")
            except Exception:
                pass
    threading.Thread(target=_worker, daemon=True).start()
