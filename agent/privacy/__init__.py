"""資料圍欄 (Data Firewall) — 確保 LLM 永遠看不到使用者的輸入資料。

見 docs/data_firewall_design.md。核心不變式:

  LLM 只能看到 (a) `privacy.facts` 定義的白名單事實, 由我們事先寫好的程式產生;
  或 (b) 使用者親自輸入的回答。除此之外沒有第三條路。

  所有送往 Claude API 的 payload 只能經由 `privacy.egress`; `privacy.sentinel`
  在執行期檢查呼叫堆疊, 繞道者一律拋 EgressViolation。
"""
from __future__ import annotations

from . import alias, egress, facts, redact, sentinel
from .context import (PrivacyContext, load_user_facts, save_user_facts)
from .egress import EgressContext
from .errors import EgressViolation, PrivacyConfigError, PrivacyError
from .facts import (AnalysisFacts, DatasetFacts, ErrorFacts, TrialFacts,
                    UserAnswer, UserFacts, UserQuestion)
from .redact import Guard

__all__ = [
    "PrivacyContext", "EgressContext", "Guard",
    "EgressViolation", "PrivacyError", "PrivacyConfigError",
    "DatasetFacts", "TrialFacts", "ErrorFacts", "AnalysisFacts",
    "UserQuestion", "UserAnswer", "UserFacts",
    "load_user_facts", "save_user_facts",
    "alias", "egress", "facts", "redact", "sentinel",
]
