"""M4 自動微調 Agent (MedClaw).

見設計文件: docs/auto_finetune_agent_design.md
資料圍欄設計:   docs/data_firewall_design.md

契約 schema + DatasetAnalyzer + EncoderRegistry + Advisor (heuristic/llm/skill) +
RecipeBuilder + Trainer(subprocess) + Evaluator + Ledger + LoopController,
另附 Flask web 界面。

**資料圍欄**: import 本套件即安裝 `privacy.sentinel` — 之後任何未經
`agent.privacy.egress` 的 Claude API 呼叫都會拋 EgressViolation。這是圍欄能成立的
前提, 不要移除。
"""

from .privacy import sentinel as _sentinel

_sentinel.install()

__version__ = "0.1.0"
