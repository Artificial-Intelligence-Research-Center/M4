"""M4 自動微調 Agent (P0 骨架).

見設計文件: docs/auto_finetune_agent_design.md
本階段 (P0) 提供: 契約 schema + DatasetAnalyzer + EncoderRegistry +
HeuristicAdvisor + RecipeBuilder + Trainer(subprocess) + Evaluator + Ledger,
串起「單一 trial」並落地 TrialResult, 另附最小 Flask web 界面.
"""

__version__ = "0.0.1"
