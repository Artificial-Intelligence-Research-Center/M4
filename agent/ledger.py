"""Experiment Ledger — trial 設定與結果落地 (設計文件 §5.9).

每個 trial 一行 JSON (ledger.jsonl)。提供 append / history / best。
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .schemas import EvalConfig, TrialResult


class Ledger:
    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, "ledger.jsonl")

    def append(self, result: TrialResult) -> None:
        with open(self.path, "a", encoding="utf8") as f:
            f.write(result.model_dump_json() + "\n")

    def history(self) -> list[TrialResult]:
        if not os.path.isfile(self.path):
            return []
        out = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(TrialResult.model_validate_json(line))
        return out

    def best(self) -> Optional[TrialResult]:
        done = [t for t in self.history() if t.status == "done"]
        return max(done, key=lambda t: t.primary_score) if done else None

    def write_profile(self, profile_json: str) -> None:
        with open(os.path.join(self.run_dir, "dataset_profile.json"), "w",
                  encoding="utf8") as f:
            f.write(profile_json)
