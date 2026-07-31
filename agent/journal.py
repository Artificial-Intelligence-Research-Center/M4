"""Journal — AIDE 式解答樹 (參考 WecoAI/aideml 的 aide/journal.py).

每個 trial 是樹上一個 Node: draft 節點無 parent; improve / debug 由既有節點
長出 children; metric 回饋引導搜尋 (greedy 改良最佳節點、機率性除錯 buggy leaf)。
LoopController._search_policy 依此結構決定下一輪 (對應 aideml Agent.search_policy)。
"""
from __future__ import annotations

from typing import Literal, Optional

from .schemas import Recipe, TrialResult

Stage = Literal["draft", "improve", "debug", "resume", "ensemble"]


class Node:
    """解答樹節點: 一份 Recipe + 其執行結果 (TrialResult)。"""

    def __init__(self, recipe: Recipe, parent: Optional["Node"] = None,
                 stage: Stage = "draft"):
        self.recipe = recipe
        self.parent = parent
        self.children: list[Node] = []
        self.stage: Stage = stage
        self.trial: Optional[TrialResult] = None   # 執行後回填
        self.debug_exhausted = False               # 無法再產生除錯配方 → policy 不再選它
        self.improve_exhausted = False             # 決策層判定此分支再變異無益 → policy 不再選它
        if parent is not None:
            parent.children.append(self)

    # ---- 狀態 ----------------------------------------------------------
    @property
    def id(self) -> Optional[str]:
        return self.trial.trial_id if self.trial else None

    @property
    def evaluated(self) -> bool:
        return self.trial is not None

    @property
    def is_buggy(self) -> bool:
        """執行失敗 (訓練炸掉 / 評估不出分數) — 對應 aideml 的 is_buggy。
        pending (dry_run) 不算 buggy 也不算 good, 不會被選去除錯/改良。"""
        return self.evaluated and self.trial.status == "failed"

    @property
    def metric(self) -> Optional[float]:
        return (self.trial.primary_score
                if self.evaluated and self.trial.status == "done" else None)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def debug_depth(self) -> int:
        """連續除錯鏈長度 (同 aideml): 非 debug 節點為 0。"""
        n, d = self, 0
        while n.stage == "debug" and n.parent is not None:
            d += 1
            n = n.parent
        return d

    @property
    def resume_depth(self) -> int:
        """連續繼續訓練鏈長度: 非 resume 節點為 0 (max_resumes 護欄用)。"""
        n, d = self, 0
        while n.stage == "resume" and n.parent is not None:
            d += 1
            n = n.parent
        return d


class Journal:
    """整個 run 的全域解答樹 (節點依執行順序排列; draft 可屬不同 encoder)。"""

    def __init__(self):
        self.nodes: list[Node] = []

    def append(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    # ---- aideml Journal 的等價 helper -----------------------------------
    @property
    def draft_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.evaluated and n.trial.status == "done"]

    @property
    def improvable_nodes(self) -> list[Node]:
        """還能當 improve 起點的成功節點 (排除決策層已放棄的分支)。"""
        return [n for n in self.good_nodes if not n.improve_exhausted]

    def get_best_node(self) -> Optional[Node]:
        good = self.good_nodes
        return max(good, key=lambda n: n.metric) if good else None

    # ---- 重建 (繼續實驗用) ----------------------------------------------
    @staticmethod
    def rebuild(trials: list[TrialResult]) -> "Journal":
        """從 ledger 的 TrialResult 重建解答樹 (resume 用)。

        每個 trial 的 recipe.provenance.search 記錄了 {stage, parent(=trial_id)},
        依執行順序重建節點與父子連結; 缺 search 資訊的舊 trial 一律視為 draft。
        (debug_exhausted / improve_exhausted 不落地, 重建後遺失 — 影響僅是
        Advisor 可能再試一次除錯或再看一次已放棄的分支。)"""
        j = Journal()
        by_id: dict[str, Node] = {}
        for t in trials:
            s = (t.recipe.provenance or {}).get("search") or {}
            stage = s.get("stage", "draft")
            if stage not in ("draft", "improve", "debug", "resume", "ensemble"):
                stage = "draft"
            parent = by_id.get(s.get("parent")) if s.get("parent") else None
            node = Node(t.recipe, parent=parent, stage=stage)
            node.trial = t
            j.append(node)
            if t.trial_id:
                by_id[t.trial_id] = node
        return j

    # ---- 落地 (供 report / web / 事後分析) ------------------------------
    def to_dict(self) -> dict:
        idx = {id(n): i for i, n in enumerate(self.nodes)}
        best = self.get_best_node()
        return {
            "best": idx.get(id(best)) if best else None,
            "nodes": [{
                "index": i,
                "trial_id": n.id,
                "encoder": n.recipe.encoder.model_key,
                "stage": n.stage,
                "parent": idx.get(id(n.parent)) if n.parent else None,
                "children": [idx[id(c)] for c in n.children],
                "metric": n.metric,
                "is_buggy": n.is_buggy,
                # 決策層判定這條分支再變異無益 → 之後不再從它長 child
                "pruned": n.improve_exhausted,
                "debug_depth": n.debug_depth,
                "resume_depth": n.resume_depth,
                "select_prob": n.recipe.provenance.get("search", {}).get("select_prob"),
                # 這個節點是 policy 抽中的, 還是決策層改選的
                "overridden": n.recipe.provenance.get("search", {}).get("overridden", False),
                "mutation": n.recipe.provenance.get("mutation"),
            } for i, n in enumerate(self.nodes)],
        }
