"""資料圍欄的例外型別 (docs/data_firewall_design.md §7)。"""
from __future__ import annotations


class PrivacyError(RuntimeError):
    """資料圍欄相關錯誤的基底。"""


class EgressViolation(PrivacyError):
    """偵測到（或即將發生）未經圍欄的資料外送 — 一律中止, 不靜默改寫後放行。

    violations: [{rule, sample, where}] — 命中的規則與樣本 (sample 已截斷)。
    """

    def __init__(self, message: str, violations: list[dict] | None = None,
                 label: str = ""):
        super().__init__(message)
        self.violations = violations or []
        self.label = label


class PrivacyConfigError(PrivacyError):
    """隱私設定本身不合法 (例: strict 模式卻要求 raw log 回饋)。"""
