"""不需 pytest 的最小測試執行器 (本 repo 目前沒有 pytest)。

    python -m tests.privacy.run_all

測試函式同時相容 pytest — 之後裝了 pytest 直接 `pytest tests/privacy` 也能跑。
"""
from __future__ import annotations

import importlib
import sys
import traceback

MODULES = [
    "tests.privacy.test_contracts",
    "tests.privacy.test_guard",
    "tests.privacy.test_error_extract",
    "tests.privacy.test_canary_leak",
]


def main() -> int:
    passed, failed = 0, []
    for name in MODULES:
        mod = importlib.import_module(name)
        for fn_name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, fn_name)
            if not callable(fn):
                continue
            label = f"{name.split('.')[-1]}.{fn_name}"
            try:
                fn()
            except Exception as e:                      # noqa: BLE001
                failed.append((label, e, traceback.format_exc()))
                print(f"✗ {label}: {type(e).__name__}: {e}")
            else:
                passed += 1
                print(f"✓ {label}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for label, _e, tb in failed:
        print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
