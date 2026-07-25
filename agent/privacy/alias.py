"""假名 (pseudonym) 產生與對照表 (docs/data_firewall_design.md §4.1 / §4.2).

資料集路徑與類別名稱不進 prompt; 改以穩定假名 `DS-xxxxxx` / `C0..Cn` 表示。
對照表只存在本機 (`runs/<run>/alias_map.json`), 永不外送 — 供 UI 把 LLM 回覆中的
假名還原成使用者看得懂的名稱。

salt 為 per-machine 固定 (預設 `~/.medclaw/privacy_salt`), 讓「同一個資料集」在不同
run 之間得到相同假名 (LLM 可跨 run 關聯), 但外人無法從假名反推來源。
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
from hashlib import sha256

_DEFAULT_SALT_FILE = os.path.join("~", ".medclaw", "privacy_salt")

# `PAPILA_seed42_fold0` → stem=PAPILA_seed42, n=0 (同一份資料的各 fold 共用前綴)
_FOLD_RE = re.compile(r"^(?P<stem>.+?)[_-]?fold[_-]?(?P<n>\d+)$", re.I)

# 產生 token 時要略過的通用詞 (不具識別性, 且常出現在正常 prompt 中)
_GENERIC_TOKENS = {
    "train", "val", "valid", "test", "data", "dataset", "datasets", "image",
    "images", "img", "fold", "folds", "seed", "class", "classes", "label",
    "labels", "split", "splits", "sample", "samples", "run", "runs", "raw",
    "processed", "final", "full", "part", "batch", "set", "sets", "new", "old",
}


def salt_path(configured: str | None = None) -> str:
    return os.path.abspath(os.path.expanduser(configured or _DEFAULT_SALT_FILE))


def load_salt(configured: str | None = None) -> bytes:
    """讀取 (必要時建立) per-machine salt。檔案權限設為 0600, 不進版控。"""
    path = salt_path(configured)
    if os.path.isfile(path):
        with open(path, encoding="utf8") as f:
            raw = f.read().strip()
        if raw:
            return bytes.fromhex(raw) if _is_hex(raw) else raw.encode("utf8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    value = secrets.token_hex(32)
    # 先以 0600 建檔再寫入, 避免短暫的可讀視窗
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        f.write(value)
    return bytes.fromhex(value)


def _is_hex(s: str) -> bool:
    try:
        bytes.fromhex(s)
        return True
    except ValueError:
        return False


def digest(salt: bytes, text: str, n: int = 6) -> str:
    return hmac.new(salt, text.encode("utf8"), sha256).hexdigest()[:n]


def dataset_ref(root: str, salt: bytes) -> str:
    """資料集假名。同一份資料的各 fold 共用前綴: `DS-a1b2c3#f0`, `DS-a1b2c3#f1`。"""
    if not root:
        return "DS-unknown"
    real = os.path.realpath(root)
    base = os.path.basename(real)
    m = _FOLD_RE.match(base)
    if m:
        stem = os.path.join(os.path.dirname(real), m.group("stem"))
        return f"DS-{digest(salt, stem)}#f{int(m.group('n'))}"
    return f"DS-{digest(salt, real)}"


def name_tokens(text: str, min_len: int = 4) -> set[str]:
    """把目錄/類別名拆成具識別性的 token (供出口掃描比對)。

    純數字、通用詞 (train/val/fold…)、過短的片段一律排除, 以免誤攔正常 prompt。
    """
    out = set()
    for tok in re.split(r"[^\w]+", text, flags=re.UNICODE):
        if not tok or tok.isdigit() or len(tok) < min_len:
            continue
        if tok.lower() in _GENERIC_TOKENS:
            continue
        out.add(tok)
    return out


class AliasMap:
    """一個 run 的假名對照表。只存本機, 永不外送。"""

    FILENAME = "alias_map.json"

    def __init__(self, dataset_ref: str, dataset_root: str,
                 class_alias: dict[str, str] | None = None,
                 revealed: list[str] | None = None):
        self.dataset_ref = dataset_ref
        self.dataset_root = dataset_root
        # 真實類別名 → 假名 (依 profile.class_names 的排序給 C0..Cn)
        self.class_alias: dict[str, str] = dict(class_alias or {})
        # 使用者明確授權可外送的真實類別名 (privacy.class_names=user_approved)
        self.revealed: list[str] = list(revealed or [])

    # ---- 建立 ---------------------------------------------------------
    @classmethod
    def build(cls, root: str, class_names: list[str], salt: bytes,
              revealed: list[str] | None = None) -> "AliasMap":
        revealed = list(revealed or [])
        alias = {}
        for i, name in enumerate(class_names):
            alias[name] = name if name in revealed else f"C{i}"
        real = os.path.realpath(root) if root else ""
        return cls(dataset_ref(root, salt), real, alias, revealed)

    # ---- 查詢 ---------------------------------------------------------
    def labels(self, class_names: list[str]) -> list[str]:
        return [self.class_alias.get(n, f"C{i}") for i, n in enumerate(class_names)]

    def real_name(self, label: str) -> str | None:
        for real, al in self.class_alias.items():
            if al == label:
                return real
        return None

    def hidden_class_names(self) -> list[str]:
        """尚未被授權外送的真實類別名 (出口掃描的比對對象)。"""
        return [n for n, al in self.class_alias.items() if al != n]

    # ---- 替換 ---------------------------------------------------------
    def secrets(self) -> list[tuple[str, str]]:
        """(真實字串, 假名) 清單, 長者優先 — 供 substitute() 與掃描器使用。"""
        pairs: list[tuple[str, str]] = []
        if self.dataset_root:
            pairs.append((self.dataset_root, self.dataset_ref))
            base = os.path.basename(self.dataset_root)
            if base:
                pairs.append((base, self.dataset_ref))
        for real in self.hidden_class_names():
            pairs.append((real, self.class_alias[real]))
        return sorted([p for p in pairs if p[0]],
                      key=lambda p: len(p[0]), reverse=True)

    def substitute(self, text: str) -> str:
        """把已知的真實字串換成假名 (用於 trial_id、provenance、討論記錄等衍生欄位)。

        這是假名機制的一部分, 不是「過濾後放行」— 出口掃描仍會再檢查一次。
        """
        if not text:
            return text
        for real, al in self.secrets():
            if real and real in text:
                text = text.replace(real, al)
        return text

    # ---- 落地 ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"dataset_ref": self.dataset_ref, "dataset_root": self.dataset_root,
                "class_alias": self.class_alias, "revealed": self.revealed}

    def save(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, self.FILENAME)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, run_dir: str) -> "AliasMap | None":
        try:
            with open(os.path.join(run_dir, cls.FILENAME), encoding="utf8") as f:
                d = json.load(f)
        except Exception:
            return None
        return cls(d.get("dataset_ref", ""), d.get("dataset_root", ""),
                   d.get("class_alias"), d.get("revealed"))
