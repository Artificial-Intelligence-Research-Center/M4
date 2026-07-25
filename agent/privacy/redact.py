"""消毒層 — DatasetProfile / TrialResult / 討論記錄 → LLM-safe facts, 以及出口掃描器
(docs/data_firewall_design.md §4 / §7.2)。

兩件事:
  1. **轉換**: 把資料平面的物件轉成 `facts` 裡的白名單型別 (允許清單, 不是過濾)。
  2. **掃描**: `Guard` 對最終 payload 做精確比對, 命中即回報違規 (由 egress 決定中止)。

掃描是縱深防禦, 不是主要防線 — 主要防線是「只有 facts 進得了 prompt」。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .alias import AliasMap, name_tokens
from .facts import (DatasetFacts, ImageSizeFacts, ProvenanceFacts, TrialFacts,
                    UserFacts)

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# 巢狀結構中一律丟棄的 key (值為路徑或本機位置)
_DROP_KEYS = {
    "workspace", "code_dir", "path", "root", "data_path", "log_path",
    "ckpt_path", "resume_from", "dest", "rel_path", "file", "filename",
}


# ---------------------------------------------------------------------------
# 出口掃描器
# ---------------------------------------------------------------------------
@dataclass
class Violation:
    rule: str
    sample: str
    start: int = 0
    end: int = 0

    def as_dict(self) -> dict:
        return {"rule": self.rule, "sample": self.sample,
                "start": self.start, "end": self.end}


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("image_filename",
     re.compile(r"[\w\-./\\]{1,160}\.(?:jpe?g|png|tiff?|bmp|webp)\b", re.I)),
    ("base64_image",
     re.compile(r"(?:/9j/|iVBORw0KGgo|R0lGOD)[A-Za-z0-9+/=]{16,}")),
    # 張量傾印: 單一連續數值序列過長 (逐 epoch 曲線最多數百點, 遠低於此門檻)
    ("numeric_dump",
     re.compile(r"(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){512,}")),
    # 台灣身分證字號
    ("national_id", re.compile(r"\b[A-Z][12]\d{8}\b")),
]


def _sample_stems(root: str, limit: int = 300) -> set[str]:
    """抽樣資料集內的影像檔名主幹 (供精確比對)。有上限, 不掃全部檔案。"""
    out: set[str] = set()
    if not os.path.isdir(root):
        return out
    try:
        splits = [d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))]
    except OSError:
        return out
    for split in splits[:4]:
        sdir = os.path.join(root, split)
        try:
            classes = [c for c in os.listdir(sdir)
                       if os.path.isdir(os.path.join(sdir, c))]
        except OSError:
            continue
        for cls in classes[:12]:
            cdir = os.path.join(sdir, cls)
            try:
                with os.scandir(cdir) as it:
                    for i, e in enumerate(it):
                        if i >= 40 or len(out) >= limit:
                            break
                        if e.name.lower().endswith(_IMG_EXT):
                            stem = os.path.splitext(e.name)[0]
                            if len(stem) >= 4 and not stem.isdigit():
                                out.add(stem)
            except OSError:
                continue
            if len(out) >= limit:
                return out
    return out


class Guard:
    """對送出前的 payload 做精確比對。

    literals: 本 run 的真實識別字串 (資料集路徑/名稱/類別名/檔名主幹) → 規則名。
    以「本 run 的真實字串集合」比對而非泛用 PII 正則, 誤判率低且不可能漏掉本 run 的
    識別字串。
    """

    def __init__(self, literals: dict[str, str] | None = None,
                 exempt: Optional[set[str]] = None):
        # {小寫字串: rule}
        self.literals: dict[str, str] = dict(literals or {})
        self.exempt: set[str] = set(exempt or ())

    # ---- 建立 ---------------------------------------------------------
    @classmethod
    def build(cls, *, data_root: Optional[str] = None,
              dataset_root: Optional[str] = None,
              class_names: Iterable[str] = (),
              revealed: Iterable[str] = (),
              exempt_text: str = "",
              scan_filenames: bool = True) -> "Guard":
        exempt = {t.lower() for t in re.split(r"[^\w]+", exempt_text or "")
                  if t and len(t) >= 3}
        lits: dict[str, str] = {}

        def add(s: str, rule: str) -> None:
            s = (s or "").strip()
            if len(s) < 4 or s.lower() in exempt:
                return
            lits.setdefault(s.lower(), rule)

        if dataset_root:
            real = os.path.realpath(dataset_root)
            add(real, "dataset_path")
            add(os.path.basename(real), "dataset_name")
            # data_root 與資料集之間的中介目錄 (例: 5_fold_PAPILA)
            if data_root:
                dr = os.path.realpath(data_root)
                add(dr, "data_root_path")
                rel = os.path.relpath(real, dr)
                if not rel.startswith(".."):
                    for part in rel.split(os.sep):
                        add(part, "dataset_name")
                        for tok in name_tokens(part):
                            add(tok, "dataset_token")
            for tok in name_tokens(os.path.basename(real)):
                add(tok, "dataset_token")

        revealed_set = {r for r in revealed}
        for name in class_names:
            if name in revealed_set:
                continue          # 使用者已明確授權外送
            add(name, "class_name")
            for tok in name_tokens(name):
                add(tok, "class_token")

        if scan_filenames and dataset_root:
            for stem in _sample_stems(dataset_root):
                add(stem, "image_filename_stem")

        return cls(lits, exempt)

    # ---- 掃描 ---------------------------------------------------------
    def scan(self, text: str) -> list[Violation]:
        if not text:
            return []
        out: list[Violation] = []
        low = text.lower()
        for lit, rule in self.literals.items():
            idx = low.find(lit)
            if idx >= 0:
                out.append(Violation(rule, _snip(text, idx, idx + len(lit)),
                                     idx, idx + len(lit)))
        for rule, pat in _PATTERNS:
            m = pat.search(text)
            if m:
                out.append(Violation(rule, _snip(text, m.start(), m.end()),
                                     m.start(), m.end()))
        return out


def _snip(text: str, start: int, end: int, pad: int = 24) -> str:
    """違規樣本 (截斷) — 稽核檔本來就會存全文, 這裡只求可讀。"""
    s = text[max(0, start - pad): min(len(text), end + pad)].replace("\n", " ")
    return (s[:120] + "…") if len(s) > 120 else s


# ---------------------------------------------------------------------------
# 轉換: DatasetProfile → DatasetFacts
# ---------------------------------------------------------------------------
def to_facts(profile, alias: AliasMap, user_facts: Optional[UserFacts] = None,
             n_folds: Optional[int] = None) -> DatasetFacts:
    """把 DatasetProfile 轉成唯一允許進 prompt 的 DatasetFacts。

    被刻意丟棄: `root` (絕對路徑)、`class_names` (真實類別名)、
    `modality_hint` (由路徑關鍵字猜測 — 改由使用者在 UI 選擇)。
    """
    uf = user_facts or UserFacts()
    names = list(profile.class_names or [])
    stats = profile.image_size_stats or {}
    w, h = stats.get("width") or {}, stats.get("height") or {}
    return DatasetFacts(
        dataset_ref=alias.dataset_ref,
        task_type=profile.task_type,
        num_classes=profile.num_classes,
        class_labels=alias.labels(names),
        class_counts=[int(profile.class_counts.get(n, 0)) for n in names],
        class_ordinal=uf.class_ordinal,
        n_train=profile.n_train, n_val=profile.n_val, n_test=profile.n_test,
        has_kfold=profile.has_kfold, n_folds=n_folds,
        imbalance_ratio=profile.imbalance_ratio,
        is_grayscale=profile.is_grayscale,
        image_size=ImageSizeFacts(
            n_sampled=int(stats.get("n_sampled") or 0),
            width_min=w.get("min"), width_median=w.get("median"),
            width_max=w.get("max"),
            height_min=h.get("min"), height_median=h.get("median"),
            height_max=h.get("max"),
        ),
        modality=uf.modality,
        anatomy=uf.anatomy,
    )


# ---------------------------------------------------------------------------
# 轉換: TrialResult → TrialFacts
# ---------------------------------------------------------------------------
def scrub(obj, alias: AliasMap):
    """遞迴: 字串做假名替換, 丟棄路徑類 key。"""
    if isinstance(obj, str):
        return alias.substitute(obj)
    if isinstance(obj, dict):
        return {k: scrub(v, alias) for k, v in obj.items()
                if k not in _DROP_KEYS}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, alias) for v in obj]
    return obj


def prov_facts(prov: dict, alias: AliasMap) -> ProvenanceFacts:
    """原始 provenance (自由 dict) → 白名單視圖。未列舉的 key 一律不外送。"""
    p = scrub(dict(prov or {}), alias)
    mf = p.get("mutated_from")
    if isinstance(mf, dict):
        mf = {k: mf.get(k) for k in ("trial", "mutation") if k in mf}
    else:
        mf = {}
    return ProvenanceFacts(
        template=_s(p.get("template")), preset=_s(p.get("preset")),
        advisor=_s(p.get("advisor")), mutation=_s(p.get("mutation")),
        reason=str(p.get("reason") or ""),
        hparam_reason=str(p.get("hparam_reason") or ""),
        hparam_changes=p.get("hparam_changes") or {},
        mutated_from=mf,
        gpu_opt=p.get("gpu_opt") or {},
        search=p.get("search") or {},
        cache_resized=p.get("cache_resized"),
        code_edits=p.get("code_edits") or {},
    )


def _s(v) -> Optional[str]:
    return None if v is None else str(v)


def trial_facts(trial, alias: AliasMap) -> TrialFacts:
    cv = trial.epoch_curve or {}
    return TrialFacts(
        trial_id=alias.substitute(trial.trial_id or ""),
        status=trial.status,
        primary_score=trial.primary_score,
        metrics={k: float(v) for k, v in (trial.metrics or {}).items()},
        epochs=trial.recipe.hparams.epochs,
        provenance=prov_facts(trial.recipe.provenance, alias),
        train_loss_curve=list(cv.get("train_loss") or []),
        val_loss_curve=list(cv.get("val_loss") or []),
        val_score_curve=list(cv.get("val_score") or []),
        gpu_stats={k: v for k, v in (trial.gpu_stats or {}).items()
                   if k != "_from"},
    )


# ---------------------------------------------------------------------------
# 討論記錄
# ---------------------------------------------------------------------------
def discussion_facts(entries: list[dict], alias: AliasMap,
                     limit: int = 20) -> list[dict]:
    """討論記錄 → 只保留 role/text, 並做假名替換。

    system 訊息含資料集名稱 (例「實驗開始：資料 XXX」), 使用者訊息可能含任何東西 —
    前者由假名替換處理, 後者由出口掃描以 `guard_on_user_input` 政策處理。
    """
    out = []
    for e in (entries or [])[-limit:]:
        if e.get("kind") == "question":
            continue                      # 問卷卡本身不回送 (答案另由 UserFacts 帶入)
        out.append({"role": e.get("role"),
                    "text": alias.substitute(str(e.get("text") or ""))})
    return out


def user_segments(entries: list[dict], alias: AliasMap,
                  limit: int = 20) -> list[str]:
    """討論記錄中由『使用者親自輸入』的片段 — 出口掃描對這些命中只警示不中止。

    回傳的是**替換後**的文字, 才能與最終 payload 中的內容逐字對應。
    """
    return [alias.substitute(str(e.get("text") or ""))
            for e in (entries or [])[-limit:] if e.get("role") == "user"]
