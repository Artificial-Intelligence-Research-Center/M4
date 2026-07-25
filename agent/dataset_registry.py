"""DatasetRegistry — 掃描 data/ 底下可用的 ImageFolder 資料集, 供 web UI 選擇.

純程式, 不呼叫 LLM。與 dataset_analyzer 的分工:
  - dataset_analyzer.analyze(root): 針對「單一已選定」資料集做完整 profile (含抽樣量測影像尺寸)
  - dataset_registry.scan(): 針對「整個 data/ 目錄」列出所有資料集 + 輕量摘要 (只數檔案, 不開圖)

資料結構慣例:
    <data_root>/[<collection>/]<dataset>/{train,val,test}/<class>/*.{jpg,png,...}
"""
from __future__ import annotations

import os
import re
import time

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_SPLITS = ("train", "val", "test")
_MAX_DEPTH = 3          # 從 data_root 往下找幾層 (5_fold_X/X_fold0 = 2 層)
_TTL_S = 300.0          # 掃描結果快取秒數 (可用 refresh=True 強制重掃)

_FOLD_RE = re.compile(r"fold[_-]?(\d+)", re.I)

# {data_root: (scanned_at, entries)}
_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _class_counts(split_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        names = sorted(os.listdir(split_dir))
    except OSError:
        return counts
    for cls in names:
        cdir = os.path.join(split_dir, cls)
        if not os.path.isdir(cdir):
            continue
        try:
            counts[cls] = sum(1 for f in os.listdir(cdir)
                              if f.lower().endswith(_IMG_EXT))
        except OSError:
            counts[cls] = 0
    return counts


def _profile(path: str, data_root: str) -> dict | None:
    """輕量摘要; 不是可訓練的 ImageFolder (缺 train/<class>/) 則回 None。"""
    counts: dict[str, dict[str, int]] = {}
    for s in _SPLITS:
        c = _class_counts(os.path.join(path, s))
        if c:
            counts[s] = c
    if "train" not in counts:
        return None

    classes = sorted(counts["train"])
    totals = {s: sum(c.values()) for s, c in counts.items()}
    nonzero = [v for v in counts["train"].values() if v > 0] or [1]
    rel = os.path.relpath(path, data_root)
    parts = rel.split(os.sep)
    fold = int(m.group(1)) if (m := _FOLD_RE.search(os.path.basename(path))) else None

    # 便宜的「能不能直接訓練」判斷 (只看已數好的 counts, 不開圖);
    # 完整檢查請用 validate()。
    issues: list[str] = []
    if (miss := [s for s in _SPLITS if s not in counts]):
        issues.append("缺少 " + "/".join(miss))
    sets = [set(c) for c in counts.values()]
    if any(s != sets[0] for s in sets):
        issues.append("各 split 類別不一致")
    if len(classes) < 2:
        issues.append("類別數 < 2")
    if any(n == 0 for cc in counts.values() for n in cc.values()):
        issues.append("有空的類別目錄")

    return {
        "ready": not issues,
        "issues": issues,
        "name": os.path.basename(path),
        "rel_path": rel.replace(os.sep, "/"),
        "path": path,
        # 上層目錄名當分組 (例: 5_fold_PAPILA); 直接放在 data/ 下的則不分組
        "group": parts[0] if len(parts) > 1 else "",
        "fold": fold,
        "classes": classes,
        "num_classes": len(classes),
        "counts": counts,
        "n_train": totals.get("train", 0),
        "n_val": totals.get("val", 0),
        "n_test": totals.get("test", 0),
        "total": sum(totals.values()),
        "imbalance_ratio": round(max(nonzero) / min(nonzero), 2),
    }


def _walk(cur: str, data_root: str, depth: int, out: list[dict]) -> None:
    try:
        names = sorted(os.listdir(cur))
    except OSError:
        return
    for n in names:
        # 略過隱藏 / 底線開頭的內部目錄 (例: _resize_cache) 與 split 目錄本身
        if n.startswith((".", "_")) or n in _SPLITS:
            continue
        d = os.path.join(cur, n)
        if not os.path.isdir(d):
            continue
        if (p := _profile(d, data_root)) is not None:
            out.append(p)          # 本身就是資料集 → 不再往下找
        elif depth < _MAX_DEPTH:
            _walk(d, data_root, depth + 1, out)


def scan(data_root: str, refresh: bool = False) -> list[dict]:
    """列出 data_root 底下所有 ImageFolder 資料集 (依 group / fold / 名稱排序)。"""
    data_root = os.path.abspath(data_root)
    hit = _CACHE.get(data_root)
    if hit and not refresh and (time.time() - hit[0]) < _TTL_S:
        return hit[1]

    out: list[dict] = []
    if os.path.isdir(data_root):
        _walk(data_root, data_root, 1, out)
    out.sort(key=lambda e: (e["group"], e["fold"] if e["fold"] is not None else -1,
                            e["name"]))
    _CACHE[data_root] = (time.time(), out)
    return out


def groups(data_root: str, refresh: bool = False) -> list[dict]:
    """同 scan(), 但依 group 收攏成 [{name, datasets:[...]}, ...] 供下拉選單分組。"""
    out: list[dict] = []
    for e in scan(data_root, refresh=refresh):
        if not out or out[-1]["name"] != e["group"]:
            out.append({"name": e["group"], "datasets": []})
        out[-1]["datasets"].append(e)
    return out


def find(data_root: str, path: str) -> dict | None:
    """以絕對路徑或相對 data_root 的路徑找出一筆資料集。"""
    target = os.path.abspath(
        path if os.path.isabs(path) else os.path.join(data_root, path))
    return next((e for e in scan(data_root) if e["path"] == target), None)


def invalidate(data_root: str | None = None) -> None:
    """清掉掃描快取 (上傳/刪除資料集後呼叫)。"""
    if data_root is None:
        _CACHE.clear()
    else:
        _CACHE.pop(os.path.abspath(data_root), None)


# --------------------------------------------------------------------------
# 格式檢查 — 「MedClaw 是否吃得下這個資料集」
# 依據 main_finetune.py: train/val/test 三個 split 都會 build_dataset(),
# 每個 split 各自用 torchvision ImageFolder (label index = 該 split 內
# 排序後的類別資料夾名) → 三個 split 的類別資料夾必須完全一致, 否則標籤錯位。
# --------------------------------------------------------------------------

_SAMPLE_N = 30          # 抽樣開圖檢查的張數
_MIN_SIDE_WARN = 224    # 影像短邊低於此值 (預設 input_size) 只能放大, 會掉細節


def _sample_paths(counts: dict[str, dict[str, int]], root: str,
                  k: int = _SAMPLE_N) -> list[str]:
    """每個 split × 類別各取幾張 (取目錄中最前面的檔案, 不做隨機以求穩定)。"""
    cells = [(s, c) for s, cc in counts.items() for c in cc]
    if not cells:
        return []
    per = max(1, k // len(cells))
    out: list[str] = []
    for split, cls in cells:
        cdir = os.path.join(root, split, cls)
        try:
            names = sorted(f for f in os.listdir(cdir)
                           if f.lower().endswith(_IMG_EXT))
        except OSError:
            continue
        out += [os.path.join(cdir, f) for f in names[:per]]
    return out[:k]


def validate(path: str) -> dict:
    """檢查資料集能否直接餵給 MedClaw 訓練。

    回傳 {ok, errors, warnings, summary}:
      - errors   非空 → 現在無法訓練, 必須修正
      - warnings 非空 → 可以訓練, 但結果可能受影響
    """
    from PIL import Image

    path = os.path.abspath(path)
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.isdir(path):
        return {"ok": False, "errors": [f"目錄不存在: {path}"],
                "warnings": [], "summary": None}

    # 1) 三個 split 都要在
    missing = [s for s in _SPLITS if not os.path.isdir(os.path.join(path, s))]
    if missing:
        errors.append(
            "缺少必要的 split 目錄: " + ", ".join(missing) +
            "（main_finetune 會同時載入 train/val/test，三者缺一不可）")

    counts: dict[str, dict[str, int]] = {}
    for s in _SPLITS:
        sdir = os.path.join(path, s)
        if os.path.isdir(sdir):
            counts[s] = _class_counts(sdir)

    # 2) 每個 split 都要有類別子目錄
    for s, cc in counts.items():
        if not cc:
            errors.append(f"{s}/ 底下沒有類別子目錄（需要 {s}/<class>/*.jpg 這樣的結構）")

    # 3) 三個 split 的類別必須一致 (ImageFolder 各自排序取 label index)
    present = {s: set(cc) for s, cc in counts.items() if cc}
    if len(present) > 1:
        base_split, base = next(iter(present.items()))
        for s, cls in present.items():
            if cls == base:
                continue
            only_a = sorted(base - cls)
            only_b = sorted(cls - base)
            detail = []
            if only_a:
                detail.append(f"{s}/ 缺少 {', '.join(only_a)}")
            if only_b:
                detail.append(f"{s}/ 多出 {', '.join(only_b)}")
            errors.append(
                f"類別與 {base_split}/ 不一致 — " + "；".join(detail) +
                "（各 split 的類別資料夾名稱必須完全相同，否則標籤會對錯）")
            break

    # 類別以 train 為準; train 缺席時退而取任一有內容的 split
    classes = sorted(counts.get("train")
                     or next(iter(present.values()), set()))

    # 4) 分類任務至少兩類
    if classes and len(classes) < 2:
        errors.append(f"只有 1 個類別（{classes[0]}）— 分類任務至少需要 2 類")

    # 5) 不能有空類別
    for s, cc in counts.items():
        empty = sorted(c for c, n in cc.items() if n == 0)
        if empty:
            errors.append(f"{s}/ 有空的類別目錄（0 張影像）: {', '.join(empty)}")

    n = {s: sum(cc.values()) for s, cc in counts.items()}
    if not sum(n.values()):
        errors.append("整個資料集找不到任何影像檔"
                      f"（支援副檔名: {', '.join(_IMG_EXT)}）")

    # ---- 以下為警告 (可訓練, 但值得注意) ----
    if n.get("train", 0) and n["train"] < 100:
        warnings.append(f"train 只有 {n['train']} 張影像，資料量偏少，建議加強增強或用 linear probe 起手")
    for s in ("val", "test"):
        if 0 < n.get(s, 0) < 10:
            warnings.append(f"{s} 只有 {n[s]} 張影像，指標會非常不穩定")

    if counts.get("train"):
        nz = [v for v in counts["train"].values() if v > 0] or [1]
        ratio = max(nz) / min(nz)
        if ratio >= 10:
            warnings.append(f"train 類別極度不平衡（最多:最少 = {ratio:.1f}:1），"
                            "建議 class weight / focal loss / 平衡取樣")
        elif ratio >= 3:
            warnings.append(f"train 類別不平衡（{ratio:.1f}:1），主要指標建議看 F1 / kappa 而非 accuracy")

    # 類別目錄下還有子目錄 → ImageFolder 會遞迴收進來並攤平
    nested = []
    for s, cc in counts.items():
        for c in cc:
            cdir = os.path.join(path, s, c)
            try:
                if any(os.path.isdir(os.path.join(cdir, x)) for x in os.listdir(cdir)):
                    nested.append(f"{s}/{c}")
            except OSError:
                pass
    if nested:
        warnings.append("類別目錄底下還有子目錄，ImageFolder 會遞迴收進來並攤平為同一類: "
                        + ", ".join(nested[:5]) + ("…" if len(nested) > 5 else ""))

    # 抽樣開圖: 讀得開嗎 / 尺寸 / 灰階
    widths, heights, gray, bad = [], [], 0, []
    sampled = _sample_paths(counts, path)
    for p in sampled:
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
                if im.mode in ("L", "1"):
                    gray += 1
        except Exception as e:                       # noqa: BLE001
            bad.append(f"{os.path.relpath(p, path)}（{type(e).__name__}）")
    if bad:
        errors.append(f"抽樣的 {len(sampled)} 張影像中有 {len(bad)} 張無法讀取: "
                      + ", ".join(bad[:3]) + ("…" if len(bad) > 3 else ""))
    if sampled and gray == len(sampled):
        warnings.append("抽樣影像皆為灰階，預訓練 encoder 以 3 通道 RGB 為主（載入時會自動轉 RGB，僅供參考）")
    if widths and min(min(widths), min(heights)) < _MIN_SIDE_WARN:
        warnings.append(f"最小影像短邊只有 {min(min(widths), min(heights))} px，"
                        f"小於預設 input_size {_MIN_SIDE_WARN}，放大後細節有限")

    summary = {
        "path": path,
        "classes": classes,
        "num_classes": len(classes),
        "counts": counts,
        "n_train": n.get("train", 0),
        "n_val": n.get("val", 0),
        "n_test": n.get("test", 0),
        "total": sum(n.values()),
        "n_sampled": len(sampled),
        "image_size": ({"min_width": min(widths), "max_width": max(widths),
                        "min_height": min(heights), "max_height": max(heights)}
                       if widths else {}),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "summary": summary}
