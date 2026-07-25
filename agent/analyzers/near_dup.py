"""近重複偵測 — 以 dHash (perceptual hash) 分群, 只回報群數與比例。

成本較高 (要開圖), 預設需要使用者同意才執行 (privacy.expensive_analyzers_need_consent)。
"""
from __future__ import annotations

from pydantic import BaseModel

from ._util import SPLITS, iter_images

_MAX_TOTAL = 4000
_SIDE = 8            # dHash: 縮到 (SIDE+1) × SIDE, 比較相鄰像素


class NearDupOut(BaseModel):
    n_hashed: int = 0
    complete: bool = True
    n_clusters: int = 0              # 含 2 張以上的群數
    n_images_in_clusters: int = 0
    max_cluster_size: int = 0
    dup_ratio: float = 0.0           # (群內張數 - 群數) / 總張數
    n_clusters_cross_split: int = 0  # 跨 split 的近重複 (等同洩漏)
    n_clusters_cross_class: int = 0  # 同一張圖被標成不同類別


def _dhash(path: str) -> int | None:
    from PIL import Image
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((_SIDE + 1, _SIDE), Image.BILINEAR)
            px = list(g.getdata())
    except Exception:                       # noqa: BLE001
        return None
    bits = 0
    for row in range(_SIDE):
        base = row * (_SIDE + 1)
        for col in range(_SIDE):
            bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
    return bits


def run(root: str) -> NearDupOut:
    out = NearDupOut()
    buckets: dict[int, list[tuple[str, str]]] = {}   # hash -> [(split, class)]
    for split in SPLITS:
        for cls, path in iter_images(root, split):
            if out.n_hashed >= _MAX_TOTAL:
                out.complete = False
                break
            h = _dhash(path)
            if h is None:
                continue
            out.n_hashed += 1
            buckets.setdefault(h, []).append((split, cls))
        if not out.complete:
            break

    clusters = [v for v in buckets.values() if len(v) > 1]
    out.n_clusters = len(clusters)
    out.n_images_in_clusters = sum(len(v) for v in clusters)
    out.max_cluster_size = max((len(v) for v in clusters), default=0)
    if out.n_hashed:
        out.dup_ratio = round(
            (out.n_images_in_clusters - out.n_clusters) / out.n_hashed, 4)
    out.n_clusters_cross_split = sum(
        1 for v in clusters if len({s for s, _ in v}) > 1)
    out.n_clusters_cross_class = sum(
        1 for v in clusters if len({c for _, c in v}) > 1)
    return out
