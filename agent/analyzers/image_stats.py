"""影像統計 — 尺寸 / 通道 / 長寬比 / 亮度分佈 (聚合後的數值, 非個別影像)。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ._util import pct, sample_paths

_N = 200
_BINS = 16


class ImageStatsOut(BaseModel):
    n_sampled: int = 0
    width_p5: Optional[int] = None
    width_p50: Optional[int] = None
    width_p95: Optional[int] = None
    height_p5: Optional[int] = None
    height_p50: Optional[int] = None
    height_p95: Optional[int] = None
    aspect_p5: Optional[float] = None      # width / height
    aspect_p50: Optional[float] = None
    aspect_p95: Optional[float] = None
    min_short_side: Optional[int] = None
    n_below_input_size: int = 0            # 短邊 < 224 的張數
    n_grayscale: int = 0
    n_rgb: int = 0
    n_other_mode: int = 0
    n_unreadable: int = 0
    mean_brightness: Optional[float] = None   # 0-1
    std_brightness: Optional[float] = None
    brightness_hist: list[int] = Field(default_factory=list)  # 16 bins, 0-1
    mean_saturation: Optional[float] = None


def run(root: str) -> ImageStatsOut:
    from PIL import Image, ImageStat

    out = ImageStatsOut(brightness_hist=[0] * _BINS)
    widths: list[float] = []
    heights: list[float] = []
    aspects: list[float] = []
    brights: list[float] = []
    sats: list[float] = []

    for _split, _cls, path in sample_paths(root, _N):
        try:
            with Image.open(path) as im:
                w, h = im.width, im.height
                mode = im.mode
                gray = im.convert("L")
                stat = ImageStat.Stat(gray)
                mean = stat.mean[0] / 255.0
                # 飽和度以 RGB 通道極差近似 (不需 colorsys 逐像素)
                if mode not in ("L", "1"):
                    rgb = ImageStat.Stat(im.convert("RGB")).mean
                    sats.append((max(rgb) - min(rgb)) / 255.0)
        except Exception:                      # noqa: BLE001
            out.n_unreadable += 1
            continue

        out.n_sampled += 1
        widths.append(w)
        heights.append(h)
        aspects.append(w / h if h else 0.0)
        brights.append(mean)
        out.brightness_hist[min(_BINS - 1, int(mean * _BINS))] += 1
        if mode in ("L", "1"):
            out.n_grayscale += 1
        elif mode in ("RGB", "RGBA"):
            out.n_rgb += 1
        else:
            out.n_other_mode += 1
        if min(w, h) < 224:
            out.n_below_input_size += 1

    if widths:
        out.width_p5 = int(pct(widths, 0.05))
        out.width_p50 = int(pct(widths, 0.50))
        out.width_p95 = int(pct(widths, 0.95))
        out.height_p5 = int(pct(heights, 0.05))
        out.height_p50 = int(pct(heights, 0.50))
        out.height_p95 = int(pct(heights, 0.95))
        out.aspect_p5 = round(pct(aspects, 0.05), 3)
        out.aspect_p50 = round(pct(aspects, 0.50), 3)
        out.aspect_p95 = round(pct(aspects, 0.95), 3)
        out.min_short_side = int(min(min(widths), min(heights)))
    if brights:
        m = sum(brights) / len(brights)
        out.mean_brightness = round(m, 4)
        out.std_brightness = round(
            (sum((b - m) ** 2 for b in brights) / len(brights)) ** 0.5, 4)
    if sats:
        out.mean_saturation = round(sum(sats) / len(sats), 4)
    return out
