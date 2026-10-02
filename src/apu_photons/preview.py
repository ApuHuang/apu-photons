"""結果預覽：讀輸出的 master FITS，縮小後做自動拉伸（只給眼睛看，不改變資料）。

拉伸方式與 APU Pick 的預覽相同（類似 PixInsight 的 STF）：黑點 = 中位數 − 2.8 × MAD，
再用 midtone transfer 把背景拉到約 25% 亮度；彩色各通道分開拉伸，背景自動變中性。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from .imageio import TOP_DOWN, row_order

TARGET_BACKGROUND = 0.25
SHADOW_CLIP = 2.8
MAX_SIDE = 2400


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def stretch_params(sample: np.ndarray) -> tuple[float, float, float] | None:
    """（黑點, 範圍, midtone）；放大檢視時整張算一次後固定，縮放、平移時亮度才不會跳。"""
    s = sample[np.isfinite(sample)]
    if not s.size:
        return None
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    top = float(np.percentile(s, 99.99))
    low = med - SHADOW_CLIP * mad
    span = max(top - low, 1e-6)
    med_n = float(np.clip((med - low) / span, 1e-6, 1 - 1e-6))
    t = TARGET_BACKGROUND
    return low, span, med_n * (1 - t) / (med_n - 2 * t * med_n + t)


def apply_stretch(channel: np.ndarray, params: tuple[float, float, float] | None) -> np.ndarray:
    if params is None:
        return np.zeros_like(channel, np.float32)
    low, span, m = params
    x = np.clip((np.nan_to_num(channel, nan=low) - low) / span, 0, 1)
    return np.clip(_mtf(m, x), 0, 1).astype(np.float32)


def stretch(channel: np.ndarray, sample: np.ndarray | None = None) -> np.ndarray:
    return apply_stretch(channel, stretch_params(channel if sample is None else sample))


def _bin(img: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return img
    h, w = img.shape[-2:]
    h, w = h - h % n, w - w % n
    v = img[..., :h, :w]
    return v.reshape(v.shape[:-2] + (h // n, n, w // n, n)).mean(axis=(-3, -1))


def load_preview(path: Path, max_side: int = MAX_SIDE) -> Image.Image:
    """master FITS → 自動拉伸的 PIL 影像（最長邊不超過 max_side）。"""
    with fits.open(path, memmap=True) as hdul:
        data, header = hdul[0].data, hdul[0].header
        if row_order(header) != TOP_DOWN:
            data = data[..., ::-1, :]  # bottom-up：第 0 列是畫面最下面，翻成由上往下顯示
        return _render(data, max_side)


def _render(data: np.ndarray, max_side: int) -> Image.Image:
    n = max(1, int(np.ceil(max(data.shape[-2:]) / max_side)))
    img = _bin(np.asarray(data, np.float32), n)
    step = max(1, max(img.shape[-2:]) // 1000)
    if img.ndim == 3:
        rgb = np.dstack([stretch(c, c[::step, ::step]) for c in img])
        return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    return Image.fromarray((stretch(img, img[::step, ::step]) * 255).astype(np.uint8), "L")
