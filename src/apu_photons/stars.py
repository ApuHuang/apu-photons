"""Stage 3 前半：在亮度代理圖上偵測星點並量 FWHM / SNR。

CFA 用 2×2 super-pixel 當亮度代理（只拿來找星，不當輸出），座標換回原始像素。
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

from .model import Stars

DETECT_SIGMA = 5.0
DETECT_FWHM = 3.0
MAX_STARS = 400
SHAPE_STARS = 60
CUTOUT = 7


def superpixel(data: np.ndarray) -> np.ndarray:
    h, w = data.shape
    h -= h % 2
    w -= w % 2
    return data[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _moment_fwhm(img: np.ndarray, x: float, y: float) -> float | None:
    xi, yi = int(round(x)), int(round(y))
    if yi < CUTOUT or xi < CUTOUT or yi + CUTOUT >= img.shape[0] or xi + CUTOUT >= img.shape[1]:
        return None
    c = img[yi - CUTOUT:yi + CUTOUT + 1, xi - CUTOUT:xi + CUTOUT + 1]
    c = np.clip(c - np.median(c), 0, None)
    tot = c.sum()
    if tot <= 0:
        return None
    yy, xx = np.indices(c.shape)
    cx, cy = (c * xx).sum() / tot, (c * yy).sum() / tot
    var = ((c * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / tot) / 2.0
    return float(2.3548 * np.sqrt(var)) if var > 0 else None


def _finder(noise: float) -> DAOStarFinder:
    kw = dict(fwhm=DETECT_FWHM, threshold=DETECT_SIGMA * noise, exclude_border=True)
    try:  # photutils 3.0 起改名
        return DAOStarFinder(n_brightest=MAX_STARS, **kw)
    except TypeError:
        return DAOStarFinder(brightest=MAX_STARS, **kw)


def detect(data: np.ndarray, bayer: str | None) -> Stars:
    img = superpixel(data) if bayer else data
    scale = 2.0 if bayer else 1.0
    box = max(16, min(img.shape) // 16)
    try:
        bkg = Background2D(img, box, filter_size=3, bkg_estimator=MedianBackground())
        sub, bg_level, noise = img - bkg.background, float(bkg.background_median), float(bkg.background_rms_median)
    except ValueError:
        mean, med, std = sigma_clipped_stats(img[::2, ::2], sigma=3.0)
        sub, bg_level, noise = img - med, float(med), float(std)
    noise = max(noise, 1e-6)
    finder = _finder(noise)
    tbl = finder(sub)
    if tbl is None or len(tbl) == 0:
        return Stars(np.empty(0), np.empty(0), np.empty(0), float("nan"), bg_level, noise, 0.0)
    tbl.sort("flux", reverse=True)
    xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    x = np.asarray(tbl[xcol], float)
    y = np.asarray(tbl[xcol.replace("x", "y", 1)], float)
    flux = np.asarray(tbl["flux"], float)
    peak = np.asarray(tbl["peak"], float)
    fw = [f for f in (_moment_fwhm(sub, xi, yi) for xi, yi in zip(x[:SHAPE_STARS], y[:SHAPE_STARS])) if f]
    fwhm = float(np.median(fw)) * scale if fw else float("nan")
    snr = float(np.median(peak[:SHAPE_STARS]) / noise)
    # super-pixel (i, j) 的中心在原始像素 2i+0.5
    return Stars(x * scale + (scale - 1) / 2, y * scale + (scale - 1) / 2, flux, fwhm, bg_level, noise, snr)
