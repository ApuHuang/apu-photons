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
CUTOUT = 12


def superpixel(data: np.ndarray) -> np.ndarray:
    h, w = data.shape
    h -= h % 2
    w -= w % 2
    return data[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _moment_fwhm(img: np.ndarray, x: float, y: float) -> float | None:
    """高斯加權的自適應二階矩：窗寬跟著星點收斂，雜訊與鄰近星不會把 FWHM 撐大。

    高斯星點 σ 配上寬 s 的高斯窗，量到的每軸加權變異數 m = σ²s²/(σ²+s²)，可反解 σ² = m·s²/(s²−m)。
    """
    xi, yi = int(round(x)), int(round(y))
    if yi < CUTOUT or xi < CUTOUT or yi + CUTOUT >= img.shape[0] or xi + CUTOUT >= img.shape[1]:
        return None
    c = img[yi - CUTOUT:yi + CUTOUT + 1, xi - CUTOUT:xi + CUTOUT + 1].astype(np.float64)
    edge = np.concatenate([c[0], c[-1], c[1:-1, 0], c[1:-1, -1]])
    c = c - np.median(edge)
    yy, xx = np.indices(c.shape)
    cx, cy = CUTOUT + (x - xi), CUTOUT + (y - yi)
    sigma = DETECT_FWHM / 2.3548
    for _ in range(30):
        s2 = 2.0 * sigma ** 2  # 窗寬取星點 σ 的 √2 倍，m 與 σ 同量級，反解最穩
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        wc = c * np.exp(-r2 / (2.0 * s2))
        tot = wc.sum()
        if tot <= 0:
            return None
        cx, cy = (wc * xx).sum() / tot, (wc * yy).sum() / tot
        m = ((wc * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / tot) / 2.0
        if not 0 < m < s2 or abs(cx - CUTOUT) > 3 or abs(cy - CUTOUT) > 3:
            return None
        new = float(np.sqrt(m * s2 / (s2 - m)))
        if abs(new - sigma) < 1e-3 * sigma:
            sigma = new
            break
        sigma = new
    return 2.3548 * sigma if 0.3 < sigma < CUTOUT / 2.5 else None


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
