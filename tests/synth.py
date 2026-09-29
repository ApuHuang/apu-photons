"""合成測試資料：已知位移／旋轉的星場、熱像素、暗角、衛星軌跡。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

H, W = 256, 320
BG = 500.0
READ_NOISE = 8.0
PEDESTAL = 300.0  # dark 的偏壓 + 暗電流，dark frame 同值


def star_catalog(n=120, seed=1):
    rng = np.random.default_rng(seed)
    # 天空座標比影像大一點，旋轉後邊緣仍有星
    x = rng.uniform(-40, W + 40, n)
    y = rng.uniform(-40, H + 40, n)
    flux = 10 ** rng.uniform(3.3, 5.0, n)
    return x, y, flux


def render(tx=0.0, ty=0.0, rot_deg=0.0, fwhm=3.0, seed=0, catalog=None, vignette=None, hot=None,
           trail=False, bayer=None, gain=1.0):
    """回傳 frame 影像；(tx, ty, rot) 是「參考 → 這張」的變換（繞影像中心旋轉）。"""
    rng = np.random.default_rng(seed)
    x, y, flux = catalog or star_catalog()
    th = np.deg2rad(rot_deg)
    cx, cy = W / 2, H / 2
    xs = np.cos(th) * (x - cx) - np.sin(th) * (y - cy) + cx + tx
    ys = np.sin(th) * (x - cx) + np.cos(th) * (y - cy) + cy + ty
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), BG, np.float64)
    sig = fwhm / 2.3548
    for xi, yi, f in zip(xs, ys, flux):
        if -10 < xi < W + 10 and -10 < yi < H + 10:
            y0, y1 = max(0, int(yi) - 12), min(H, int(yi) + 13)
            x0, x1 = max(0, int(xi) - 12), min(W, int(xi) + 13)
            g = np.exp(-((xx[y0:y1, x0:x1] - xi) ** 2 + (yy[y0:y1, x0:x1] - yi) ** 2) / (2 * sig ** 2))
            img[y0:y1, x0:x1] += f * gain * g / (2 * np.pi * sig ** 2)
    if trail:
        for t in np.linspace(0, 1, 4000):
            px, py = int(10 + t * (W - 20)), int(30 + t * (H - 60))
            img[py, px] += 3000
    if bayer:
        # 各色響應不同，模擬 OSC
        resp = {"R": 0.8, "G": 1.0, "B": 0.6}
        for i, c in enumerate(bayer):
            img[i // 2::2, i % 2::2] *= resp[c]
    if vignette is not None:
        img *= vignette
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float64) + rng.normal(PEDESTAL, READ_NOISE, img.shape)
    if hot is not None:
        img[hot] += 20000
    return img.astype(np.float32)


def vignette_map():
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2) ** 2
    return 1.0 - 0.3 * r2


def hot_pixels(n=40, seed=7):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), bool)
    m[rng.integers(5, H - 5, n), rng.integers(5, W - 5, n)] = True
    return m


def write(path: Path, data: np.ndarray, **hdr):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k.replace("_", "-")] = v
    fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), header=h).writeto(path, overwrite=True)
