"""沿著 frame 軸的 robust 整合（master calibration 與 Stage 6 共用）。

剔除代碼（SPEC §4.4）：0 valid、1 low、2 high、3 saturated、4 out_of_bounds、5 cosmetic。

`xp` 可以是 numpy 或 cupy：同一份程式在 CPU 與 GPU 上跑。中位數用排序取中間值
（比 numpy 的 nanmedian 快約 7 倍），標準差用遮罩加總，不用 nanstd。
"""

from __future__ import annotations

import numpy as np

VALID, LOW, HIGH, SATURATED, OUT_OF_BOUNDS, COSMETIC = 0, 1, 2, 3, 4, 5
MIN_FOR_CLIP = 3


def _median(xp, x, valid, n):
    """有效值的中位數；無效值排到最後（+inf），再依每個像素的有效張數取中間。"""
    s = xp.sort(xp.where(valid, x, xp.inf), axis=0)
    lo = xp.clip((n - 1) // 2, 0, None)[None]
    hi = xp.clip(n // 2, 0, None)[None]
    med = 0.5 * (xp.take_along_axis(s, lo, 0)[0] + xp.take_along_axis(s, hi, 0)[0])
    return xp.where(n > 0, med, xp.nan).astype(xp.float32)


def _masked_std(xp, v, valid, n):
    nn = xp.maximum(n, 1)
    m = xp.where(valid, v, 0).sum(axis=0) / nn
    d = xp.where(valid, v - m, 0)
    return xp.sqrt((d * d).sum(axis=0) / nn)


def _winsorized_sigma(xp, x, valid, n, med):
    """Winsorized 標準差：把 1.5σ 以外的值壓回邊界再算，重複到收斂。

    張數少時會低估（5 張約 0.76σ、8 張約 0.87σ），用 1/(1 − 1.2/n) 修正（模擬常態分布得出），
    否則少量 frame 時會誤剔 1~3% 的正常像素。
    """
    s = _masked_std(xp, x, valid, n)
    for _ in range(5):
        w = xp.clip(x, med - 1.5 * s, med + 1.5 * s)
        s_new = 1.134 * _masked_std(xp, w, valid, n)
        if bool(xp.allclose(s_new, s, rtol=5e-4)):
            s = s_new
            break
        s = s_new
    return s / (1.0 - 1.2 / xp.maximum(n, 2))


def clip_combine(stack, weights=None, low: float = 4.0, high: float = 3.0,
                 method: str = "winsorized", max_iter: int = 5, xp=np):
    """stack: (N, ...) float32，NaN 代表沒有資料。回傳（加權平均, 剔除代碼 (N, ...) uint8），型別跟 xp 一致。

    method: "winsorized"、"sigma"、"average"（不剔除）、"median"。
    """
    x = xp.asarray(stack, dtype=xp.float32)
    n_frames = x.shape[0]
    valid = ~xp.isnan(x)
    codes = xp.where(valid, VALID, OUT_OF_BOUNDS).astype(xp.uint8)
    count = valid.sum(axis=0)
    if method == "median":
        return _median(xp, x, valid, count), codes
    if method in ("winsorized", "sigma"):
        # winsorized：σ 本身已對離群值穩健，只估一次；反覆重估會讓 σ 越縮越小、越剔越多
        for _ in range(1 if method == "winsorized" else max_iter):
            med = _median(xp, x, valid, count)
            s = _winsorized_sigma(xp, x, valid, count, med) if method == "winsorized" else _masked_std(xp, x, valid, count)
            can = count >= MIN_FOR_CLIP
            lo = valid & can & (x < med - low * s)
            hi = valid & can & (x > med + high * s)
            if not bool((lo | hi).any()):
                break
            codes[lo] = LOW
            codes[hi] = HIGH
            valid &= ~(lo | hi)
            count = valid.sum(axis=0)
    w = xp.ones(n_frames, xp.float32) if weights is None else xp.asarray(weights, dtype=xp.float32)
    w = w.reshape((n_frames,) + (1,) * (x.ndim - 1))
    ww = xp.where(valid, w, 0)
    den = ww.sum(axis=0)
    num = (xp.where(valid, x, 0) * ww).sum(axis=0)
    out = xp.where(den > 0, num / xp.where(den > 0, den, 1), xp.nan)
    return out.astype(xp.float32), codes
