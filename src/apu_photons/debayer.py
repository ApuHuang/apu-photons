"""Bilinear 解馬賽克（MVP；VNG / AHD 列在 V1）。輸入的切片起點必須對齊偶數列、偶數行。

可在 CPU（numpy + scipy.ndimage）或 GPU（cupy + cupyx.scipy.ndimage）上跑。
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as _ndi

_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0


def _sparse(cfa, pattern: str, color: str, xp):
    """只留這個顏色的像素、其他填 0。用跨步切片，不建整張的座標與遮罩
    （以前每次都在 CPU 上建，Seestar 這種幾百張小圖時占整合時間的大半）。"""
    out = xp.zeros(tuple(cfa.shape), xp.float32)
    for i, c in enumerate(pattern):
        if c == color:
            dy, dx = i // 2, i % 2
            out[dy::2, dx::2] = cfa[dy::2, dx::2]
    return out


def bilinear(cfa, pattern: str, xp=np, ndi=_ndi):
    """回傳 (3, H, W) 的 R, G, B。"""
    out = xp.empty((3,) + tuple(cfa.shape), xp.float32)
    for i, (color, k) in enumerate((("R", _K_RB), ("G", _K_G), ("B", _K_RB))):
        out[i] = ndi.convolve(_sparse(cfa, pattern, color, xp), xp.asarray(k), mode="mirror")
    return out
