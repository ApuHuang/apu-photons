"""Bilinear 解馬賽克（MVP；VNG / AHD 列在 V1）。輸入的切片起點必須對齊偶數列、偶數行。

可在 CPU（numpy + scipy.ndimage）或 GPU（cupy + cupyx.scipy.ndimage）上跑。
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as _ndi

from .imageio import cfa_masks

_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0


def bilinear(cfa, pattern: str, xp=np, ndi=_ndi):
    """回傳 (3, H, W) 的 R, G, B。"""
    masks = cfa_masks(pattern, tuple(cfa.shape))
    out = xp.empty((3,) + tuple(cfa.shape), xp.float32)
    for i, (color, k) in enumerate((("R", _K_RB), ("G", _K_G), ("B", _K_RB))):
        m = xp.asarray(masks[color])
        out[i] = ndi.convolve(xp.where(m, cfa, 0.0).astype(xp.float32), xp.asarray(k), mode="mirror")
    return out
