"""解馬賽克：Bayer 用 bilinear（MVP；VNG / AHD 列在 V1），富士 X-Trans 用色差內插。
輸入的切片起點必須對齊排列的週期（Bayer 偶數列、偶數行；X-Trans 6 的倍數）。

可在 CPU（numpy + scipy.ndimage）或 GPU（cupy + cupyx.scipy.ndimage）上跑。
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as _ndi

from . import cfa

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


# X-Trans 色差內插的權重（可分離的二項式核）：綠用 3×3（每個 3×3 都有綠），色差用 7×7（每個 7×7 都有紅與藍）
_W_G = np.array([1, 2, 1], np.float32)
_W_D = np.array([1, 6, 15, 20, 15, 6, 1], np.float32)


def _smooth(a, w, xp, ndi):
    w = xp.asarray(w)
    return ndi.convolve1d(ndi.convolve1d(a, w, axis=0, mode="mirror"), w, axis=1, mode="mirror")


def _nconv(values, mask, w, xp, ndi):
    """正規化卷積：只用 mask 上的像素做加權平均（權重和永遠大於 0：核夠大，每個窗都有這個顏色）。"""
    return _smooth(values * mask, w, xp, ndi) / _smooth(mask, w, xp, ndi)


def xtrans_colordiff(data, pattern: str, xp=np, ndi=_ndi, wb: tuple[float, float] | None = None):
    """X-Trans 解馬賽克：先內插完整的綠（綠佔 55%），再內插 R−G、B−G 的「色差」，加回綠。
    色差在星點邊緣變化比各色本身平緩得多。與 LibRaw 的 Markesteijn 3-pass 比：一樣銳利、雜訊較低、
    不會在低訊噪比的背景做出迷宮紋；星點色彩略遜（富士 X-E5 實拍比較，SPEC §10.11）。輸入的起點必須對齊 6×6 週期。

    wb：(紅, 藍) 相對綠的白平衡增益。色差平緩的前提是各色大致平衡；沒做白平衡的原始資料裡紅約只有綠的一半，
    R−G 在星點是一個陡的負峰，平滑後紅與藍比綠窄，亮星周圍出現綠圈。所以先乘增益再內插、最後除回去
    （輸出仍是原始色彩，同 LibRaw 在解馬賽克前先套白平衡）。None = 不調整。
    回傳 (3, H, W) 的 R, G, B；原本就是這個顏色的像素保持原值。"""
    data = xp.asarray(data, xp.float32)
    ch = cfa.channel_map(pattern, tuple(data.shape), xp)
    work = data
    if wb is not None:
        gain = xp.asarray(np.array([wb[0], 1.0, wb[1]], np.float32))[ch]
        work = data * gain
    m_g = (ch == 1).astype(xp.float32)
    out = xp.empty((3,) + tuple(data.shape), xp.float32)
    g = xp.where(ch == 1, work, _nconv(work, m_g, _W_G, xp, ndi))
    out[1] = g
    diff = work - g
    for i, c in enumerate((0, 2)):
        m = (ch == c).astype(xp.float32)
        est = g + _nconv(diff, m, _W_D, xp, ndi)
        if wb is not None:
            est = est / xp.float32(wb[i])
        out[c] = xp.where(ch == c, data, est)
    return out


def demosaic(data, pattern: str, xp=np, ndi=_ndi, wb: tuple[float, float] | None = None):
    """依排列選解馬賽克的方法：Bayer 用 bilinear（照舊，不用 wb），X-Trans 用色差內插。回傳 (3, H, W)。"""
    if cfa.is_xtrans(pattern):
        return xtrans_colordiff(data, pattern, xp, ndi, wb)
    return bilinear(data, pattern, xp, ndi)
