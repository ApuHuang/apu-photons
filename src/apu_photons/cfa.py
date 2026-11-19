"""CFA 排列：2×2 Bayer 與富士的 6×6 X-Trans 共用的工具。

排列以字串表示，依列優先列出一個週期內每格的顏色（"R"、"G"、"B"）：
Bayer 是 4 個字（"RGGB"），X-Trans 是 36 個字（"GGRGGB…"，與 PixInsight 紀錄的相同）。
排列一律相對於陣列的 (0, 0)；header 的 XBAYROFF／YBAYROFF 已在 `imageio.bayer_pattern` 換算掉。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


def period(pattern: str | None) -> int:
    """排列的週期（單色當成 1）。"""
    if not pattern:
        return 1
    return 6 if len(pattern) == 36 else 2


def is_xtrans(pattern: str | None) -> bool:
    return bool(pattern) and len(pattern) == 36


def is_valid(pattern: str) -> bool:
    if pattern in BAYER_PATTERNS:
        return True
    return len(pattern) == 36 and set(pattern) == {"R", "G", "B"}


def label(pattern: str | None) -> str:
    """顯示用：Bayer 照寫，X-Trans 的 36 個字太長，寫 X-Trans。"""
    return "X-Trans" if is_xtrans(pattern) else (pattern or "")


def shift(pattern: str, dx: int, dy: int) -> str:
    """影像原點移動 (dx, dy) 後的排列（新的 (0, 0) 是原本的 (dy, dx)）。"""
    p = period(pattern)
    return "".join(pattern[((r + dy) % p) * p + (c + dx) % p] for r in range(p) for c in range(p))


@lru_cache(maxsize=32)
def color_index(pattern: str) -> np.ndarray:
    """(p, p) 的通道編號（R=0, G=1, B=2）。"""
    p = period(pattern)
    return np.array(["RGB".index(c) for c in pattern], np.int64).reshape(p, p)


def channel_map(pattern: str, shape: tuple[int, int], xp=np):
    """整張的通道編號 (H, W)。"""
    p = period(pattern)
    h, w = shape
    idx = xp.asarray(color_index(pattern))
    return xp.tile(idx, (-(-h // p), -(-w // p)))[:h, :w]


def masks(pattern: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """每個顏色在整張影像上的布林遮罩。"""
    ch = channel_map(pattern, shape)
    return {c: ch == i for i, c in enumerate("RGB")}


def color_counts(pattern: str) -> dict[str, int]:
    """一個週期內每個顏色的像素數（Bayer 是 R1 G2 B1，X-Trans 是 R8 G20 B8）。"""
    return {c: pattern.count(c) for c in "RGB"}


def phases(pattern: str, color: str | None = None) -> list[tuple[int, int]]:
    """週期內的位置 (dy, dx)；給 color 時只列這個顏色的。"""
    p = period(pattern)
    return [(i // p, i % p) for i, c in enumerate(pattern) if color is None or c == color]


@lru_cache(maxsize=32)
def same_color_offsets(pattern: str, radius: int = 2) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    """每個位置 (dy, dx) → 半徑內（不含自己）同色像素的相對位置。"""
    p = period(pattern)
    idx = color_index(pattern)
    out = {}
    for py in range(p):
        for px in range(p):
            c = idx[py, px]
            out[(py, px)] = tuple((dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
                                  if (dy or dx) and idx[(py + dy) % p, (px + dx) % p] == c)
    return out


def pad_periodic(a: np.ndarray, p: int) -> np.ndarray:
    """四邊各加 p 格，用最靠邊的那個週期平移複製：每格的顏色不變，內容也是附近的像素。"""
    a = np.concatenate([a[:p], a, a[-p:]], axis=0)
    return np.concatenate([a[:, :p], a, a[:, -p:]], axis=1)


def bin_factor(pattern: str | None) -> int:
    """找星用的亮度圖合併倍數：單色 1、Bayer 2×2、X-Trans 3×3（每個 3×3 都是 2 紅 5 綠 2 藍）。"""
    return {1: 1, 2: 2, 6: 3}[period(pattern)]


def bin_origin(pattern: str | None) -> tuple[int, int]:
    """3×3 合併的起點 (oy, ox)：從這裡開始，每個 3×3 區塊都剛好含 2 紅 5 綠 2 藍。Bayer 與單色是 (0, 0)。"""
    if not is_xtrans(pattern):
        return 0, 0
    idx = color_index(pattern)
    for oy in range(3):
        for ox in range(3):
            ok = True
            for by in (0, 3):
                for bx in (0, 3):
                    blk = np.roll(idx, (-oy - by, -ox - bx), axis=(0, 1))[:3, :3]
                    if np.bincount(blk.ravel(), minlength=3).tolist() != [2, 5, 2]:
                        ok = False
            if ok:
                return oy, ox
    return 0, 0
