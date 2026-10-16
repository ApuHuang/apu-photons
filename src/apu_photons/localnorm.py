"""Stage 4 的 local normalization：每張 frame 的大尺度背景對齊到正規化參考 frame 的背景。

全域正規化每個通道只有一個 scale 與一個 offset。各晚（或同一晚不同高度）的光害梯度不同時對不齊，
剔除會把梯度差當成離群值，邊緣與梯度大的那幾張被多剔或少剔。

做法：scale 照舊用星點的 flux 比；offset 改成平滑的面。
1. 校正與找星時（工作行程）把每張分成格子，取每格的背景（剔除星點後的中位數）→ `background_grid`
2. Stage 4 在參考座標上取一組格點，每點算「正規化參考的背景 − k × 這張的背景」（各自用逆變換到自己的座標取樣），
   用低階多項式擬合（反覆剔除離群的格點，例如衛星軌跡、雲）→ `fit_offset`
   星雲在兩張都有，相減後抵消；留下的是兩張的梯度差。最後成品保留正規化參考那張的梯度，交給後製去除。
3. 整合（`integrate.warp_band`）與 drizzle 用 `eval_offset` 在參考座標上算出每個像素的 offset。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

TILE = 128            # 背景格子邊長（單色是像素；CFA 是每個色版平面的像素，等於原始的 256 像素）
DEGREE = 3            # 多項式階數（Seestar 沒有 flat、天亮前的片：2 階剔除 7～8%，3 階 1～2%；4 階以上沒有更好）
SAMPLES = (24, 36)    # 參考座標上取樣的格點數（列, 行）
MIN_POINTS = 30       # 重疊的格點少於這個數就不做（退回全域 offset）
CLIP_SIGMA = 3.0


def _tile_background(plane: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳（每格背景 (ny, nx), 格子中心的列座標, 行座標），座標以這個平面的像素計。
    小影像時格子跟著縮小，短邊至少分成 8 格。"""
    h, w = plane.shape
    tile = max(8, min(tile, min(h, w) // 8))
    ny, nx = max(1, h // tile), max(1, w // tile)
    th, tw = h // ny, w // nx
    t = np.asarray(plane[: ny * th, : nx * tw], np.float32).reshape(ny, th, nx, tw)[:, ::2, :, ::2]
    t = t.transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    med = np.nanmedian(t, axis=-1)
    mad = 1.4826 * np.nanmedian(np.abs(t - med[..., None]), axis=-1)
    clipped = np.where(np.abs(t - med[..., None]) < CLIP_SIGMA * np.maximum(mad, 1e-6)[..., None], t, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(clipped, axis=-1)
    cy = (np.arange(ny) + 0.5) * th - 0.5
    cx = (np.arange(nx) + 0.5) * tw - 0.5
    return bg.astype(np.float32), cy, cx


def background_grid(data: np.ndarray, bayer: str | None) -> dict:
    """每張 frame 的背景格點：{"y": 中心列座標, "x": 中心行座標（原始像素）, "v": (通道, ny, nx)}。"""
    if not bayer:
        bg, cy, cx = _tile_background(data, TILE)
        return {"y": cy, "x": cx, "v": bg[None]}
    planes: dict[str, list] = {"R": [], "G": [], "B": []}
    for i, color in enumerate(bayer):
        dy, dx = i // 2, i % 2
        planes[color].append(_tile_background(data[dy::2, dx::2], TILE))
    v = np.stack([np.mean([p[0] for p in planes[c]], axis=0) for c in "RGB"])
    _, cy, cx = planes["R"][0]
    # 色版平面的像素 i 在原始座標約是 2i+0.5（2×2 格的中心）
    return {"y": cy * 2 + 0.5, "x": cx * 2 + 0.5, "v": v}


def _sample(grid: dict, xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """在 frame 自己的座標 xy (P, 2) 取背景（雙線性）；落在影像外的點是 NaN。回傳 (通道, P)。"""
    ys, xs = grid["y"], grid["x"]
    iy = np.interp(xy[:, 1], ys, np.arange(len(ys))) if len(ys) > 1 else np.zeros(len(xy))
    ix = np.interp(xy[:, 0], xs, np.arange(len(xs))) if len(xs) > 1 else np.zeros(len(xy))
    h, w = shape
    outside = (xy[:, 0] < 0) | (xy[:, 0] > w - 1) | (xy[:, 1] < 0) | (xy[:, 1] > h - 1)
    out = np.stack([map_coordinates(np.asarray(v, np.float64), [iy, ix], order=1, mode="nearest")
                    for v in np.asarray(grid["v"])])
    out[:, outside] = np.nan
    return out


def _basis(x: np.ndarray, y: np.ndarray, shape: tuple[int, int], degree: int, xp=np) -> list:
    """多項式的各項（座標換成 −0.5～0.5，數值才穩定）。"""
    h, w = shape
    u = x / max(w - 1, 1) - 0.5
    v = y / max(h - 1, 1) - 0.5
    return [u ** i * v ** (d - i) for d in range(degree + 1) for i in range(d, -1, -1)]


def _degree(n_terms: int) -> int:
    d = 0
    while (d + 1) * (d + 2) // 2 < n_terms:
        d += 1
    return d


def fit_offset(grid: dict, inv: np.ndarray, shape: tuple[int, int], ref_grid: dict, ref_inv: np.ndarray,
               ref_frame_shape: tuple[int, int], k: float, out_shape: tuple[int, int],
               degree: int = DEGREE) -> list[list[float]] | None:
    """這張 frame 的 offset 面（每通道一組多項式係數，參考座標）。重疊太少時回傳 None。

    grid / inv / shape：這張的背景格點、參考座標 → 這張座標的變換、這張的尺寸；ref_*：正規化參考 frame 的；
    k：這張的 scale（相對正規化參考）；out_shape：參考座標（輸出網格）的尺寸。"""
    h, w = out_shape
    gy, gx = np.meshgrid(np.linspace(0, h - 1, SAMPLES[0]), np.linspace(0, w - 1, SAMPLES[1]), indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    mine = _sample(grid, (pts @ inv.T)[:, :2], shape)
    target = _sample(ref_grid, (pts @ ref_inv.T)[:, :2], ref_frame_shape)
    diff = target - k * mine
    a = np.column_stack(_basis(pts[:, 0], pts[:, 1], out_shape, degree))
    coeffs = []
    for d in diff:
        ok = np.isfinite(d)
        if ok.sum() < max(MIN_POINTS, 3 * a.shape[1]):
            return None
        for _ in range(5):
            c, *_ = np.linalg.lstsq(a[ok], d[ok], rcond=None)
            resid = d - a @ c
            sig = 1.4826 * np.median(np.abs(resid[ok] - np.median(resid[ok])))
            keep = ok & (np.abs(resid) < CLIP_SIGMA * max(sig, 1e-6))
            if keep.sum() < max(MIN_POINTS, 3 * a.shape[1]) or np.array_equal(keep, ok):
                break
            ok = keep
        coeffs.append([float(v) for v in c])
    return coeffs


def eval_offset(xp, coeffs: list[float], x, y, shape: tuple[int, int]):
    """在參考座標 (x, y)（可廣播的 xp 陣列）算 offset 面。"""
    terms = _basis(x, y, shape, _degree(len(coeffs)), xp)  # 第一項是常數 1（形狀已廣播成 x、y 的形狀）
    out = coeffs[0] * terms[0]
    for c, t in zip(coeffs[1:], terms[1:]):
        out = out + c * t
    return out
