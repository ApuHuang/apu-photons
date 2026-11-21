"""畫質評測：把 APU Photons 的成品和參考成品（例如 PixInsight WBPP 的 master）放在同一把尺上比較。

開發用，不在 GUI 裡。用法：

    python -m apu_photons.benchmark 參考.xisf 成品1.fits [成品2.fits ...] [--labels PI A B] [--json 結果.json]

指標（每張各自量，再和參考配對比較）：
- 星點：FWHM（高斯加權自適應二階矩，與疊圖引擎相同）、偏心率、星點峰值 SNR（峰值 ÷ 背景雜訊，與亮度單位無關）
- 背景雜訊：去掉背景與星點後的像素雜訊；用配對星的亮度比例換成參考的單位
- 熱像素殘留：比星點尖很多的尖點、其中只出現在單一色版的（RGB）、明顯的負向孤立點（冷像素）；
  JSON 的 examples 列出各類最明顯的位置，可以打開成品確認
- 亮星振鈴：最亮的星周圍、方位角中位數輪廓低於背景多少 σ
- 衛星軌跡殘留：遮掉星點後沿各角度投影，最強直線的顯著度（純雜訊約 6～7，超過 10 就是有一條線）

配對比較時，成品用相似變換＋3 階多項式對到參考，所以 Drizzle 2×、畸變不同的成品也能比；
FWHM 一律換成參考的像素。比值都是「成品 ÷ 參考」的配對星中位數。
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .register import RegistrationError, register, scale_of

STAR_SIGMA = 30.0       # 量星點形狀用的星：峰值至少 30σ
PEAK_BOX = 11
CUTOUT = 12
MAX_SHAPE_STARS = 4000
BORDER = 0.03           # 邊緣不算（疊圖邊緣張數少、雜訊大）
ARTIFACT_SIGMA = 10.0
RING_STARS = 40
TRAIL_BIN = 4
TRAIL_MIN_LENGTH = 100  # 縮小後的格數；太短的線不算
MATCH_RADIUS = 2.0
REGISTER_STARS = 300
NOISE_BIN = 8
EXAMPLES = 10
BACKGROUND_BLOCK = 16
POLY_DEGREE = 3


# ---------- 讀檔 ----------

def read_xisf(path: Path) -> np.ndarray:
    """讀單一影像的 XISF（PixInsight 格式），回傳 (色版, 高, 寬) float32。支援不壓縮、zlib、zlib+sh。"""
    with open(path, "rb") as fh:
        if fh.read(8) != b"XISF0100":
            raise ValueError(f"{path.name}: not an XISF file")
        hlen = struct.unpack("<I", fh.read(4))[0]
        fh.read(4)
        xml = fh.read(hlen).decode("utf-8", "replace")
        tag = re.search(r"<Image\b[^>]*>", xml)
        if tag is None:
            raise ValueError(f"{path.name}: no image")
        tag = tag.group(0)
        geom = [int(v) for v in re.search(r'geometry="([^"]+)"', tag).group(1).split(":")]
        loc = re.search(r'location="attachment:(\d+):(\d+)"', tag)
        if loc is None:
            raise ValueError(f"{path.name}: only attached image data is supported")
        fmt = re.search(r'sampleFormat="([^"]+)"', tag).group(1)
        order = re.search(r'pixelStorage="([^"]+)"', tag)
        comp = re.search(r'compression="([^"]+)"', tag)
        fh.seek(int(loc.group(1)))
        raw = fh.read(int(loc.group(2)))
    dt = np.dtype({"Float32": "<f4", "Float64": "<f8", "UInt8": "u1", "UInt16": "<u2", "UInt32": "<u4"}[fmt])
    if comp:
        parts = comp.group(1).split(":")
        codec = parts[0]
        if codec not in ("zlib", "zlib+sh"):
            raise ValueError(f"{path.name}: compression {codec} not supported (save as uncompressed or zlib)")
        raw = zlib.decompress(raw)
        if codec == "zlib+sh":
            item = int(parts[2]) if len(parts) > 2 else dt.itemsize
            n = len(raw) // item
            raw = np.frombuffer(raw[: n * item], np.uint8).reshape(item, n).T.tobytes() + raw[n * item:]
    w, h = geom[0], geom[1]
    c = geom[2] if len(geom) > 2 else 1
    a = np.frombuffer(raw, dt, count=w * h * c)
    if order and order.group(1).lower() == "normal":
        a = a.reshape(h, w, c).transpose(2, 0, 1)
    else:
        a = a.reshape(c, h, w)
    a = a.astype(np.float32)
    if dt.kind == "u":
        a /= np.iinfo(dt).max
    return a


def load(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".xisf":
        a = read_xisf(path)
    else:
        with fits.open(path, memmap=False) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            a = np.asarray(hdu.data, dtype=np.float32)
            if str(hdu.header.get("ROWORDER", "")).upper() == "BOTTOM-UP":
                a = a[..., ::-1, :]
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"{path.name}: unexpected shape {a.shape}")
    return np.ascontiguousarray(a)


# ---------- 單張量測 ----------

def _mad(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else float("nan")


def _upsample(a: np.ndarray, n: int, axis: int, step: int) -> np.ndarray:
    """雙線性放大，取樣點在每個 step 區塊的中心。不能用區塊複製：每 step 像素一個台階，會被當成橫線直線。"""
    pos = (np.arange(n) - (step - 1) / 2) / step
    i0 = np.clip(np.floor(pos).astype(int), 0, a.shape[axis] - 1)
    i1 = np.clip(i0 + 1, 0, a.shape[axis] - 1)
    f = np.clip(pos - np.floor(pos), 0.0, 1.0).astype(np.float32)
    f = np.where(pos < 0, 0.0, f).astype(np.float32)
    shape = [1, 1]
    shape[axis] = n
    f = f.reshape(shape)
    return np.take(a, i0, axis=axis) * (1 - f) + np.take(a, i1, axis=axis) * f


def _background(img: np.ndarray) -> np.ndarray:
    """16×16 區塊中位數 → 5×5 中位數平滑（約 80 像素）→ 雙線性放大。

    用區塊中位數而不是每 16 點取一點：背景模型自己的雜訊會原封不動地進到合併後的雜訊裡。"""
    b = BACKGROUND_BLOCK
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    blocks = img[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, b * b)
    small = ndi.median_filter(np.median(blocks, axis=2), 5, mode="nearest").astype(np.float32)
    return _upsample(_upsample(small, img.shape[0], 0, b), img.shape[1], 1, b)


def _sample(a: np.ndarray, n: int = 2_000_000) -> np.ndarray:
    return a[:: max(1, a.size // n)]


def _star_mask(hp: np.ndarray, valid: np.ndarray, s0: float) -> np.ndarray:
    """只遮點狀光源（星點），不遮直線。

    星點往每個方向都是凸的；直線沿著線的方向是平的。取四個方向二次微分（取負號）的最小值，
    只有點狀光源會很大。每顆星依亮度遮一個圓，半徑涵蓋到星點降到約 1σ 的地方。"""
    smooth = ndi.gaussian_filter(hp, 1.0)
    point = None
    for kern in (np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]), np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),
                 np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]]) / 2, np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]]) / 2):
        c = -ndi.correlate(smooth, kern.astype(np.float32), mode="nearest")
        point = c if point is None else np.minimum(point, c, out=point)
    sp = _mad(_sample(point[valid]))
    ss = _mad(_sample(smooth[valid]))
    seeds = (point > 5 * sp) & (smooth > 5 * ss)
    lab, n = ndi.label(seeds)
    mask = np.zeros(hp.shape, bool)
    if not n:
        return mask
    peak = ndi.maximum(hp, lab, index=np.arange(1, n + 1))
    # 高斯星點（FWHM 約 4）降到 1σ 的半徑，再放寬 1.5 倍容納星點的外暈
    radius = 2 + 1.5 * 1.7 * np.sqrt(2 * np.log(np.maximum(np.asarray(peak) / s0, 1.0)))
    classes = np.array([3, 5, 8, 12, 18, 27, 40])
    cls = np.searchsorted(classes, radius)
    cls = np.minimum(cls, len(classes) - 1)
    lut = np.concatenate([[-1], cls])
    which = lut[lab]
    for i, r in enumerate(classes):
        sel = which == i
        if sel.any():
            mask |= ndi.distance_transform_edt(~sel) <= r
    return mask


def _highpass(img: np.ndarray, valid: np.ndarray, with_mask: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
    """去背景；回傳（高通, 背景雜訊 σ, 星點遮罩）。"""
    filled = np.where(valid, img, np.median(img[valid]))
    hp = (filled - _background(filled)).astype(np.float32)
    s0 = _mad(_sample(hp[valid]))
    stars = _star_mask(hp, valid, s0) if with_mask else ndi.binary_dilation(hp > 3 * s0, iterations=3)
    sel = valid & ~stars
    sigma = _mad(_sample(hp[sel]))
    return hp, sigma, stars


def _moments(img: np.ndarray, x: float, y: float) -> tuple[float, float, float, float] | None:
    """與 stars._moment_fwhm 相同的自適應高斯加權二階矩，另外回傳偏心率。回傳（x, y, FWHM, 偏心率）。"""
    xi, yi = int(round(x)), int(round(y))
    if yi < CUTOUT or xi < CUTOUT or yi + CUTOUT >= img.shape[0] or xi + CUTOUT >= img.shape[1]:
        return None
    c = img[yi - CUTOUT:yi + CUTOUT + 1, xi - CUTOUT:xi + CUTOUT + 1].astype(np.float64)
    edge = np.concatenate([c[0], c[-1], c[1:-1, 0], c[1:-1, -1]])
    c = c - np.median(edge)
    yy, xx = np.indices(c.shape)
    cx, cy = CUTOUT + (x - xi), CUTOUT + (y - yi)
    sigma = 3.0 / 2.3548
    cov = None
    for _ in range(30):
        s2 = 2.0 * sigma ** 2
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            wc = c * np.exp(-r2 / (2.0 * s2))
        tot = wc.sum()
        if not tot > 0:
            return None
        cx, cy = (wc * xx).sum() / tot, (wc * yy).sum() / tot
        dx, dy = xx - cx, yy - cy
        mxx, myy, mxy = (wc * dx * dx).sum() / tot, (wc * dy * dy).sum() / tot, (wc * dx * dy).sum() / tot
        m = (mxx + myy) / 2.0
        if not 0 < m < s2 or abs(cx - CUTOUT) > 3 or abs(cy - CUTOUT) > 3:
            return None
        cov = np.array([[mxx, mxy], [mxy, myy]])
        new = float(np.sqrt(m * s2 / (s2 - m)))
        done = abs(new - sigma) < 1e-3 * sigma
        sigma = new
        if done:
            break
    if not 0.3 < sigma < CUTOUT / 2.5 or cov is None:
        return None
    # 去掉高斯窗的影響：C⁻¹ = C_meas⁻¹ − I/s²
    try:
        inv = np.linalg.inv(cov) - np.eye(2) / s2
        lam = np.linalg.eigvalsh(np.linalg.inv(inv))
    except np.linalg.LinAlgError:
        return None
    if lam.min() <= 0:
        ecc = float("nan")
    else:
        ecc = float(np.sqrt(max(0.0, 1.0 - lam.min() / lam.max())))
    return xi - CUTOUT + cx, yi - CUTOUT + cy, 2.3548 * sigma, ecc


def _peaks(hp: np.ndarray, valid: np.ndarray, threshold: float, box: int = PEAK_BOX) -> tuple[np.ndarray, np.ndarray]:
    pk = (hp == ndi.maximum_filter(hp, box)) & (hp > threshold) & valid
    return np.nonzero(pk)


@dataclass
class Report:
    label: str
    path: str
    shape: list
    stars: int = 0
    fwhm: float = float("nan")
    eccentricity: float = float("nan")
    star_snr: float = float("nan")
    noise: float = float("nan")
    noise_channels: list = field(default_factory=list)
    sharp_per_mp: float = float("nan")
    color_isolated_per_mp: float = float("nan")
    cold_per_mp: float = float("nan")
    ring_sigma_median: float = float("nan")
    ring_sigma_max: float = float("nan")
    noise_binned: float = float("nan")
    trail_z: float = float("nan")
    trail_angle: float = float("nan")
    trail_point: list | None = None
    pattern_z: float = float("nan")
    examples: dict = field(default_factory=dict)
    # 與參考配對（參考自己沒有）
    matched: int | None = None
    geometry_rms: float | None = None
    pixel_scale: float | None = None
    brightness_scale: float | None = None
    fwhm_ratio: float | None = None
    eccentricity_diff: float | None = None
    star_snr_ratio: float | None = None
    noise_ratio: float | None = None
    noise_binned_ratio: float | None = None


@dataclass
class Measured:
    report: Report
    lum_hp: np.ndarray
    sigma: float
    cat: np.ndarray     # 欄位：x, y, FWHM, 偏心率, 峰值 SNR, 光通量
    bright: np.ndarray  # 對齊用：最亮的星（含飽和），由亮到暗的 x, y
    mask: np.ndarray    # 背景（有效範圍內、不是星點）


def _binned_noise(hp: np.ndarray, mask: np.ndarray, k: int = NOISE_BIN) -> float:
    """k×k 平均後的背景雜訊（只用整塊都是背景的格子），再乘 k 換回單一像素的尺度。

    內插讓相鄰像素的雜訊相關，逐像素雜訊會被低估；合併後的雜訊才是暗弱細節真正看到的。
    雜訊不相關時這個值與逐像素雜訊相同。"""
    h, w = (hp.shape[0] // k) * k, (hp.shape[1] // k) * k
    v = np.where(mask, hp, 0.0)[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    full = mask[:h, :w].reshape(h // k, k, w // k, k).all(axis=(1, 3))
    return _mad(v[full]) * k if full.sum() > 100 else float("nan")


def _valid_mask(img: np.ndarray) -> np.ndarray:
    ok = np.all(np.isfinite(img), axis=0) & np.any(img != 0, axis=0)
    h, w = ok.shape
    by, bx = int(h * BORDER), int(w * BORDER)
    ok[:by] = ok[h - by:] = False
    ok[:, :bx] = ok[:, w - bx:] = False
    return ndi.binary_erosion(ok, iterations=8)


def _star_catalog(hp: np.ndarray, sigma: float, valid: np.ndarray, sat: float) -> np.ndarray:
    ys, xs = _peaks(hp, valid, STAR_SIGMA * sigma)
    v = hp[ys, xs]
    keep = v < sat
    ys, xs, v = ys[keep], xs[keep], v[keep]
    order = np.argsort(-v, kind="stable")
    if len(order) > MAX_SHAPE_STARS:  # 從各亮度均勻抽，不只挑最亮的
        order = order[np.linspace(0, len(order) - 1, MAX_SHAPE_STARS).astype(int)]
    rows = []
    for i in order:
        m = _moments(hp, float(xs[i]), float(ys[i]))
        if m is None:
            continue
        x, y, fw, ecc = m
        r = max(3, int(round(1.5 * fw)))
        yi, xi = ys[i], xs[i]
        flux = float(hp[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].sum())
        rows.append((x, y, fw, ecc, float(v[i]) / sigma, flux))
    return np.array(rows, float).reshape(-1, 6)


def _artifacts(img: np.ndarray, valid: np.ndarray, star_fwhm: float) -> tuple[float, float, float, dict]:
    """回傳（尖點, 單一色版尖點, 冷像素）每百萬像素的個數，以及各類最明顯的幾個位置（x, y, 色版, σ）。"""
    mp = valid.sum() / 1e6
    hps, sigs = [], []
    for ch in img:
        hp, s, _ = _highpass(ch, valid, with_mask=False)
        hps.append(hp)
        sigs.append(s)
    found = {"sharp": [], "single_channel": [], "cold": []}
    inner = valid.copy()
    inner[:2] = inner[-2:] = False
    inner[:, :2] = inner[:, -2:] = False
    g = np.exp(-1.0 / (2 * ((star_fwhm / 2) / 2.3548) ** 2))
    star_limit = 1.0 / ((4 * g + 4 * g * g) / 8)  # 高斯星點中心 ÷ 周圍 8 點平均（FWHM 只有一半時）；比它還尖就不是星
    for c, (hp, s) in enumerate(zip(hps, sigs)):
        ys, xs = _peaks(hp, inner, ARTIFACT_SIGMA * s, box=5)
        if len(ys):
            ring = np.stack([hp[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]).mean(0)
            sharp = (ring <= 0) | (hp[ys, xs] > star_limit * ring)
            found["sharp"] += [(int(x), int(y), c, float(hp[y, x] / s)) for y, x in zip(ys[sharp], xs[sharp])]
            if len(hps) > 1:
                others = np.max([hps[o][ys, xs] / sigs[o] for o in range(len(hps)) if o != c], axis=0)
                iso = (others < 3) & sharp  # 也要比星點尖：發射星雲的小結只在紅色版，但不會比星點尖
                found["single_channel"] += [(int(x), int(y), c, float(hp[y, x] / s))
                                            for y, x in zip(ys[iso], xs[iso])]
        nys, nxs = _peaks(-hp, inner, ARTIFACT_SIGMA * s, box=5)
        found["cold"] += [(int(x), int(y), c, float(-hp[y, x] / s)) for y, x in zip(nys, nxs)]
    n = len(hps)
    counts = (len(found["sharp"]) / mp / n,
              len(found["single_channel"]) / mp if n > 1 else float("nan"),
              len(found["cold"]) / mp / n)
    examples = {k: sorted(v, key=lambda t: -t[3])[:EXAMPLES] for k, v in found.items()}
    return (*counts, examples)


def _ringing(hp: np.ndarray, sigma: float, valid: np.ndarray, fwhm: float) -> tuple[float, float]:
    """最亮的星（含飽和）周圍，方位角中位數輪廓的最低點低於背景幾個 σ。"""
    ys, xs = _peaks(hp, valid, STAR_SIGMA * sigma, box=31)
    if not len(ys):
        return float("nan"), float("nan")
    order = np.argsort(-hp[ys, xs], kind="stable")[:RING_STARS]
    R = int(max(8, round(6 * fwhm)))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rad = np.hypot(yy, xx)
    rbin = np.round(rad).astype(int)
    depths = []
    for i in order:
        y, x = ys[i], xs[i]
        if y < R or x < R or y + R >= hp.shape[0] or x + R >= hp.shape[1]:
            continue
        cut = hp[y - R:y + R + 1, x - R:x + R + 1]
        prof = [np.median(cut[rbin == r]) for r in range(1, R + 1)]
        depths.append(-min(prof) / sigma)
    if not depths:
        return float("nan"), float("nan")
    return float(np.median(depths)), float(np.max(depths))


def _trail(hp: np.ndarray, sigma: float, stars: np.ndarray, valid: np.ndarray) -> tuple[float, float, list, float]:
    """遮掉星點後每 1° 沿直線加總，找最顯著的直線。

    回傳（斜線的顯著度, 角度, 線上最靠近影像中心的一點 [x, y], 水平／垂直線的顯著度）。
    水平與垂直（±2°）另外算：那通常是感光元件的條紋或裁切邊緣，不是衛星軌跡。"""
    k = TRAIL_BIN
    h, w = (hp.shape[0] // k) * k, (hp.shape[1] // k) * k
    mask = valid & ~ndi.binary_dilation(stars, iterations=2)
    v = np.where(mask, np.clip(hp, -5 * sigma, 5 * sigma), 0.0)[:h, :w]
    m = mask[:h, :w].astype(np.float32)
    v = v.reshape(h // k, k, w // k, k).sum(axis=(1, 3)).ravel()
    m = m.reshape(h // k, k, w // k, k).sum(axis=(1, 3)).ravel()
    inside = valid[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3)).ravel().astype(np.float32)
    yy, xx = np.indices((h // k, w // k))
    xx, yy = xx.ravel() + 0.5, yy.ravel() + 0.5
    cx, cy = w / k / 2, h / k / 2
    best, best_angle, best_point, pattern = 0.0, float("nan"), None, 0.0
    for angle in range(0, 180):
        t = np.deg2rad(angle)
        nx, ny = np.cos(t), np.sin(t)  # 線的法向量；同一條線上 d = x·nx + y·ny 相同
        d = np.floor(xx * nx + yy * ny - (cx * nx + cy * ny)).astype(int)
        off = d.min()
        sv = np.bincount(d - off, weights=v)
        sm = np.bincount(d - off, weights=m)
        cells = np.bincount(d - off, weights=inside)  # 線在有效範圍內的長度（縮小後的格數），不受遮掉多少影響
        ok = (cells >= TRAIL_MIN_LENGTH) & (sm > 0.1 * cells * k * k)
        if ok.sum() < 20:
            continue
        mean = np.where(ok, sv / np.maximum(sm, 1), 0.0)
        mean = mean - ndi.median_filter(mean, 31, mode="nearest")  # 去掉背景殘差造成的緩慢起伏
        z = np.zeros_like(mean)
        z[ok] = mean[ok] * np.sqrt(sm[ok])  # 長短不同的線雜訊不同：換成同一個尺度再估 σ
        z[ok] /= _mad(z[ok])
        i = int(np.argmax(np.abs(z)))
        top = float(abs(z[i]))
        if min(angle % 90, 90 - angle % 90) <= 2:
            pattern = max(pattern, top)
        elif top > best:
            dist = i + off + 0.5
            best, best_angle = top, float(angle)
            best_point = [round((cx + dist * nx) * k, 1), round((cy + dist * ny) * k, 1)]
    return best, best_angle, best_point, pattern


def measure(img: np.ndarray, label: str, path: str = "", artifacts: bool = True, trail: bool = True) -> Measured:
    valid = _valid_mask(img)
    lum = np.nan_to_num(img.mean(0))
    hp, sigma, stars = _highpass(lum, valid)
    sat = 0.5 * float(np.nanmax(hp[valid]))
    cat = _star_catalog(hp, sigma, valid, sat)
    rep = Report(label=label, path=str(path), shape=list(img.shape))
    rep.stars = len(cat)
    rep.noise = sigma
    rep.noise_binned = _binned_noise(hp, valid & ~stars)
    rep.noise_channels = [_highpass(np.nan_to_num(ch), valid, with_mask=False)[1] for ch in img] if img.shape[0] > 1 else [sigma]
    if len(cat):
        rep.fwhm = float(np.median(cat[:, 2]))
        rep.eccentricity = float(np.nanmedian(cat[:, 3]))
        rep.star_snr = float(np.median(cat[:, 4]))
    if artifacts:
        rep.sharp_per_mp, rep.color_isolated_per_mp, rep.cold_per_mp, rep.examples = _artifacts(
            np.nan_to_num(img), valid, rep.fwhm if np.isfinite(rep.fwhm) else 3.0)
    rep.ring_sigma_median, rep.ring_sigma_max = _ringing(hp, sigma, valid, rep.fwhm if np.isfinite(rep.fwhm) else 3.0)
    if trail:
        rep.trail_z, rep.trail_angle, rep.trail_point, rep.pattern_z = _trail(hp, sigma, stars, valid)
    ys, xs = _peaks(hp, valid, STAR_SIGMA * sigma, box=31)
    order = np.argsort(-hp[ys, xs], kind="stable")[:REGISTER_STARS]
    bright = np.column_stack([xs[order], ys[order]]).astype(float)
    return Measured(rep, hp, sigma, cat, bright, valid & ~stars)


# ---------- 配對 ----------

def _poly_terms(xy: np.ndarray, scale: float) -> np.ndarray:
    x, y = xy[:, 0] / scale, xy[:, 1] / scale
    return np.column_stack([x ** i * y ** j for i in range(POLY_DEGREE + 1) for j in range(POLY_DEGREE + 1 - i)])


def match(ref: Measured, cand: Measured, cand_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """配對星點。回傳（參考索引, 成品索引, 幾何殘差 RMS（參考像素）, 成品→參考的像素尺度）。"""
    rc, cc = ref.cat, cand.cat
    if len(rc) < 8 or len(cc) < 8:
        raise RegistrationError("too few stars")
    m = None
    for flip in (False, True):
        src = cand.bright.copy()
        if flip:
            src[:, 1] = cand_shape[0] - 1 - src[:, 1]
        try:
            m, *_ = register(src, ref.bright, expected_scale=None, max_residual=99)
            break
        except RegistrationError:
            continue
    if m is None:
        raise RegistrationError("could not align the image to the reference")
    pts = cc[:, :2].copy()
    if flip:
        pts[:, 1] = cand_shape[0] - 1 - pts[:, 1]
    pred = pts @ m[:2, :2].T + m[:2, 2]
    tree = cKDTree(rc[:, :2])
    norm = float(max(cand_shape))
    radius = 4.0
    for _ in range(4):  # 相似變換之後再擬合 3 階多項式，容忍兩邊的畸變處理不同
        dist, j = tree.query(pred)
        ok = dist < radius
        if ok.sum() < 30:
            break
        A = _poly_terms(pts[ok], norm)
        coef, *_ = np.linalg.lstsq(A, rc[j[ok], :2], rcond=None)
        pred = _poly_terms(pts, norm) @ coef
        radius = MATCH_RADIUS
    dist, j = tree.query(pred)
    ok = dist < MATCH_RADIUS
    rms = float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("nan")
    return j[ok], np.flatnonzero(ok), rms, scale_of(m)


def compare(ref: Measured, cand: Measured, cand_shape: tuple[int, int]) -> None:
    rep = cand.report
    ri, ci, rms, s = match(ref, cand, cand_shape)
    rep.matched, rep.geometry_rms, rep.pixel_scale = int(len(ri)), rms, s
    if not len(ri):
        return
    r, c = ref.cat[ri], cand.cat[ci]
    rep.fwhm_ratio = float(np.median(c[:, 2] * s / r[:, 2]))
    rep.eccentricity_diff = float(np.nanmedian(c[:, 3] - r[:, 3]))
    rep.star_snr_ratio = float(np.median(c[:, 4] / r[:, 4]))
    # 亮度單位：參考像素值 ≈ a × 成品像素值。表面亮度不受像素大小影響（光通量要除以面積比 1/s²）
    good = (r[:, 5] > 0) & (c[:, 5] > 0)
    if good.any():
        a = float(np.median(r[good, 5] / (c[good, 5] * s * s)))
        rep.brightness_scale = a
        rep.noise_ratio = cand.sigma * a / ref.sigma
        # 合併 k×k 的格子在成品是 k·s 個參考像素寬；Drizzle 2× 時要合併兩倍大的格子才是同一片天空
        k = max(1, int(round(NOISE_BIN / s)))
        cand_binned = _binned_noise(cand.lum_hp, cand.mask, k) * NOISE_BIN / k
        rep.noise_binned_ratio = cand_binned * a / ref.report.noise_binned


# ---------- 命令列 ----------

ROWS = [
    ("stars", "stars ≥30σ", "{:.0f}"),
    ("fwhm", "median FWHM (own px)", "{:.3f}"),
    ("eccentricity", "median eccentricity", "{:.3f}"),
    ("star_snr", "median star peak SNR", "{:.0f}"),
    ("sharp_per_mp", "sharp peaks /MP", "{:.2f}"),
    ("color_isolated_per_mp", "single-channel peaks /MP", "{:.2f}"),
    ("cold_per_mp", "cold pixels /MP", "{:.2f}"),
    ("ring_sigma_median", "bright-star ring depth σ (median)", "{:.2f}"),
    ("ring_sigma_max", "bright-star ring depth σ (max)", "{:.2f}"),
    ("trail_z", "strongest slanted line z (noise ≈6-7)", "{:.1f}"),
    ("pattern_z", "strongest row/column line z", "{:.1f}"),
    ("matched", "matched stars", "{:.0f}"),
    ("geometry_rms", "geometry rms (ref px)", "{:.2f}"),
    ("pixel_scale", "pixel scale vs ref", "{:.3f}"),
    ("brightness_scale", "brightness scale vs ref", "{:.4g}"),
    ("fwhm_ratio", "FWHM ÷ ref (matched)", "{:.3f}"),
    ("eccentricity_diff", "eccentricity − ref", "{:+.3f}"),
    ("star_snr_ratio", "star peak SNR ÷ ref", "{:.3f}"),
    ("noise_ratio", "pixel noise ÷ ref", "{:.3f}"),
    ("noise_binned_ratio", "8×8-binned noise ÷ ref", "{:.3f}"),
]


def format_table(reports: list[Report]) -> str:
    width = max(12, *(len(r.label) for r in reports))
    lines = ["".ljust(36) + "".join(r.label.rjust(width + 2) for r in reports)]
    for key, name, fmt in ROWS:
        cells = []
        for r in reports:
            v = getattr(r, key)
            cells.append(("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v))
                         .rjust(width + 2))
        lines.append(name.ljust(36) + "".join(cells))
    return "\n".join(lines)


def run(paths: list[Path], labels: list[str] | None = None, artifacts: bool = True, trail: bool = True,
        echo=print) -> list[Report]:
    labels = labels or [Path(p).stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError("--labels must have one name per image")
    ref = None
    reports = []
    for path, label in zip(paths, labels):
        if echo:
            echo(f"measuring {label} …")
        img = load(Path(path))
        m = measure(img, label, path, artifacts, trail)
        if ref is None:
            ref = m
        else:
            try:
                compare(ref, m, img.shape[1:])
            except RegistrationError as e:
                if echo:
                    echo(f"  {label}: could not match stars with the reference ({e})")
        reports.append(m.report)
        del img
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m apu_photons.benchmark",
                                description="Compare stacked images against a reference (the first image).")
    p.add_argument("images", nargs="+", type=Path, help="reference first, then the images to compare (.fits/.xisf)")
    p.add_argument("--labels", nargs="+", help="short names, one per image")
    p.add_argument("--json", type=Path, help="also write the results as JSON")
    p.add_argument("--no-artifacts", action="store_true", help="skip hot/cold pixel counts (faster)")
    p.add_argument("--no-trail", action="store_true", help="skip the satellite-trail search (faster)")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    reports = run(a.images, a.labels, not a.no_artifacts, not a.no_trail)
    print()
    print(format_table(reports))
    if a.json:
        a.json.write_text(json.dumps([asdict(r) for r in reports], indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
