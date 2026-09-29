"""Stage 1（master calibration）與 Stage 2（校正 + cosmetic correction），全部在 CFA 原始資料上進行。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, uniform_filter

from .combine import clip_combine
from .i18n import Msg
from .imageio import cfa_masks, load_image, write_fits

ALGO_VERSION = "calib-1"
HOT_SIGMA = 5.0
# 熱像素：3×3（同色）內中心那格佔的比例；星點再小也會分到周圍（比照 APU Pick）
HOT_CONCENTRATION = 0.7
COLD_FLAT = 0.5  # 正規化後的 flat 低於此值視為壞像素


def _mad_sigma(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def _cache_key(kind: str, paths: list[Path], extra: dict) -> str:
    h = hashlib.sha256(f"{ALGO_VERSION}|{kind}|{json.dumps(extra, sort_keys=True)}".encode())
    for p in paths:
        st = p.stat()
        h.update(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _combine_files(paths: list[Path], rows_per_band: int = 256, preprocess=None) -> np.ndarray:
    """把一批 calibration frame 用 winsorized sigma clip 整合；先轉成暫存 memmap，再分帶整合避免吃光記憶體。"""
    first, _ = load_image(paths[0])
    h, w = first.shape
    tmp = np.lib.format.open_memmap(_tmp_path(paths[0]), mode="w+", dtype=np.float32, shape=(len(paths), h, w))
    try:
        for i, p in enumerate(paths):
            data = first if i == 0 else load_image(p)[0]
            if data.shape != (h, w):
                raise ValueError(Msg("msg.size_mismatch", name=p.name, shape=data.shape, first=paths[0].name, first_shape=(h, w)))
            tmp[i] = preprocess(data) if preprocess else data
        out = np.empty((h, w), np.float32)
        for r in range(0, h, rows_per_band):
            out[r:r + rows_per_band], _ = clip_combine(np.asarray(tmp[:, r:r + rows_per_band]), low=3.0, high=3.0)
        return out
    finally:
        del tmp
        _tmp_path(paths[0]).unlink(missing_ok=True)


def _tmp_path(p: Path) -> Path:
    import tempfile

    return Path(tempfile.gettempdir()) / f"apu_photons_{abs(hash(str(p)))}.npy"


def normalize_flat(flat: np.ndarray, bayer: str | None) -> np.ndarray:
    """每個顏色各自除以自己的中位數（CFA 各通道的響應不同）。"""
    out = flat.astype(np.float32, copy=True)
    if bayer:
        for m in cfa_masks(bayer, flat.shape).values():
            out[m] /= np.median(flat[m])
    else:
        out /= np.median(flat)
    return out


class Masters:
    """master bias / dark / flat（flat 已正規化）與壞像素遮罩。"""

    def __init__(self, bias=None, dark=None, flat=None, bad=None):
        self.bias, self.dark, self.flat, self.bad = bias, dark, flat, bad

    @property
    def any(self) -> bool:
        return any(m is not None for m in (self.bias, self.dark, self.flat))

    def describe(self) -> dict:
        return {k: getattr(self, k) is not None for k in ("bias", "dark", "flat")}


def build_masters(cal, bayer: str | None, cache: Path, warnings: list[str], log) -> Masters:
    """Stage 1。cal: model.Calibration。結果以 cache key 存在 cache/masters。"""
    mdir = cache / "masters"
    mdir.mkdir(parents=True, exist_ok=True)

    def master(kind: str, paths: list[Path], extra: dict | None = None, preprocess=None):
        if not paths:
            return None
        key = _cache_key(kind, paths, extra or {})
        path = mdir / f"master_{kind}_{key}.fits"
        if path.is_file():
            log(Msg("msg.master_cached", kind=kind, name=path.name))
            return fits.getdata(path).astype(np.float32)
        log(Msg("msg.master_combine", kind=kind, n=len(paths)))
        data = _combine_files(paths, preprocess=preprocess)
        hdr = fits.Header()
        hdr["IMAGETYP"] = f"MASTER {kind.upper()}"
        hdr["NCOMBINE"] = len(paths)
        write_fits(path, data, hdr)
        cal.masters[kind] = path
        return data

    bias = master("bias", cal.bias)
    dark = master("dark", cal.dark)
    flat_dark = master("flat_dark", cal.flat_dark)
    flat = None
    if cal.flat:
        sub = flat_dark if flat_dark is not None else bias
        if sub is None:
            warnings.append(Msg("msg.flat_no_sub"))
        flat_raw = master("flat", cal.flat, {"sub": None if sub is None else float(np.mean(sub))},
                          preprocess=(lambda d: d - sub) if sub is not None else None)
        flat = normalize_flat(flat_raw, bayer)

    if dark is None and bias is None:
        warnings.append(Msg("msg.no_dark_bias"))
    if flat is None:
        warnings.append(Msg("msg.no_flat"))

    bad = None
    if dark is not None:
        med, sig = float(np.median(dark)), _mad_sigma(dark[::4, ::4])
        bad = dark > med + HOT_SIGMA * max(sig, 1e-6)
    if flat is not None:
        cold = flat < COLD_FLAT
        bad = cold if bad is None else (bad | cold)
    if bad is not None:
        log(Msg("msg.bad_pixels", n=int(bad.sum())))
    return Masters(bias=bias, dark=dark, flat=flat, bad=bad)


def _plane_views(shape, bayer):
    """CFA 的四個子平面（每個都是單一顏色），單色時就是整張。"""
    if not bayer:
        return [(slice(None), slice(None))]
    return [(slice(dy, None, 2), slice(dx, None, 2)) for dy in (0, 1) for dx in (0, 1)]


def cosmetic(data: np.ndarray, bayer: str | None, bad: np.ndarray | None) -> tuple[np.ndarray, int]:
    """把 master 標出的壞像素與單張偵測到的熱像素，換成同色鄰近像素的中位數。"""
    out = data
    fixed = 0
    for sy, sx in _plane_views(data.shape, bayer):
        plane = out[sy, sx]
        med = median_filter(plane, size=3, mode="reflect")
        diff = plane - med
        sig = _mad_sigma(diff[::2, ::2])
        box = uniform_filter(np.clip(plane - np.median(med), 0, None), size=3, mode="reflect") * 9.0
        with np.errstate(all="ignore"):
            conc = np.clip(plane - np.median(med), 0, None) / box
        hot = (diff > HOT_SIGMA * max(sig, 1e-6)) & (conc > HOT_CONCENTRATION)
        if bad is not None:
            hot |= bad[sy, sx]
        plane[hot] = med[hot]
        fixed += int(hot.sum())
    return out, fixed


def calibrate(data: np.ndarray, masters: Masters, bayer: str | None) -> tuple[np.ndarray, int]:
    """Stage 2：(light − dark 或 bias) / flat，再做 cosmetic correction。輸出 float32。"""
    out = data.astype(np.float32, copy=True)
    if masters.dark is not None:
        out -= masters.dark
    elif masters.bias is not None:
        out -= masters.bias
    if masters.flat is not None:
        with np.errstate(all="ignore"):
            out /= np.where(masters.flat > 0.05, masters.flat, 1.0)
    return cosmetic(out, bayer, masters.bad)
