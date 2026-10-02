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

ALGO_VERSION = "calib-2"  # 2：熱像素改用局部中位數判斷
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


def _prepared_master(path: Path, kind: str, log) -> np.ndarray:
    """做好的 master 直接讀；0～1 的浮點 master（Siril、PixInsight 常見）換算成 16-bit ADU。"""
    data, _ = load_image(path)
    if kind != "flat" and np.nanmax(data) <= 1.0 + 1e-3:
        data = data * 65535.0
        log(Msg("msg.master_rescaled", kind=kind, name=path.name))
    log(Msg("msg.master_file", kind=kind, name=path.name))
    return data.astype(np.float32, copy=False)


def hot_from_dark(dark: np.ndarray, bayer: str | None) -> np.ndarray:
    """master dark 的熱像素：比同色鄰近像素的中位數高出 5σ。

    用局部中位數而不是整張的中位數：amp glow 那一側整片偏亮，但 dark 扣掉就沒了，不是壞像素
    （整張門檻在 QHY183M 上會把右側 17% 的像素標成壞像素）。"""
    bad = np.zeros(dark.shape, bool)
    for sy, sx in _plane_views(dark.shape, bayer):
        plane = dark[sy, sx]
        resid = plane - median_filter(plane, size=5, mode="reflect")
        sig = _mad_sigma(resid[::3, ::3])
        bad[sy, sx] = resid > HOT_SIGMA * max(sig, 1e-6)
    return bad


class MasterBuilder:
    """Stage 1：依 Stage 0 配好的校正套建 master；同一套只建一次（也存在 cache/masters 跨次共用）。"""

    def __init__(self, sets: dict, cache: Path, warnings: list, log):
        self.sets, self.warnings, self.log = sets, warnings, log
        self.mdir = cache / "masters"
        self.mdir.mkdir(parents=True, exist_ok=True)
        self._memo: dict = {}

    def _combined(self, sid: str, extra: dict | None = None, preprocess=None) -> np.ndarray:
        s = self.sets[sid]
        if s.is_master:
            return _prepared_master(s.frames[0].path, s.kind, self.log)
        paths = s.paths
        key = _cache_key(s.kind, paths, extra or {})
        path = self.mdir / f"master_{s.kind}_{key}.fits"
        if path.is_file():
            self.log(Msg("msg.master_cached", kind=sid, name=path.name))
            return fits.getdata(path).astype(np.float32)
        self.log(Msg("msg.master_combine", kind=sid, n=len(paths)))
        data = _combine_files(paths, preprocess=preprocess)
        hdr = fits.Header()
        hdr["IMAGETYP"] = f"MASTER {s.kind.upper().replace('_', '')}"
        hdr["NCOMBINE"] = len(paths)
        write_fits(path, data, hdr)
        return data

    def get(self, sid: str | None, sub: str | None = None) -> np.ndarray | None:
        if sid is None:
            return None
        key = (sid, sub)
        if key not in self._memo:
            if self.sets[sid].kind == "flat":
                sub_data = self.get(sub)
                extra = {"sub": None if sub_data is None else [sub, float(np.mean(sub_data))]}
                self._memo[key] = self._combined(sid, extra, (lambda d: d - sub_data) if sub_data is not None else None)
            else:
                self._memo[key] = self._combined(sid)
        return self._memo[key]

    def masters(self, entry: dict, bayer: str | None, shape: tuple[int, int]) -> Masters:
        """一個校正組（Stage 0 的 calib_groups 一筆）的 master；尺寸不合的不用並提醒。"""
        def fit(kind, data, sid):
            if data is not None and data.shape != tuple(shape):
                self.warnings.append(Msg("msg.master_shape", kind=kind, name=sid, shape=data.shape, light=tuple(shape)))
                return None
            return data

        dark = fit("dark", self.get(entry.get("dark")), entry.get("dark"))
        bias = fit("bias", self.get(entry.get("bias")), entry.get("bias"))
        flat_raw = fit("flat", self.get(entry.get("flat"), entry.get("flat_sub")), entry.get("flat"))
        flat = normalize_flat(flat_raw, bayer) if flat_raw is not None else None
        bad = hot_from_dark(dark, bayer) if dark is not None else None
        if flat is not None:
            cold = flat < COLD_FLAT
            bad = cold if bad is None else (bad | cold)
        if bad is not None:
            self.log(Msg("msg.bad_pixels", n=int(bad.sum())))
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
