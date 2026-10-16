"""Stage 1（master calibration）與 Stage 2（校正 + cosmetic correction），全部在 CFA 原始資料上進行。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, uniform_filter

from .backend import CPU, Backend
from .combine import clip_combine
from .i18n import Msg
from .imageio import cfa_masks, load_image, write_fits

ALGO_VERSION = "calib-2"  # 2：熱像素改用局部中位數判斷
HOT_SIGMA = 5.0
# 熱像素：3×3（同色）內中心那格佔的比例；星點再小也會分到周圍（比照 APU Pick）
HOT_CONCENTRATION = 0.7
COLD_FLAT = 0.5  # 正規化後的 flat 低於此值視為壞像素
# CPU 每段最多幾列：小段比較能留在 CPU 快取（NGC1499 40 張 dark、6 核心：256 列 42 秒、32 列以下約 31 秒）
MASTER_ROWS = 32
READ_THREADS = 8    # 讀校正檔的執行緒數


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


def _combine_files(paths: list[Path], preprocess=None, be: Backend = CPU, workers: int = 1, memory_mb: int = 2048,
                   progress=None) -> np.ndarray:
    """把一批 calibration frame 用 winsorized sigma clip 整合。

    先用多個執行緒讀進暫存 memmap，再分段整合：有 GPU 在顯示卡上做，否則分給多個行程（與 Stage 6 整合同一套做法）。
    progress(done, total)：已整合的列數。"""
    first, _ = load_image(paths[0])
    h, w = first.shape
    path = _tmp_path(paths[0])
    tmp = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(paths), h, w))
    try:
        def load(i: int) -> None:
            data = first if i == 0 else load_image(paths[i])[0]
            if data.shape != (h, w):
                raise ValueError(Msg("msg.size_mismatch", name=paths[i].name, shape=data.shape, first=paths[0].name,
                                     first_shape=(h, w)))
            tmp[i] = preprocess(data) if preprocess else data

        with ThreadPoolExecutor(max_workers=max(1, min(READ_THREADS, workers, len(paths)))) as io:
            list(io.map(load, range(len(paths))))
        tmp.flush()
        return _combine_memmap(tmp, path, be, workers, memory_mb, progress)
    finally:
        del tmp
        path.unlink(missing_ok=True)


def _combine_memmap(tmp: np.ndarray, path: Path, be: Backend, workers: int, memory_mb: int, progress) -> np.ndarray:
    n, h, w = tmp.shape
    out = np.empty((h, w), np.float32)
    bytes_per_row = n * w * 4 * 4  # 堆疊 + 排序 + 暫存（同 integrate）
    if be.gpu:
        budget = min(memory_mb * 2 ** 20 * 4, int((be.memory_bytes() or 0) * 0.45))
        band = int(max(8, min(h, budget // bytes_per_row)))
        for r0 in range(0, h, band):
            m, _ = clip_combine(be.xp.asarray(tmp[:, r0:r0 + band]), low=3.0, high=3.0, xp=be.xp)
            out[r0:r0 + band] = be.to_numpy(m)
            del m
            be.free()
            if progress:
                progress(min(h, r0 + band), h)
        return out
    band = int(max(8, min(MASTER_ROWS, memory_mb * 2 ** 20 // max(1, workers) // bytes_per_row)))
    bands = [(r0, min(h, r0 + band)) for r0 in range(0, h, band)]
    if workers <= 1:
        _init_combine(str(path), tmp)
        results = map(_combine_band, bands)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(bands)), initializer=_init_combine,
                                   initargs=(str(path),))
        results = pool.map(_combine_band, bands)
    try:
        done = 0
        for r0, m in results:
            out[r0:r0 + m.shape[0]] = m
            done += m.shape[0]
            if progress:
                progress(done, h)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        _C.clear()
    return out


_C: dict = {}


def _init_combine(path: str, tmp: np.ndarray | None = None) -> None:
    _C["tmp"] = tmp if tmp is not None else np.load(path, mmap_mode="r")


def _combine_band(band: tuple[int, int]) -> tuple[int, np.ndarray]:
    r0, r1 = band
    m, _ = clip_combine(np.asarray(_C["tmp"][:, r0:r1]), low=3.0, high=3.0)
    return r0, m


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


def hot_from_dark(dark: np.ndarray, bayer: str | None, be: Backend = CPU) -> np.ndarray:
    """master dark 的熱像素：比同色鄰近像素的中位數高出 5σ。

    用局部中位數而不是整張的中位數：amp glow 那一側整片偏亮，但 dark 扣掉就沒了，不是壞像素
    （整張門檻在 QHY183M 上會把右側 17% 的像素標成壞像素）。中位數濾波有 GPU 時在顯示卡上做（結果相同）。"""
    bad = np.zeros(dark.shape, bool)
    for sy, sx in _plane_views(dark.shape, bayer):
        plane = dark[sy, sx]
        if be.gpu:
            med = be.to_numpy(be.ndi.median_filter(be.xp.asarray(plane), size=5, mode="reflect"))
            be.free()
        else:
            med = median_filter(plane, size=5, mode="reflect")
        resid = plane - med
        sig = _mad_sigma(resid[::3, ::3])
        bad[sy, sx] = resid > HOT_SIGMA * max(sig, 1e-6)
    return bad


class MasterBuilder:
    """Stage 1：依 Stage 0 配好的校正套建 master；同一套只建一次（也存在 cache/masters 跨次共用）。"""

    def __init__(self, sets: dict, cache: Path, warnings: list, log, be: Backend = CPU, workers: int = 1,
                 memory_mb: int = 2048, progress=None):
        self.sets, self.warnings, self.log = sets, warnings, log
        self.be, self.workers, self.memory_mb, self.progress = be, workers, memory_mb, progress
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
        data = _combine_files(paths, preprocess=preprocess, be=self.be, workers=self.workers,
                              memory_mb=self.memory_mb, progress=self.progress)
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
        bad = hot_from_dark(dark, bayer, self.be) if dark is not None else None
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


# 9 個數取中位數的比較網路（每一步把一對排成小、大）
_MED9 = ((1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3), (5, 8), (4, 7),
         (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2))


def median3(plane: np.ndarray) -> np.ndarray:
    """3×3 中位數濾波，邊緣照 scipy 的 reflect。結果與 `median_filter(size=3, mode="reflect")` 完全相同
    （取到的是同一個值），但快約 4 倍：每張都要做，Seestar 一次幾百張、單色大圖每張兩千萬像素時很明顯。"""
    if not np.isfinite(plane).all():
        return median_filter(plane, size=3, mode="reflect")  # 有 NaN 時比較網路的結果會不同，照舊
    h, w = plane.shape
    pad = np.pad(plane, 1, mode="symmetric")  # numpy 的 symmetric 就是 scipy 的 reflect
    p = [pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    for a, b in _MED9:
        p[a], p[b] = np.minimum(p[a], p[b]), np.maximum(p[a], p[b])
    return p[4].astype(plane.dtype, copy=False)


def cosmetic(data: np.ndarray, bayer: str | None, bad: np.ndarray | None) -> tuple[np.ndarray, int]:
    """把 master 標出的壞像素與單張偵測到的熱像素，換成同色鄰近像素的中位數。"""
    out = data
    fixed = 0
    for sy, sx in _plane_views(data.shape, bayer):
        plane = out[sy, sx]
        med = median3(plane)
        diff = plane - med
        sig = _mad_sigma(diff[::2, ::2])
        level = np.median(med)
        box = uniform_filter(np.clip(plane - level, 0, None), size=3, mode="reflect") * 9.0
        with np.errstate(all="ignore"):
            conc = np.clip(plane - level, 0, None) / box
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
