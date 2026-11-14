"""Stage 6 整合：把每張 frame 變形到參考座標、正規化、加權剔除整合。

影像依列切成一段一段（band），每段互相獨立：
- CPU：多個行程各自處理不同的段（每個行程自己讀 memmap、寫回輸出 memmap）
- GPU：主行程依序處理每一段，變形與整合都在顯示卡上，讀檔用執行緒預先載入
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backend import CPU, Backend
from .combine import HIGH, LOW, OUT_OF_BOUNDS, clip_combine
from .debayer import bilinear
from .i18n import Msg
from .localnorm import eval_offset

WARP_MARGIN = 8


@dataclass
class FrameJob:
    """送進工作行程的精簡資料（可 pickle）。"""
    cal_path: str
    transform: np.ndarray
    scale: list[float]
    offset: list[float]
    weight: float
    bayer: str | None = None          # 這張自己的 Bayer 排列；None = 與輸出相同（合併不同相機時才會不同）
    local: list | None = None         # local normalization：每通道一組多項式係數（參考座標），取代 offset
    distortion: dict | None = None    # 畸變修正（register.fit_distortion）；None = 只有 transform
    noise: float = 0.0                # 這張每個像素的背景雜訊（校正後的單位）；內插限幅的門檻用，0 = 從資料估

    @property
    def margin(self) -> int:
        """畸變修正讓取樣位置偏離仿射變換的最大量（像素，留一點餘量）：讀檔範圍要多讀這麼多。"""
        return int(np.ceil(self.distortion["max"] * 1.5)) + 1 if self.distortion else 0

    def pattern(self, default: str | None) -> str | None:
        return self.bayer or default

    def offset_at(self, xp, c: int, x, y, shape: tuple[int, int]):
        """通道 c 在參考座標 (x, y) 的 offset。"""
        if self.local:
            return eval_offset(xp, self.local[c], x, y, shape).astype(xp.float32)
        return self.offset[c]


@dataclass
class BandSpec:
    bayer: str | None
    shape: tuple[int, int]
    rejection: str
    low: float
    high: float
    master_path: str                  # (C, H, W) float32 .npy
    map_paths: list[str] | None       # 每張 (H, W) uint8 .npy
    clamp: bool = True                # 內插結果不低於來源鄰近像素的最小值（去掉亮星旁的暗環）


def _patch_bounds(inv: np.ndarray, r0: int, r1: int, width: int, shape: tuple[int, int], margin: int = 0):
    corners = np.array([[0, r0, 1], [width - 1, r0, 1], [0, r1 - 1, 1], [width - 1, r1 - 1, 1]], float)
    src = corners @ inv.T
    h, w = shape
    margin += WARP_MARGIN
    x0 = int(np.floor(src[:, 0].min())) - margin
    x1 = int(np.ceil(src[:, 0].max())) + margin + 1
    y0 = int(np.floor(src[:, 1].min())) - margin
    y1 = int(np.ceil(src[:, 1].max())) + margin + 1
    x0, y0 = max(0, x0 - x0 % 2), max(0, y0 - y0 % 2)  # 對齊 Bayer 格
    return x0, y0, min(w, x1), min(h, y1)


def read_patch(job: FrameJob, cal: np.ndarray, r0: int, r1: int, width: int):
    """從校正後的 memmap 讀出這段需要的區域；不在影像內就回傳 None。"""
    inv = np.linalg.inv(job.transform)
    x0, y0, x1, y1 = _patch_bounds(inv, r0, r1, width, cal.shape, job.margin)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return np.array(cal[y0:y1, x0:x1], np.float32), x0, y0


CLAMP_CONTRAST = 8.0  # 附近（約 10×10 像素）的最大 − 最小超過雜訊的這個倍數才限幅（亮星、熱像素；只有背景雜訊時約 5 倍）
CLAMP_WINDOW = 5      # 半解析度上的視窗邊長


def _clamp_floor(xp, ndi, chan, noise: float = 0.0):
    """內插限幅用的下限（來源座標）：亮星、殘留的熱像素附近是 3×3 鄰近像素的最小值，其他地方不限。

    不能整張都限：背景的雜訊本來就會讓內插值略低於鄰近像素，全部墊高會讓背景偏亮，
    解馬賽克後鄰近像素很像時尤其明顯（實測剔除比例會多一倍）。
    noise：這張的背景雜訊（找星時量的，整張一個值：分段整合時每一段的門檻都一樣，結果才與怎麼分段無關）；
    沒給時用隔兩格的差估計（不受梯度與星雲影響）。"""
    if noise <= 0:
        d = chan[::2, 2::4] - chan[::2, :-2:4]
        noise = 1.4826 * float(xp.median(xp.abs(d))) / 1.4142
    # 3×3 最小值用切片做（與 minimum_filter(size=3, mode="nearest") 相同，快約 4 倍）
    p = xp.pad(chan, 1, mode="edge")
    r = xp.minimum(xp.minimum(p[:-2], p[1:-1]), p[2:])
    lo = xp.minimum(xp.minimum(r[:, :-2], r[:, 1:-1]), r[:, 2:])
    # 哪裡要限幅：過衝的起伏離亮星每遠一格約剩四分之一，星點外 4 格內都要管，再遠就低於雜訊了。
    # 對比在 2×2 合併後的半解析度上看（5×5 格 = 原本的 10×10），只要四分之一的運算；整張都用原解析度會讓變形慢七成
    h, w = chan.shape
    h2, w2 = h // 2, w // 2
    even, odd = chan[0:h2 * 2:2, : w2 * 2], chan[1:h2 * 2:2, : w2 * 2]
    top, bottom = xp.maximum(even, odd), xp.minimum(even, odd)
    wide = (ndi.maximum_filter(xp.maximum(top[:, 0::2], top[:, 1::2]), size=CLAMP_WINDOW, mode="nearest")
            - ndi.minimum_filter(xp.minimum(bottom[:, 0::2], bottom[:, 1::2]), size=CLAMP_WINDOW, mode="nearest"))
    near = xp.repeat(xp.repeat(wide > CLAMP_CONTRAST * max(noise, 1e-6), 2, axis=0), 2, axis=1)
    near = xp.pad(near, ((0, h - h2 * 2), (0, w - w2 * 2)), mode="edge")
    return xp.where(near, lo, -xp.inf).astype(xp.float32)


def warp_band(be: Backend, job: FrameJob, patch, bayer: str | None, r0: int, r1: int, width: int,
              shape: tuple[int, int] | None = None, clamp: bool = True):
    """把一張 frame 在參考座標第 r0~r1 列的部分算出來（已正規化），回傳 (C, rows, width)，範圍外為 NaN。
    shape：參考座標的尺寸（local normalization 的座標換算用）。

    clamp：三次樣條在很陡的地方（欠取樣的星、飽和星的邊緣、殘留的熱像素）會衝到比周圍還低，疊起來是亮星旁一圈暗環。
    對比很大的地方，內插結果不低於來源 3×3 鄰近像素的最小值（`_clamp_floor`）；只限低的那一側，峰值不會被削。"""
    xp, ndi = be.xp, be.ndi
    nch = 3 if bayer else 1
    rows = r1 - r0
    if patch is None:
        return xp.full((nch, rows, width), xp.nan, xp.float32)
    data, x0, y0 = patch
    data = xp.asarray(data)
    chans = bilinear(data, bayer, xp, ndi) if bayer else data[None]
    a = np.linalg.inv(job.transform)
    matrix = xp.asarray([[a[1, 1], a[1, 0]], [a[0, 1], a[0, 0]]])
    off = (a[1, 1] * r0 + a[1, 2] - y0, a[0, 1] * r0 + a[0, 2] - x0)
    ones = xp.ones(chans.shape[1:], xp.float32)
    xx = xp.arange(width, dtype=xp.float32)[None, :]
    yy = xp.arange(r0, r1, dtype=xp.float32)[:, None]
    if job.distortion:
        # 有畸變修正：每個輸出像素的取樣位置 = 仿射反變換 + 多項式修正，逐點取樣（比仿射慢，只有這些 frame 這樣做）
        d = job.distortion
        sx = a[0, 0] * xx + a[0, 1] * yy + (a[0, 2] - x0) + eval_offset(xp, d["inv"][0], xx, yy, d["ref_shape"])
        sy = a[1, 0] * xx + a[1, 1] * yy + (a[1, 2] - y0) + eval_offset(xp, d["inv"][1], xx, yy, d["ref_shape"])
        coords = xp.stack([sy.astype(xp.float32), sx.astype(xp.float32)])

        def resample(img, order, **kw):
            return ndi.map_coordinates(img, coords, order=order, **kw)
    else:
        def resample(img, order, **kw):
            return ndi.affine_transform(img, matrix, off, output_shape=(rows, width), order=order, **kw)
    valid = resample(ones, 1, mode="constant", cval=0.0) > 0.999
    out = xp.empty((nch, rows, width), xp.float32)
    for c in range(nch):
        warped = resample(chans[c], 3, mode="nearest", prefilter=True)
        if clamp:
            warped = xp.maximum(warped, resample(_clamp_floor(xp, ndi, chans[c], job.noise), 0, mode="nearest"))
        offset = job.offset_at(xp, c, xx, yy, shape or (r1, width))
        out[c] = xp.where(valid, warped * job.scale[c] + offset, xp.nan)
    return out


def combine_band(be: Backend, stack, weights: np.ndarray, spec: BandSpec):
    """stack：(N, C, rows, W)。回傳（master (C, rows, W), 剔除代碼 (N, rows, W)）皆為 xp 陣列。"""
    xp = be.xp
    n, nch = stack.shape[:2]
    master = xp.empty(stack.shape[1:], xp.float32)
    codes_all = xp.zeros((n,) + tuple(stack.shape[2:]), xp.uint8)
    for c in range(nch):
        master[c], codes = clip_combine(stack[:, c], weights, spec.low, spec.high, spec.rejection, xp=xp)
        codes_all = xp.maximum(codes_all, codes)
    return master, codes_all


def _write_band(spec: BandSpec, r0: int, master: np.ndarray, codes: np.ndarray):
    """寫進輸出 memmap，回傳（覆蓋張數 (rows, W), 每張 [剔除數, 有效數]）。"""
    r1 = r0 + master.shape[1]
    out = np.load(spec.master_path, mmap_mode="r+")
    out[:, r0:r1] = master
    out.flush()
    del out
    if spec.map_paths:
        for i, p in enumerate(spec.map_paths):
            m = np.load(p, mmap_mode="r+")
            m[r0:r1] = codes[i]
            m.flush()
            del m
    inb = codes != OUT_OF_BOUNDS
    rej = ((codes == LOW) | (codes == HIGH)).sum(axis=(1, 2))
    return inb.sum(axis=0).astype(np.uint16), np.column_stack([rej, inb.sum(axis=(1, 2))])


# ---------- CPU：多行程 ----------

_W: dict = {}


def _init_worker(jobs: list[FrameJob], weights: np.ndarray, spec: BandSpec) -> None:
    _W["jobs"], _W["weights"], _W["spec"] = jobs, weights, spec
    _W["cals"] = [np.load(j.cal_path, mmap_mode="r") for j in jobs]


def _cpu_band(band: tuple[int, int]):
    r0, r1 = band
    jobs, spec = _W["jobs"], _W["spec"]
    w = spec.shape[1]
    stack = np.stack([warp_band(CPU, j, read_patch(j, c, r0, r1, w), j.pattern(spec.bayer), r0, r1, w, spec.shape,
                                spec.clamp)
                      for j, c in zip(jobs, _W["cals"])])
    master, codes = combine_band(CPU, stack, _W["weights"], spec)
    cov, counts = _write_band(spec, r0, master, codes)
    return r0, cov, counts


def integrate(be: Backend, jobs: list[FrameJob], spec: BandSpec, workers: int, memory_mb: int, log, progress=None):
    """回傳（覆蓋張數 (H, W) uint16, 每張 [剔除數, 有效數] (N, 2)）；master 與剔除遮罩寫在 spec 指定的檔案。"""
    h, w = spec.shape
    n = len(jobs)
    nch = 3 if spec.bayer else 1
    np.lib.format.open_memmap(spec.master_path, mode="w+", dtype=np.float32, shape=(nch, h, w)).flush()
    if spec.map_paths:
        for p in spec.map_paths:
            np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(h, w)).flush()
    weights = np.array([j.weight for j in jobs], np.float32)
    coverage = np.zeros((h, w), np.uint16)
    counts = np.zeros((n, 2), np.int64)
    bytes_per_row = n * nch * w * 4 * 4  # 堆疊 + 排序 + 暫存

    if be.gpu:
        budget = min(memory_mb * 2 ** 20 * 4, int((be.memory_bytes() or 0) * 0.45))
        band = int(max(8, min(h, budget // bytes_per_row)))
        log(Msg("msg.integrate_gpu", device=be.device, band=band))
        cals = [np.load(j.cal_path, mmap_mode="r") for j in jobs]
        bands = [(r0, min(h, r0 + band)) for r0 in range(0, h, band)]
        with ThreadPoolExecutor(max_workers=max(2, workers)) as io:
            def fetch(b):
                return [io.submit(read_patch, j, c, b[0], b[1], w) for j, c in zip(jobs, cals)]
            pending = fetch(bands[0])
            for k, (r0, r1) in enumerate(bands):
                patches = [f.result() for f in pending]
                if k + 1 < len(bands):
                    pending = fetch(bands[k + 1])  # 算這段時先讀下一段
                stack = be.xp.stack([warp_band(be, j, p, j.pattern(spec.bayer), r0, r1, w, spec.shape, spec.clamp)
                                     for j, p in zip(jobs, patches)])
                master, codes = combine_band(be, stack, weights, spec)
                del stack
                cov, cnt = _write_band(spec, r0, be.to_numpy(master), be.to_numpy(codes))
                coverage[r0:r1], counts = cov, counts + cnt
                be.free()
                log(Msg("msg.integrate_rows", done=r1, total=h))
                if progress:
                    progress(r1, h)
        return coverage, counts

    band = int(max(8, min(h, memory_mb * 2 ** 20 // max(1, workers) // bytes_per_row)))
    bands = [(r0, min(h, r0 + band)) for r0 in range(0, h, band)]
    log(Msg("msg.integrate_cpu", workers=workers, band=band, bands=len(bands)))
    if workers <= 1 or len(bands) <= 1:  # 只有 1 段：開行程池比算還久
        _init_worker(jobs, weights, spec)
        results = map(_cpu_band, bands)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(bands)), initializer=_init_worker,
                                   initargs=(jobs, weights, spec))
        results = pool.map(_cpu_band, bands)
    try:
        done = 0
        for r0, cov, cnt in results:
            coverage[r0:r0 + cov.shape[0]] = cov
            counts += cnt
            done += cov.shape[0]
            log(Msg("msg.integrate_rows", done=done, total=h))
            if progress:
                progress(done, h)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        _W.clear()
    return coverage, counts


def load_master(path: str | Path, bayer: str | None) -> np.ndarray:
    m = np.load(path)
    return m if bayer else m[0]
