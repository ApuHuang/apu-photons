"""Stage 7：Drizzle（SPEC §3 Stage 7）。

- OSC 用 CFA (Bayer) drizzle：不解馬賽克，每個 CFA 像素依自己的顏色投進 R / G / B 輸出網格。
- 單色就是一般的 drizzle。
- 沿用 Stage 4 的正規化、Stage 5 的權重、Stage 6 的剔除遮罩（參考座標）；本身不做 rejection。
- drop 以「軸對齊正方形」近似：面積正確，旋轉角小（或 180° 中天翻轉）時與真正的旋轉正方形一致。
  大角度旋轉時邊緣分配會略有差異，對結果影響很小，換來可以完全向量化。

輸出依列切成條帶（strip），每條只讀各 frame 會落進這條的區域，彼此獨立：
CPU 用多個行程分條處理，GPU 用同一份程式（xp = cupy）依序處理。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .backend import CPU, Backend
from .combine import HIGH, LOW
from .integrate import FrameJob

ALGO_VERSION = "drizzle-1"


@dataclass
class DrizzleResult:
    image: np.ndarray    # (C, Ho, Wo)，沒有資料的地方是 NaN
    weight: np.ndarray   # (C, Ho, Wo) 權重總和（覆蓋率）
    holes: list[float]   # 每個通道沒有資料的比例


@dataclass
class DrizzleSpec:
    bayer: str | None
    shape: tuple[int, int]
    scale: int
    pixfrac: float
    map_paths: list[str] | None


def _chan_lut(bayer: str | None) -> np.ndarray:
    """(2, 2) 查表：CFA 位置 → 通道（R=0, G=1, B=2）。"""
    lut = np.zeros((2, 2), np.int64)
    if bayer:
        for i, color in enumerate(bayer):
            lut[i // 2, i % 2] = "RGB".index(color)
    return lut


def _input_bounds(t: np.ndarray, q0: int, q1: int, spec: DrizzleSpec, hw: float):
    """輸出第 q0~q1 列會用到的輸入區域（輸入座標，含 drop 半徑與邊界餘量）。"""
    s = spec.scale
    h, w = spec.shape
    # 輸出列 → 參考座標的 y 範圍
    ry0 = (q0 - 0.5 - hw) / s - 0.5 + 0.5 / s
    ry1 = (q1 - 0.5 + hw) / s - 0.5 + 0.5 / s
    inv = np.linalg.inv(t)
    corners = np.array([[-1, ry0, 1], [w, ry0, 1], [-1, ry1, 1], [w, ry1, 1]], float) @ inv.T
    y0 = max(0, int(np.floor(corners[:, 1].min())) - 2)
    y1 = min(h, int(np.ceil(corners[:, 1].max())) + 3)
    x0 = max(0, int(np.floor(corners[:, 0].min())) - 2)
    x1 = min(w, int(np.ceil(corners[:, 0].max())) + 3)
    return y0 - y0 % 2, y1, x0 - x0 % 2, x1


def drizzle_strip(be: Backend, jobs: list[FrameJob], cals, rmaps, spec: DrizzleSpec, q0: int, q1: int):
    """算輸出第 q0~q1 列：回傳（加權和, 權重和），皆為 (C, q1-q0, Wo) 的 xp 陣列。"""
    xp = be.xp
    h, w = spec.shape
    s = spec.scale
    wo = w * s
    rows = q1 - q0
    nch = 3 if spec.bayer else 1
    npix = rows * wo
    acc = xp.zeros(nch * npix, xp.float32)
    wsum = xp.zeros(nch * npix, xp.float32)
    lut = xp.asarray(_chan_lut(spec.bayer))
    for k, (job, cal) in enumerate(zip(jobs, cals)):
        t = job.transform
        hw = spec.pixfrac * float(np.sqrt(abs(np.linalg.det(t[:2, :2])))) * s / 2.0  # drop 半寬（輸出像素）
        reach = int(np.ceil(hw + 0.5))
        y0, y1, x0, x1 = _input_bounds(t, q0, q1, spec, hw)
        if y1 - y0 < 1 or x1 - x0 < 1:
            continue
        v = xp.asarray(np.array(cal[y0:y1, x0:x1], np.float32))
        yy = xp.arange(y0, y1, dtype=xp.float32)[:, None]
        xx = xp.arange(x0, x1, dtype=xp.float32)[None, :]
        ch = lut[(xp.arange(y0, y1) % 2)[:, None], (xp.arange(x0, x1) % 2)[None, :]]
        v = v * job.scale[0] + xp.asarray(np.asarray(job.offset, np.float32))[ch]
        xr = t[0, 0] * xx + t[0, 1] * yy + t[0, 2]
        yr = t[1, 0] * xx + t[1, 1] * yy + t[1, 2]
        keep = xp.isfinite(v)
        if rmaps is not None and rmaps[k] is not None:
            xi = xp.rint(xr).astype(xp.int64)
            yi = xp.rint(yr).astype(xp.int64)
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            # 剔除遮罩只讀這些像素落到的參考座標範圍
            ry0 = int(xp.clip(yi, 0, h - 1).min())
            ry1 = int(xp.clip(yi, 0, h - 1).max()) + 1
            rmap = xp.asarray(np.array(rmaps[k][ry0:ry1]))
            code = xp.zeros(v.shape, xp.uint8)
            code[inside] = rmap[yi[inside] - ry0, xi[inside]]
            keep &= (code != LOW) & (code != HIGH)
        u = ((xr + 0.5) * s - 0.5)[keep]
        q = ((yr + 0.5) * s - 0.5)[keep] - q0
        val = v[keep]
        chk = ch[keep]
        ju0 = xp.floor(u + 0.5).astype(xp.int64)
        jq0 = xp.floor(q + 0.5).astype(xp.int64)
        wgt = float(job.weight)
        for dy in range(-reach, reach + 1):
            jq = jq0 + dy
            oy = xp.minimum(q + hw, jq + 0.5) - xp.maximum(q - hw, jq - 0.5)
            oky = (oy > 0) & (jq >= 0) & (jq < rows)
            if not bool(oky.any()):
                continue
            for dx in range(-reach, reach + 1):
                ju = ju0 + dx
                ox = xp.minimum(u + hw, ju + 0.5) - xp.maximum(u - hw, ju - 0.5)
                ok = oky & (ox > 0) & (ju >= 0) & (ju < wo)
                a = (ox[ok] * oy[ok] * wgt).astype(xp.float64)
                idx = chk[ok] * npix + jq[ok] * wo + ju[ok]
                if idx.size:
                    acc += xp.bincount(idx, weights=a * val[ok], minlength=nch * npix).astype(xp.float32)
                    wsum += xp.bincount(idx, weights=a, minlength=nch * npix).astype(xp.float32)
    return acc.reshape(nch, rows, wo), wsum.reshape(nch, rows, wo)


# ---------- CPU：多行程 ----------

_W: dict = {}


def _init_worker(jobs, spec: DrizzleSpec, img_path: str, wgt_path: str) -> None:
    _W.update(jobs=jobs, spec=spec, img=img_path, wgt=wgt_path,
              cals=[np.load(j.cal_path, mmap_mode="r") for j in jobs],
              rmaps=[np.load(p, mmap_mode="r") for p in spec.map_paths] if spec.map_paths else None)


def _cpu_strip(strip):
    q0, q1 = strip
    acc, wsum = drizzle_strip(CPU, _W["jobs"], _W["cals"], _W["rmaps"], _W["spec"], q0, q1)
    _store(_W["img"], _W["wgt"], q0, acc, wsum)
    return q1 - q0


def _store(img_path, wgt_path, q0, acc, wsum):
    q1 = q0 + acc.shape[1]
    img = np.load(img_path, mmap_mode="r+")
    wgt = np.load(wgt_path, mmap_mode="r+")
    with np.errstate(all="ignore"):
        img[:, q0:q1] = np.where(wsum > 0, acc / wsum, np.nan)
    wgt[:, q0:q1] = wsum
    img.flush()
    wgt.flush()


def drizzle(be: Backend, jobs: list[FrameJob], spec: DrizzleSpec, workdir, workers: int, memory_mb: int,
            log=None, progress=None) -> DrizzleResult:
    h, w = spec.shape
    ho, wo = h * spec.scale, w * spec.scale
    nch = 3 if spec.bayer else 1
    img_path, wgt_path = str(workdir / "drizzle_image.npy"), str(workdir / "drizzle_weight.npy")
    np.lib.format.open_memmap(img_path, mode="w+", dtype=np.float32, shape=(nch, ho, wo)).flush()
    np.lib.format.open_memmap(wgt_path, mode="w+", dtype=np.float32, shape=(nch, ho, wo)).flush()
    # 每個輸入像素運算時約用 ~150 bytes 暫存；輸入列數約為輸出列數 / scale
    per_out_row = w / spec.scale * 150 + nch * wo * 8
    log = log or (lambda _m: None)

    if be.gpu:
        budget = min(memory_mb * 2 ** 20 * 4, int((be.memory_bytes() or 0) * 0.4))
        rows = int(max(16, min(ho, budget // per_out_row)))
        strips = [(q, min(ho, q + rows)) for q in range(0, ho, rows)]
        log(f"Drizzle（GPU {be.device}），每條 {rows} 列、共 {len(strips)} 條")
        cals = [np.load(j.cal_path, mmap_mode="r") for j in jobs]
        rmaps = [np.load(p, mmap_mode="r") for p in spec.map_paths] if spec.map_paths else None
        for i, (q0, q1) in enumerate(strips, 1):
            acc, wsum = drizzle_strip(be, jobs, cals, rmaps, spec, q0, q1)
            _store(img_path, wgt_path, q0, be.to_numpy(acc), be.to_numpy(wsum))
            del acc, wsum
            be.free()
            log(f"Drizzle {q1}/{ho} 列")
            if progress:
                progress(q1, ho)
    else:
        rows = int(max(16, min(ho, memory_mb * 2 ** 20 // max(1, workers) // per_out_row)))
        # 條數至少是行程數的幾倍，工作量才平均
        rows = min(rows, max(16, -(-ho // (max(1, workers) * 3))))
        strips = [(q, min(ho, q + rows)) for q in range(0, ho, rows)]
        log(f"Drizzle（CPU {workers} 個行程），每條 {rows} 列、共 {len(strips)} 條")
        if workers <= 1:
            _init_worker(jobs, spec, img_path, wgt_path)
            results, pool = map(_cpu_strip, strips), None
        else:
            pool = ProcessPoolExecutor(max_workers=min(workers, len(strips)), initializer=_init_worker,
                                       initargs=(jobs, spec, img_path, wgt_path))
            results = pool.map(_cpu_strip, strips)
        try:
            done = 0
            for n in results:
                done += n
                log(f"Drizzle {done}/{ho} 列")
                if progress:
                    progress(done, ho)
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
            _W.clear()

    img = np.load(img_path)
    wsum = np.load(wgt_path)
    holes = [float((wsum[c] <= 0).mean()) for c in range(nch)]
    return DrizzleResult(img, wsum, holes)


def fill_holes(img: np.ndarray) -> np.ndarray:
    """把沒有資料的像素用周圍 3×3 有資料的平均補上（只補零星的洞；大片空白留 NaN）。"""
    from scipy.ndimage import uniform_filter

    out = img.copy()
    for c in range(out.shape[0]):
        ch = out[c]
        bad = ~np.isfinite(ch)
        if not bad.any():
            continue
        z = np.where(bad, 0.0, ch)
        n = uniform_filter((~bad).astype(np.float32), 3, mode="constant")
        s = uniform_filter(z.astype(np.float32), 3, mode="constant")
        with np.errstate(all="ignore"):
            fill = s / n
        ch[bad & (n > 0)] = fill[bad & (n > 0)]
    return out
