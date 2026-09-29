"""把 Stage 0~8 串起來（SPEC §3）。Drizzle（Stage 7）尚未實作。

記憶體策略：校正後的 frame 存成 float32 .npy（memmap 讀取），整合時一次只處理一段列（band），
每段只從各 frame 讀出需要的區域、就地解馬賽克與變形，所以幾百張大圖也不會一次載入。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import __version__, backend
from . import drizzle as drizzle_mod
from . import integrate as integrate_mod
from .backend import default_workers
from .calibration import ALGO_VERSION as CALIB_VERSION, build_masters
from .drizzle import DrizzleSpec
from .imageio import cfa_masks, write_fits
from .ingest import ingest
from .integrate import BandSpec, FrameJob
from .model import ACCEPTED, Calibration, Frame, Project
from .prepare import prepare
from .register import ALGO_VERSION as REG_VERSION, RegistrationError, register
MIN_STARS = 10


@dataclass
class Settings:
    rejection: str = "winsorized"      # winsorized / sigma / average / median
    low: float = 4.0
    high: float = 3.0
    weighting: str = "snr2_over_fwhm2"  # snr2_over_fwhm2 / none
    pick_boost: bool = False
    reference: str | None = None       # 檔名；None = 自動
    downsample: float = 1.0            # 1.0 或 0.5
    crop_common: bool = True
    output_bits: int = 32
    memory_mb: int = 2048
    split_nights: bool = True
    keep_rejection_maps: bool = True
    preview: int = 0                   # >0：只抽樣這麼多張試跑
    drizzle: int = 0                   # 0 = 不做；1 或 2 = 輸出倍率
    pixfrac: float = 0.9
    fill_holes: bool = True            # 零星沒資料的像素用鄰近平均補
    gpu: str = "auto"                  # auto / gpu / cpu：整合與 drizzle 用的運算後端
    workers: int = 0                   # CPU 平行處理數；0 = 自動（核心數 − 1）


@dataclass
class Result:
    output: Path
    recipe: Path
    project: Project
    warnings: list[str] = field(default_factory=list)
    qc: dict = field(default_factory=dict)


def _log_to(path: Path, echo):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        fh.write(line + "\n")
        fh.flush()
        if echo:
            echo(line)
    return log, fh


def _calib_key(f: Frame, cal: Calibration) -> str:
    st = f.path.stat()
    h = hashlib.sha256(f"{CALIB_VERSION}|{f.path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    for k in sorted(cal.masters):
        h.update(f"{k}={cal.masters[k].name}".encode())
    return h.hexdigest()[:20]


def _sample(frames: list[Frame], n: int) -> list[Frame]:
    if n <= 0 or n >= len(frames):
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in sorted(set(idx))]


class Cancelled(RuntimeError):
    """使用者按了停止。"""


# 進度回報的階段代號（視窗依此顯示當下語言的文字）
STAGES = ("ingest", "masters", "prepare", "register", "integrate", "drizzle", "output")


def run(light_dirs: list[Path], calibration: Calibration, output: Path, settings: Settings | None = None,
        cache: Path | None = None, echo=print, progress=None, cancel=None) -> Result:
    """progress(stage, done, total)：進度回報；cancel：threading.Event，設定後在下一個檢查點丟出 Cancelled。"""
    s = settings or Settings()

    def report(stage: str, done: int = 0, total: int = 0) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress is not None:
            progress(stage, done, total)
    t0 = time.time()
    cache = cache or output.parent / ".photons_cache"
    cache.mkdir(parents=True, exist_ok=True)
    log, log_fh = _log_to(output.with_suffix(".log"), echo)
    warnings: list[str] = []
    try:
        return _run(light_dirs, calibration, output, s, cache, log, warnings, t0, report)
    finally:
        log_fh.close()


def _run(light_dirs, calibration, output, s, cache, log, warnings, t0, report) -> Result:
    log(f"APU Photons {__version__}")
    report("ingest")
    if s.drizzle not in (0, 1, 2):
        raise RuntimeError("Drizzle 倍率目前只支援 1× 或 2×")
    if s.drizzle and s.downsample != 1.0:
        warnings.append("Drizzle 與 Downsample 0.5× 不能同時使用，這次不做 Downsample")
        s.downsample = 1.0
    # ---- Stage 0 ----
    project = ingest(light_dirs, calibration, warnings, name=output.stem, split_nights=s.split_nights)
    frames = [f for f in project.frames if f.accepted]
    if s.preview:
        chosen = set(map(id, _sample(frames, s.preview)))
        for f in frames:
            if id(f) not in chosen:
                f.reject("not_in_preview")
        frames = [f for f in frames if f.accepted]
        log(f"Preview：抽樣 {len(frames)} 張")
    if not frames:
        raise RuntimeError("沒有可用的 light frame")
    bayer, shape = frames[0].bayer, frames[0].shape
    log(f"Light {len(project.frames)} 張、{len(project.sessions)} 個 session；"
        f"{'OSC ' + bayer if bayer else '單色'}，{shape[1]}×{shape[0]}")
    for sess in project.sessions:
        log(f"  session {sess.id}：{len(sess.frames)} 張，像素尺度 "
            f"{'%.2f″/px' % sess.pixel_scale_arcsec if sess.pixel_scale_arcsec else '未知'}")

    # ---- Stage 1 ----
    report("masters")
    masters = build_masters(calibration, bayer, cache, warnings, log)

    # ---- Stage 2 + Stage 3 偵測 ----
    cdir = cache / "calibrated"
    cdir.mkdir(exist_ok=True)
    workers = s.workers or default_workers()
    be = backend.select(s.gpu)
    log(f"運算：CPU {workers} 個行程" + (f"、GPU {be.device}" if be.gpu else ""))
    tasks = [(str(f.path), str(cdir / f"{_calib_key(f, calibration)}.npy")) for f in frames]
    for f, (src, dst), (stars, err) in zip(frames, tasks, prepare(tasks, masters, bayer, workers, log,
                                                                          lambda d, t: report("prepare", d, t))):
        f.calibrated_path = Path(dst)
        if err is not None:
            f.reject("unreadable")
            warnings.append(f"{f.name}: 校正失敗（{err}）")
            continue
        f.stars = stars
        if len(f.stars) < MIN_STARS:
            f.reject("too_few_stars")
            warnings.append(f"{f.name}: 只偵測到 {len(f.stars)} 顆星，不使用")
    frames = [f for f in frames if f.accepted]
    if not frames:
        raise RuntimeError("所有 frame 都找不到足夠的星點")

    # ---- Stage 3 參考 frame + 對齊 ----
    ref = _choose_reference(frames, s.reference, warnings)
    project.reference = ref
    log(f"參考 frame：{ref.name}（FWHM {ref.stars.fwhm:.2f} px、{len(ref.stars)} 顆星）")
    ref_xy = np.column_stack([ref.stars.x, ref.stars.y])
    flux_ratio: dict[int, float] = {}
    for i, f in enumerate(frames, 1):
        report("register", i, len(frames))
        if f is ref:
            f.transform, f.registration_residual_px = np.eye(3), 0.0
            flux_ratio[id(f)] = 1.0
            continue
        try:
            m, rms, si, ri = register(np.column_stack([f.stars.x, f.stars.y]), ref_xy)
        except RegistrationError as exc:
            f.reject("registration_failed")
            warnings.append(f"{f.name}: 對齊失敗（{exc}）")
            continue
        f.transform, f.registration_residual_px = m, round(rms, 4)
        with np.errstate(all="ignore"):
            r = ref.stars.flux[ri] / f.stars.flux[si]
        r = r[np.isfinite(r) & (r > 0)]
        flux_ratio[id(f)] = float(np.median(r)) if len(r) else 1.0
    frames = [f for f in frames if f.accepted]
    log(f"對齊成功 {len(frames)} 張")

    # ---- Stage 4 Normalize ----
    ref_loc = _channel_locations(np.load(ref.calibrated_path, mmap_mode="r"), bayer)
    for f in frames:
        loc = ref_loc if f is ref else _channel_locations(np.load(f.calibrated_path, mmap_mode="r"), bayer)
        k = flux_ratio[id(f)]
        f.norm = {"scale": [k] * len(loc), "offset": [lr - k * li for lr, li in zip(ref_loc, loc)]}

    # ---- Stage 5 Weight ----
    for f in frames:
        pm = f.pick_metrics
        if pm is not None and pm.verdict == "reject":
            f.reject("pick")
    frames = [f for f in frames if f.accepted]
    if not frames:
        raise RuntimeError("所有 frame 都被 APU Pick 淘汰")
    _weights(frames, s)

    # ---- Stage 6 Integrate ----
    for f in frames:
        f.status = ACCEPTED
    master, coverage, rej_counts, map_paths = _integrate(frames, bayer, shape, s, cache, be, workers, log,
                                                            lambda d, t: report("integrate", d, t))
    for f, (n_rej, n_valid) in zip(frames, rej_counts):
        f.rejected_pixel_fraction = round(n_rej / n_valid, 6) if n_valid else None

    qc = _qc(project, frames, coverage, None)

    # ---- Stage 7 Drizzle ----
    driz = None
    if s.drizzle:
        driz = _run_drizzle(frames, bayer, shape, s, map_paths, qc, warnings, cache, be, workers, log,
                            lambda d, t: report("drizzle", d, t))

    # ---- Stage 8 Post ----
    report("output")
    crop = None
    hdr = _output_header(project, frames, s, bayer)
    if driz is not None:
        stack_out = output.with_name(output.stem + "_stack" + output.suffix)
        std, std_crop = (_crop_common(master, coverage, len(frames)) if s.crop_common else (master, None))
        write_fits(stack_out, std, hdr, bits=s.output_bits)
        log(f"一般疊圖另存 {stack_out.name}")
        img, weight = driz.image, driz.weight
        if s.fill_holes:
            img = drizzle_mod.fill_holes(img)
        if not bayer:
            img, weight = img[0], weight[0]
        if std_crop:
            x0, y0, x1, y1 = (v * s.drizzle for v in std_crop)
            img, weight, crop = img[..., y0:y1, x0:x1], weight[..., y0:y1, x0:x1], [x0, y0, x1, y1]
        master = img
        hdr["DRIZZLE"] = (s.drizzle, "drizzle output scale")
        hdr["PIXFRAC"] = s.pixfrac
        if "XPIXSZ" in hdr:
            hdr["XPIXSZ"] = float(hdr["XPIXSZ"]) / s.drizzle
        write_fits(output.with_name(output.stem + "_weight" + output.suffix), weight)
    else:
        if s.crop_common:
            master, crop = _crop_common(master, coverage, len(frames))
        if s.downsample == 0.5:
            master = _downsample(master)
    write_fits(output, master, hdr, bits=s.output_bits)
    log(f"輸出 {output}（{master.shape[-1]}×{master.shape[-2]}）")
    qc["crop"] = crop
    recipe = _write_recipe(output, project, frames, calibration, s, qc, warnings)
    for w in warnings:
        log(f"⚠ {w}")
    log(f"完成，耗時 {time.time() - t0:.0f} 秒")
    return Result(output=output, recipe=recipe, project=project, warnings=warnings, qc=qc)


def _choose_reference(frames: list[Frame], name: str | None, warnings: list[str]) -> Frame:
    if name:
        hit = next((f for f in frames if f.name == name), None)
        if hit:
            return hit
        warnings.append(f"指定的參考 frame {name} 不可用，改用自動選擇")
    good = [f for f in frames if np.isfinite(f.stars.fwhm)] or frames

    def rank(vals, reverse=False):
        order = np.argsort(np.argsort(-np.asarray(vals) if reverse else np.asarray(vals)))
        return order

    fw = rank([f.stars.fwhm for f in good])
    ns = rank([len(f.stars) for f in good], reverse=True)
    bg = rank([f.stars.background / (f.exposure or 1.0) for f in good])
    return good[int(np.argmin(fw * 2 + ns + bg))]


def _channel_locations(data: np.ndarray, bayer: str | None) -> list[float]:
    """每個輸出通道的背景位置（中位數，先剔除 5σ 以上的星點與軌跡）。"""
    if not bayer:
        return [_robust_loc(np.asarray(data[::3, ::3]))]
    h, w = data.shape
    arr = np.asarray(data[: h - h % 2, : w - w % 2])
    masks = cfa_masks(bayer, arr.shape)
    return [_robust_loc(arr[masks[c]][::7]) for c in "RGB"]


def _robust_loc(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    med = np.median(a)
    sig = 1.4826 * np.median(np.abs(a - med))
    a = a[np.abs(a - med) < 5 * max(sig, 1e-6)]
    return float(np.median(a))


def _weights(frames: list[Frame], s: Settings) -> None:
    if s.weighting == "none":
        raw = np.ones(len(frames))
    else:
        noise = np.array([f.stars.noise * f.norm["scale"][0] for f in frames])
        fwhm = np.array([f.stars.fwhm if np.isfinite(f.stars.fwhm) else np.nan for f in frames])
        fwhm = np.where(np.isfinite(fwhm), fwhm, np.nanmedian(fwhm) if np.isfinite(fwhm).any() else 1.0)
        # 正規化後的 SNR ∝ 1/noise；權重 = SNR² / FWHM²
        raw = 1.0 / (noise ** 2 * fwhm ** 2)
    if s.pick_boost:
        boost = np.array([(f.pick_metrics.score / 100.0) if f.pick_metrics and f.pick_metrics.score else 1.0
                          for f in frames])
        raw = raw * boost
    w = raw / raw.sum()
    for f, v in zip(frames, w):
        f.weight = float(v)


def _jobs(frames: list[Frame]) -> list[FrameJob]:
    return [FrameJob(str(f.calibrated_path), f.transform, list(f.norm["scale"]), list(f.norm["offset"]), float(f.weight))
            for f in frames]


def _integrate(frames, bayer, shape, s: Settings, cache: Path, be, workers: int, log, progress=None):
    n = len(frames)
    log(f"整合 {n} 張（{s.rejection}，low {s.low}σ / high {s.high}σ）")
    map_paths = None
    if s.keep_rejection_maps or s.drizzle:
        rdir = cache / "rejection"
        rdir.mkdir(exist_ok=True)
        for old in rdir.glob("*.npy"):
            old.unlink()
        map_paths = [str(rdir / f"{i:04d}_{f.path.stem}.npy") for i, f in enumerate(frames)]
    spec = BandSpec(bayer, shape, s.rejection, s.low, s.high, str(cache / "master_stack.npy"), map_paths)
    coverage, counts = integrate_mod.integrate(be, _jobs(frames), spec, workers, s.memory_mb, log, progress)
    master = integrate_mod.load_master(spec.master_path, bayer)
    return master, coverage, counts, map_paths or [None] * n


def _run_drizzle(frames, bayer, shape, s: Settings, map_paths, qc, warnings, cache: Path, be, workers: int, log,
                 progress=None):
    """Stage 7。條件不理想時只提醒、照樣執行（APU 的原則）。"""
    n = len(frames)
    if not qc["drizzle_suitable"]:
        warnings.append(f"Drizzle 在張數夠多（建議 20 張以上）且有 dither 時效果最好；目前 {n} 張、"
                        f"dither 範圍 {qc['dither_spread_px']} px，結果可能出現格紋或空洞")
    if s.drizzle >= 2 and qc["fwhm_px"] and qc["fwhm_px"]["median"] > 3.0 * (2 if bayer else 1):
        warnings.append(f"星點 FWHM {qc['fwhm_px']['median']:.1f} px 已經取樣充足，{s.drizzle}× drizzle 能增加的細節有限")
    log(f"{'CFA ' if bayer else ''}Drizzle {s.drizzle}×，pixfrac {s.pixfrac}，{n} 張")
    spec = DrizzleSpec(bayer, shape, s.drizzle, s.pixfrac, [p for p in map_paths] if map_paths[0] else None)
    res = drizzle_mod.drizzle(be, _jobs(frames), spec, cache, workers, s.memory_mb, log, progress)
    qc["drizzle"] = {"scale": s.drizzle, "pixfrac": s.pixfrac,
                     "holes_fraction": [round(v, 5) for v in res.holes]}
    worst = max(res.holes)
    if worst > 0.01:
        warnings.append(f"Drizzle 輸出有 {worst:.1%} 的像素沒有資料"
                        f"{'（已用鄰近像素補上）' if s.fill_holes else ''}；可以提高 pixfrac 或增加張數")
    return res


def _crop_common(master: np.ndarray, coverage: np.ndarray, n: int):
    full = coverage >= n
    rows = np.flatnonzero(full.mean(axis=1) > 0.98)
    cols = np.flatnonzero(full.mean(axis=0) > 0.98)
    if not len(rows) or not len(cols):
        return master, None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return master[..., y0:y1, x0:x1], [x0, y0, x1, y1]


def _downsample(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[-2:]
    h, w = h - h % 2, w - w % 2
    v = img[..., :h, :w]
    return v.reshape(v.shape[:-2] + (h // 2, 2, w // 2, 2)).mean(axis=(-3, -1))


def _output_header(project: Project, frames: list[Frame], s: Settings, bayer) -> fits.Header:
    hdr = fits.Header()
    ref = project.reference.header
    for k in ("INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ", "FILTER", "GAIN"):
        if k in ref:
            hdr[k] = ref[k]
    if s.downsample == 0.5 and "XPIXSZ" in hdr:
        hdr["XPIXSZ"] = float(hdr["XPIXSZ"]) * 2
    hdr["NCOMBINE"] = (len(frames), "frames integrated")
    hdr["EXPTOTAL"] = (sum(f.exposure or 0 for f in frames), "[s] total exposure")
    hdr["DATE-OBS"] = min((f.date_obs for f in frames if f.date_obs), default="")
    hdr["SOFTWARE"] = f"APU Photons {__version__}"
    hdr["REJECT"] = s.rejection
    if bayer:
        hdr["COLORTYP"] = "RGB"
    hdr["HISTORY"] = "Calibrated, registered, normalized, weighted and integrated by APU Photons"
    return hdr


def _qc(project: Project, frames: list[Frame], coverage: np.ndarray, crop) -> dict:
    all_f = project.frames
    reasons: dict[str, int] = {}
    for f in all_f:
        if f.reject_reason:
            reasons[f.reject_reason] = reasons.get(f.reject_reason, 0) + 1
    fwhm = [f.stars.fwhm for f in frames if f.stars and np.isfinite(f.stars.fwhm)]
    tx = np.array([f.transform[0, 2] for f in frames])
    ty = np.array([f.transform[1, 2] for f in frames])
    # dither：平移量的小數部分分布是否夠散（drizzle 適用性，只提醒）
    frac = np.column_stack([tx % 1, ty % 1])
    dither_spread = float(np.hypot(np.std(tx), np.std(ty)))
    subpix_spread = float(np.mean(np.std(frac, axis=0))) if len(frames) > 1 else 0.0
    rej = [f.rejected_pixel_fraction for f in frames if f.rejected_pixel_fraction is not None]
    return {
        "frames_total": len(all_f),
        "frames_integrated": len(frames),
        "rejected_by_reason": reasons,
        "fwhm_px": {"median": float(np.median(fwhm)), "min": float(np.min(fwhm)), "max": float(np.max(fwhm))} if fwhm else None,
        "dither_spread_px": round(dither_spread, 2),
        "subpixel_spread": round(subpix_spread, 3),  # 均勻分布約 0.29
        "drizzle_suitable": len(frames) >= 20 and subpix_spread > 0.2 and dither_spread > 1.0,
        "common_area_fraction": round(float((coverage >= len(frames)).mean()), 4),
        "crop": crop,
        "rejected_pixel_fraction": {"median": float(np.median(rej)), "max": float(np.max(rej))} if rej else None,
        "suspicious_frames": [f.name for f in frames
                              if rej and f.rejected_pixel_fraction and f.rejected_pixel_fraction > max(0.01, 5 * float(np.median(rej)))],
    }


def _write_recipe(output: Path, project: Project, frames, cal: Calibration, s: Settings, qc, warnings) -> Path:
    doc = {
        "schema": "apuphotons-recipe/1",
        "photons_version": __version__,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions": [{"id": x.id, "pixel_scale_arcsec": x.pixel_scale_arcsec, "frames": [str(f.path) for f in x.frames]}
                     for x in project.sessions],
        "inputs": [f.summary() for f in project.frames],
        "calibration": {k: [str(p) for p in getattr(cal, k)] for k in ("bias", "dark", "flat", "flat_dark")},
        "masters": {k: str(v) for k, v in cal.masters.items()},
        "reference_frame": project.reference.name,
        "stages": {
            "calibration": {"version": CALIB_VERSION},
            "registration": {"algorithm": "triangle-ransac", "version": REG_VERSION, "model": "similarity"},
            "normalize": {"algorithm": "star-flux-scale+background-offset", "version": "1"},
            "weight": {"formula": s.weighting, "pick_boost": s.pick_boost},
            "integrate": {"method": "weighted_average", "rejection": s.rejection, "low": s.low, "high": s.high},
            "drizzle": {"enabled": bool(s.drizzle), "scale": s.drizzle, "pixfrac": s.pixfrac,
                        "mode": "cfa" if project.reference.bayer else "mono", "version": drizzle_mod.ALGO_VERSION},
            "post": {"downsample": s.downsample, "crop_common": s.crop_common, "bits": s.output_bits},
        },
        "settings": s.__dict__,
        "qc": qc,
        "warnings": warnings,
        "outputs": [output.name],
    }
    path = output.with_suffix(".recipe.json")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
