"""把 Stage 0~8 串起來（SPEC §3）。

0.2：輸入是檔案，Stage 0 分出光學系統／對齊組／整合組並配好校正檔。每個校正組各用自己的 master 校正；
每個對齊組選一張參考 frame，組內所有濾鏡都對齊到它；每個整合組各自正規化、加權、整合，輸出一個 master。
同一對齊組的 drizzle、downsample、裁切範圍統一，各濾鏡的 master 像素對齊。

記憶體策略：校正後的 frame 存成 float32 .npy（memmap 讀取），整合時一次只處理一段列（band），
每段只從各 frame 讀出需要的區域、就地解馬賽克與變形，所以幾百張大圖也不會一次載入。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import __version__, backend
from . import drizzle as drizzle_mod
from . import integrate as integrate_mod
from . import localnorm
from .backend import default_workers
from .combine import HIGH, LOW, OUT_OF_BOUNDS
from .calibration import ALGO_VERSION as CALIB_VERSION, MasterBuilder
from .drizzle import DrizzleSpec
from .imageio import row_order, write_fits
from .i18n import Msg, tr
from .ingest import filter_names, ingest, safe_name, unused_text
from .integrate import BandSpec, FrameJob
from .model import ACCEPTED, AlignGroup, Frame, IntegrationGroup, Project
from .prepare import channel_locations, prepare
from .register import ALGO_VERSION as REG_VERSION, overlap_fraction, register_many

MIN_STARS = 10
MIN_GROUP_FRAMES = 2
MIN_OVERLAP = 0.3  # 與參考重疊不到這個比例：對齊結果是錯的（配到別處的星）
RECIPE_SCHEMA = "apuphotons-recipe/2"
# 每個對齊組可以各自設定的輸出項目（SPEC §3 Stage 7、8）
ALIGN_OUTPUT_KEYS = ("drizzle", "pixfrac", "fill_holes", "downsample", "crop_common", "crop_min_coverage")


@dataclass
class Settings:
    rejection: str = "winsorized"      # winsorized / sigma / average / median
    low: float = 4.0
    high: float = 3.0
    weighting: str = "snr2_over_fwhm2"  # snr2_over_fwhm2 / none
    normalization: str = "global"      # global（每通道一個 offset）/ local（offset 是平滑的面，對齊各張的梯度）
    dual_band: bool = True             # 彩色相機＋雙窄帶濾鏡（認得出的名稱）時，另存 Ha 與 OIII 兩個單色 master
    reference: str | list | None = None  # 參考 frame 的檔名（可以多個，各自套用到所在的對齊組）；None = 自動
    downsample: float = 1.0            # 1.0 或 0.5
    crop_common: bool = True
    crop_min_coverage: float = 0.9     # 裁到至少這個比例的 frame 覆蓋的範圍
    output_bits: int = 32
    memory_mb: int = 2048
    split_nights: bool = True
    keep_rejection_maps: bool = True
    preview: int = 0                   # >0：每個整合組只抽樣這麼多張試跑
    drizzle: int = 0                   # 0 = 不做；1 或 2 = 輸出倍率
    pixfrac: float = 0.9
    fill_holes: bool = True            # 零星沒資料的像素用鄰近平均補
    gpu: str = "auto"                  # auto / gpu / cpu：整合與 drizzle 用的運算後端
    workers: int = 0                   # CPU 平行處理數；0 = 自動（核心數 − 1）
    # ---- 0.2：輸入與分組 ----
    target: str | None = None          # 輸出檔名用的目標名稱；None = light 的 OBJECT 或資料夾名稱
    temp_tolerance: float = 2.0        # dark 與 light 的溫度容許差距（°C）
    flat_any_night: bool = False       # 這晚沒有 flat 時改用日期最近那晚的（預設由使用者逐組選）
    kinds: dict = field(default_factory=dict)            # {檔案路徑: 類型}，使用者指定
    filter_aliases: dict = field(default_factory=dict)   # {header 裡的寫法: 歸併到的名稱}
    calib_overrides: dict = field(default_factory=dict)  # {校正組 key: {"dark"/"bias"/"flat"/"flat_sub": set id 或 None}}
    file_calib: dict = field(default_factory=dict)       # {light 路徑: {同上}}，使用者選幾張 light 指定校正檔
    merge_trains: list = field(default_factory=list)     # [[光學系統 id, …], …] 合併成同一對齊組
    grid_train: dict = field(default_factory=dict)       # {對齊組 id: 光學系統 id} 合併時的參考網格
    align_output: dict = field(default_factory=dict)     # {對齊組 id: {ALIGN_OUTPUT_KEYS 的子集}}

    def for_align(self, align_id: str) -> "Settings":
        s = copy.copy(self)
        for k, v in (self.align_output.get(align_id) or {}).items():
            if k in ALIGN_OUTPUT_KEYS:
                setattr(s, k, v)
        return s


@dataclass
class GroupResult:
    """一個整合組的輸出。"""
    align: str
    filter: str | None
    output: Path
    coverage: Path | None
    stack: Path | None                 # drizzle 時另存的一般疊圖
    weight: Path | None                # drizzle 的 weight map
    frames: list[Frame]
    qc: dict
    channels: dict = field(default_factory=dict)  # 雙窄帶濾鏡另存的單色 master：{"Ha": 路徑, "OIII": 路徑}


@dataclass
class Result:
    output_dir: Path
    outputs: list[GroupResult]
    recipe: Path
    log: Path
    project: Project
    warnings: list = field(default_factory=list)
    qc: dict = field(default_factory=dict)

    @property
    def output(self) -> Path | None:
        """第一個 master（只有一組時就是唯一的輸出）。"""
        return self.outputs[0].output if self.outputs else None


class _Log:
    """目標名稱要等 Stage 0 才知道，紀錄檔在那之後才開；之前的內容先暫存。"""

    def __init__(self, echo):
        self.echo, self.fh, self.buffer, self.path = echo, None, [], None

    def __call__(self, msg) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        if self.fh is None:
            self.buffer.append(line)
        else:
            self.fh.write(line + "\n")
            self.fh.flush()
        if self.echo:
            self.echo(line)

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fh = path.open("a", encoding="utf-8")
        for line in self.buffer:
            self.fh.write(line + "\n")
        self.fh.flush()
        self.buffer.clear()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


class Cancelled(RuntimeError):
    """使用者按了停止。"""


# 進度回報的階段代號（視窗依此顯示當下語言的文字）
STAGES = ("ingest", "masters", "prepare", "register", "integrate", "drizzle", "output")


def run(inputs: list[Path], output_dir: Path, settings: Settings | None = None, cache: Path | None = None,
        echo=print, progress=None, cancel=None) -> Result:
    """inputs：light 與校正檔（檔案；資料夾等同其中的影像）。output_dir：輸出資料夾。

    progress(stage, done, total)：進度回報；cancel：threading.Event，設定後在下一個檢查點丟出 Cancelled。"""
    s = settings or Settings()
    output_dir = Path(output_dir)

    def report(stage: str, done: int = 0, total: int = 0) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress is not None:
            progress(stage, done, total)
    t0 = time.time()
    cache = Path(cache) if cache else output_dir / ".photons_cache"
    cache.mkdir(parents=True, exist_ok=True)
    log = _Log(echo)
    warnings: list = []
    try:
        return _run([Path(p) for p in inputs], output_dir, s, cache, log, warnings, t0, report)
    except Cancelled:
        raise
    except Exception:
        # 視窗只顯示一行錯誤；完整的經過寫進紀錄檔，才查得出是哪一步、哪一張出錯
        if log.fh is None:
            log.open(output_dir / "photons.log")
        log(traceback.format_exc())
        raise
    finally:
        log.close()


def _sample(frames: list[Frame], n: int) -> list[Frame]:
    if n <= 0 or n >= len(frames):
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in sorted(set(idx))]


def _run(inputs, output_dir: Path, s: Settings, cache: Path, log: _Log, warnings, t0, report) -> Result:
    log(f"APU Photons {__version__}")
    report("ingest")
    if s.drizzle not in (0, 1, 2) or any((o or {}).get("drizzle", 0) not in (0, 1, 2) for o in s.align_output.values()):
        raise RuntimeError(Msg("msg.drizzle_scale"))

    # ---- Stage 0 ----
    project = ingest(inputs, warnings, kinds=s.kinds, filter_aliases=s.filter_aliases,
                     temp_tolerance=s.temp_tolerance, split_nights=s.split_nights, overrides=s.calib_overrides,
                     flat_any_night=s.flat_any_night, file_calib=s.file_calib,
                     merge_trains=s.merge_trains, name=safe_name(s.target) if s.target else None)
    log.open(output_dir / f"{project.name}.log")
    if s.preview:
        for g in project.groups:
            chosen = set(map(id, _sample([f for f in g.frames if f.accepted], s.preview)))
            for f in g.frames:
                if f.accepted and id(f) not in chosen:
                    f.reject("not_in_preview")
        log(Msg("msg.preview_sample", n=len(project.accepted)))
    if not project.accepted:
        raise RuntimeError(Msg("msg.no_lights"))
    _log_plan(project, log)

    # ---- Stage 1 + 2 + 3 偵測：每個校正組各自的 master ----
    report("masters")
    cdir = cache / "calibrated"
    cdir.mkdir(exist_ok=True)
    workers = s.workers or default_workers()
    be = backend.select(s.gpu)
    log(Msg("msg.compute", workers=workers, gpu=Msg("msg.compute_gpu", device=be.device) if be.gpu else ""))
    builder = MasterBuilder(project.calibration_sets, cache, warnings, log, be=be, workers=workers,
                            memory_mb=s.memory_mb, progress=lambda d, t: report("masters", d, t))
    by_cal: dict[str, list[Frame]] = {}
    for f in project.accepted:
        by_cal.setdefault(f.calib["group"], []).append(f)
    total = sum(len(v) for v in by_cal.values())
    done = 0
    for gk, frames in by_cal.items():
        entry = project.calib_groups[gk]
        log(Msg("msg.calib_group", group=gk, n=len(frames), dark=entry["dark"] or "—", bias=entry["bias"] or "—",
                flat=entry["flat"] or "—"))
        if entry["flat"]:
            log(Msg("msg.calib_group_flat_sub", sub=entry["flat_sub"] or "—"))
        masters = builder.masters(entry, frames[0].bayer, frames[0].shape)
        report("masters")
        sig = _calib_signature(entry, project)
        tasks = [(str(f.path), str(cdir / f"{_calib_key(f, sig)}.npy")) for f in frames]
        base = done
        results = prepare(tasks, masters, frames[0].bayer, workers, log,
                          lambda d, t, base=base: report("prepare", base + d, total), grid=s.normalization == "local")
        done += len(frames)
        for f, (_src, dst), (stars, loc, bg, err) in zip(frames, tasks, results):
            f.calibrated_path = Path(dst)
            f.bg_loc, f.bg_grid = loc, bg
            if err is not None:
                f.reject("unreadable")
                warnings.append(Msg("msg.calib_failed", name=f.name, error=err))
                continue
            f.stars = stars
            if len(f.stars) < MIN_STARS:
                f.reject("too_few_stars")
                warnings.append(Msg("msg.too_few_stars", name=f.name, n=len(f.stars)))
        del masters
    if not project.accepted:
        raise RuntimeError(Msg("msg.no_stars_all"))

    # ---- Stage 3～6：每個對齊組 ----
    planned: list[tuple[AlignGroup, Settings, list]] = []
    for ag in project.align_groups:
        sa = s.for_align(ag.id)
        if sa.drizzle and sa.downsample != 1.0:
            warnings.append(Msg("msg.drizzle_no_downsample"))
            sa.downsample = 1.0
        items = _align_and_integrate(ag, sa, project, cache, be, workers, log, warnings, report)
        if items:
            planned.append((ag, sa, items))
    if not planned:
        raise RuntimeError(Msg("msg.no_output_groups"))

    # ---- Stage 7～8：drizzle 與輸出（對齊組共用裁切範圍）----
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[GroupResult] = []
    multi_align = len(planned) > 1
    used_names: set[str] = set()
    for ag, sa, items in planned:
        crop = _shared_crop([(it["coverage"], len(it["frames"])) for it in items], sa) if sa.crop_common else None
        if sa.crop_common and crop is None:
            warnings.append(Msg("msg.crop_none", group=ag.id))
        for it in items:
            g: IntegrationGroup = it["group"]
            name = _output_name(project.name, g.filter, ag.id if multi_align else None, used_names)
            g.output_name = name
            outputs.append(_write_group(ag, g, it, sa, crop, output_dir / f"{name}.fits", cache, be, workers, log,
                                        warnings, report))

    qc = _project_qc(project, outputs)
    recipe = _write_recipe(output_dir, project, outputs, s, qc, warnings)
    for w in warnings:
        log(f"⚠ {w}")
    log(Msg("msg.done", seconds=time.time() - t0))
    return Result(output_dir=output_dir, outputs=outputs, recipe=recipe, log=log.path, project=project,
                  warnings=warnings, qc=qc)


def _log_plan(project: Project, log) -> None:
    for tid, t in project.trains.items():
        scale = f"{t.pixel_scale_arcsec:.2f}″/px" if t.pixel_scale_arcsec else Msg("msg.unknown")
        kind = f"OSC {t.bayer}" if t.bayer else Msg("msg.mono")
        w, h = (t.shape[1], t.shape[0]) if t.shape else ("?", "?")
        log(Msg("msg.train", id=tid, kind=kind, w=w, h=h, scale=scale))
    for ag in project.align_groups:
        for g in ag.groups:
            nights = sorted({f.night for f in g.frames if f.accepted})
            log(Msg("msg.integration_group", align=ag.id, filter=g.label,
                    n=sum(f.accepted for f in g.frames), nights=", ".join(nights)))
    for alias, spellings in filter_names(project).items():
        if len(spellings) > 1:
            log(Msg("msg.filter_merged", name=alias, spellings=" / ".join(spellings)))
    # 沒用到的校正檔只寫進紀錄（不是警告）：整包校正檔庫丟進來時多半有很多套用不到
    for sid, cs in project.calibration_sets.items():
        if not cs.used_by:
            log(Msg("msg.set_unused", set=sid, n=len(cs.frames), reason=unused_text(cs)))


def _calib_signature(entry: dict, project: Project) -> str:
    """校正組用到的校正檔內容（路徑、大小、修改時間）＋演算法版本；任何一項變了，校正後的快取就失效。"""
    h = hashlib.sha256(CALIB_VERSION.encode())
    for kind in ("dark", "bias", "flat", "flat_sub"):
        sid = entry.get(kind)
        h.update(f"|{kind}=".encode())
        if sid:
            for p in sorted(project.calibration_sets[sid].paths):
                st = p.stat()
                h.update(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns};".encode())
    return h.hexdigest()


def _calib_key(f: Frame, signature: str) -> str:
    st = f.path.stat()
    h = hashlib.sha256(f"{signature}|{f.path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def _rank_frames(frames: list[Frame]) -> list[Frame]:
    """FWHM 低、星點多、背景低綜合評分，由好到壞。"""
    good = [f for f in frames if np.isfinite(f.stars.fwhm)] or list(frames)

    def rank(vals, reverse=False):
        return np.argsort(np.argsort(-np.asarray(vals) if reverse else np.asarray(vals)))

    fw = rank([f.stars.fwhm for f in good])
    ns = rank([len(f.stars) for f in good], reverse=True)
    bg = rank([f.stars.background / (f.exposure or 1.0) for f in good])
    score = fw * 2 + ns + bg
    return [good[i] for i in np.argsort(score, kind="stable")]


def _choose_reference(frames: list[Frame], name: str | None, warnings: list, quiet: bool = False) -> Frame:
    if name:
        hit = next((f for f in frames if f.name == name), None)
        if hit:
            return hit
        if not quiet:
            warnings.append(Msg("msg.bad_reference", name=name))
    return _rank_frames(frames)[0]


def _align_and_integrate(ag: AlignGroup, s: Settings, project: Project, cache: Path, be, workers, log, warnings,
                         report) -> list[dict]:
    """Stage 3（對齊到對齊組的參考）＋每個整合組的 Stage 4～6。回傳每組的整合結果（master 在 cache 裡）。"""
    frames = [f for f in ag.frames if f.accepted]
    if not frames:
        return []
    names = [s.reference] if isinstance(s.reference, str) else list(s.reference or [])
    mine = next((n for n in names if any(f.name == n for f in frames)), None)
    others = any(f.name == n for n in names for f in project.lights)
    grid = _grid_train(ag, project, s)
    if len(ag.trains) > 1:
        log(Msg("msg.merge_grid", group=ag.id, train=grid))
    candidates = [f for f in frames if f.train == grid] or frames
    if mine:
        candidates = frames  # 使用者指定的參考 frame 決定網格
    # 指定的參考 frame 在別的對齊組：這組自動選；完全找不到才提醒
    ref = _choose_reference(candidates, mine or (None if others else (names[0] if names else None)), warnings)
    ag.reference = ref
    log(Msg("msg.reference_align", group=ag.id, name=ref.name, fwhm=ref.stars.fwhm, n=len(ref.stars)))
    ref_xy = np.column_stack([ref.stars.x, ref.stars.y])
    flux_ratio: dict[int, float] = {id(ref): 1.0}
    ref.transform, ref.registration_residual_px = np.eye(3), 0.0
    others = [f for f in frames if f is not ref]
    tasks = [(np.column_stack([f.stars.x, f.stars.y]), ref_xy, _expected_scale(f, ref, project)) for f in others]
    report("register", 0, len(others))
    results = register_many(tasks, workers, lambda d, t: report("register", d, t))
    for f, (res, err) in zip(others, results):
        if res is None:
            f.reject("registration_failed")
            warnings.append(Msg("msg.reg_failed", name=f.name, error=err))
            continue
        m, rms, si, ri = res
        overlap = overlap_fraction(m, f.shape, ref.shape)
        if overlap < MIN_OVERLAP:
            f.reject("registration_failed")
            warnings.append(Msg("msg.reg_failed", name=f.name, error=Msg("msg.reg.overlap", pct=overlap * 100)))
            continue
        f.transform, f.registration_residual_px = m, round(rms, 4)
        with np.errstate(all="ignore"):
            r = ref.stars.flux[ri] / f.stars.flux[si]
        r = r[np.isfinite(r) & (r > 0)]
        flux_ratio[id(f)] = float(np.median(r)) if len(r) else 1.0
    log(Msg("msg.registered", n=sum(f.accepted for f in frames)))

    out = []
    for gi, g in enumerate(ag.groups):
        gframes = [f for f in g.frames if f.accepted]
        for f in gframes:
            pm = f.pick_metrics
            if pm is not None and pm.verdict == "reject":
                f.reject("pick")
        gframes = [f for f in gframes if f.accepted]
        if len(gframes) < MIN_GROUP_FRAMES:
            warnings.append(Msg("msg.group_too_few", group=f"{ag.id} / {g.label}", n=len(gframes)))
            continue
        bayer, shape = ref.bayer, ref.shape
        # ---- Stage 4：整合組內正規化（不同濾鏡的亮度不互相比）----
        nref = ref if ref in gframes else _rank_frames(gframes)[0]
        k_ref = flux_ratio[id(nref)]
        ref_loc = _bg_loc(nref)
        for f in gframes:
            loc = _bg_loc(f)
            k = flux_ratio[id(f)] / k_ref
            f.norm = {"scale": [k] * len(loc), "offset": [lr - k * li for lr, li in zip(ref_loc, loc)]}
        if s.normalization == "local":
            _local_normalize(gframes, nref, shape, f"{ag.id} / {g.label}", log, warnings)
        # ---- Stage 5 ----
        # 權重算不出來的那張排除：留著的話正規化後其他張的權重全變 0，整個 master 變成無效值
        for f in _weights(gframes, s):
            f.reject("bad_weight")
            warnings.append(Msg("msg.bad_weight", name=f.name))
        gframes = [f for f in gframes if f.accepted]
        if len(gframes) < MIN_GROUP_FRAMES:
            warnings.append(Msg("msg.group_too_few", group=f"{ag.id} / {g.label}", n=len(gframes)))
            continue
        # ---- Stage 6 ----
        for f in gframes:
            f.status = ACCEPTED
        log(Msg("msg.integrate_group", group=f"{ag.id} / {g.label}", n=len(gframes)))
        gdir = cache / "groups" / f"{_slug(ag.id)}_{gi:02d}"
        gdir.mkdir(parents=True, exist_ok=True)
        master_path, coverage, rej_counts, map_paths = _integrate(gframes, bayer, shape, s, gdir, be, workers, log,
                                                                  lambda d, t: report("integrate", d, t))
        if not _has_data(master_path):
            warnings.append(Msg("msg.group_no_data", group=f"{ag.id} / {g.label}"))
            continue
        for f, (n_rej, n_valid) in zip(gframes, rej_counts):
            f.rejected_pixel_fraction = round(n_rej / n_valid, 6) if n_valid else None
        out.append({"group": g, "frames": gframes, "master": master_path, "coverage": coverage,
                    "maps": map_paths, "dir": gdir, "bayer": bayer, "shape": shape, "norm_ref": nref})
    return out


def _expected_scale(f: Frame, ref: Frame, project: Project) -> float | None:
    """frame → 參考的預期縮放：同一套器材是 1；不同器材是像素尺度的比例（不知道就回傳 None）。"""
    if f.train == ref.train:
        return 1.0
    a, b = project.trains.get(f.train), project.trains.get(ref.train)
    if a and b and a.pixel_scale_arcsec and b.pixel_scale_arcsec:
        return a.pixel_scale_arcsec / b.pixel_scale_arcsec
    return None


def _grid_train(ag: AlignGroup, project: Project, s: Settings) -> str:
    """合併光學系統時的參考網格：使用者指定的，否則像素尺度最細的（不損失解析度）。"""
    if s.grid_train.get(ag.id) in ag.trains:
        return s.grid_train[ag.id]
    trains = [project.trains[t] for t in ag.trains]
    known = [t for t in trains if t.pixel_scale_arcsec]
    if known:
        return min(known, key=lambda t: t.pixel_scale_arcsec).id
    return max(trains, key=lambda t: (t.focal_mm or 0)).id


def _slug(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def _local_normalize(frames: list[Frame], nref: Frame, shape, label: str, log, warnings) -> None:
    """local normalization：每張的 offset 改成平滑的面（f.norm["local"]，每通道一組多項式係數）。
    算不出來的（重疊太少、沒有背景格點）保留全域 offset。"""
    nref_inv = np.linalg.inv(nref.transform)
    failed = []
    for f in frames:
        if f is nref:
            continue
        coeffs = None
        if f.bg_grid is not None and nref.bg_grid is not None:
            coeffs = localnorm.fit_offset(f.bg_grid, np.linalg.inv(f.transform), f.shape, nref.bg_grid, nref_inv,
                                          nref.shape, f.norm["scale"][0], shape)
        if coeffs is None:
            failed.append(f.name)
        else:
            f.norm["local"] = coeffs
    log(Msg("msg.local_norm", group=label, n=len(frames) - 1 - len(failed), ref=nref.name))
    if failed:
        warnings.append(Msg("msg.local_norm_failed", group=label, n=len(failed), names=", ".join(failed[:5])))


def _bg_loc(f: Frame) -> list[float]:
    """背景位置：校正與找星時已經在工作行程裡算好；沒有時（不應該發生）才在這裡讀檔計算。"""
    if f.bg_loc is None:
        f.bg_loc = channel_locations(np.load(f.calibrated_path, mmap_mode="r"), f.bayer)
    return f.bg_loc


def _weights(frames: list[Frame], s: Settings) -> list[Frame]:
    """設定每張的權重（總和 1）；回傳權重算不出來（不是有限的正數）的 frame，它們的權重是 None。"""
    if s.weighting == "none":
        raw = np.ones(len(frames))
    else:
        noise = np.array([f.stars.noise * f.norm["scale"][0] for f in frames])
        # FWHM 換算成參考像素（不同光學系統合併時，變換含縮放）
        fwhm = np.array([f.stars.fwhm * _scale_of(f.transform) if np.isfinite(f.stars.fwhm) else np.nan
                         for f in frames])
        fwhm = np.where(np.isfinite(fwhm), fwhm, np.nanmedian(fwhm) if np.isfinite(fwhm).any() else 1.0)
        # 正規化後的 SNR ∝ 1/noise；權重 = SNR² / FWHM²
        with np.errstate(all="ignore"):
            raw = 1.0 / (noise ** 2 * fwhm ** 2)
    ok = np.isfinite(raw) & (raw > 0)
    total = raw[ok].sum()
    for f, v, good in zip(frames, raw, ok):
        f.weight = float(v / total) if good else None
    return [f for f, good in zip(frames, ok) if not good]


def _has_data(master_path) -> bool:
    """每個色版都至少有一個有效像素。整張無效時不要默默存成 0。"""
    m = np.load(master_path, mmap_mode="r")
    return all(bool(np.isfinite(m[c]).any()) for c in range(m.shape[0]))


def _scale_of(t: np.ndarray | None) -> float:
    return 1.0 if t is None else float(np.sqrt(abs(np.linalg.det(t[:2, :2]))))


def _jobs(frames: list[Frame]) -> list[FrameJob]:
    return [FrameJob(str(f.calibrated_path), f.transform, list(f.norm["scale"]), list(f.norm["offset"]), float(f.weight),
                     f.bayer, f.norm.get("local")) for f in frames]


def _integrate(frames, bayer, shape, s: Settings, gdir: Path, be, workers: int, log, progress=None):
    n = len(frames)
    log(Msg("msg.integrate_start", n=n, method=s.rejection, low=s.low, high=s.high))
    map_paths = None
    if s.keep_rejection_maps or s.drizzle:
        rdir = gdir / "rejection"
        rdir.mkdir(exist_ok=True)
        for old in rdir.glob("*.npy"):
            old.unlink()
        map_paths = [str(rdir / f"{i:04d}_{f.path.stem}.npy") for i, f in enumerate(frames)]
    spec = BandSpec(bayer, shape, s.rejection, s.low, s.high, str(gdir / "master_stack.npy"), map_paths)
    coverage, counts = integrate_mod.integrate(be, _jobs(frames), spec, workers, s.memory_mb, log, progress)
    return spec.master_path, coverage, counts, map_paths or [None] * n


EDGE_COVERED = 0.995  # 裁切框的每一條邊，至少這個比例的像素要達到覆蓋門檻


def _crop_box(coverage: np.ndarray, n: int, min_coverage: float):
    """覆蓋至少 min_coverage × n 張的最大範圍：從整張開始，每次把四條邊裡達標比例最低的那條往內收一格，
    直到四條邊都幾乎全部達標。用積分影像，每一步只要常數時間。

    （0.1 是分別看每一列、每一行有沒有 98% 達標；中天翻轉後上下各缺一條、邊又有點斜時，
    幾乎每一行都差一點不到 98%，會裁掉大半張。）"""
    need = max(1, math.ceil(min_coverage * n - 1e-9))
    full = (coverage >= need).astype(np.int32)
    h, w = full.shape
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = full.cumsum(axis=0).cumsum(axis=1)

    def ok(y0, y1, x0, x1) -> float:  # [y0, y1) × [x0, x1) 裡達標的比例
        area = (y1 - y0) * (x1 - x0)
        return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area if area > 0 else 1.0

    y0, y1, x0, x1 = 0, h, 0, w
    while y1 - y0 > 16 and x1 - x0 > 16:
        edges = (ok(y0, y0 + 1, x0, x1), ok(y1 - 1, y1, x0, x1), ok(y0, y1, x0, x0 + 1), ok(y0, y1, x1 - 1, x1))
        worst = int(np.argmin(edges))
        if edges[worst] >= EDGE_COVERED:
            return [x0, y0, x1, y1]
        if worst == 0:
            y0 += 1
        elif worst == 1:
            y1 -= 1
        elif worst == 2:
            x0 += 1
        else:
            x1 -= 1
    return None


def _shared_crop(items: list[tuple[np.ndarray, int]], s: Settings):
    """同一對齊組各整合組裁切範圍的交集，各濾鏡的 master 才會像素對齊。"""
    boxes = [_crop_box(cov, n, s.crop_min_coverage) for cov, n in items]
    if any(b is None for b in boxes):
        return None
    x0, y0 = max(b[0] for b in boxes), max(b[1] for b in boxes)
    x1, y1 = min(b[2] for b in boxes), min(b[3] for b in boxes)
    return [x0, y0, x1, y1] if x1 - x0 > 16 and y1 - y0 > 16 else None


def _downsample(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[-2:]
    h, w = h - h % 2, w - w % 2
    v = img[..., :h, :w]
    return v.reshape(v.shape[:-2] + (h // 2, 2, w // 2, 2)).mean(axis=(-3, -1))


def _downsample_min(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[-2:]
    h, w = h - h % 2, w - w % 2
    v = img[..., :h, :w]
    return v.reshape(v.shape[:-2] + (h // 2, 2, w // 2, 2)).min(axis=(-3, -1))


def _output_name(target: str, filt: str | None, align: str | None, used: set[str]) -> str:
    parts = [target]
    if filt:
        parts.append(safe_name(filt))
    if align:
        parts.append(safe_name(align.replace(" @ ", "_").replace(" + ", "+")))
    base = "_".join(p for p in parts if p) or "stack"
    name, n = base, 2
    while name.casefold() in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name.casefold())
    return name


def _write_group(ag: AlignGroup, g: IntegrationGroup, it: dict, s: Settings, crop, output: Path, cache: Path, be,
                 workers, log, warnings, report) -> GroupResult:
    frames, bayer, shape = it["frames"], it["bayer"], it["shape"]
    master = integrate_mod.load_master(it["master"], bayer)
    coverage = it["coverage"]
    qc = _qc(g, frames, coverage)
    label = f"{ag.id} / {g.label}"
    qc["suspicious_causes"] = _suspicious_causes(frames, it["maps"], set(qc["suspicious_frames"]))
    qc["advice"] = _advice(label, frames, qc, s, bayer, crop, shape)
    if s.drizzle and not qc["drizzle_suitable"]:
        warnings.append(Msg("msg.drizzle_few_group", group=label, n=len(frames), dither=qc["dither_spread_px"]))

    hdr = _output_header(ag.reference, g, frames, s, bayer)
    stack_out = weight_out = None
    if s.drizzle:
        driz = _run_drizzle(frames, bayer, shape, s, it["maps"], qc, warnings, it["dir"], be, workers, log, label,
                            lambda d, t: report("drizzle", d, t))
        report("output")
        stack_out = output.with_name(output.stem + "_stack" + output.suffix)
        std = master if crop is None else master[..., crop[1]:crop[3], crop[0]:crop[2]]
        write_fits(stack_out, std, hdr, bits=s.output_bits)
        log(Msg("msg.stack_saved", name=stack_out.name))
        img, weight = driz.image, driz.weight
        if s.fill_holes:
            img = drizzle_mod.fill_holes(img)
        if not bayer:
            img, weight = img[0], weight[0]
        k = s.drizzle
        cov = np.repeat(np.repeat(coverage, k, axis=0), k, axis=1)
        if crop is not None:
            x0, y0, x1, y1 = (v * k for v in crop)
            img, weight, cov = img[..., y0:y1, x0:x1], weight[..., y0:y1, x0:x1], cov[y0:y1, x0:x1]
        master = img
        hdr["DRIZZLE"] = (k, "drizzle output scale")
        hdr["PIXFRAC"] = s.pixfrac
        if "XPIXSZ" in hdr:
            hdr["XPIXSZ"] = float(hdr["XPIXSZ"]) / k
        whdr = fits.Header()
        whdr["ROWORDER"] = hdr["ROWORDER"], hdr.comments["ROWORDER"]
        whdr["DRIZZLE"] = (k, "drizzle output scale")
        weight_out = output.with_name(output.stem + "_weight" + output.suffix)
        write_fits(weight_out, weight, whdr)
    else:
        report("output")
        cov = coverage
        if crop is not None:
            master = master[..., crop[1]:crop[3], crop[0]:crop[2]]
            cov = cov[crop[1]:crop[3], crop[0]:crop[2]]
        if s.downsample == 0.5:
            master, cov = _downsample(master), _downsample_min(cov)
    write_fits(output, master, hdr, bits=s.output_bits)
    log(Msg("msg.output", path=output, w=master.shape[-1], h=master.shape[-2]))
    channels = {}
    if s.dual_band and bayer and master.ndim == 3 and is_dual_band(g.filter, ag.reference.header.get("INSTRUME")):
        channels = _write_dual_band(master, hdr, output, s, log)
        qc["advice"].append(Msg("msg.advice.dual_band", group=label, files=_names([p.name for p in channels.values()])))
    cov_out = output.with_name(output.stem + "_coverage" + output.suffix)
    chdr = fits.Header()
    chdr["ROWORDER"] = hdr["ROWORDER"], hdr.comments["ROWORDER"]
    chdr["NCOMBINE"] = (len(frames), "frames integrated")
    fits.PrimaryHDU(cov.astype(np.uint16), header=chdr).writeto(cov_out, overwrite=True)
    qc["crop"] = crop
    return GroupResult(align=ag.id, filter=g.filter, output=output, coverage=cov_out, stack=stack_out,
                       weight=weight_out, frames=frames, qc=qc, channels=channels)


# 雙窄帶（Ha + OIII）濾鏡的名稱；Seestar 的「LP」也是雙窄帶，但別家的 LP 常是寬頻光害濾鏡，只在 Seestar 時認
_DUAL_BAND = re.compile(r"(?<![a-z])(duo|dual|l-?extreme|l-?enhance|l-?ultimate|alp-?t|nbz|ha-?oiii)(?![a-z])", re.I)


def is_dual_band(filter_name: str | None, instrument=None) -> bool:
    """雙窄帶濾鏡：Seestar 的 LP、Optolong L-eXtreme／L-eNhance／L-Ultimate、Antlia ALP-T、IDAS NBZ 等。"""
    if not filter_name:
        return False
    name = str(filter_name).strip()
    if name.upper() == "LP" and "seestar" in str(instrument or "").lower():
        return True
    return bool(_DUAL_BAND.search(name))


def _write_dual_band(master: np.ndarray, hdr: fits.Header, output: Path, s: Settings, log) -> dict[str, Path]:
    """雙窄帶濾鏡：Ha 取紅色版；OIII 取綠與藍，依 Bayer 上的像素數 2:1 加權（同 Siril 的 Ha／OIII 擷取）。
    各存一個單色 master，與彩色 master 像素對齊，可直接在 APU Processing 做 HOO 合成。"""
    out = {}
    for name, data in (("Ha", master[0]), ("OIII", (2.0 * master[1] + master[2]) / 3.0)):
        h = hdr.copy()
        h["FILTER"] = name
        h.remove("COLORTYP", ignore_missing=True)
        h["HISTORY"] = f"{name} extracted from the dual-band colour master {output.name}"
        path = output.with_name(f"{output.stem}_{name}{output.suffix}")
        write_fits(path, data.astype(np.float32), h, bits=s.output_bits)
        log(Msg("msg.dual_band_output", channel=name, name=path.name))
        out[name] = path
    return out


def _run_drizzle(frames, bayer, shape, s: Settings, map_paths, qc, warnings, workdir: Path, be, workers: int, log,
                 label: str, progress=None):
    """Stage 7。條件不理想時只提醒、照樣執行（APU 的原則）。"""
    n = len(frames)
    if s.drizzle >= 2 and qc["fwhm_px"] and qc["fwhm_px"]["median"] > 3.0 * (2 if bayer else 1):
        warnings.append(Msg("msg.drizzle_oversampled_group", group=label, fwhm=qc["fwhm_px"]["median"],
                            scale=s.drizzle))
    log(Msg("msg.drizzle_start", cfa="CFA " if bayer else "", scale=s.drizzle, pixfrac=s.pixfrac, n=n))
    spec = DrizzleSpec(bayer, shape, s.drizzle, s.pixfrac, list(map_paths) if map_paths[0] else None)
    res = drizzle_mod.drizzle(be, _jobs(frames), spec, workdir, workers, s.memory_mb, log, progress)
    qc["drizzle"] = {"scale": s.drizzle, "pixfrac": s.pixfrac,
                     "holes_fraction": [round(v, 5) for v in res.holes]}
    worst = max(res.holes)
    if worst > 0.01:
        warnings.append(Msg("msg.drizzle_holes", fraction=worst,
                            filled=Msg("msg.drizzle_holes_filled") if s.fill_holes else ""))
    return res


def _output_header(ref: Frame, g: IntegrationGroup, frames: list[Frame], s: Settings, bayer) -> fits.Header:
    hdr = fits.Header()
    rh = ref.header
    for k in ("INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ", "GAIN", "OBJECT"):
        if k in rh:
            hdr[k] = rh[k]
    if g.filter:
        hdr["FILTER"] = g.filter  # 這組的濾鏡，不是參考 frame 的
    if s.downsample == 0.5 and "XPIXSZ" in hdr:
        hdr["XPIXSZ"] = float(hdr["XPIXSZ"]) * 2
    hdr["NCOMBINE"] = (len(frames), "frames integrated")
    hdr["EXPTOTAL"] = (sum(f.exposure or 0 for f in frames), "[s] total exposure")
    hdr["DATE-OBS"] = min((f.date_obs for f in frames if f.date_obs), default="")
    hdr["SOFTWARE"] = f"APU Photons {__version__}"
    hdr["REJECT"] = s.rejection
    # 輸出照參考 frame 的列順序（所有 frame 都對齊到它），標明方向讓後製軟體不會上下翻反
    hdr["ROWORDER"] = (row_order(rh), "Order of pixel rows stored in the image array")
    if bayer:
        hdr["COLORTYP"] = "RGB"
    hdr["HISTORY"] = "Calibrated, registered, normalized, weighted and integrated by APU Photons"
    return hdr


def _qc(g: IntegrationGroup, frames: list[Frame], coverage: np.ndarray) -> dict:
    reasons: dict[str, int] = {}
    for f in g.frames:
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
        "filter": g.filter,
        "frames_total": len(g.frames),
        "frames_integrated": len(frames),
        "rejected_by_reason": reasons,
        "fwhm_px": {"median": float(np.median(fwhm)), "min": float(np.min(fwhm)), "max": float(np.max(fwhm))} if fwhm else None,
        "dither_spread_px": round(dither_spread, 2),
        "subpixel_spread": round(subpix_spread, 3),  # 均勻分布約 0.29
        "drizzle_suitable": len(frames) >= 20 and subpix_spread > 0.2 and dither_spread > 1.0,
        "common_area_fraction": round(float((coverage >= len(frames)).mean()), 4),
        "rejected_pixel_fraction": {"median": float(np.median(rej)), "max": float(np.max(rej))} if rej else None,
        "suspicious_frames": [f.name for f in frames
                              if rej and f.rejected_pixel_fraction and f.rejected_pixel_fraction > max(0.01, 5 * float(np.median(rej)))],
    }


CLOUD_SCALE = 2.0   # 星點亮度只剩正規化參考的一半以下：雲或霧
WIDE_TILE = 0.05    # 8×8 格裡，一格有 5% 以上的像素被剔除算「這格大片被剔除」
WIDE_AREA = 0.1     # 這種格子占 10% 以上：大片（背景梯度、天光變化），不是衛星或飛機的細線
CROP_HINT = 0.85    # 裁切後剩不到原本的 85%：提示可以調低覆蓋門檻


def _suspicious_causes(frames: list[Frame], maps: list, names: set[str]) -> dict[str, str]:
    """剔除像素特別多的片是什麼原因：cloud（星點明顯變暗）、sky（大片被剔除：背景梯度或天光變化，例如天亮、月光）、
    trail（只有細線或零星：衛星、飛機，已經剔除）；沒有剔除遮罩時是 unknown。"""
    causes: dict[str, str] = {}
    for f, path in zip(frames, maps):
        if f.name not in names:
            continue
        if f.norm and f.norm["scale"][0] > CLOUD_SCALE:
            causes[f.name] = "cloud"
            continue
        if not path:
            causes[f.name] = "unknown"
            continue
        m = np.asarray(np.load(path, mmap_mode="r")[::2, ::2])
        h, w = (m.shape[0] // 8) * 8, (m.shape[1] // 8) * 8
        m = m[:h, :w].reshape(8, h // 8, 8, w // 8)
        rej = ((m == LOW) | (m == HIGH)).sum(axis=(1, 3))
        valid = (m != OUT_OF_BOUNDS).sum(axis=(1, 3))
        frac = np.where(valid > 0, rej / np.maximum(valid, 1), 0.0)
        causes[f.name] = "sky" if (frac > WIDE_TILE).mean() >= WIDE_AREA else "trail"
    return causes


class _Names:
    """檔名清單（最多列 3 個）；顯示時才依目前語言加分隔符號，切換語言後也對。"""

    __slots__ = ("names",)

    def __init__(self, names: list[str]):
        self.names = list(names)

    def __str__(self) -> str:
        return tr("cli.sep").join(self.names[:3]) + ("…" if len(self.names) > 3 else "")


def _names(names: list[str]) -> _Names:
    return _Names(names)


def _advice(label: str, frames: list[Frame], qc: dict, s: Settings, bayer, crop, shape) -> list:
    """結果裡的建議（只是提示，不自動排除、不改設定）：可疑的片怎麼處理、適不適合 Drizzle、裁掉太多時怎麼辦。"""
    out = []
    by: dict[str, list[str]] = {}
    for name in qc["suspicious_frames"]:
        by.setdefault(qc["suspicious_causes"].get(name, "unknown"), []).append(name)
    if by.get("sky"):
        key = "msg.advice.sky_local" if s.normalization == "local" else "msg.advice.sky"
        out.append(Msg(key, group=label, n=len(by["sky"]), files=_names(by["sky"])))
    if by.get("cloud"):
        out.append(Msg("msg.advice.cloud", group=label, n=len(by["cloud"]), files=_names(by["cloud"])))
    if by.get("trail"):
        out.append(Msg("msg.advice.trail", group=label, n=len(by["trail"]), files=_names(by["trail"])))
    if by.get("unknown"):
        out.append(Msg("msg.advice.suspicious", group=label, n=len(by["unknown"]), files=_names(by["unknown"])))
    if bayer and not s.drizzle and qc["drizzle_suitable"]:
        out.append(Msg("msg.advice.drizzle_osc", group=label, n=len(frames)))
    if crop is not None and shape:
        x0, y0, x1, y1 = crop
        kept = (x1 - x0) * (y1 - y0) / float(shape[0] * shape[1])
        if kept < CROP_HINT:
            out.append(Msg("msg.advice.crop", group=label, pct=kept * 100, cov=s.crop_min_coverage * 100))
    return out


def _project_qc(project: Project, outputs: list[GroupResult]) -> dict:
    """整個專案的摘要（各組的細節在 groups）。"""
    reasons: dict[str, int] = {}
    for f in project.lights:
        if f.reject_reason:
            reasons[f.reject_reason] = reasons.get(f.reject_reason, 0) + 1
    return {
        "frames_total": len(project.lights),
        "frames_integrated": sum(len(o.frames) for o in outputs),
        "rejected_by_reason": reasons,
        "suspicious_frames": [n for o in outputs for n in o.qc["suspicious_frames"]],
        "drizzle_suitable": all(o.qc["drizzle_suitable"] for o in outputs),
        "groups": [{"align": o.align, "output": o.output.name, **o.qc} for o in outputs],
        "unknown_files": [str(f.path) for f in project.unknown],
        "unused_calibration": {sid: s.unused_reason for sid, s in project.calibration_sets.items() if not s.used_by},
    }


def _write_recipe(output_dir: Path, project: Project, outputs: list[GroupResult], s: Settings, qc, warnings) -> Path:
    settings = asdict(s)
    doc = {
        "schema": RECIPE_SCHEMA,
        "photons_version": __version__,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target": project.name,
        "settings": settings,
        "filter_aliases": project.filter_aliases,
        "optical_trains": [t.summary() for t in project.trains.values()],
        "calibration_sets": [cs.summary() for cs in project.calibration_sets.values()],
        "calibration_groups": [{"key": k, **{x: v.get(x) for x in ("dark", "bias", "flat", "flat_sub", "source")},
                                "frames": len(v["frames"])} for k, v in project.calib_groups.items()],
        "align_groups": [{"id": ag.id, "trains": ag.trains,
                          "reference_frame": ag.reference.name if ag.reference else None,
                          "output": {k: getattr(s.for_align(ag.id), k) for k in ALIGN_OUTPUT_KEYS},
                          "crop": next((o.qc.get("crop") for o in outputs if o.align == ag.id), None)}
                         for ag in project.align_groups],
        "inputs": [f.summary() for f in project.files],
        "stages": {
            "calibration": {"version": CALIB_VERSION},
            "registration": {"algorithm": "triangle-ransac", "version": REG_VERSION, "model": "similarity"},
            "normalize": {"algorithm": "star-flux-scale+background-offset", "version": "1",
                          "method": s.normalization,
                          **({"local": {"tile": localnorm.TILE, "degree": localnorm.DEGREE}}
                             if s.normalization == "local" else {})},
            "weight": {"formula": s.weighting},
            "integrate": {"method": "weighted_average", "rejection": s.rejection, "low": s.low, "high": s.high},
            "drizzle": {"version": drizzle_mod.ALGO_VERSION},
        },
        "outputs": [{"align_group": o.align, "filter": o.filter, "frames": len(o.frames), "file": o.output.name,
                     "coverage": o.coverage.name if o.coverage else None,
                     "stack": o.stack.name if o.stack else None, "weight": o.weight.name if o.weight else None,
                     **({"channels": {k: v.name for k, v in o.channels.items()}} if o.channels else {})}
                    for o in outputs],
        "qc": qc,
        "warnings": [str(w) for w in warnings],
    }
    path = output_dir / f"{project.name}.recipe.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
