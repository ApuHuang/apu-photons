"""Stage 0：讀 header、分 session、檢查相容性、接上 APU Pick sidecar。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .imageio import bayer_pattern, file_sha256, list_images, read_header
from .integrations.pick import load_pick_sidecar
from .model import Calibration, Frame, Project, Session

HEADER_KEYS = ("EXPTIME", "EXPOSURE", "GAIN", "OFFSET", "CCD-TEMP", "XBINNING", "YBINNING", "BAYERPAT",
               "XBAYROFF", "YBAYROFF", "DATE-OBS", "FILTER", "INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ",
               "PIERSIDE", "SITELONG", "NAXIS1", "NAXIS2", "SATLEVEL", "ISO")


def night_of(date_obs: str | None, site_long: float | None) -> str:
    """觀測夜：地方時中午切換日期（DATE-OBS 視為 UTC；有經度時換算成地方平時）。"""
    if not date_obs:
        return "unknown"
    try:
        t = datetime.fromisoformat(date_obs.replace("Z", ""))
    except ValueError:
        return "unknown"
    if site_long is not None:
        t += timedelta(hours=site_long / 15.0)
    return (t - timedelta(hours=12)).date().isoformat()


def pixel_scale(header: dict) -> float | None:
    """arcsec/px = 206.265 × XPIXSZ(µm) / FOCALLEN(mm)；XPIXSZ 通常已含 binning。"""
    try:
        px, fl = float(header["XPIXSZ"]), float(header["FOCALLEN"])
    except (KeyError, TypeError, ValueError):
        return None
    return 206.265 * px / fl if fl > 0 else None


def _frame(path: Path) -> Frame:
    hdr = read_header(path)
    h = {k: hdr[k] for k in HEADER_KEYS if k in hdr}
    exp = h.get("EXPTIME", h.get("EXPOSURE"))
    shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"])) if "NAXIS1" in hdr else None
    return Frame(path=path, header=h, exposure=float(exp) if exp is not None else None,
                 date_obs=h.get("DATE-OBS"), bayer=bayer_pattern(hdr), shape=shape)


def ingest(light_dirs: list[Path], calibration: Calibration, warnings: list[str],
           name: str = "project", split_nights: bool = True) -> Project:
    frames: list[Frame] = []
    for folder in light_dirs:
        paths = list_images(folder)
        if not paths:
            warnings.append(f"{folder}: 找不到影像檔")
        pick = load_pick_sidecar(folder, warnings)
        for p in paths:
            try:
                f = _frame(p)
            except Exception as exc:  # noqa: BLE001  單張壞檔不中斷
                f = Frame(path=p)
                f.reject("unreadable")
                warnings.append(f"{p.name}: 讀取失敗（{exc}）")
            if pick is not None:
                f.hash = file_sha256(p)
                hit = pick.get(f.hash)
                if hit is None:
                    warnings.append(f"{p.name}: APU Pick sidecar 裡沒有對應的 hash，略過該張的 Pick 資料")
                else:
                    f.pick_metrics = hit[1]
                    f.pick_group = hit[1].group
            frames.append(f)

    _check_compatible(frames, warnings)

    groups: dict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        key = night_of(f.date_obs, _float(f.header.get("SITELONG"))) if split_nights else "all"
        groups[key].append(f)
    sessions = []
    for key in sorted(groups):
        fs = sorted(groups[key], key=lambda f: (f.date_obs or "", f.name))
        scale = next((s for s in (pixel_scale(f.header) for f in fs) if s), None)
        sessions.append(Session(id=key, frames=fs, calibration=calibration, pixel_scale_arcsec=scale))
    return Project(name=name, sessions=sessions)


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _check_compatible(frames: list[Frame], warnings: list[str]) -> None:
    """尺寸與 Bayer 排列以多數為準，不一致的標記 incompatible。"""
    ok = [f for f in frames if f.accepted]
    if not ok:
        return
    from collections import Counter

    shape, _ = Counter(f.shape for f in ok).most_common(1)[0]
    bayer, _ = Counter(f.bayer for f in ok).most_common(1)[0]
    for f in ok:
        if f.shape != shape or f.bayer != bayer:
            f.reject("incompatible")
            warnings.append(f"{f.name}: 尺寸或 Bayer 排列與其他 frame 不同（{f.shape}, {f.bayer}），不使用")
