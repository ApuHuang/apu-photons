"""Stage 0：分類加入的檔案、分光學系統／對齊組／整合組，並配對校正檔（SPEC §3 Stage 0、§3.1、Stage 1）。

輸入是檔案（給資料夾時等同其中的影像，不往子資料夾找）。light 與校正檔一起加入，類型依序由
header 的 IMAGETYP、檔名、上層資料夾名稱判斷；判斷不出來的標成 unknown，由使用者指定。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .i18n import Msg
from .imageio import bayer_pattern, file_sha256, is_image, list_images, read_header, row_order
from .integrations.pick import load_pick_sidecar
from .model import UNKNOWN, AlignGroup, CalibrationSet, Frame, IntegrationGroup, OpticalTrain, Project

HEADER_KEYS = ("IMAGETYP", "EXPTIME", "EXPOSURE", "GAIN", "OFFSET", "CCD-TEMP", "XBINNING", "YBINNING", "BAYERPAT",
               "XBAYROFF", "YBAYROFF", "DATE-OBS", "FILTER", "INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ",
               "PIERSIDE", "SITELONG", "NAXIS1", "NAXIS2", "SATLEVEL", "ISO", "ROWORDER", "OBJECT", "LENS")

CAL_KINDS = ("dark", "bias", "flat", "flat_dark")


# ---------- 觀測夜與像素尺度 ----------

def night_of(date_obs: str | None, site_long: float | None) -> str:
    """觀測夜：地方時中午切換日期（DATE-OBS 視為 UTC；有經度時換算成地方平時）。"""
    t = _time(date_obs)
    if t is None:
        return "unknown"
    if site_long is not None:
        t += timedelta(hours=site_long / 15.0)
    return (t - timedelta(hours=12)).date().isoformat()


def pixel_scale(header: dict) -> float | None:
    """arcsec/px = 206.265 × XPIXSZ(µm) / FOCALLEN(mm)；XPIXSZ 已含 binning（SBFITSEXT），不再乘。"""
    try:
        px, fl = float(header["XPIXSZ"]), float(header["FOCALLEN"])
    except (KeyError, TypeError, ValueError):
        return None
    return 206.265 * px / fl if fl > 0 else None


def _time(date_obs: str | None) -> datetime | None:
    if not date_obs:
        return None
    try:
        t = datetime.fromisoformat(str(date_obs).replace("Z", ""))
    except ValueError:
        return None
    return t.replace(tzinfo=None)


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------- 輸入與類型判斷 ----------

def expand_inputs(paths: list[Path]) -> list[Path]:
    """檔案照原樣；資料夾展開成其中的影像（不往子資料夾找）。去掉重複，保持順序。"""
    out, seen = [], set()
    for p in map(Path, paths):
        items = list_images(p) if p.is_dir() else ([p] if is_image(p) else [])
        for q in items:
            key = str(q.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                out.append(q)
    return out


_MASTER = re.compile(r"master", re.I)
# 依序比對（flat-dark 必須在 flat、dark 之前）
_NAME_RULES = (
    ("flat_dark", re.compile(r"(?<![a-z])(flat[\s_\-]*darks?|dark[\s_\-]*flats?)(?![a-z])", re.I)),
    ("bias", re.compile(r"(?<![a-z])(bias(es)?|zeros?)(?![a-z])", re.I)),
    ("flat", re.compile(r"(?<![a-z])flats?(?![a-z])", re.I)),
    ("dark", re.compile(r"(?<![a-z])darks?(?![a-z])", re.I)),
    ("light", re.compile(r"(?<![a-z])lights?(?![a-z])", re.I)),
)


def _kind_from_text(text: str) -> str | None:
    text = _MASTER.sub(" ", text)  # MasterDark → " Dark"
    for kind, rx in _NAME_RULES:
        if rx.search(text):
            return kind
    return None


def _kind_from_imagetyp(value) -> str | None:
    t = str(value or "").strip().upper()
    if not t:
        return None
    if "FLAT" in t and "DARK" in t:
        return "flat_dark"
    if "BIAS" in t or "ZERO" in t or t == "OFFSET":
        return "bias"
    if "FLAT" in t:
        return "flat"
    if "DARK" in t:
        return "dark"
    if "LIGHT" in t or "OBJECT" in t or "SCIENCE" in t:
        return "light"
    return None


def classify(path: Path, header: dict) -> tuple[str, str | None, bool]:
    """（類型, 來源, 是否做好的 master）。來源：header / filename / folder；判斷不出來是 (unknown, None)。"""
    imagetyp = header.get("IMAGETYP")
    is_master = bool(_MASTER.search(path.stem)) or bool(_MASTER.search(str(imagetyp or "")))
    by_name, source = _kind_from_name(path)
    kind = _kind_from_imagetyp(imagetyp)
    if kind == "light" and by_name not in (None, "light"):
        # 很多拍攝軟體的預設類型就是 LIGHT（例如在 NINA 用一般序列拍 flat）；檔名或資料夾明確寫了校正檔就以名稱為準
        return by_name, source, is_master
    if kind:
        return kind, "header", is_master and kind != "light"
    if by_name:
        return by_name, source, is_master and by_name != "light"
    return UNKNOWN, None, False


def _kind_from_name(path: Path) -> tuple[str | None, str | None]:
    kind = _kind_from_text(path.stem)
    if kind:
        return kind, "filename"
    for parent in (path.parent, path.parent.parent):
        kind = _kind_from_text(parent.name)
        if kind:
            return kind, "folder"
    return None, None


_HEADERS: dict[tuple, dict] = {}
_HASHES: dict[tuple, str] = {}


def _stat_key(path: Path) -> tuple:
    st = path.stat()
    return str(path.resolve()), st.st_size, st.st_mtime_ns


def _header(path: Path) -> dict:
    """讀 header（快取：介面每次改設定都重新分組，相機 RAW 的 header 要讀整個檔案）。"""
    key = _stat_key(path)
    if key not in _HEADERS:
        if len(_HEADERS) > 20000:
            _HEADERS.clear()
        hdr = read_header(path)
        _HEADERS[key] = {k: hdr[k] for k in HEADER_KEYS + ("NAXIS1", "NAXIS2") if k in hdr}
    return dict(_HEADERS[key])


def cached_sha256(path: Path) -> str:
    key = _stat_key(path)
    if key not in _HASHES:
        _HASHES[key] = file_sha256(path)
    return _HASHES[key]


def _frame(path: Path) -> Frame:
    hdr = _header(path)
    h = {k: hdr[k] for k in HEADER_KEYS if k in hdr}
    exp = h.get("EXPTIME", h.get("EXPOSURE"))
    shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"])) if "NAXIS1" in hdr else None
    return Frame(path=path, header=h, exposure=float(exp) if exp is not None else None,
                 date_obs=h.get("DATE-OBS"), bayer=bayer_pattern(hdr), shape=shape)


# ---------- 光學系統與濾鏡 ----------

def _camera(f: Frame) -> tuple[str, str]:
    """（比對鍵, 顯示名稱）。RAW 沒有機身型號，用尺寸＋Bayer 區分。"""
    h = f.header
    inst = str(h.get("INSTRUME", "") or "").strip()
    px = _float(h.get("XPIXSZ"))
    binning = int(_float(h.get("XBINNING")) or 1)
    shape = f.shape or (0, 0)
    key = f"{inst}|{px}|{shape[1]}x{shape[0]}|{f.bayer}|bin{binning}"
    name = inst or f"{shape[1]}×{shape[0]}{' ' + f.bayer if f.bayer else ''}"
    if binning > 1:
        name += f" bin{binning}"
    return key, name


def _focal(f: Frame) -> float | None:
    v = _float(f.header.get("FOCALLEN"))
    return round(v, 1) if v and v > 0 else None


def filter_key(name: str | None) -> str | None:
    """自動歸併只忽略大小寫與前後空白（SPEC §3.1.2）。"""
    if name is None:
        return None
    s = str(name).strip()
    return s.casefold() if s else None


def _raw_filter(f: Frame) -> str | None:
    v = f.header.get("FILTER")
    s = str(v).strip() if v is not None else ""
    return s or None


def _apply_filters(lights_and_flats: list[Frame], aliases: dict[str, str]) -> None:
    """設定每張的 filter：先套手動對應，再依比對鍵歸併；顯示名稱用出現最多的寫法。"""
    alias = {filter_key(k): v for k, v in (aliases or {}).items() if filter_key(k)}
    keyed: list[tuple[Frame, str | None]] = []
    spellings: dict[str, Counter] = defaultdict(Counter)
    for f in lights_and_flats:
        raw = _raw_filter(f)
        if raw is not None and filter_key(raw) in alias:
            raw = alias[filter_key(raw)].strip() or raw
        k = filter_key(raw)
        keyed.append((f, k))
        if k is not None:
            spellings[k][raw] += 1
    for f, k in keyed:
        f.filter = None if k is None else spellings[k].most_common(1)[0][0]


def filter_names(project: Project) -> dict[str, list[str]]:
    """每個歸併後的濾鏡名稱 → header 裡出現過的寫法（介面顯示「自動歸併了哪些」）。"""
    out: dict[str, set] = defaultdict(set)
    for f in project.files:
        if f.kind in ("light", "flat") and f.filter:
            out[f.filter].add(_raw_filter(f) or f.filter)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _trains(lights: list[Frame]) -> dict[str, OpticalTrain]:
    by_key: dict[tuple, list[Frame]] = defaultdict(list)
    for f in lights:
        by_key[(f.camera, _focal(f))].append(f)
    trains: dict[str, OpticalTrain] = {}
    for (cam_key, focal), fs in sorted(by_key.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        f0 = fs[0]
        _, cam_name = _camera(f0)
        lens = str(f0.header.get("LENS", "") or "").strip() or None
        label = cam_name if focal is None else f"{cam_name} @ {focal:g}mm"
        tid, n = label, 2
        while tid in trains:
            tid, n = f"{label} ({n})", n + 1
        scale = next((s for s in (pixel_scale(x.header) for x in fs) if s), None)
        trains[tid] = OpticalTrain(id=tid, camera=cam_name, camera_key=cam_key, focal_mm=focal, lens=lens,
                                   pixel_um=_float(f0.header.get("XPIXSZ")), shape=f0.shape, bayer=f0.bayer,
                                   pixel_scale_arcsec=scale)
        for x in fs:
            x.train = tid
    return trains


# ---------- 校正檔：分套與配對 ----------

def _same(a, b, rel: float = 0.0) -> bool | None:
    """True / False；任一邊未知回傳 None。"""
    if a is None or b is None:
        return None
    fa, fb = _float(a), _float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= max(1e-6, rel * max(abs(fa), abs(fb)))
    return str(a).strip().casefold() == str(b).strip().casefold()


def _camera_same(a: str | None, b: str | None) -> bool | None:
    """相機比對鍵逐項比：尺寸必須相同；型號、像素大小、Bayer、binning 任一邊缺就算未知（校正檔常少寫幾個欄位）。"""
    if a is None or b is None:
        return None
    pa, pb = a.split("|"), b.split("|")
    if len(pa) != len(pb) or pa[2] != pb[2]:
        return False
    unknown = False
    for x, y in zip(pa, pb):
        if x in ("", "None") or y in ("", "None"):
            unknown = True
        elif x != y:
            return False
    return None if unknown else True


def _exp_same(a, b) -> bool | None:
    if a is None or b is None:
        return None
    return abs(a - b) <= max(0.01, 0.005 * max(a, b))


def _fmt(v) -> str:
    if v is None:
        return "?"
    fv = _float(v)
    return f"{fv:g}" if fv is not None else str(v)


def _set_conditions(kind: str, f: Frame) -> dict:
    c = {"camera": f.camera, "gain": f.gain, "offset": f.offset}
    if kind in ("dark", "flat_dark", "flat"):
        c["exptime"] = f.exposure
    if kind in ("dark", "flat_dark"):
        c["temp"] = None if f.temp is None else round(f.temp)
    if kind == "flat":
        c["filter"] = filter_key(f.filter)
        c["focal_mm"] = _focal(f)
        c["night"] = None if f.night == "unknown" else f.night
    return c


def _set_id(kind: str, c: dict, master: Frame | None) -> str:
    if master is not None:
        return f"{kind} master {master.path.stem}"
    parts = [kind]
    if c.get("exptime") is not None:
        parts.append(f"{_fmt(c['exptime'])}s")
    if c.get("gain") is not None:
        parts.append(f"g{_fmt(c['gain'])}")
    if c.get("offset") is not None:
        parts.append(f"o{_fmt(c['offset'])}")
    if c.get("temp") is not None:
        parts.append(f"{_fmt(c['temp'])}°C")
    if c.get("filter"):
        parts.append(str(c["filter"]))
    if c.get("night") and kind == "flat":
        parts.append(str(c["night"]))
    return " ".join(parts)


def build_sets(cal_frames: list[Frame]) -> dict[str, CalibrationSet]:
    """同類型、同條件的校正檔歸成一套；做好的 master 各自一套；只有一張的套視為 master（SPEC Stage 0）。"""
    groups: dict[tuple, list[Frame]] = defaultdict(list)
    masters: list[Frame] = []
    for f in cal_frames:
        if f.is_master:
            masters.append(f)
        else:
            c = _set_conditions(f.kind, f)
            if f.kind == "flat":
                c.pop("exptime")  # 天光 flat 常常每張曝光不同，仍是同一套
            groups[(f.kind, tuple(sorted((k, str(v)) for k, v in c.items())))].append(f)
    sets: dict[str, CalibrationSet] = {}

    def add(kind, frames, is_master):
        c = _set_conditions(kind, frames[0])
        if kind == "flat":
            exps = {f.exposure for f in frames}
            c["exptime"] = exps.pop() if len(exps) == 1 else None  # 曝光不一致時 flat-dark 無從配對，改用 bias
        temps = [f.temp for f in frames if f.temp is not None]
        if temps:
            c["temp_mean"] = round(sum(temps) / len(temps), 2)
        times = sorted(t for t in (_time(f.date_obs) for f in frames) if t)
        if times:
            c["date"] = times[len(times) // 2].isoformat(timespec="seconds")
        base = _set_id(kind, c, frames[0] if is_master else None)
        sid, n = base, 2
        while sid in sets:
            sid, n = f"{base} ({n})", n + 1
        sets[sid] = CalibrationSet(id=sid, kind=kind, frames=frames, is_master=is_master, conditions=c)

    for (kind, _), frames in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        frames.sort(key=lambda f: f.name)
        if len(frames) == 1:
            frames[0].is_master = True  # 判斷不出來、但這套只有一張：就是 master
        add(kind, frames, len(frames) == 1)
    for f in sorted(masters, key=lambda f: f.name):
        add(f.kind, [f], True)
    return sets


def _hours_apart(a: str | None, b: str | None) -> float:
    ta, tb = _time(a), _time(b)
    return abs((ta - tb).total_seconds()) / 3600.0 if ta and tb else 1e9


def _rank(cands: list[CalibrationSet], want: dict, date: str | None, temp: float | None, tol: float):
    """依必須相同的條件篩選；回傳（排好的候選, 只因溫度超出而落選的候選）。未知的條件算吻合，但排在後面。"""
    ok, too_far = [], []
    for s in cands:
        c = s.conditions
        unknown = 0
        good = True
        for key, val in want.items():
            if key == "exptime":
                same = _exp_same(val, c.get(key))
            elif key == "camera":
                same = _camera_same(val, c.get(key))
            else:
                same = _same(val, c.get(key))
            if same is None:
                unknown += 1
            elif not same:
                good = False
                break
        if not good:
            continue
        dt = 0.0
        st = c.get("temp_mean")
        if temp is not None and st is not None:
            dt = abs(temp - st)
            if dt > tol:
                too_far.append(s)
                continue
        ok.append((unknown, dt, _hours_apart(date, c.get("date")), s.id, s))
    ok.sort(key=lambda t: t[:4])
    return [t[-1] for t in ok], too_far


def light_group_key(f: Frame) -> str:
    """條件相同的 light 共用一個校正組（使用者手動修改也以這個為單位）。"""
    t = None if f.temp is None else round(f.temp)
    return (f"{f.train} | {f.filter or '—'} | {f.night} | {_fmt(f.exposure)}s | gain {_fmt(f.gain)} | "
            f"offset {_fmt(f.offset)} | {_fmt(t)}°C")


def match_calibration(project: Project, warnings: list, temp_tolerance: float = 2.0,
                      overrides: dict[str, dict] | None = None, flat_any_night: bool = False) -> None:
    """每張 light 配 dark / bias / flat，每套 flat 配 flat-dark（或曝光相同的 dark、或 bias）。結果寫進
    Frame.calib 與 project.calib_groups；沒用到的校正檔標上原因。overrides：{校正組 key: {kind: set id 或 None}}。
    flat_any_night：這晚沒有 flat 時，改用日期最接近那晚的（使用者開啟才會；預設由使用者逐組選）。"""
    sets = project.calibration_sets
    by_kind: dict[str, list[CalibrationSet]] = defaultdict(list)
    for s in sets.values():
        by_kind[s.kind].append(s)
    groups: dict[str, list[Frame]] = defaultdict(list)
    for f in project.lights:
        if f.accepted:
            groups[light_group_key(f)].append(f)

    flat_sub: dict[str, str | None] = {}

    def sub_for(flat: CalibrationSet) -> str | None:
        if flat.id in flat_sub:
            return flat_sub[flat.id]
        sid = None
        if not flat.is_master:  # 做好的 master flat 視為已扣過
            c = flat.conditions
            want = {"camera": c["camera"], "gain": c["gain"], "offset": c["offset"], "exptime": c.get("exptime")}
            temp = c.get("temp_mean")
            for kind in ("flat_dark", "dark") if c.get("exptime") is not None else ():
                hit, _ = _rank(by_kind[kind], want, c.get("date"), temp, temp_tolerance)
                if hit:
                    sid = hit[0].id
                    break
            if sid is None:
                hit, _ = _rank(by_kind["bias"], {k: want[k] for k in ("camera", "gain", "offset")},
                               c.get("date"), None, temp_tolerance)
                sid = hit[0].id if hit else None
            if sid is None:
                warnings.append(Msg("msg.flat_no_sub_set", set=flat.id))
        flat_sub[flat.id] = sid
        return sid

    calib_groups: dict[str, dict] = {}
    for gk, frames in sorted(groups.items()):
        f = frames[0]
        base = {"camera": f.camera, "gain": f.gain, "offset": f.offset}
        dark_hit, dark_far = _rank(by_kind["dark"], {**base, "exptime": f.exposure}, f.date_obs, f.temp,
                                   temp_tolerance)
        bias_hit, _ = _rank(by_kind["bias"], base, f.date_obs, None, temp_tolerance)
        flat_want = {"camera": f.camera, "filter": filter_key(f.filter), "focal_mm": _focal(f),
                     "night": None if f.night == "unknown" else f.night}
        flat_hit, _ = _rank(by_kind["flat"], flat_want, f.date_obs, None, temp_tolerance)
        auto = {"dark": dark_hit[0].id if dark_hit else None,
                "bias": bias_hit[0].id if bias_hit else None,
                "flat": flat_hit[0].id if flat_hit else None}
        other_night = []
        if not flat_hit:  # 別晚的 flat 不自動代用，列給使用者選（SPEC Stage 1）
            other_night, _ = _rank(by_kind["flat"], {k: v for k, v in flat_want.items() if k != "night"},
                                   f.date_obs, None, temp_tolerance)
        borrowed = None
        if auto["flat"] is None and other_night and flat_any_night:
            borrowed = other_night[0]
            auto["flat"] = borrowed.id
        chosen, source = dict(auto), "auto"
        if overrides and gk in overrides:
            for kind, sid in overrides[gk].items():
                if kind in chosen and (sid is None or sid in sets):
                    chosen[kind] = sid
                    source = "user"
        if chosen["dark"] is None and dark_far:
            warnings.append(Msg("msg.dark_temp_far", group=gk, set=dark_far[0].id,
                                temp=_fmt(dark_far[0].conditions.get("temp_mean")), tol=_fmt(temp_tolerance)))
        if chosen["dark"]:
            chosen["bias"] = chosen["bias"] if source == "user" and overrides[gk].get("bias") else None
        sub = sub_for(sets[chosen["flat"]]) if chosen["flat"] else None
        entry = {**chosen, "flat_sub": sub, "source": source, "frames": frames,
                 "flat_other_nights": [s.id for s in other_night],
                 "candidates": {"dark": [s.id for s in dark_hit], "bias": [s.id for s in bias_hit],
                                "flat": [s.id for s in flat_hit] + [s.id for s in other_night]}}
        calib_groups[gk] = entry
        for kind in ("dark", "bias", "flat"):
            if chosen[kind]:
                sets[chosen[kind]].used_by.add(gk)
        if sub:
            sets[sub].used_by.add(gk)
        for x in frames:
            x.calib = {"group": gk, "dark": chosen["dark"], "bias": chosen["bias"], "flat": chosen["flat"],
                       "flat_sub": sub, "source": source}
        if not chosen["dark"] and not chosen["bias"]:
            warnings.append(Msg("msg.group_no_dark", group=gk, n=len(frames)))
        if borrowed is not None and chosen["flat"] == borrowed.id:
            warnings.append(Msg("msg.group_flat_other_night", group=gk, n=len(frames), set=borrowed.id,
                                night=borrowed.conditions.get("night") or "?"))
        if not chosen["flat"]:
            if other_night:
                warnings.append(Msg("msg.group_no_flat_night", group=gk, n=len(frames), night=f.night))
            else:
                warnings.append(Msg("msg.group_no_flat", group=gk, n=len(frames)))
    project.calib_groups = calib_groups

    for s in sets.values():
        if s.used_by:
            continue
        if s.kind == "bias" and any(s.id in g["candidates"]["bias"] for g in calib_groups.values()):
            s.unused_reason = "dark_present"
        elif s.kind in ("dark", "flat_dark") and not by_kind["flat"]:
            s.unused_reason = "no_match_no_flat"
        else:
            s.unused_reason = "no_match"
        warnings.append(Msg("msg.set_unused", set=s.id, n=len(s.frames), reason=Msg(f"msg.unused.{s.unused_reason}")))


# ---------- 對齊組與整合組 ----------

def build_groups(project: Project, merge: list[list[str]] | None = None, warnings: list | None = None) -> None:
    """預設每個光學系統一個對齊組；merge 列出要合併成同一對齊組的光學系統（SPEC §3.1.3）。"""
    merged_of: dict[str, str] = {}
    for trains in merge or []:
        trains = [t for t in trains if t in project.trains and t not in merged_of]
        if len(trains) < 2:
            continue
        if len({project.trains[t].bayer is None for t in trains}) > 1:  # 單色與彩色不能疊成同一個 master
            if warnings is not None:
                warnings.append(Msg("msg.merge_mono_color", trains=" + ".join(trains)))
            continue
        if len({project.trains[t].bayer for t in trains}) > 1 and warnings is not None:
            warnings.append(Msg("msg.merge_color_cast", trains=" + ".join(trains)))
        for t in trains:
            merged_of[t] = " + ".join(trains)
    by_align: dict[str, list[Frame]] = defaultdict(list)
    members: dict[str, list[str]] = {}
    for tid in project.trains:
        aid = merged_of.get(tid, tid)
        members.setdefault(aid, []).append(tid)
    for f in project.lights:
        if f.accepted and f.train:
            by_align[merged_of.get(f.train, f.train)].append(f)
    aligns = []
    for aid, tids in members.items():
        frames = by_align.get(aid, [])
        by_filter: dict[str | None, list[Frame]] = defaultdict(list)
        for f in frames:
            by_filter[f.filter].append(f)
        groups = [IntegrationGroup(align=aid, filter=k, frames=sorted(v, key=lambda f: (f.date_obs or "", f.name)))
                  for k, v in sorted(by_filter.items(), key=lambda kv: (kv[0] is None, str(kv[0]).casefold()))]
        aligns.append(AlignGroup(id=aid, trains=tids, groups=groups))
    project.align_groups = aligns


def _check_rows(project: Project, warnings: list) -> None:
    """同一光學系統裡，列順序（ROWORDER）與多數相反的 light 是上下鏡像，對不上，標成 incompatible。"""
    by_train: dict[str, list[Frame]] = defaultdict(list)
    for f in project.lights:
        if f.accepted:
            by_train[f.train].append(f)
    for frames in by_train.values():
        order, _ = Counter(row_order(f.header) for f in frames).most_common(1)[0]
        for f in frames:
            if row_order(f.header) != order:
                f.reject("incompatible")
                warnings.append(Msg("msg.roworder_mismatch", name=f.name, order=row_order(f.header), majority=order))


def target_name(lights: list[Frame]) -> str:
    """輸出用的目標名稱：light 最常見的 OBJECT，沒有時用第一張 light 的資料夾名稱。"""
    objs = Counter(str(f.header.get("OBJECT")).strip() for f in lights if str(f.header.get("OBJECT") or "").strip())
    name = objs.most_common(1)[0][0] if objs else (lights[0].path.parent.name if lights else "stack")
    return safe_name(name) or "stack"


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip()
    return re.sub(r"\s+", "", name).strip("._")


# ---------- 主流程 ----------

def ingest(paths: list[Path], warnings: list, *, kinds: dict | None = None, filter_aliases: dict | None = None,
           temp_tolerance: float = 2.0, split_nights: bool = True, overrides: dict | None = None,
           merge_trains: list[list[str]] | None = None, name: str | None = None,
           flat_any_night: bool = False) -> Project:
    """kinds：{路徑: 類型} 強制指定（使用者在介面上改的、或命令列 --dark 等）。"""
    forced = {str(Path(k).resolve()).casefold(): v for k, v in (kinds or {}).items()}
    files = expand_inputs(paths)
    if not files:
        warnings.append(Msg("msg.no_input_files"))
    frames: list[Frame] = []
    for p in files:
        try:
            f = _frame(p)
        except Exception as exc:  # noqa: BLE001  單張壞檔不中斷
            f = Frame(path=p)
            f.reject("unreadable")
            warnings.append(Msg("msg.read_failed", name=p.name, error=exc))
        kind, source, is_master = classify(p, f.header)
        user = forced.get(str(p.resolve()).casefold())
        if user:
            kind, source = user, "user"
        f.kind, f.kind_source, f.is_master = kind, source, is_master and kind != "light"
        frames.append(f)

    unknown = [f for f in frames if f.kind == UNKNOWN]
    if unknown:
        warnings.append(Msg("msg.unknown_kind", n=len(unknown), names=", ".join(f.name for f in unknown[:3])
                            + ("…" if len(unknown) > 3 else "")))
    for f in frames:
        f.camera = _camera(f)[0]
        f.night = night_of(f.date_obs, _float(f.header.get("SITELONG"))) if split_nights else "all"
    _apply_filters([f for f in frames if f.kind in ("light", "flat")], filter_aliases or {})

    lights = [f for f in frames if f.kind == "light"]
    project = Project(name=name or target_name(lights), files=frames, filter_aliases=dict(filter_aliases or {}))
    project.trains = _trains([f for f in lights if f.shape])

    # APU Pick sidecar：每個 light 所在的資料夾各一份
    sidecars: dict[Path, dict | None] = {}
    for f in lights:
        folder = f.path.parent
        if folder not in sidecars:
            sidecars[folder] = load_pick_sidecar(folder, warnings)
        pick = sidecars[folder]
        if pick is None:
            continue
        f.hash = f.hash or cached_sha256(f.path)
        hit = pick.get(f.hash)
        if hit is None:
            warnings.append(Msg("msg.pick_no_hash", name=f.name))
        else:
            f.pick_metrics, f.pick_group = hit[1], hit[1].group

    _check_rows(project, warnings)
    project.calibration_sets = build_sets([f for f in frames if f.kind in CAL_KINDS and f.accepted])
    match_calibration(project, warnings, temp_tolerance, overrides, flat_any_night)
    build_groups(project, merge_trains, warnings)
    return project
