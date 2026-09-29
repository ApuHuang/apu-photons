"""命令列：apu-photons [--lang en] stack LIGHT資料夾... --dark D --flat F --bias B -o master.fits"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import Settings, run
from .i18n import LANGUAGES, set_language, tr
from .imageio import list_images
from .model import Calibration


def _files(folder: str | None) -> list[Path]:
    return list_images(Path(folder)) if folder else []


def _language(argv: list[str]) -> str:
    """說明文字要在建立 argparse 之前就決定語言，所以先自己找 --lang。"""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return "zh"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lang = _language(argv)
    if lang in LANGUAGES:
        set_language(lang)
    ap = argparse.ArgumentParser(prog="apu-photons", description=tr("cli.description"))
    ap.add_argument("--lang", choices=list(LANGUAGES), default="zh", help=tr("cli.lang"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stack", help=tr("cli.stack"))
    st.add_argument("lights", nargs="+", help=tr("cli.lights"))
    st.add_argument("-o", "--output", required=True, help=tr("cli.output"))
    st.add_argument("--bias", help=tr("cli.bias"))
    st.add_argument("--dark", help=tr("cli.dark"))
    st.add_argument("--flat", help=tr("cli.flat"))
    st.add_argument("--flat-dark", help=tr("cli.flat_dark"))
    st.add_argument("--rejection", choices=["winsorized", "sigma", "average", "median"], default="winsorized",
                    help=tr("cli.rejection"))
    st.add_argument("--low", type=float, default=4.0, help=tr("cli.low"))
    st.add_argument("--high", type=float, default=3.0, help=tr("cli.high"))
    st.add_argument("--no-weights", action="store_true", help=tr("cli.no_weights"))
    st.add_argument("--pick-boost", action="store_true", help=tr("cli.pick_boost"))
    st.add_argument("--reference", help=tr("cli.reference"))
    st.add_argument("--downsample", action="store_true", help=tr("cli.downsample"))
    st.add_argument("--drizzle", type=int, choices=[1, 2], default=0, help=tr("cli.drizzle"))
    st.add_argument("--pixfrac", type=float, default=0.9, help=tr("cli.pixfrac"))
    st.add_argument("--no-fill-holes", action="store_true", help=tr("cli.no_fill_holes"))
    st.add_argument("--gpu", choices=["auto", "gpu", "cpu"], default="auto", help=tr("cli.gpu"))
    st.add_argument("--workers", type=int, default=0, help=tr("cli.workers"))
    st.add_argument("--no-crop", action="store_true", help=tr("cli.no_crop"))
    st.add_argument("--bits", type=int, choices=[16, 32], default=32, help=tr("cli.bits"))
    st.add_argument("--memory", type=int, default=2048, help=tr("cli.memory"))
    st.add_argument("--one-session", action="store_true", help=tr("cli.one_session"))
    st.add_argument("--no-rejection-maps", action="store_true", help=tr("cli.no_rejection_maps"))
    st.add_argument("--preview", type=int, default=0, metavar="N", help=tr("cli.preview"))
    st.add_argument("--cache", help=tr("cli.cache"))
    a = ap.parse_args(argv)

    cal = Calibration(bias=_files(a.bias), dark=_files(a.dark), flat=_files(a.flat), flat_dark=_files(a.flat_dark))
    s = Settings(rejection=a.rejection, low=a.low, high=a.high, weighting="none" if a.no_weights else "snr2_over_fwhm2",
                 pick_boost=a.pick_boost, reference=a.reference, downsample=0.5 if a.downsample else 1.0,
                 crop_common=not a.no_crop, output_bits=a.bits, memory_mb=a.memory, split_nights=not a.one_session,
                 keep_rejection_maps=not a.no_rejection_maps, preview=a.preview,
                 drizzle=a.drizzle, pixfrac=a.pixfrac, fill_holes=not a.no_fill_holes,
                 gpu=a.gpu, workers=a.workers)
    try:
        res = run([Path(p) for p in a.lights], cal, Path(a.output), s, Path(a.cache) if a.cache else None)
    except RuntimeError as exc:
        print(tr("cli.error", error=exc), file=sys.stderr)
        return 1
    q = res.qc
    sep = tr("cli.sep")
    print()
    print(tr("cli.summary", used=q["frames_integrated"], total=q["frames_total"], output=res.output))
    if q["rejected_by_reason"]:
        items = sep.join(f"{tr('reason.' + k)} {v}" for k, v in q["rejected_by_reason"].items())
        print(tr("cli.unused", items=items))
    if q["suspicious_frames"]:
        print(tr("cli.suspicious", files=sep.join(q["suspicious_frames"])))
    if "drizzle" in q:
        holes = sep.join(f"{v:.2%}" for v in q["drizzle"]["holes_fraction"])
        print(tr("cli.drizzle_holes", scale=q["drizzle"]["scale"], holes=holes))
    else:
        print(tr("cli.drizzle_ok") if q["drizzle_suitable"] else tr("cli.drizzle_no"))
    for w in res.warnings:
        print(f"⚠ {w}")
    print(tr("cli.recipe", path=res.recipe))
    return 0
