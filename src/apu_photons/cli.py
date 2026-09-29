"""命令列：apu-photons stack LIGHT資料夾... --dark D --flat F --bias B -o master.fits"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import Settings, run
from .imageio import list_images
from .model import Calibration


def _files(folder: str | None) -> list[Path]:
    return list_images(Path(folder)) if folder else []


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="apu-photons", description="APU Photons：天文攝影校正、對齊、疊圖")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stack", help="校正、對齊並整合 light frames")
    st.add_argument("lights", nargs="+", help="light frame 資料夾（可以多個，例如不同晚）")
    st.add_argument("-o", "--output", required=True, help="輸出的 master FITS")
    st.add_argument("--bias", help="bias 資料夾")
    st.add_argument("--dark", help="dark 資料夾")
    st.add_argument("--flat", help="flat 資料夾")
    st.add_argument("--flat-dark", help="flat-dark 資料夾")
    st.add_argument("--rejection", choices=["winsorized", "sigma", "average", "median"], default="winsorized")
    st.add_argument("--low", type=float, default=4.0, help="低端剔除 σ（預設 4）")
    st.add_argument("--high", type=float, default=3.0, help="高端剔除 σ（預設 3）")
    st.add_argument("--no-weights", action="store_true", help="不加權（每張權重相同）")
    st.add_argument("--pick-boost", action="store_true", help="用 APU Pick 分數加成權重（預設只當篩選）")
    st.add_argument("--reference", help="指定參考 frame 的檔名")
    st.add_argument("--downsample", action="store_true", help="輸出 0.5×（整合後 2×2 平均）")
    st.add_argument("--drizzle", type=int, choices=[1, 2], default=0,
                    help="Drizzle 輸出倍率（OSC 自動用 CFA drizzle）；一般疊圖另存為 *_stack.fits")
    st.add_argument("--pixfrac", type=float, default=0.9, help="Drizzle 的 drop 大小（預設 0.9）")
    st.add_argument("--no-fill-holes", action="store_true", help="Drizzle 沒資料的像素保留為空（NaN→0）")
    st.add_argument("--gpu", choices=["auto", "gpu", "cpu"], default="auto",
                    help="整合與 drizzle 用 NVIDIA GPU（auto：有就用）")
    st.add_argument("--workers", type=int, default=0, help="CPU 平行處理數（預設：核心數 − 1）")
    st.add_argument("--no-crop", action="store_true", help="不裁切到共同區域")
    st.add_argument("--bits", type=int, choices=[16, 32], default=32)
    st.add_argument("--memory", type=int, default=2048, help="整合時的記憶體上限（MB，預設 2048）")
    st.add_argument("--one-session", action="store_true", help="不依觀測夜分 session")
    st.add_argument("--no-rejection-maps", action="store_true", help="不保存每張的剔除遮罩（省硬碟；之後 drizzle 需要）")
    st.add_argument("--preview", type=int, default=0, metavar="N", help="只抽樣 N 張快速試跑")
    st.add_argument("--cache", help="快取資料夾（預設：輸出旁的 .photons_cache）")
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
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    q = res.qc
    print(f"\n整合 {q['frames_integrated']}/{q['frames_total']} 張 → {res.output}")
    if q["rejected_by_reason"]:
        print("未使用：" + "、".join(f"{k} {v}" for k, v in q["rejected_by_reason"].items()))
    if q["suspicious_frames"]:
        print("剔除像素特別多（可能有衛星、飛機或雲）：" + "、".join(q["suspicious_frames"]))
    if "drizzle" in q:
        holes = "、".join(f"{v:.2%}" for v in q["drizzle"]["holes_fraction"])
        print(f"Drizzle {q['drizzle']['scale']}×：沒有資料的像素 {holes}")
    else:
        print(f"Drizzle 適用性：{'適合' if q['drizzle_suitable'] else '不建議（張數或 dither 不足）'}")
    for w in res.warnings:
        print(f"⚠ {w}")
    print(f"Recipe：{res.recipe}")
    return 0
