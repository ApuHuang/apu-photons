"""讀寫影像：FITS 與相機 RAW，一律轉成 2D float32 CFA（或單色）加上 FITS 欄位名稱的 header。

RAW 的讀法比照 APU Pick 的 rawfile.py（rawpy 扣黑位後當 CFA），將來兩邊可以抽成共用套件。
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import cfa
from .cfa import BAYER_PATTERNS  # noqa: F401  （舊的匯入位置）
from .i18n import Msg

FITS_SUFFIXES = {".fit", ".fits", ".fts"}
RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".pef", ".dng", ".srw"}
IMAGE_SUFFIXES = FITS_SUFFIXES | RAW_SUFFIXES
TOP_DOWN, BOTTOM_UP = "TOP-DOWN", "BOTTOM-UP"


def is_image(path: Path) -> bool:
    """能處理的檔案；略過 macOS 在 exFAT 上留下的「._檔名」附屬檔。"""
    return path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and is_image(p))


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> fits.Header:
    """只讀 header（FITS 不載入影像；RAW 只能整個讀）。"""
    if path.suffix.lower() in RAW_SUFFIXES:
        return load_image(path)[1]
    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if h.header.get("NAXIS", 0) >= 2)
        return hdu.header.copy()


def load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    if path.suffix.lower() in RAW_SUFFIXES:
        return _load_raw(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    if data.ndim != 2:
        raise ValueError(Msg("msg.not_2d", name=path.name, shape=data.shape))
    return data, header


def _load_raw(path: Path) -> tuple[np.ndarray, fits.Header]:
    import rawpy

    header = fits.Header()
    with rawpy.imread(str(path)) as raw:
        data = raw.raw_image_visible.astype(np.float32)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        pattern = raw.raw_pattern
        if data.ndim != 2 or pattern is None or pattern.shape not in ((2, 2), (6, 6)):
            # 已解馬賽克的 DNG 與其他排列不支援；2×2 是 Bayer、6×6 是富士 X-Trans
            raise ValueError(Msg("msg.raw_bayer_only", name=path.name))
        data -= black[raw.raw_colors_visible]
        desc = raw.color_desc.decode()
        header["BAYERPAT"] = "".join(desc[c] for c in pattern.flatten())
        header["SATLEVEL"] = float(raw.white_level) - float(black.max())
        wb = [float(v) for v in (getattr(raw, "camera_whitebalance", None) or [])][:3]
        if len(wb) == 3 and min(wb) > 0:
            # 相機的白平衡（相對綠）：只給 X-Trans 解馬賽克用（debayer.xtrans_colordiff），輸出仍是原始色彩
            header["WBGAINR"] = (round(wb[0] / wb[1], 5), "camera white balance R/G (demosaic only)")
            header["WBGAINB"] = (round(wb[2] / wb[1], 5), "camera white balance B/G (demosaic only)")
        other = getattr(raw, "other", None)
        if other is not None:
            if other.shutter_speed:
                header["EXPTIME"] = float(other.shutter_speed)
            if other.iso_speed:
                header["ISO"] = float(other.iso_speed)
            if isinstance(other.timestamp, datetime):
                header["DATE-OBS"] = other.timestamp.isoformat(timespec="seconds")
            if getattr(other, "focal_length", 0):
                header["FOCALLEN"] = float(other.focal_length)  # 分光學系統用；RAW 沒有像素大小，不會算出像素尺度
        lens = getattr(raw, "lens", None)
        model = str(getattr(lens, "model", "") or "").strip()
        if model:
            header["LENS"] = model[:68]
    header["NAXIS2"], header["NAXIS1"] = data.shape
    header["ROWORDER"] = TOP_DOWN  # rawpy 的第 0 列是畫面最上面
    if "DATE-OBS" not in header:
        header["DATE-OBS"] = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    return data, header


def row_order(header) -> str:
    """影像陣列的列順序。FITS 沒寫 ROWORDER 就照標準當 bottom-up；相機 RAW 讀進來時已標 TOP-DOWN。

    Photons 不翻轉資料（CFA 上下翻轉會改變 Bayer 排列，校正檔也得一起翻），只在輸出時標明實際的方向。
    """
    return TOP_DOWN if str(header.get("ROWORDER", "")).strip().upper() == TOP_DOWN else BOTTOM_UP


def bayer_pattern(header: fits.Header) -> str | None:
    """header 裡的 CFA 排列（Bayer 4 個字、X-Trans 36 個字），已依 XBAYROFF / YBAYROFF 位移；單色回傳 None。"""
    pat = str(header.get("BAYERPAT", "")).strip().upper()
    if not cfa.is_valid(pat):
        return None
    p = cfa.period(pat)
    dx = int(header.get("XBAYROFF", 0)) % p
    dy = int(header.get("YBAYROFF", 0)) % p
    if dx or dy:
        pat = cfa.shift(pat, dx, dy)
    return pat


def cfa_masks(pattern: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """每個顏色在整張影像上的布林遮罩（同色的各格合在一起）。"""
    return cfa.masks(pattern, shape)


def write_fits(path: Path, data: np.ndarray, header: fits.Header | None = None, bits: int = 32) -> None:
    """輸出 FITS。RGB 為 (3, H, W)。bits=16 時線性縮到 0~65535，原本的範圍記在 LINMIN／LINMAX：
    線性值 = LINMIN + 像素值 × (LINMAX − LINMIN) / 65535，後製才還原得出校正後的數值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else fits.Header()
    out = np.nan_to_num(data, nan=0.0).astype(np.float32, copy=False)  # nan_to_num 已經是新的陣列，不用再複製一份
    if bits == 16:
        lo, hi = float(out.min()), float(out.max())
        out = ((out - lo) / (hi - lo or 1.0) * 65535.0).round().astype(np.uint16)
        hdr["LINMIN"] = (lo, "linear value stored as 0")
        hdr["LINMAX"] = (hi, "linear value stored as 65535")
    fits.PrimaryHDU(out, header=hdr).writeto(path, overwrite=True)
