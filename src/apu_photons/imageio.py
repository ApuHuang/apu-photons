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

from .i18n import Msg

FITS_SUFFIXES = {".fit", ".fits", ".fts"}
RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".pef", ".dng", ".srw"}
IMAGE_SUFFIXES = FITS_SUFFIXES | RAW_SUFFIXES
BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


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
        if data.ndim != 2 or pattern is None or pattern.shape != (2, 2):
            # X-Trans 與已解馬賽克的 DNG：MVP 不支援
            raise ValueError(Msg("msg.raw_bayer_only", name=path.name))
        data -= black[raw.raw_colors_visible]
        desc = raw.color_desc.decode()
        header["BAYERPAT"] = "".join(desc[c] for c in pattern.flatten())
        header["SATLEVEL"] = float(raw.white_level) - float(black.max())
        other = getattr(raw, "other", None)
        if other is not None:
            if other.shutter_speed:
                header["EXPTIME"] = float(other.shutter_speed)
            if other.iso_speed:
                header["ISO"] = float(other.iso_speed)
            if isinstance(other.timestamp, datetime):
                header["DATE-OBS"] = other.timestamp.isoformat(timespec="seconds")
    header["NAXIS2"], header["NAXIS1"] = data.shape
    if "DATE-OBS" not in header:
        header["DATE-OBS"] = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    return data, header


def bayer_pattern(header: fits.Header) -> str | None:
    """header 裡的 Bayer 排列，已依 XBAYROFF / YBAYROFF 位移；單色回傳 None。"""
    pat = str(header.get("BAYERPAT", "")).strip().upper()
    if pat not in BAYER_PATTERNS:
        return None
    dx = int(header.get("XBAYROFF", 0)) % 2
    dy = int(header.get("YBAYROFF", 0)) % 2
    if dx or dy:
        grid = [[pat[0], pat[1]], [pat[2], pat[3]]]
        pat = "".join(grid[(r + dy) % 2][(c + dx) % 2] for r in range(2) for c in range(2))
    return pat


def cfa_masks(pattern: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """每個顏色在整張影像上的布林遮罩；G 兩格合在一起。"""
    h, w = shape
    yy, xx = np.indices((h, w))
    masks = {c: np.zeros((h, w), dtype=bool) for c in "RGB"}
    for i, color in enumerate(pattern):
        masks[color] |= ((yy % 2) == i // 2) & ((xx % 2) == i % 2)
    return masks


def write_fits(path: Path, data: np.ndarray, header: fits.Header | None = None, bits: int = 32) -> None:
    """輸出 FITS。RGB 為 (3, H, W)。bits=16 時線性縮到 0~65535。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else fits.Header()
    out = np.nan_to_num(data, nan=0.0).astype(np.float32)
    if bits == 16:
        lo, hi = float(out.min()), float(out.max())
        out = ((out - lo) / (hi - lo or 1.0) * 65535.0).round().astype(np.uint16)
    fits.PrimaryHDU(out, header=hdr).writeto(path, overwrite=True)
