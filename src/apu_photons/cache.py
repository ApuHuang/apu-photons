"""快取資料夾的管理：大小、清除、清掉中斷時留下的暫存檔、磁碟空間檢查。

快取（預設在輸出資料夾的 .photons_cache）放校正後的 frame、master 校正檔、整合與 drizzle 的工作檔，
跨次執行共用；條件一變就會多出一批，不會自己變小，所以要能看大小、能清除。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from .i18n import Msg

CACHE_NAME = ".photons_cache"
# 快取裡由 Photons 建立的子資料夾；清除時只刪這些（--cache 可能指到使用者自己的資料夾）
SUBDIRS = ("calibrated", "masters", "groups", "tmp")
STALE_SECONDS = 6 * 3600   # 暫存檔超過這麼久沒動：是中斷時留下的（建一個 master 不會這麼久）
MARGIN = 200 * 2 ** 20     # 檢查磁碟空間時多留的餘量


def default_cache(output_dir: Path) -> Path:
    return Path(output_dir) / CACHE_NAME


def tmp_dir(cache: Path) -> Path:
    path = Path(cache) / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _files(folder: Path):
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    yield from _files(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield entry
    except OSError:
        return


def cache_size(cache: Path) -> int:
    """快取占用的位元組數；資料夾不存在是 0。"""
    total = 0
    for sub in SUBDIRS:
        for entry in _files(Path(cache) / sub):
            try:
                total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def clear_cache(cache: Path) -> int:
    """刪掉快取裡 Photons 建立的內容，回傳釋放的位元組數。刪不掉的（別的程式正在用）留著。"""
    cache = Path(cache)
    before = cache_size(cache)
    for sub in SUBDIRS:
        shutil.rmtree(cache / sub, ignore_errors=True)
    try:
        cache.rmdir()  # 空了才刪得掉；裡面有別的東西就留著
    except OSError:
        pass
    return before - cache_size(cache)


def clean_stale(cache: Path) -> None:
    """清掉中斷（當掉、被強制結束）時留下的暫存檔：快取的 tmp，以及 0.2.2 以前放在系統暫存資料夾的。
    只刪放了一段時間的，不會動到另一個正在執行的 Photons。"""
    old = time.time() - STALE_SECONDS
    candidates = list((Path(cache) / "tmp").glob("*.npy"))
    candidates += list(Path(tempfile.gettempdir()).glob("apu_photons_*.npy"))
    for p in candidates:
        try:
            if p.stat().st_mtime < old:
                p.unlink()
        except OSError:
            pass


def free_bytes(path: Path) -> int | None:
    """path 所在磁碟的可用空間（path 還不存在時看最近的上層資料夾）；查不到回傳 None。"""
    p = Path(path).absolute()
    for folder in (p, *p.parents):
        if folder.exists():
            try:
                return shutil.disk_usage(folder).free
            except OSError:
                return None
    return None


def format_size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def require_space(folder: Path, need: int, what: str) -> None:
    """folder 所在的磁碟放不下 need 位元組就報錯（寫到一半才滿，錯誤訊息看不出原因，還留下半成品）。
    what：i18n 代號，說明這些空間是做什麼用的。"""
    if need <= 0:
        return
    free = free_bytes(folder)
    if free is not None and free < need + MARGIN:
        raise RuntimeError(Msg("msg.disk_full", folder=folder, what=Msg(what), need=format_size(need),
                               free=format_size(free)))
