"""Stage 2 + Stage 3 偵測：每張 frame 校正、存成 float32 .npy、找星。每張互相獨立，用多個行程平行處理。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .calibration import Masters, calibrate
from .i18n import Msg, get_language, set_language
from .imageio import load_image
from .localnorm import background_grid
from .stars import detect

_W: dict = {}


def _init(masters: Masters, bayer: str | None, lang: str | None = None, grid: bool = False) -> None:
    _W["masters"], _W["bayer"], _W["grid"] = masters, bayer, grid
    if lang:
        set_language(lang)  # 子行程的錯誤訊息用跟主行程相同的語言


def channel_locations(data: np.ndarray, bayer: str | None) -> list[float]:
    """每個輸出通道的背景位置（中位數，先剔除 5σ 以上的星點與軌跡）。Stage 4 正規化用。"""
    if not bayer:
        return [_robust_loc(np.asarray(data[::3, ::3]))]
    h, w = data.shape
    arr = np.asarray(data[: h - h % 2, : w - w % 2])
    return [_robust_loc(_color_samples(arr, bayer, c)[::7]) for c in "RGB"]


def _color_samples(arr: np.ndarray, bayer: str, color: str) -> np.ndarray:
    """這個顏色的所有像素，依原本的列優先順序排（與 `arr[cfa_masks(...)[color]]` 相同），用跨步切片取，不建整張遮罩。"""
    pos = [(i // 2, i % 2) for i, c in enumerate(bayer) if c == color]
    if len(pos) == 1:
        dy, dx = pos[0]
        return arr[dy::2, dx::2].ravel()
    out = np.empty((arr.shape[0], arr.shape[1] // 2), arr.dtype)  # G：偶數列與奇數列各一個位置
    for dy, dx in pos:
        out[dy::2] = arr[dy::2, dx::2]
    return out.ravel()


def _robust_loc(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    med = np.median(a)
    sig = 1.4826 * np.median(np.abs(a - med))
    a = a[np.abs(a - med) < 5 * max(sig, 1e-6)]
    return float(np.median(a))


def _one(task: tuple[str, str]):
    """(原始檔, 校正後的 .npy) → (校正後的 .npy, Stars 或 None, 背景位置或 None, 背景格點或 None, 錯誤訊息)

    背景位置與背景格點（Stage 4 正規化用）在這裡順便算：校正完的資料已經在記憶體裡，而且是多行程。"""
    src, dst = task
    try:
        dst_p = Path(dst)
        if dst_p.is_file():
            data = np.load(dst_p, mmap_mode="r")
        else:
            raw, _ = load_image(Path(src))
            data, _fixed = calibrate(raw, _W["masters"], _W["bayer"])
            tmp = dst_p.with_suffix(".partial.npy")
            np.save(tmp, data)
            tmp.replace(dst_p)  # 中斷時不會留下寫一半的快取
        data = np.asarray(data)
        bayer = _W["bayer"]
        grid = background_grid(data, bayer) if _W.get("grid") else None
        return dst, detect(data, bayer), channel_locations(data, bayer), grid, None
    except Exception as exc:  # noqa: BLE001  單張失敗不中斷整批
        return dst, None, None, None, str(exc)


def prepare(tasks: list[tuple[str, str]], masters: Masters, bayer: str | None, workers: int, log, progress=None,
            grid: bool = False):
    """依 tasks 順序回傳 [(Stars 或 None, 背景位置或 None, 背景格點或 None, 錯誤訊息)]。
    grid：順便算 local normalization 用的背景格點。"""
    out: list = [None] * len(tasks)
    if workers <= 1 or len(tasks) <= 1:
        _init(masters, bayer, None, grid)
        it = map(_one, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(tasks)), initializer=_init,
                                   initargs=(masters, bayer, get_language(), grid))
        it = pool.map(_one, tasks)
    try:
        for i, (_dst, stars, loc, bg, err) in enumerate(it):
            out[i] = (stars, loc, bg, err)
            done = i + 1
            if progress:
                progress(done, len(tasks))
            if done % 10 == 0 or done == len(tasks):
                log(Msg("msg.prepare_progress", done=done, total=len(tasks)))
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        _W.clear()
    return out
