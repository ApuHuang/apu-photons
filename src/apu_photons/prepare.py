"""Stage 2 + Stage 3 偵測：每張 frame 校正、存成 float32 .npy、找星。每張互相獨立，用多個行程平行處理。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .calibration import Masters, calibrate
from .i18n import Msg, get_language, set_language
from .imageio import load_image
from .stars import detect

_W: dict = {}


def _init(masters: Masters, bayer: str | None, lang: str | None = None) -> None:
    _W["masters"], _W["bayer"] = masters, bayer
    if lang:
        set_language(lang)  # 子行程的錯誤訊息用跟主行程相同的語言


def _one(task: tuple[str, str]):
    """(原始檔, 校正後的 .npy) → (校正後的 .npy, Stars 或 None, 錯誤訊息)"""
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
        return dst, detect(np.asarray(data), _W["bayer"]), None
    except Exception as exc:  # noqa: BLE001  單張失敗不中斷整批
        return dst, None, str(exc)


def prepare(tasks: list[tuple[str, str]], masters: Masters, bayer: str | None, workers: int, log, progress=None):
    """依 tasks 順序回傳 [(Stars 或 None, 錯誤訊息)]。"""
    out: list = [None] * len(tasks)
    if workers <= 1 or len(tasks) <= 1:
        _init(masters, bayer)
        it = map(_one, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(tasks)), initializer=_init,
                                   initargs=(masters, bayer, get_language()))
        it = pool.map(_one, tasks)
    try:
        for i, (_dst, stars, err) in enumerate(it):
            out[i] = (stars, err)
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
