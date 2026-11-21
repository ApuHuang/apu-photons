"""選參考張的排名：同分時的順序固定（0.4.1），跟輸入順序、排序演算法無關。"""

from itertools import permutations
from pathlib import Path

import numpy as np

from apu_photons.engine import _choose_reference, _rank_frames
from apu_photons.model import Frame, Stars


def _frame(name: str, fwhm: float, n: int, bg: float, folder: str = "night") -> Frame:
    z = np.zeros(n)
    return Frame(path=Path(folder) / name, exposure=60.0, kind="light",
                 stars=Stars(z, z, z, fwhm, bg, 1.0, 10.0))


def _names(frames):
    return [f.name for f in frames]


def test_identical_frames_pick_first_name_in_any_order():
    frames = [_frame(n, 2.5, 30, 100.0) for n in ("c.fits", "a.fits", "b.fits")]
    for perm in permutations(frames):
        assert _names(_rank_frames(list(perm))) == ["a.fits", "b.fits", "c.fits"]
        assert _choose_reference(list(perm), None, []).name == "a.fits"


def test_score_tie_broken_by_fwhm_then_name():
    # z：FWHM 第 1（0 分×2），星數、背景都第 2 → 2 分；a：FWHM 第 2（2 分），星數、背景都第 1 → 2 分。同分 → FWHM 小的 z
    z = _frame("z.fits", 2.0, 20, 200.0)
    a = _frame("a.fits", 2.1, 40, 100.0)
    for perm in permutations([z, a]):
        assert _names(_rank_frames(list(perm))) == ["z.fits", "a.fits"]
    # 總分、FWHM 都一樣 → 檔名
    b = _frame("b.fits", 2.0, 20, 200.0)
    for perm in permutations([z, a, b]):
        assert _names(_rank_frames(list(perm))) == ["b.fits", "z.fits", "a.fits"]


def test_tied_values_share_rank():
    """同值同名次：三張星數一樣時，星數這一項不分高下，由 FWHM 與背景決定。"""
    frames = [_frame("x.fits", 2.4, 30, 120.0), _frame("y.fits", 2.4, 30, 110.0), _frame("w.fits", 2.6, 30, 100.0)]
    want = None
    for perm in permutations(frames):
        got = _names(_rank_frames(list(perm)))
        want = want or got
        assert got == want
    assert want[0] == "y.fits"     # FWHM 並列第一（0）×2 ＋ 星數 0 ＋ 背景第 2 名（1）＝ 1；x＝2；w＝4＋0＋0＝4


def test_same_name_in_two_folders_broken_by_path():
    a = _frame("IMG_0001.CR2", 2.5, 30, 100.0, "n2")
    b = _frame("IMG_0001.CR2", 2.5, 30, 100.0, "n1")
    for perm in permutations([a, b]):
        assert _rank_frames(list(perm))[0].path.parent.name == "n1"


def test_nan_fwhm_frames_still_ranked():
    frames = [_frame("b.fits", float("nan"), 0, 100.0), _frame("a.fits", float("nan"), 0, 100.0)]
    assert _names(_rank_frames(frames)) == ["a.fits", "b.fits"]
