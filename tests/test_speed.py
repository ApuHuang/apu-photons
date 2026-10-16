"""加速的寫法必須與原本的做法結果完全相同（Seestar 這類幾百張小圖、單色大圖時每張都要做的步驟）。"""

import numpy as np
import pytest
from scipy.ndimage import median_filter

from apu_photons.calibration import median3
from apu_photons.debayer import bilinear
from apu_photons.imageio import cfa_masks
from apu_photons.prepare import _robust_loc, channel_locations


@pytest.mark.parametrize("shape", [(7, 9), (64, 51), (1, 5), (2, 2), (120, 160)])
def test_median3_equals_scipy(shape):
    rng = np.random.default_rng(0)
    x = rng.normal(1000, 50, shape).astype(np.float32)
    x[rng.random(shape) < 0.05] = 60000.0  # 熱像素
    x[rng.random(shape) < 0.2] = 1000.0    # 重複值
    np.testing.assert_array_equal(median3(x), median_filter(x, size=3, mode="reflect"))
    y = x.copy()
    y.flat[3] = np.nan  # 有 NaN 時照舊用 scipy
    np.testing.assert_array_equal(median3(y), median_filter(y, size=3, mode="reflect"))


@pytest.mark.parametrize("bayer", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_channel_locations_equal_mask_version(bayer):
    rng = np.random.default_rng(1)
    data = rng.normal(1000, 30, (101, 134)).astype(np.float32)
    data[:, :20] += 200  # 各色版的取樣若不同，結果就會不同
    data[rng.random(data.shape) < 0.01] = 50000
    h, w = data.shape
    arr = data[: h - h % 2, : w - w % 2]
    masks = cfa_masks(bayer, arr.shape)
    old = [_robust_loc(arr[masks[c]][::7]) for c in "RGB"]
    assert channel_locations(data, bayer) == old


@pytest.mark.parametrize("bayer", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_bilinear_equals_mask_version(bayer):
    from scipy.ndimage import convolve

    from apu_photons.debayer import _K_G, _K_RB
    rng = np.random.default_rng(2)
    cfa = rng.normal(1000, 30, (40, 56)).astype(np.float32)
    masks = cfa_masks(bayer, cfa.shape)
    old = np.stack([convolve(np.where(masks[c], cfa, 0.0).astype(np.float32), k, mode="mirror")
                    for c, k in (("R", _K_RB), ("G", _K_G), ("B", _K_RB))])
    np.testing.assert_array_equal(bilinear(cfa, bayer), old)
