"""富士 X-Trans（6×6 CFA）：讀排列、熱像素、找星、色差解馬賽克、整合與 CFA Drizzle。"""

import numpy as np
import pytest
from astropy.io import fits

from apu_photons import cfa
from apu_photons.calibration import cosmetic, hot_from_dark
from apu_photons.debayer import bilinear, demosaic, xtrans_colordiff
from apu_photons.engine import Settings, run
from apu_photons.imageio import bayer_pattern
from apu_photons.stars import detect

from . import synth
from .test_drizzle import _dithered
from .test_pipeline import _make

X = synth.XTRANS


def test_pattern_basics():
    assert cfa.period(X) == 6 and cfa.is_xtrans(X) and cfa.label(X) == "X-Trans"
    assert cfa.color_counts(X) == {"R": 8, "G": 20, "B": 8}
    assert bayer_pattern(fits.Header({"BAYERPAT": X})) == X
    # 位移後的排列：新的 (0, 0) 是原本的 (dy, dx)
    shifted = bayer_pattern(fits.Header({"BAYERPAT": X, "XBAYROFF": 1, "YBAYROFF": 2}))
    assert shifted == cfa.shift(X, 1, 2)
    assert np.array_equal(cfa.color_index(shifted), np.roll(cfa.color_index(X), (-2, -1), axis=(0, 1)))
    assert bayer_pattern(fits.Header({"BAYERPAT": "XYZ"})) is None
    # 每列、每行都有三種顏色（X-Trans 的特性，7×7 的色差內插窗一定有紅與藍）
    idx = cfa.color_index(X)
    for k in range(6):
        assert set(idx[k]) == {0, 1, 2} and set(idx[:, k]) == {0, 1, 2}


@pytest.mark.parametrize("dx, dy", [(0, 0), (1, 0), (2, 5), (4, 3)])
def test_bin_origin_gives_balanced_blocks(dx, dy):
    pat = cfa.shift(X, dx, dy)
    oy, ox = cfa.bin_origin(pat)
    ch = cfa.channel_map(pat, (30, 30))[oy:oy + 24, ox:ox + 24]
    blocks = ch.reshape(8, 3, 8, 3).transpose(0, 2, 1, 3).reshape(64, 9)
    for b in blocks:
        assert np.bincount(b, minlength=3).tolist() == [2, 5, 2]


def test_bayer_unchanged():
    rng = np.random.default_rng(0)
    img = rng.normal(1000, 30, (40, 56)).astype(np.float32)
    np.testing.assert_array_equal(demosaic(img, "RGGB"), bilinear(img, "RGGB"))
    assert cfa.bin_factor("RGGB") == 2 and cfa.bin_origin("RGGB") == (0, 0) and cfa.bin_factor(None) == 1


def test_colordiff_reconstructs_smooth_colour():
    """均勻的顏色完全還原；平緩的梯度誤差很小；原本是這個顏色的像素保持原值。"""
    h, w = 60, 72
    ch = cfa.channel_map(X, (h, w))
    level = np.array([800.0, 1000.0, 600.0], np.float32)
    flat = level[ch]
    out = xtrans_colordiff(flat, X)
    for c in range(3):
        np.testing.assert_allclose(out[c], level[c], rtol=1e-5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    truth = np.stack([800 + 2 * xx, 1000 + 3 * yy, 600 + xx + yy])
    mosaic = np.take_along_axis(truth, ch[None], axis=0)[0]
    out = xtrans_colordiff(mosaic, X)
    inner = (slice(None), slice(4, -4), slice(4, -4))
    assert np.abs(out - truth)[inner].max() < 0.5
    for c in range(3):
        np.testing.assert_array_equal(out[c][ch == c], mosaic[ch == c])


def test_colordiff_less_fringe_than_plain_interpolation():
    """白色星點：色差內插的假色比各色分別內插少很多（實拍比較見 SPEC §10.11）。"""
    from scipy import ndimage as ndi
    img = synth.render(0.3, 0.2, 0, seed=3, fwhm=2.2, bayer=None) - synth.PEDESTAL
    ch = cfa.channel_map(X, img.shape)
    mosaic = img.astype(np.float32)
    out = xtrans_colordiff(mosaic, X)
    k = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]).astype(np.float32)
    plain = np.stack([np.where(ch == c, mosaic, ndi.convolve(mosaic * (ch == c), k, mode="mirror")
                               / ndi.convolve((ch == c).astype(np.float32), k, mode="mirror")) for c in range(3)])
    stars = img > np.median(img) + 400

    def fringe(rgb):
        return np.abs(rgb[0] - rgb[1])[stars].mean() + np.abs(rgb[2] - rgb[1])[stars].mean()
    assert fringe(out) < 0.6 * fringe(plain)


def test_white_balance_removes_green_rim():
    """沒有白平衡的原始資料（紅、藍只有綠的一半左右）：不調整時紅藍的星點比綠窄、周圍一圈偏綠；
    依相機白平衡調整後各色的星點形狀一致，而且輸出仍是原始色彩（總亮度比例不變）。"""
    resp = np.array([0.45, 1.0, 0.6], np.float32)
    img = synth.render(0.3, 0.2, 0, seed=3, fwhm=2.5, bayer=None) - synth.PEDESTAL
    ch = cfa.channel_map(X, img.shape)
    mosaic = (img * resp[ch]).astype(np.float32)
    truth = img[None] * resp[:, None, None]
    stars = img > np.median(img) + 300

    def shape_err(rgb):  # 各色除以自己的響應後與真值的差（形狀不同就會有差）
        return np.abs(rgb / resp[:, None, None] - img[None])[:, stars].mean()
    plain = xtrans_colordiff(mosaic, X)
    balanced = xtrans_colordiff(mosaic, X, wb=(1 / 0.45, 1 / 0.6))
    assert shape_err(balanced) < 0.5 * shape_err(plain)
    from scipy.ndimage import binary_dilation
    ap = binary_dilation(stars, iterations=4)  # 整顆星（內插只是在星點內重新分配亮度）
    flux = balanced[:, ap].sum(axis=1)
    np.testing.assert_allclose(flux / flux[1], truth[:, ap].sum(axis=1) / truth[1, ap].sum(), rtol=0.02)
    for c in range(3):  # 原本是這個顏色的像素保持原值
        np.testing.assert_array_equal(balanced[c][ch == c], mosaic[ch == c])


def test_cosmetic_removes_hot_keeps_stars():
    hot = synth.hot_pixels()
    clean = synth.render(0, 0, 0, seed=9, fwhm=2.5, bayer=X)
    dirty = synth.render(0, 0, 0, seed=9, fwhm=2.5, bayer=X, hot=hot)
    fixed, n = cosmetic(dirty.copy(), X, None)
    assert n >= hot.sum() * 0.9
    assert np.abs(fixed[hot] - clean[hot]).max() < 500  # 熱像素加了 20000
    # 星點的總亮度幾乎不變（沒有把星點核心當成熱像素削掉）
    stars = clean > np.median(clean) + 1000
    assert abs(fixed[stars & ~hot].sum() / clean[stars & ~hot].sum() - 1) < 0.01


def test_hot_from_dark_xtrans():
    rng = np.random.default_rng(4)
    hot = synth.hot_pixels()
    dark = rng.normal(synth.PEDESTAL, synth.READ_NOISE, hot.shape).astype(np.float32)
    dark[hot] += 2000
    bad = hot_from_dark(dark, X)
    assert bad[hot].all() and (bad & ~hot).sum() < 5


def test_fwhm_xtrans():
    for fwhm in (3.0, 4.5, 6.0):
        got = detect(synth.render(0.3, 0.6, 0, seed=5, fwhm=fwhm, bayer=X), X).fwhm
        assert abs(got - fwhm) < 0.2 * fwhm, (fwhm, got)


def test_star_positions_xtrans():
    """3×3 合併後換回原始像素的座標正確（與單色找到的位置比）。"""
    a = detect(synth.render(0.3, 0.6, 0, seed=5, fwhm=3.5, bayer=None), None)
    b = detect(synth.render(0.3, 0.6, 0, seed=5, fwhm=3.5, bayer=X), X)
    pa = np.column_stack([a.x, a.y])[:40]
    pb = np.column_stack([b.x, b.y])
    d = np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1)).min(axis=1)
    assert np.median(d) < 0.25


def test_full_stack_xtrans(tmp_path):
    light, inputs = _make(tmp_path, bayer=X)
    res = run(inputs, tmp_path / "out", Settings(crop_common=False, reference="L_000.fits"), echo=None)
    data = fits.getdata(res.output)
    assert data.shape == (3, synth.H, synth.W)
    assert res.qc["frames_integrated"] == 8
    inner = data[:, 40:-40, 40:-40]
    bg = np.median(inner, axis=(1, 2))
    # 各色響應 R 0.8、G 1、B 0.6（flat 每色各自正規化，不改顏色）
    assert abs(bg[0] / bg[1] - 0.8) < 0.03 and abs(bg[2] / bg[1] - 0.6) < 0.03
    # 熱像素與暗角已修正
    lum = data.mean(axis=0)
    assert abs(np.median(lum[20:50, 20:60]) / np.median(lum[110:150, 140:180]) - 1) < 0.05
    trail = res.project.frames[3]
    others = [f.rejected_pixel_fraction for f in res.project.frames if f is not trail]
    assert trail.rejected_pixel_fraction > 1.8 * max(others)


def test_cfa_drizzle_1x_xtrans(tmp_path):
    light, inputs = _make(tmp_path, bayer=X, shifts=_dithered(16))
    res = run(inputs, tmp_path / "c", Settings(drizzle=1, pixfrac=1.0, crop_common=False), echo=None)
    driz = fits.getdata(res.output)
    stack = fits.getdata(res.outputs[0].stack)
    assert driz.shape == stack.shape == (3, synth.H, synth.W)
    holes = res.qc["groups"][0]["drizzle"]["holes_fraction"]
    assert holes[1] <= min(holes[0], holes[2])  # 綠的取樣最多
    inner = (slice(30, -30), slice(30, -30))
    for c in range(3):
        assert abs(np.median(driz[c][inner]) / np.median(stack[c][inner]) - 1) < 0.02
    assert np.isfinite(driz).all()
