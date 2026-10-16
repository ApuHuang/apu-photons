"""Local normalization：各張梯度不同時，把每張的大尺度背景對齊到參考 frame。"""

import numpy as np
import pytest
from astropy.io import fits

from apu_photons import localnorm
from apu_photons.engine import Settings, run

from . import synth
from .test_pipeline import _make


def _plane(h, w, base, gx, gy, seed):
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(seed)
    return (base + gx * (xx / (w - 1) - 0.5) + gy * (yy / (h - 1) - 0.5) + rng.normal(0, 20, (h, w))).astype(np.float32)


def test_fit_offset_recovers_gradient_difference():
    h, w = 512, 768
    ref = _plane(h, w, 1000.0, 0.0, 0.0, 1)
    img = _plane(h, w, 800.0, 300.0, -200.0, 2)
    eye = np.eye(3)
    coeffs = localnorm.fit_offset(localnorm.background_grid(img, None), eye, (h, w),
                                  localnorm.background_grid(ref, None), eye, (h, w), 1.0, (h, w))
    assert coeffs is not None and len(coeffs) == 1
    yy, xx = np.mgrid[h // 10: h - h // 10: 16, w // 10: w - w // 10: 16].astype(np.float32)
    got = localnorm.eval_offset(np, coeffs[0], xx, yy, (h, w))
    want = 200.0 - 300.0 * (xx / (w - 1) - 0.5) + 200.0 * (yy / (h - 1) - 0.5)
    assert np.abs(got - want).max() < 10.0


def test_fit_offset_without_overlap_returns_none():
    h, w = 256, 320
    img = _plane(h, w, 500.0, 0.0, 0.0, 3)
    grid = localnorm.background_grid(img, None)
    away = np.eye(3)
    away[0, 2] = 5000.0  # 參考座標 → 這張座標：全部落在影像外
    assert localnorm.fit_offset(grid, away, (h, w), grid, np.eye(3), (h, w), 1.0, (h, w)) is None


def _add_gradients(light_dir, amplitudes):
    """每張 light 加上不同的線性梯度（模擬各晚光害方向不同）。第一張（參考）不加。"""
    for path, (gx, gy) in zip(sorted(light_dir.iterdir()), amplitudes):
        with fits.open(path) as hdul:
            data, hdr = hdul[0].data.astype(np.float64), hdul[0].header.copy()
        yy, xx = np.mgrid[0:synth.H, 0:synth.W]
        data += gx * (xx / (synth.W - 1) - 0.5) + gy * (yy / (synth.H - 1) - 0.5)
        fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), header=hdr).writeto(path, overwrite=True)


def _tilt(img):
    """影像背景的大尺度梯度：左右、上下四分之一的中位數差。"""
    img = img if img.ndim == 2 else img.mean(axis=0)
    h, w = img.shape
    return (np.median(img[:, : w // 4]) - np.median(img[:, -w // 4:]),
            np.median(img[: h // 4]) - np.median(img[-h // 4:]))


@pytest.mark.parametrize("bayer, drizzle", [(None, 0), ("RGGB", 1)])
def test_local_normalization_matches_reference_background(tmp_path, bayer, drizzle):
    light, inputs = _make(tmp_path, bayer=bayer, with_cal=False, trail_index=-1)
    amps = [(0, 0)] + [(180 + 20 * i, -120 + 10 * i) for i in range(len(list(light.iterdir())) - 1)]
    _add_gradients(light, amps)
    base = dict(crop_common=False, reference="L_000.fits", drizzle=drizzle, target="m")
    res_g = run(inputs, tmp_path / "global", Settings(**base), echo=None)
    res_l = run(inputs, tmp_path / "local", Settings(normalization="local", **base), echo=None)
    frames = [f for f in res_l.project.lights if f.accepted]
    assert all(f.norm.get("local") for f in frames if f.name != "L_000.fits")
    assert all(len(f.norm["local"]) == (3 if bayer else 1) for f in frames if f.norm.get("local"))
    # 全域正規化的成品帶著各張梯度的平均；局部正規化對齊到參考那張（沒有梯度）
    tg = _tilt(fits.getdata(res_g.output))
    tl = _tilt(fits.getdata(res_l.output))
    assert abs(tg[0]) > 60 and abs(tl[0]) < 0.25 * abs(tg[0]), (tg, tl)
    assert abs(tl[1]) < 0.25 * abs(tg[1]) + 5, (tg, tl)
    rej_g = np.mean([f.rejected_pixel_fraction for f in res_g.project.lights if f.accepted])
    rej_l = np.mean([f.rejected_pixel_fraction for f in frames])
    assert rej_l <= rej_g
