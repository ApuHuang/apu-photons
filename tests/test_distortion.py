"""0.3：內插限幅（亮星旁的暗環）與畸變修正（選用）。合成資料。"""

import numpy as np
import pytest
from astropy.io import fits

from apu_photons import backend
from apu_photons.engine import Settings, run
from apu_photons.integrate import FrameJob, warp_band
from apu_photons.register import MAX_RESIDUAL, _register_task, apply, fit_distortion, register
from apu_photons.stars import detect

from . import synth

K = 0.02  # 徑向畸變：離中心半個畫面寬的地方往外 2%（角落約 5 px）
SHIFTS = [(0, 0, 0), (38, -25, 0.4), (-35, 28, -0.3), (30, 33, 0.2), (-28, -36, 0.5), (41, 8, -0.2),
          (-12, 40, 0.1), (20, -38, -0.4)]


def _stars(tx, ty, rot, seed, catalog):
    s = detect(synth.render(tx, ty, rot, seed=seed, catalog=catalog, distort=K) - synth.PEDESTAL, None)
    return np.column_stack([s.x, s.y])


def test_fit_distortion_recovers_the_corners():
    """位移很大、鏡頭又有畸變：相似變換在角落差了好幾個像素，加上多項式修正後全畫面都對得上。"""
    catalog = synth.star_catalog(n=420, seed=3)
    ref = _stars(0, 0, 0, 1, catalog)
    src = _stars(38, -25, 0.4, 2, catalog)
    shape = (synth.H, synth.W)
    m, rms_sim, *_ = register(src, ref, max_residual=2.0)
    info, rms, si, ri = fit_distortion(src, ref, m, shape, shape)
    assert rms < 0.25 and rms < rms_sim * 0.6
    assert len(si) > (np.linalg.norm(apply(m, src)[:, None] - ref[None], axis=2).min(axis=1) < 2).sum() * 0.95
    assert 0.5 < info["max"] < 20
    # 正向與反向互相對得回來
    pts = np.column_stack([np.linspace(5, synth.W - 5, 9), np.linspace(5, synth.H - 5, 9)])
    from apu_photons.register import _poly
    fwd = apply(m, pts) + _poly(info["fwd"], pts, shape)
    back = apply(np.linalg.inv(m), fwd) + _poly(info["inv"], fwd, shape)
    assert np.abs(back - pts).max() < 0.1
    # 行程池用的入口：沒開畸變修正時這一張過不了原本的殘差門檻，或殘差明顯比較大
    plain, _err = _register_task((src, ref, 1.0, None))
    fixed, err = _register_task((src, ref, 1.0, (shape, shape)))
    assert err is None and fixed[4] is not None and fixed[1] < MAX_RESIDUAL
    assert plain is None or plain[1] > fixed[1] * 1.5


def test_no_distortion_means_no_correction():
    """沒有畸變的資料：修正不會比相似變換好，就不用（回傳 None），結果與沒開一樣。"""
    catalog = synth.star_catalog(n=420, seed=3)

    def stars(tx, ty, rot, seed):
        s = detect(synth.render(tx, ty, rot, seed=seed, catalog=catalog) - synth.PEDESTAL, None)
        return np.column_stack([s.x, s.y])

    ref, src = stars(0, 0, 0, 1), stars(12, -9, 0.3, 2)
    res, err = _register_task((src, ref, 1.0, ((synth.H, synth.W),) * 2))
    assert err is None
    if res[4] is not None:                      # 有擬合也只能是很小的修正
        assert res[4]["max"] < 0.5
    few = fit_distortion(src[:12], ref[:12], np.eye(3), (synth.H, synth.W), (synth.H, synth.W))
    assert few is None                          # 星太少不擬合


def _corner_fwhm(path):
    data = fits.getdata(path).astype(np.float32)
    s = detect(data, None)
    return s.fwhm, len(s)


@pytest.mark.parametrize("drizzle", [0, 1])
def test_stack_with_distortion_is_sharper(tmp_path, drizzle):
    catalog = synth.star_catalog(n=420, seed=3)
    light = tmp_path / "light"
    for i, (tx, ty, rot) in enumerate(SHIFTS):
        img = synth.render(tx, ty, rot, seed=50 + i, catalog=catalog, distort=K)
        synth.write(light / f"L_{i:03d}.fits", img, EXPTIME=60.0, DATE_OBS=f"2026-09-20T14:{i:02d}:00")
    common = dict(workers=1, gpu="cpu", crop_common=False, reference="L_000.fits", drizzle=drizzle)
    off = run([light], tmp_path / "off", Settings(**common), echo=None)
    on = run([light], tmp_path / "on", Settings(distortion=True, **common), echo=None)
    assert on.qc["frames_integrated"] == 8
    assert sum(f.distortion is not None for f in on.project.lights) >= 6
    assert all(f.distortion is None for f in off.project.lights)
    f_on, n_on = _corner_fwhm(on.output)
    if off.outputs:                             # 沒開時有些張對不上被排除，剩下的星也比較胖
        f_off, _n = _corner_fwhm(off.output)
        assert off.qc["frames_integrated"] < 8 or f_on < f_off
    assert f_on < 3.6 and n_on > 100            # 單張 FWHM 3.0；疊完還是圓的


def test_clamp_removes_the_dark_ring():
    """欠取樣的亮星位移半個像素：三次樣條在星旁衝到比天空低很多；限幅後不會，峰值不變。"""
    yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
    sig = 1.5 / 2.3548
    img = (1000 + 20000 * np.exp(-((xx - 80.3) ** 2 + (yy - 59.6) ** 2) / (2 * sig * sig))).astype(np.float32)
    t = np.eye(3)
    t[0, 2], t[1, 2] = 0.5, 0.5
    job = FrameJob("", t, [1.0], [0.0], 1.0, noise=15.0)      # 背景雜訊 15 ADU 的 frame
    raw = warp_band(backend.CPU, job, (img, 0, 0), None, 0, 120, 160, clamp=False)[0]
    fixed = warp_band(backend.CPU, job, (img, 0, 0), None, 0, 120, 160, clamp=True)[0]
    inner = (slice(10, 110), slice(10, 150))
    assert raw[inner].min() < 1000 - 200            # 沒限幅：低於天空幾百 ADU
    assert fixed[inner].min() > 1000 - 5            # 限幅：剩下的起伏遠小於雜訊
    flat = np.full((120, 160), 1000, np.float32) + np.random.default_rng(0).normal(0, 15, (120, 160)).astype(np.float32)
    a = warp_band(backend.CPU, job, (flat, 0, 0), None, 0, 120, 160, clamp=False)[0]
    b = warp_band(backend.CPU, job, (flat, 0, 0), None, 0, 120, 160, clamp=True)[0]
    assert np.array_equal(a[inner], b[inner])       # 只有雜訊的背景完全不動
    assert fixed[inner].max() == raw[inner].max()   # 只限低的那一側，峰值不被削
    assert np.array_equal(np.isnan(raw), np.isnan(fixed))
