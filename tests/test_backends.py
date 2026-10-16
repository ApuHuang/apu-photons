"""CPU 單行程、CPU 多行程、GPU 三種跑法的結果必須一致。"""

import numpy as np
import pytest
from astropy.io import fits

from apu_photons import backend
from apu_photons.combine import clip_combine
from apu_photons.engine import Settings, run

from .test_drizzle import _dithered
from .test_pipeline import _make

HAS_GPU = backend.select("auto").gpu
gpu_only = pytest.mark.skipif(not HAS_GPU, reason="沒有 NVIDIA GPU / CuPy")


def _run(tmp_path, name, **kw):
    res = run(_run.inputs, tmp_path / name, Settings(crop_common=False, target="m", **kw), echo=None)
    return res, fits.getdata(res.output)


@pytest.fixture
def data(tmp_path):
    _light, inputs = _make(tmp_path, bayer="RGGB", shifts=_dithered(12))
    _run.inputs = inputs
    return tmp_path


def test_cpu_single_vs_multi(data):
    _, a = _run(data, "one", gpu="cpu", workers=1, drizzle=2, memory_mb=64)
    _, b = _run(data, "many", gpu="cpu", workers=3, drizzle=2, memory_mb=64)
    # 行程數不同時分段大小不同，樣條內插在段落邊界有極小差異（約 1e-5）
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=0.05)
    np.testing.assert_allclose(fits.getdata(data / "one" / "m_stack.fits"),
                               fits.getdata(data / "many" / "m_stack.fits"), rtol=1e-4, atol=0.05)


@gpu_only
def test_gpu_matches_cpu(data):
    rc, c = _run(data, "cpu", gpu="cpu", workers=2, drizzle=2)
    rg, g = _run(data, "gpu", gpu="gpu", workers=2, drizzle=2)
    sc = fits.getdata(data / "cpu" / "m_stack.fits")
    sg = fits.getdata(data / "gpu" / "m_stack.fits")
    # 浮點誤差讓極少數剛好落在剔除門檻上的像素判斷不同（實測 25 萬像素中 2 個），其餘幾乎完全相同
    for x, y in ((sc, sg), (c, g)):
        diff = np.abs(x.astype(float) - y.astype(float))
        assert (diff > 0.1).mean() < 1e-4
        assert np.percentile(diff, 99.9) < 0.05
    assert rc.qc["groups"][0]["drizzle"]["holes_fraction"] == pytest.approx(
        rg.qc["groups"][0]["drizzle"]["holes_fraction"], abs=1e-4)


@gpu_only
def test_local_norm_gpu_matches_cpu(data):
    """局部正規化的 offset 面在 CPU 多行程與 GPU（整合與 drizzle）算出來相同。"""
    _, c = _run(data, "lcpu", gpu="cpu", workers=2, drizzle=2, normalization="local")
    _, g = _run(data, "lgpu", gpu="gpu", workers=2, drizzle=2, normalization="local")
    for x, y in ((fits.getdata(data / "lcpu" / "m_stack.fits"), fits.getdata(data / "lgpu" / "m_stack.fits")), (c, g)):
        diff = np.abs(x.astype(float) - y.astype(float))
        assert (diff > 0.1).mean() < 1e-4
        assert np.percentile(diff, 99.9) < 0.05


@gpu_only
def test_clip_combine_gpu_equals_cpu():
    import cupy as cp

    rng = np.random.default_rng(1)
    x = rng.normal(100, 5, (15, 64, 64)).astype(np.float32)
    x[2, 5, 5] = 9000
    x[:, 0, 0] = np.nan
    x[:4, 1, 1] = np.nan
    w = rng.uniform(0.5, 1.5, 15)
    a, ca = clip_combine(x, w)
    b, cb = clip_combine(x, w, xp=cp)
    np.testing.assert_allclose(a, b.get(), rtol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(ca, cb.get())


def _cal_frames(tmp_path, n=7):
    from . import synth
    rng = np.random.default_rng(5)
    paths = []
    for i in range(n):
        d = rng.normal(synth.PEDESTAL, synth.READ_NOISE, (synth.H, synth.W)).astype(np.float32)
        d[rng.integers(0, synth.H, 30), rng.integers(0, synth.W, 30)] += 20000  # 宇宙射線，要被剔除
        p = tmp_path / f"D_{i}.fits"
        synth.write(p, d, EXPTIME=300.0)
        paths.append(p)
    return paths


def test_master_combine_parallel_matches_single(tmp_path):
    """master 校正檔：多行程與單行程結果相同；扣 flat-dark 的前處理照樣套用；進度報到最後一列。"""
    from apu_photons.calibration import _combine_files
    paths = _cal_frames(tmp_path)
    one = _combine_files(paths, workers=1)
    seen = []
    many = _combine_files(paths, workers=3, memory_mb=1, progress=lambda d, t: seen.append((d, t)))
    np.testing.assert_array_equal(one, many)
    assert seen[-1][0] == seen[-1][1] == one.shape[0] and len(seen) > 1
    sub = _combine_files(paths, preprocess=lambda d: d - 100.0, workers=2)
    np.testing.assert_allclose(sub, one - 100.0, atol=1e-3)


@gpu_only
def test_master_combine_and_hot_pixels_on_gpu(tmp_path):
    from apu_photons.calibration import _combine_files, hot_from_dark
    be = backend.select("auto")
    paths = _cal_frames(tmp_path)
    cpu = _combine_files(paths, workers=1)
    gpu = _combine_files(paths, be=be)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-3)
    np.testing.assert_array_equal(hot_from_dark(cpu, None), hot_from_dark(cpu, None, be))
    np.testing.assert_array_equal(hot_from_dark(cpu, "RGGB"), hot_from_dark(cpu, "RGGB", be))
