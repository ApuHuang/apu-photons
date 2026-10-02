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
    out = tmp_path / name / "m.fits"
    res = run([tmp_path / "light"], _run.cal, out, Settings(crop_common=False, **kw), echo=None)
    return res, fits.getdata(out)


@pytest.fixture
def data(tmp_path):
    _light, cal = _make(tmp_path, bayer="RGGB", shifts=_dithered(12))
    _run.cal = cal
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
    assert rc.qc["drizzle"]["holes_fraction"] == pytest.approx(rg.qc["drizzle"]["holes_fraction"], abs=1e-4)


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
