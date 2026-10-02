import numpy as np
import pytest
from astropy.io import fits

from apu_photons.engine import Settings, run

from . import synth
from .test_pipeline import _make


def _dithered(n, seed=11):
    rng = np.random.default_rng(seed)
    return [(0.0, 0.0, 0.0)] + [(float(rng.uniform(-6, 6)), float(rng.uniform(-6, 6)), float(rng.uniform(-0.3, 0.3)))
                                for _ in range(n - 1)]


def _brightest_centroid(img, box=6):
    from scipy.ndimage import gaussian_filter

    sm = gaussian_filter(np.nan_to_num(img), 1.5)
    edge = 3 * box
    inner = sm[edge:-edge, edge:-edge]
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    y, x = y + edge, x + edge
    c = img[y - box:y + box + 1, x - box:x + box + 1]
    c = np.clip(c - np.median(img), 0, None)
    yy, xx = np.indices(c.shape)
    return (c * xx).sum() / c.sum() + x - box, (c * yy).sum() / c.sum() + y - box


def test_mono_drizzle_2x(tmp_path):
    light, inputs = _make(tmp_path, shifts=_dithered(20), trail_index=4)
    out_dir = tmp_path / "out"
    res = run(inputs, out_dir, Settings(drizzle=2, crop_common=False, target="m"), echo=None)
    driz = fits.getdata(out_dir / "m.fits")
    stack = fits.getdata(out_dir / "m_stack.fits")
    assert driz.shape == (2 * synth.H, 2 * synth.W)
    assert (out_dir / "m_weight.fits").is_file()
    assert fits.getdata(out_dir / "m_coverage.fits").shape == driz.shape
    assert res.qc["groups"][0]["drizzle"]["holes_fraction"][0] < 0.01
    # 輸出都標明列順序（light 沒寫 ROWORDER ＝ bottom-up）
    for name in ("m.fits", "m_stack.fits", "m_weight.fits", "m_coverage.fits"):
        assert fits.getheader(out_dir / name)["ROWORDER"] == "BOTTOM-UP"

    # 幾何：一般疊圖的 (x, y) 對應 drizzle 的 (2x+0.5, 2y+0.5)
    sx, sy = _brightest_centroid(stack)
    dx, dy = _brightest_centroid(driz, box=12)
    assert abs(dx - (2 * sx + 0.5)) < 0.3 and abs(dy - (2 * sy + 0.5)) < 0.3

    # 亮度：背景與一般疊圖一致
    inner = (slice(60, -60), slice(60, -60))
    assert abs(np.median(driz[inner]) / np.median(stack[30:-30, 30:-30]) - 1) < 0.01

    # 衛星軌跡沒有進到 drizzle：比較沿軌跡的 drizzle 與一般疊圖（已剔除）
    t = np.linspace(0.2, 0.8, 200)
    px, py = (10 + t * (synth.W - 20)).astype(int), (30 + t * (synth.H - 60)).astype(int)
    ref = res.project.align_groups[0].reference
    # 參考 frame 是第 0 張（位移 0）或其他張；軌跡在第 4 張的座標，換到參考座標
    trail = res.project.frames[4]
    xy = np.column_stack([px, py]) @ trail.transform[:2, :2].T + trail.transform[:2, 2]
    ok = (xy[:, 0] > 5) & (xy[:, 0] < synth.W - 5) & (xy[:, 1] > 5) & (xy[:, 1] < synth.H - 5)
    xs, ys = xy[ok, 0], xy[ok, 1]
    d_vals = driz[np.rint(2 * ys + 0.5).astype(int), np.rint(2 * xs + 0.5).astype(int)]
    s_vals = stack[np.rint(ys).astype(int), np.rint(xs).astype(int)]
    assert np.median(d_vals) < np.median(s_vals) * 1.05
    assert ref is not None


def test_cfa_drizzle_1x(tmp_path):
    light, inputs = _make(tmp_path, bayer="RGGB", shifts=_dithered(16))
    res = run(inputs, tmp_path / "c", Settings(drizzle=1, pixfrac=1.0, crop_common=False), echo=None)
    driz = fits.getdata(res.output)
    stack = fits.getdata(res.outputs[0].stack)
    assert driz.shape == stack.shape == (3, synth.H, synth.W)
    holes = res.qc["groups"][0]["drizzle"]["holes_fraction"]
    # G 的取樣是 R、B 的兩倍，洞應該最少
    assert holes[1] <= min(holes[0], holes[2])
    inner = (slice(30, -30), slice(30, -30))
    for c in range(3):
        assert abs(np.median(driz[c][inner]) / np.median(stack[c][inner]) - 1) < 0.02
    # 解馬賽克以外的顏色比例相同
    assert np.isfinite(driz).all()  # 已補洞


def test_drizzle_warns_when_few_frames(tmp_path):
    light, inputs = _make(tmp_path, with_cal=False)
    res = run(inputs, tmp_path / "w", Settings(drizzle=2, downsample=0.5), echo=None)
    assert any("20 張" in w for w in res.warnings)
    assert any("Downsample" in w for w in res.warnings)
    assert fits.getdata(res.output).shape[-1] > synth.W  # 還是 2×，沒有被降採樣


def test_drizzle_rejects_bad_scale(tmp_path):
    light, inputs = _make(tmp_path, with_cal=False)
    with pytest.raises(RuntimeError):
        run(inputs, tmp_path / "x", Settings(drizzle=3), echo=None)
