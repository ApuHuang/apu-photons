"""畫質評測工具：用合成星場確認每個指標朝對的方向變化。"""

import struct

import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage as ndi

from apu_photons import benchmark as bm

H, W = 700, 900


def _field(fwhm=3.0, seed=0, n=500, noise=2.0, shape=(H, W), scale=1.0, shift=(0.0, 0.0), rgb=True):
    """同一片天空（星點位置與亮度只看 seed=0），可以改 FWHM、像素尺度、平移與雜訊。"""
    sky = np.random.default_rng(0)
    xs = sky.uniform(0, W, n)
    ys = sky.uniform(0, H, n)
    flux = 10 ** sky.uniform(3.0, 5.0, n)
    rng = np.random.default_rng(seed)
    h, w = int(shape[0] * scale), int(shape[1] * scale)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w), 100.0, np.float32)
    s = fwhm * scale / 2.3548
    for x, y, f in zip(xs, ys, flux):
        cx, cy = (x + shift[0]) * scale, (y + shift[1]) * scale
        x0, x1 = int(max(0, cx - 15 * scale)), int(min(w, cx + 15 * scale + 1))
        y0, y1 = int(max(0, cy - 15 * scale)), int(min(h, cy + 15 * scale + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        r2 = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2
        img[y0:y1, x0:x1] += f / (2 * np.pi * s * s) * np.exp(-r2 / (2 * s * s)) * scale * scale  # 表面亮度不變（Drizzle）
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    if not rgb:
        return img[None]
    return np.stack([img * 0.8 + rng.normal(0, noise, img.shape), img, img * 0.7 + rng.normal(0, noise, img.shape)]
                    ).astype(np.float32)


def _measure(img, label="x"):
    return bm.measure(img, label, trail=False)


def test_identical_images_compare_as_one():
    img = _field()
    ref, cand = _measure(img, "ref"), _measure(img.copy(), "cand")
    bm.compare(ref, cand, img.shape[1:])
    r = cand.report
    assert r.matched > 100
    assert r.geometry_rms < 0.05
    assert r.fwhm_ratio == pytest.approx(1.0, abs=1e-3)
    assert r.star_snr_ratio == pytest.approx(1.0, abs=1e-3)
    assert r.brightness_scale == pytest.approx(1.0, abs=1e-3)
    assert r.noise_ratio == pytest.approx(1.0, abs=1e-3)
    assert ref.report.fwhm == pytest.approx(3.0, rel=0.05)


def test_blur_raises_fwhm_ratio():
    ref = _measure(_field(3.0), "ref")
    cand = _measure(_field(3.6, seed=1), "cand")
    bm.compare(ref, cand, (H, W))
    assert cand.report.fwhm_ratio == pytest.approx(1.2, rel=0.04)


def test_drizzle_scale_and_shift_are_handled():
    """2× 像素、平移、亮度單位不同、上下顛倒（FITS 的另一種列順序）都要能配對，FWHM 換回參考像素。"""
    ref = _measure(_field(3.0), "ref")
    big = _field(3.0, seed=2, scale=2.0, shift=(3.3, -2.7)) * 0.01
    big = big[:, ::-1, :].copy()
    cand = _measure(big, "cand")
    bm.compare(ref, cand, big.shape[1:])
    r = cand.report
    assert r.pixel_scale == pytest.approx(0.5, rel=0.01)
    assert r.fwhm_ratio == pytest.approx(1.0, abs=0.03)
    assert r.brightness_scale == pytest.approx(100.0, rel=0.05)
    assert r.geometry_rms < 0.3


def test_noise_ratio_tracks_extra_noise():
    ref = _measure(_field(noise=2.0), "ref")
    cand = _measure(_field(noise=3.0, seed=3), "cand")
    bm.compare(ref, cand, (H, W))
    assert cand.report.noise_ratio == pytest.approx(1.5, rel=0.06)
    assert cand.report.star_snr_ratio < 0.75


def test_hot_and_cold_pixels_are_counted():
    clean = _field()
    dirty = clean.copy()
    rng = np.random.default_rng(5)
    ys, xs = rng.integers(40, H - 40, 200), rng.integers(40, W - 40, 200)
    dirty[0, ys[:100], xs[:100]] += 2000.0          # 只在紅色版：熱像素殘留
    dirty[:, ys[100:], xs[100:]] -= 2000.0          # 冷像素
    a, b = _measure(clean).report, _measure(dirty).report
    mp = (H * W * (1 - 2 * bm.BORDER) ** 2) / 1e6
    assert b.color_isolated_per_mp - a.color_isolated_per_mp > 60 / mp
    assert b.sharp_per_mp > a.sharp_per_mp
    assert b.cold_per_mp - a.cold_per_mp > 60 / mp


def test_dark_rings_around_bright_stars():
    clean = _field()
    m = _measure(clean)
    ringed = clean.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    ys, xs = bm._peaks(m.lum_hp, np.ones((H, W), bool), 30 * m.sigma, box=31)
    top = np.argsort(-m.lum_hp[ys, xs])[:100]
    for x, y in zip(xs[top], ys[top]):
        r = np.hypot(xx - x, yy - y)
        ringed -= ((r > 4) & (r < 6)) * 30.0
    a, b = m.report, _measure(ringed).report
    assert b.ring_sigma_median > a.ring_sigma_median + 5


def test_trail_is_found():
    clean = _field()
    trail = clean.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.abs((yy - 100) - np.tan(np.deg2rad(20)) * xx) * np.cos(np.deg2rad(20))
    trail += (np.exp(-d ** 2 / 2) * 3.0).astype(np.float32)  # 1.5σ 的細線，單看像素看不出來
    a = bm.measure(clean, "clean").report
    b = bm.measure(trail, "trail").report
    assert a.trail_z < 8.5
    assert b.trail_z > 18


def _write_xisf(path, data, shuffle_zlib=False):
    c, h, w = data.shape
    payload = data.astype("<f4").tobytes()
    comp = ""
    if shuffle_zlib:
        import zlib
        n = len(payload) // 4
        shuffled = np.frombuffer(payload, np.uint8).reshape(n, 4).T.tobytes()
        comp = f' compression="zlib+sh:{len(payload)}:4"'
        payload = zlib.compress(shuffled)
    offset = 4096
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><xisf version="1.0"><Image geometry="{w}:{h}:{c}" '
           f'sampleFormat="Float32" colorSpace="RGB" location="attachment:{offset}:{len(payload)}"{comp}/></xisf>')
    head = b"XISF0100" + struct.pack("<I", len(xml)) + b"\0\0\0\0" + xml.encode()
    path.write_bytes(head + b"\0" * (offset - len(head)) + payload)


@pytest.mark.parametrize("shuffle", [False, True])
def test_xisf_and_fits_load_the_same(tmp_path, shuffle):
    data = np.random.default_rng(0).random((3, 20, 30)).astype(np.float32)
    _write_xisf(tmp_path / "a.xisf", data, shuffle)
    fits.PrimaryHDU(data).writeto(tmp_path / "b.fits")
    np.testing.assert_array_equal(bm.load(tmp_path / "a.xisf"), data)
    np.testing.assert_array_equal(bm.load(tmp_path / "b.fits"), data)


def test_command_line_writes_table_and_json(tmp_path, capsys):
    img = _field(rgb=False)
    fits.PrimaryHDU(img[0]).writeto(tmp_path / "ref.fits")
    fits.PrimaryHDU(ndi.gaussian_filter(img[0], 0.8)).writeto(tmp_path / "soft.fits")
    out = tmp_path / "r.json"
    assert bm.main([str(tmp_path / "ref.fits"), str(tmp_path / "soft.fits"), "--labels", "ref", "soft",
                    "--json", str(out), "--no-trail"]) == 0
    text = capsys.readouterr().out
    assert "FWHM ÷ ref" in text and "soft" in text
    import json
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[1]["fwhm_ratio"] > 1.05


def test_trail_location_and_banding_are_separate():
    clean = _field()
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.abs((yy - 100) - np.tan(np.deg2rad(20)) * xx) * np.cos(np.deg2rad(20))
    trail = clean + (np.exp(-d ** 2 / 2) * 3.0).astype(np.float32)
    r = bm.measure(trail, "t").report
    assert abs(r.trail_angle - 110) <= 1  # 線的方向 20°，法向量 110°
    px, py = r.trail_point
    assert abs((py - 100) - np.tan(np.deg2rad(20)) * px) * np.cos(np.deg2rad(20)) < 4
    banded = clean.copy()
    banded[:, :, 451] += 5.0  # 一整欄偏亮：感光元件的條紋
    b = bm.measure(banded, "b").report
    assert b.pattern_z > 15
    assert b.trail_z < 8.5


def test_binned_noise_sees_through_interpolation_smoothing():
    """內插讓逐像素雜訊變小，但合併後的雜訊不變：逐像素比值會騙人，合併後的不會。"""
    ref = _measure(_field(noise=2.0), "ref")
    soft = _field(noise=2.0, seed=4)
    soft = ndi.shift(soft, (0, 0.5, 0.5), order=1, mode="nearest").astype(np.float32)  # 半像素雙線性內插
    cand = _measure(soft, "soft")
    bm.compare(ref, cand, (H, W))
    assert cand.report.noise_ratio < 0.75
    assert cand.report.noise_binned_ratio == pytest.approx(1.0, abs=0.1)
