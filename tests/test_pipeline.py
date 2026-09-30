import json

import numpy as np
import pytest
from astropy.io import fits

from apu_photons.combine import HIGH, clip_combine
from apu_photons.engine import Settings, run
from apu_photons.imageio import bayer_pattern, list_images
from apu_photons.ingest import night_of
from apu_photons.model import Calibration
from apu_photons.register import register
from apu_photons.stars import detect

from . import synth

SHIFTS = [(0, 0, 0), (3.4, -2.7, 0.3), (-5.2, 4.1, -0.4), (7.7, 1.3, 0.2), (-2.1, -6.6, 0.1),
          (1.5, 5.5, -0.2), (-6.3, -1.8, 0.5), (4.9, 3.3, -0.3)]


def _make(tmp_path, bayer=None, trail_index=3, with_cal=True, shifts=SHIFTS):
    hot = synth.hot_pixels()
    vig = synth.vignette_map()
    light = tmp_path / "light"
    extra = {"BAYERPAT": bayer} if bayer else {}
    for i, (tx, ty, rot) in enumerate(shifts):
        img = synth.render(tx, ty, rot, seed=100 + i, vignette=vig, hot=hot, trail=(i == trail_index), bayer=bayer)
        synth.write(light / f"L_{i:03d}.fits", img, EXPTIME=300.0,
                    DATE_OBS=f"2026-09-20T14:{i:02d}:00", XPIXSZ=3.76, FOCALLEN=400.0, **extra)
    cal = Calibration()
    if with_cal:
        rng = np.random.default_rng(3)
        for i in range(5):
            d = rng.normal(synth.PEDESTAL, synth.READ_NOISE, (synth.H, synth.W)).astype(np.float32)
            d[hot] += 20000
            synth.write(tmp_path / "dark" / f"D_{i}.fits", d, EXPTIME=300.0, **extra)
            f = rng.poisson(20000 * vig).astype(np.float32)
            synth.write(tmp_path / "flat" / f"F_{i}.fits", f, EXPTIME=1.0, **extra)
        cal = Calibration(dark=list_images(tmp_path / "dark"), flat=list_images(tmp_path / "flat"))
    return light, cal


@pytest.mark.parametrize("bayer", [None, "RGGB"])
def test_fwhm_matches_truth(bayer):
    # CFA 在 super-pixel 上量，會稍微偏大
    tol = 0.1 if bayer is None else 0.15
    for fwhm in (2.5, 4.0, 6.0):
        got = detect(synth.render(0.3, 0.6, 0, seed=5, fwhm=fwhm, bayer=bayer), bayer).fwhm
        assert abs(got - fwhm) < tol * fwhm, (fwhm, got)


def test_register_recovers_transform():
    a = synth.render(0, 0, 0, seed=1)
    b = synth.render(4.3, -2.2, 1.0, seed=2)
    sa, sb = detect(a, None), detect(b, None)
    m, rms, *_ = register(np.column_stack([sb.x, sb.y]), np.column_stack([sa.x, sa.y]))
    # b → a：套回去後 b 的星應落在 a 的位置
    assert np.isclose(np.degrees(np.arctan2(m[1, 0], m[0, 0])), -1.0, atol=0.02)
    assert rms < 0.3
    assert abs(np.hypot(m[0, 0], m[1, 0]) - 1.0) < 1e-3


def test_register_meridian_flip():
    a = synth.render(0, 0, 0, seed=1)
    b = synth.render(1.0, 2.0, 180.0, seed=2)
    sa, sb = detect(a, None), detect(b, None)
    m, rms, *_ = register(np.column_stack([sb.x, sb.y]), np.column_stack([sa.x, sa.y]))
    assert abs(abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))) - 180.0) < 0.1
    assert rms < 0.3


def test_clip_combine_rejects_outlier():
    rng = np.random.default_rng(0)
    stack = rng.normal(100, 5, (10, 4, 4)).astype(np.float32)
    stack[3, 1, 1] = 5000
    out, codes = clip_combine(stack)
    assert codes[3, 1, 1] == HIGH
    assert abs(out[1, 1] - 100) < 10


def test_bayer_offset_and_night():
    h = fits.Header({"BAYERPAT": "RGGB", "XBAYROFF": 1})
    assert bayer_pattern(h) == "GRBG"
    # 台灣經度 120.4：UTC 14:00 = 地方時約 22:02，屬於當天晚上
    assert night_of("2026-09-20T14:00:00", 120.4) == "2026-09-20"
    assert night_of("2026-09-20T20:00:00", 120.4) == "2026-09-20"  # 地方時凌晨 4 點


@pytest.mark.parametrize("bayer", [None, "RGGB"])
def test_full_stack(tmp_path, bayer):
    light, cal = _make(tmp_path, bayer=bayer)
    out = tmp_path / "out" / "master.fits"
    # 固定參考 frame：合成星場每張 FWHM 幾乎一樣，自動選可能選到有軌跡的那張，
    # 參考 frame 不內插、軌跡較細，剔除比例就不到其他張的 2 倍
    res = run([light], cal, out, Settings(crop_common=False, reference="L_000.fits"), echo=None)
    data = fits.getdata(out)
    assert data.shape == ((3, synth.H, synth.W) if bayer else (synth.H, synth.W))
    assert res.qc["frames_integrated"] == len(SHIFTS)

    lum = data.mean(axis=0) if bayer else data
    ref = next(f for f in res.project.frames if f is res.project.reference)
    # 衛星軌跡被剔除：沿著軌跡取樣，不應比附近背景亮很多
    inner = lum[40:-40, 40:-40]
    bg, noise = np.median(inner), 1.4826 * np.median(np.abs(inner - np.median(inner)))
    # 那張有軌跡的 frame 剔除像素比例應該最高
    trail = res.project.frames[3]
    others = [f.rejected_pixel_fraction for f in res.project.frames if f is not trail]
    assert trail.rejected_pixel_fraction > 2 * max(others)
    # 熱像素與暗角已修正：背景平坦（邊角與中央差不到幾個雜訊）
    corner = np.median(lum[20:50, 20:60])
    center = np.median(lum[110:150, 140:180])
    assert abs(corner - center) < 0.05 * center
    # 沒有殘留的熱像素：不能有「亮度幾乎全集中在單一像素」的尖點（星點至少會分到周圍）
    from scipy.ndimage import uniform_filter
    pos = np.clip(inner - bg, 0, None)
    conc = pos / np.maximum(uniform_filter(pos, 3) * 9, 1e-6)
    assert ((pos > 30 * noise) & (conc > 0.6)).sum() == 0

    recipe = json.loads(res.recipe.read_text(encoding="utf-8"))
    assert recipe["schema"] == "apuphotons-recipe/1"
    assert recipe["reference_frame"] == ref.name
    assert (out.parent / ".photons_cache" / "rejection").is_dir()


def test_pick_sidecar_gate(tmp_path):
    from apu_photons.imageio import file_sha256

    light, cal = _make(tmp_path, with_cal=False)
    files = list_images(light)
    doc = {"schema": "apupick/1", "pick_version": "0.5.0", "frames": [
        {"file": p.name, "sha256": file_sha256(p), "score": 90.0, "fwhm_px": 3.0,
         "verdict": "reject" if i == 5 else "keep"} for i, p in enumerate(files)]}
    (light / "apupick.json").write_text(json.dumps(doc), encoding="utf-8")
    res = run([light], cal, tmp_path / "m.fits", Settings(), echo=None)
    assert res.qc["rejected_by_reason"].get("pick") == 1
    assert res.qc["frames_integrated"] == len(SHIFTS) - 1
    assert any("沒有 flat" in w for w in res.warnings)


def test_preview_and_downsample(tmp_path):
    light, cal = _make(tmp_path, with_cal=False)
    res = run([light], cal, tmp_path / "p.fits", Settings(preview=4, downsample=0.5, crop_common=False), echo=None)
    assert res.qc["frames_integrated"] == 4
    assert fits.getdata(res.output).shape == (synth.H // 2, synth.W // 2)
