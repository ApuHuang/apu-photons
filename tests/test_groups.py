"""0.2：檔案分類、多組資料、校正配對、做好的 master、合併光學系統（SPEC §3.1、Stage 0、Stage 1）。"""

import numpy as np
import pytest
from astropy.io import fits

from apu_photons.engine import Settings, run
from apu_photons.ingest import classify, filter_key, ingest
from apu_photons.register import register
from apu_photons.stars import detect

from . import synth

SHIFTS = [(0, 0, 0), (3.4, -2.7, 0.3), (-5.2, 4.1, -0.4), (7.7, 1.3, 0.2), (-2.1, -6.6, 0.1)]
CAM = {"INSTRUME": "CamA", "XPIXSZ": 3.76, "FOCALLEN": 400.0, "GAIN": 100, "OFFSET": 10}


def _lights(folder, filt, night=20, seed=0, shifts=SHIFTS, cam=CAM, catalog=None, fwhm=3.0, **extra):
    for i, (tx, ty, rot) in enumerate(shifts):
        img = synth.render(tx, ty, rot, seed=seed + i, catalog=catalog, fwhm=fwhm)
        synth.write(folder / f"{filt}_{night}_{i:03d}.fits", img, IMAGETYP="LIGHT", EXPTIME=300.0, FILTER=filt,
                    DATE_OBS=f"2026-09-{night}T14:{i:02d}:00", CCD_TEMP=-10.0, **cam, **extra)
    return folder


def _darks(folder, exp=300.0, n=3, temp=-10.0, filt="Ha", cam=CAM):
    rng = np.random.default_rng(5)
    for i in range(n):
        d = rng.normal(synth.PEDESTAL, synth.READ_NOISE, (synth.H, synth.W)).astype(np.float32)
        # 濾鏡輪停在 Ha：dark 的 header 也寫 FILTER，配對時要忽略
        synth.write(folder / f"D_{exp:g}_{i}.fits", d, IMAGETYP="DARK", EXPTIME=exp, FILTER=filt, CCD_TEMP=temp,
                    DATE_OBS="2026-09-21T03:00:00", **cam)
    return folder


def _flats(folder, filt, night, n=3, cam=CAM):
    rng = np.random.default_rng(9)
    vig = synth.vignette_map()
    for i in range(n):
        f = rng.poisson(20000 * vig).astype(np.float32) + synth.PEDESTAL
        synth.write(folder / f"F_{filt}_{night}_{i}.fits", f, IMAGETYP="FLAT", EXPTIME=1.0, FILTER=filt,
                    DATE_OBS=f"2026-09-{night + 1}T00:30:00", **cam)
    return folder


@pytest.mark.parametrize("path, header, expected", [
    ("x/LIGHT/a.fits", {"IMAGETYP": "Light Frame"}, ("light", "header", False)),
    ("x/a.fits", {"IMAGETYP": "DARKFLAT"}, ("flat_dark", "header", False)),
    ("x/a.fits", {"IMAGETYP": "Master Dark"}, ("dark", "header", True)),
    ("x/a.fits", {"IMAGETYP": "Bias Frame"}, ("bias", "header", False)),
    ("x/MasterFlat_Ha.fits", {}, ("flat", "filename", True)),
    ("x/2024_flat-dark_001.fits", {}, ("flat_dark", "filename", False)),
    ("M31/Darks/DSC0001.ARW", {}, ("dark", "folder", False)),
    ("M31/Lights/night1/DSC0001.ARW", {}, ("light", "folder", False)),
    ("M31/DSC0001.ARW", {}, ("unknown", None, False)),
    ("x/darkness.fits", {}, ("unknown", None, False)),
    # NINA 用一般序列拍的 flat：header 是預設的 LIGHT，資料夾寫 FLAT → 以資料夾為準（NGC2244 實拍資料）
    ("G/NGC2244/FLAT/2026-02-15_S_2.50s_0000.fits", {"IMAGETYP": "LIGHT"}, ("flat", "folder", False)),
    ("x/DARK/MasterDark_300s.fits", {"IMAGETYP": "LIGHT"}, ("dark", "filename", True)),
    ("x/NGC2244/H/a.fits", {"IMAGETYP": "LIGHT"}, ("light", "header", False)),
])
def test_classify(path, header, expected):
    from pathlib import Path
    assert classify(Path(path), header) == expected


def test_filter_key_only_ignores_case_and_spaces():
    assert filter_key(" Ha ") == filter_key("HA") == filter_key("ha")
    assert filter_key("H-alpha") != filter_key("Ha")
    assert filter_key("L") != filter_key("L-Pro")
    assert filter_key("") is None and filter_key(None) is None


def test_filters_dark_matching_and_alignment(tmp_path):
    """單色＋濾鏡：Ha、OIII 各自輸出 master，像素對齊；dark 的 FILTER=Ha 不影響 OIII 配對；
    多出來的短曝光 dark 列為未使用；有 dark 時 bias 不需要。"""
    _lights(tmp_path / "LIGHT", "Ha", seed=0)
    _lights(tmp_path / "LIGHT", "OIII", seed=50, shifts=[(x + 1.3, y - 0.7, r) for x, y, r in SHIFTS])
    # 同一個濾鏡不同寫法：自動歸併（只差大小寫與空白）
    img = synth.render(2.0, 1.0, 0.1, seed=99)
    synth.write(tmp_path / "LIGHT" / "oiii_extra.fits", img, IMAGETYP="LIGHT", EXPTIME=300.0, FILTER=" oiii ",
                DATE_OBS="2026-09-20T15:00:00", CCD_TEMP=-10.0, **CAM)
    _darks(tmp_path / "DARK", 300.0)
    _darks(tmp_path / "DARK", 1.0)
    for i in range(3):
        synth.write(tmp_path / "BIAS" / f"B_{i}.fits", np.full((synth.H, synth.W), synth.PEDESTAL, np.float32),
                    IMAGETYP="BIAS", EXPTIME=0.0001, **CAM)
    res = run([tmp_path / "LIGHT", tmp_path / "DARK", tmp_path / "BIAS"], tmp_path / "out",
              Settings(target="T", crop_common=False), echo=None)
    names = sorted(o.output.name for o in res.outputs)
    assert names == ["T_Ha.fits", "T_OIII.fits"]
    o3 = next(o for o in res.outputs if o.filter == "OIII")
    assert len(o3.frames) == 6  # " oiii " 併進 OIII
    for gk, entry in res.project.calib_groups.items():
        assert entry["dark"] and "300s" in entry["dark"], gk
        assert entry["bias"] is None  # 有 dark 就不扣 bias
    unused = res.qc["unused_calibration"]
    assert any("1s" in k for k in unused) and any(k.startswith("bias") for k in unused)
    assert fits.getheader(o3.output)["FILTER"] == "OIII"
    # 同一對齊組：兩個 master 像素對齊
    ha = fits.getdata(tmp_path / "out" / "T_Ha.fits")
    oo = fits.getdata(tmp_path / "out" / "T_OIII.fits")
    assert ha.shape == oo.shape
    sa, so = detect(ha, None), detect(oo, None)
    m, _rms, *_ = register(np.column_stack([so.x, so.y]), np.column_stack([sa.x, sa.y]))
    assert np.hypot(*m[:2, 2]) < 0.15


def test_flat_per_night_and_user_choice(tmp_path):
    """flat 依濾鏡與觀測夜配對；某晚沒有 flat 時不自動用別晚的，使用者選了才用。"""
    _lights(tmp_path / "LIGHT", "Ha", night=20, seed=0)
    _lights(tmp_path / "LIGHT", "Ha", night=22, seed=20)
    _flats(tmp_path / "FLAT", "Ha", night=20)
    _flats(tmp_path / "FLAT", "OIII", night=22)  # 濾鏡不同，不能用
    warnings = []
    project = ingest([tmp_path / "LIGHT", tmp_path / "FLAT"], warnings)
    by_night = {g["frames"][0].night: g for g in project.calib_groups.values()}
    assert by_night["2026-09-20"]["flat"] is not None
    n22 = by_night["2026-09-22"]
    assert n22["flat"] is None and n22["flat_other_nights"] == [by_night["2026-09-20"]["flat"]]
    assert any("2026-09-22" in str(w) and "別晚" in str(w) for w in warnings)
    # 使用者選擇用 9/20 的 flat
    key = next(k for k, g in project.calib_groups.items() if g is n22)
    project = ingest([tmp_path / "LIGHT", tmp_path / "FLAT"], [],
                     overrides={key: {"flat": by_night["2026-09-20"]["flat"]}})
    g = project.calib_groups[key]
    assert g["flat"] == by_night["2026-09-20"]["flat"] and g["source"] == "user"


def test_prepared_masters(tmp_path):
    """做好的 master：檔名有 master，或那一套只有一張；0～1 的浮點 dark 換算成 ADU。"""
    _lights(tmp_path / "LIGHT", "Ha")
    dark = np.full((synth.H, synth.W), synth.PEDESTAL / 65535.0, np.float32)
    hot = synth.hot_pixels()
    dark[hot] = 0.5
    path = tmp_path / "cal" / "MasterDark_300s.fits"
    path.parent.mkdir()
    fits.PrimaryHDU(dark, header=fits.Header({"EXPTIME": 300.0, "INSTRUME": "CamA"})).writeto(path)
    flat = (synth.vignette_map() * 20000).astype(np.float32)
    synth.write(tmp_path / "cal" / "flat_Ha_stacked.fits", flat, FILTER="Ha", **CAM)
    res = run([tmp_path / "LIGHT", tmp_path / "cal"], tmp_path / "out", Settings(crop_common=False), echo=None)
    sets = res.project.calibration_sets
    assert all(s.is_master for s in sets.values()) and len(sets) == 2
    entry = next(iter(res.project.calib_groups.values()))
    assert entry["dark"] and entry["flat"] and entry["flat_sub"] is None  # master flat 不再扣
    log = res.log.read_text(encoding="utf-8")
    assert "×65535" in log
    master = fits.getdata(res.output)
    assert abs(np.median(master) - synth.BG * np.median(synth.vignette_map()) / np.median(flat / flat.mean())) < 200


def _scaled_catalog(k):
    x, y, flux = synth.star_catalog()
    cx, cy = synth.W / 2, synth.H / 2
    return (cx + (x - cx) * k, cy + (y - cy) * k, flux)


def test_merge_optical_trains(tmp_path):
    """兩台相機拍同一個濾鏡：預設各自輸出；合併時疊成一個 master，對齊到像素較細的網格。"""
    cam_b = {**CAM, "INSTRUME": "CamB", "XPIXSZ": 4.7}  # 像素較粗：同一片天在 B 上縮成 0.8 倍
    _lights(tmp_path / "A" / "LIGHT", "Ha", seed=0)
    _lights(tmp_path / "B" / "LIGHT", "Ha", seed=30, cam=cam_b, catalog=_scaled_catalog(3.76 / 4.7), fwhm=2.4)
    inputs = [tmp_path / "A" / "LIGHT", tmp_path / "B" / "LIGHT"]
    res = run(inputs, tmp_path / "sep", Settings(target="T", crop_common=False), echo=None)
    assert len(res.outputs) == 2 and all("Cam" in o.output.name for o in res.outputs)

    trains = list(res.project.trains)
    res = run(inputs, tmp_path / "merged", Settings(target="T", merge_trains=[trains]), echo=None)
    assert len(res.outputs) == 1
    out = res.outputs[0]
    assert len(out.frames) == 2 * len(SHIFTS)
    ag = res.project.align_groups[0]
    assert res.project.trains[ag.reference.train].camera == "CamA"  # 較細的網格
    b_scales = [np.hypot(*f.transform[:2, 0]) for f in out.frames if f.train != ag.reference.train]
    assert np.allclose(b_scales, 4.7 / 3.76, atol=0.01)


def test_register_rejects_implausible_scale():
    """縮放倍率不合理就判定失敗，不能讓近乎退化的變換進到整合（NGC2244：flat 被當成 light 時整合出現 Singular matrix）。"""
    from apu_photons.register import RegistrationError, scale_of

    a = synth.render(0, 0, 0, seed=1)
    sa = detect(a, None)
    ref = np.column_stack([sa.x, sa.y])
    cx, cy = synth.W / 2, synth.H / 2
    half = np.column_stack([cx + (sa.x - cx) * 0.5, cy + (sa.y - cy) * 0.5])  # 同一片天、縮成一半
    with pytest.raises(RegistrationError):
        register(half, ref)                          # 同一套器材：倍率應該是 1
    m, *_ = register(half, ref, expected_scale=2.0)  # 合併不同器材：預期的倍率
    assert abs(scale_of(m) - 2.0) < 0.01
    noise = np.random.default_rng(3).uniform(0, [synth.W, synth.H], (60, 2))
    with pytest.raises(RegistrationError):
        register(noise, ref)                         # 沒有星點的影像（例如 flat）配不上
