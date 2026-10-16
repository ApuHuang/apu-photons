"""壞 frame 不能拖垮整個 master：權重算不出來、對齊到別處、整合結果沒有有效像素。"""

from pathlib import Path

import numpy as np

from apu_photons import engine
from apu_photons.engine import Settings, _has_data, _weights, run
from apu_photons.model import Frame, Stars
from apu_photons.register import overlap_fraction

from .test_pipeline import _make


def _frame(name, noise=10.0, fwhm=3.0, transform=None):
    f = Frame(Path(name))
    f.stars = Stars(np.zeros(20), np.zeros(20), np.ones(20), fwhm, 100.0, noise, 20.0)
    f.norm = {"scale": [1.0], "offset": [0.0]}
    f.transform = np.eye(3) if transform is None else transform
    return f


def test_weights_skip_frames_without_a_weight():
    """縮放 0 的變換（所有星擠到同一點）讓 FWHM 變 0、權重無限大；雜訊是 NaN 也算不出來。
    以前正規化後其他張全變 0，整個 master 是 NaN。"""
    collapsed = np.eye(3)
    collapsed[:2, :2] = 0.0
    frames = [_frame("a"), _frame("b", noise=12.0), _frame("c", transform=collapsed), _frame("d", noise=float("nan"))]
    bad = _weights(frames, Settings())
    assert [f.name for f in bad] == ["c", "d"]
    good = [f.weight for f in frames[:2]]
    assert all(np.isfinite(good)) and min(good) > 0 and abs(sum(good) - 1.0) < 1e-9
    assert frames[2].weight is None and frames[3].weight is None


def test_overlap_fraction():
    shape = (1000, 1500)
    eye = np.eye(3)
    assert overlap_fraction(eye, shape, shape) == 1.0
    half = eye.copy()
    half[0, 2] = 750.0
    assert abs(overlap_fraction(half, shape, shape) - 0.5) < 0.05
    away = eye.copy()
    away[0, 2] = 5000.0
    assert overlap_fraction(away, shape, shape) == 0.0
    # 合併不同器材：像素較粗（視野較大）的 frame 縮放 2 倍後蓋住整張參考，算 100%
    wide = eye.copy()
    wide[:2, :2] *= 2.0
    wide[:2, 2] = (-500.0, -400.0)
    assert overlap_fraction(wide, shape, shape) > 0.95


def test_has_data(tmp_path):
    p = tmp_path / "m.npy"
    m = np.full((3, 4, 4), np.nan, np.float32)
    m[0, 1, 1] = m[1, 2, 2] = 1.0
    np.save(p, m)
    assert not _has_data(p)  # 有一個色版整張無效
    m[2, 0, 0] = 1.0
    np.save(p, m)
    assert _has_data(p)


def test_frame_registered_elsewhere_is_rejected(tmp_path, monkeypatch):
    """對齊結果平移到參考影像外（配到別處的星）：那張判對齊失敗，其他張照常整合。"""
    real = engine.register_many

    def fake(tasks, workers, progress=None):
        out = real(tasks, workers, progress)
        (m, rms, si, ri), err = out[1]
        m = m.copy()
        m[0, 2] += 10000.0
        out[1] = ((m, rms, si, ri), err)
        return out

    monkeypatch.setattr(engine, "register_many", fake)
    _light, inputs = _make(tmp_path, with_cal=False)
    res = run(inputs, tmp_path / "out", Settings(crop_common=False, reference="L_000.fits"), echo=None)
    rejected = [f for f in res.project.lights if f.reject_reason == "registration_failed"]
    assert len(rejected) == 1
    assert any("重疊" in str(w) or "overlap" in str(w) for w in res.warnings)
    from astropy.io import fits
    data = fits.getdata(res.outputs[0].output)
    assert np.isfinite(data).all() and np.median(data) > 0


def test_register_many_parallel_matches_sequential():
    """對齊分給多個行程，結果（含失敗原因）與逐張相同。"""
    from apu_photons.register import register_many
    rng = np.random.default_rng(3)
    ref = rng.uniform(0, 1000, (60, 2))
    tasks = []
    for i in range(9):
        th = np.radians(rng.uniform(-2, 2))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        src = (ref - 500) @ rot.T + 500 + rng.uniform(-20, 20, 2) + rng.normal(0, 0.05, ref.shape)
        tasks.append((src, ref, 1.0))
    tasks.append((ref[:3], ref, 1.0))  # 星太少：失敗
    one = register_many(tasks, 1)
    many = register_many(tasks, 2, progress=lambda d, t: None)
    assert len(one) == len(many) == len(tasks)
    for (a, ea), (b, eb) in zip(one, many):
        assert (a is None) == (b is None) and str(ea) == str(eb)
        if a is not None:
            np.testing.assert_array_equal(a[0], b[0])
    assert one[-1][0] is None and "星" in str(one[-1][1]) or "star" in str(one[-1][1])


def test_suspicious_causes_and_advice(tmp_path):
    """剔除特別多的片分原因給建議：大片（梯度、天光）、星點變暗（雲）、細線（衛星、飛機）；彩色適合 Drizzle、裁掉太多也提示。"""
    from apu_photons.combine import HIGH, LOW, OUT_OF_BOUNDS
    from apu_photons.engine import _advice, _suspicious_causes
    from apu_photons.i18n import set_language

    h, w = 400, 640
    maps = []
    for i, kind in enumerate(("sky", "trail", "cloud")):
        m = np.zeros((h, w), np.uint8)
        m[:, :20] = OUT_OF_BOUNDS
        if kind == "sky":
            m[: h // 3, : w // 2] = HIGH       # 一角整片被剔除
        elif kind == "trail":
            for x in range(w):                  # 斜的一條細線
                m[min(h - 1, x * h // w), x] = HIGH
            m[10, 30] = LOW
        p = tmp_path / f"{i}.npy"
        np.save(p, m)
        maps.append(str(p))
    frames = [_frame(n) for n in ("sky", "trail", "cloud")]
    frames[2].norm["scale"] = [3.5]
    causes = _suspicious_causes(frames, maps, {"sky", "trail", "cloud"})
    assert causes == {"sky": "sky", "trail": "trail", "cloud": "cloud"}

    qc = {"suspicious_frames": ["sky", "trail", "cloud"], "suspicious_causes": causes, "drizzle_suitable": True}
    set_language("zh")
    keys = [a.key for a in _advice("G", frames, qc, Settings(), "RGGB", (0, 0, 500, 300), (h, w))]
    assert keys == ["msg.advice.sky", "msg.advice.cloud", "msg.advice.trail", "msg.advice.drizzle_osc",
                    "msg.advice.crop"]
    keys = [a.key for a in _advice("G", frames, qc, Settings(normalization="local", drizzle=1), None, None, (h, w))]
    assert keys == ["msg.advice.sky_local", "msg.advice.cloud", "msg.advice.trail"]
    many = _advice("G", frames, {**qc, "suspicious_frames": ["a", "b", "c", "d"], "suspicious_causes": {}},
                   Settings(), None, None, (h, w))
    assert str(many[0]).endswith("a、b、c…")
    set_language("en")
    assert str(many[0]).endswith("a, b, c…")
    set_language("zh")
