"""0.3 的修正：參考 frame 用路徑比對、同名檔的標示、16-bit 輸出記錄縮放。"""

import json
import shutil

import numpy as np
from astropy.io import fits

from apu_photons.engine import Settings, _find_frame, _frame_labels, run
from apu_photons.imageio import write_fits
from apu_photons.recipe import load_recipe

from .test_pipeline import _make


def test_reference_is_matched_by_path_when_names_repeat(tmp_path):
    """兩晚的資料夾有同名的檔：指定第二晚那張當參考，不能指到第一晚的同名檔。"""
    light, _inputs = _make(tmp_path, with_cal=False)
    night2 = tmp_path / "night2"
    night2.mkdir()
    for p in sorted(light.iterdir())[:4]:
        shutil.copy(p, night2 / p.name)
    wanted = night2 / "L_002.fits"
    kinds = {str(p): "light" for p in night2.iterdir()}     # 資料夾名稱看不出類型，直接指定
    res = run([light, night2], tmp_path / "out", Settings(workers=1, gpu="cpu", split_nights=False, kinds=kinds,
                                                         reference=str(wanted)), echo=None)
    ref = res.project.align_groups[0].reference
    assert ref.path == wanted
    doc = json.loads(res.recipe.read_text(encoding="utf-8"))
    assert doc["align_groups"][0]["reference_path"] == str(wanted)
    assert doc["align_groups"][0]["reference_frame"] == "L_002.fits"
    # 載入 recipe「完全重現」：參考 frame 是路徑
    assert load_recipe(res.recipe, reproduce=True).settings.reference == [str(wanted)]
    # 只給檔名（0.2 的 recipe、命令列）還是找得到
    frames = res.project.lights
    assert _find_frame(frames, "L_005.fits").name == "L_005.fits"
    assert _find_frame(frames, "nothing.fits") is None
    labels = _frame_labels(frames)
    assert "light/L_002.fits" in labels and "night2/L_002.fits" in labels and "L_005.fits" in labels


def test_16bit_output_records_the_linear_range(tmp_path):
    data = np.linspace(-12.5, 4000.0, 64 * 48, dtype=np.float32).reshape(48, 64)
    write_fits(tmp_path / "a.fits", data, bits=16)
    with fits.open(tmp_path / "a.fits") as hdul:
        hdr, stored = hdul[0].header, hdul[0].data.astype(np.float64)
    assert stored.min() == 0 and stored.max() == 65535
    back = hdr["LINMIN"] + stored * (hdr["LINMAX"] - hdr["LINMIN"]) / 65535.0
    assert np.abs(back - data).max() < (4012.5 / 65535.0)       # 還原到量化誤差以內
    write_fits(tmp_path / "b.fits", data, bits=32)
    assert "LINMIN" not in fits.getheader(tmp_path / "b.fits")
