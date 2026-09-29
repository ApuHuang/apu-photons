"""視窗流程測試：不點滑鼠，直接呼叫方法。

Tk 在同一個行程只建一次（APU Pick 踩過：反覆建立、關閉 Tk 在 Windows 偶爾失敗、Mac 雲端機會卡死）。
"""

import time
import tkinter as tk

import pytest

from apu_photons import gui
from apu_photons.i18n import _EN, _ZH, set_language
from apu_photons.model import ACCEPTED

from .test_pipeline import _make


@pytest.fixture(scope="session")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有顯示環境")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root, tmp_path, monkeypatch):
    monkeypatch.setenv("APU_PHOTONS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: True)
    set_language("zh")
    a = gui.App(tk_root)
    yield a
    a.close()
    set_language("zh")


def _wait(app, timeout=120):
    end = time.time() + timeout
    while app._busy() and time.time() < end:
        time.sleep(0.05)
    app.drain()
    assert not app._busy()


def test_catalogs_match():
    assert set(_ZH) == set(_EN)


def test_output_rules(app):
    app.scale_var.set("2")
    assert app.drizzle_var.get() and app.settings().drizzle == 2
    app.scale_var.set("0.5")
    s = app.settings()
    assert not app.drizzle_var.get() and s.drizzle == 0 and s.downsample == 0.5
    app.scale_var.set("1")
    app.drizzle_var.set(True)
    assert app.settings().drizzle == 1
    app.gpu_var.set(False)
    app.memory_var.set(4)
    s = app.settings(trial=True)
    assert s.gpu == "cpu" and s.memory_mb == 4096 and s.preview == app.preview_n_var.get()


def test_stack_flow_and_language(app, tmp_path):
    light, cal = _make(tmp_path)
    app.add_light_dir(light)
    assert app.output_var.get().endswith(".fits")
    assert len(app.tree.get_children()) == 8          # 還沒跑：列出檔案
    app.output_var.set(str(tmp_path / "out" / "m.fits"))
    app.cal_vars["dark"].set(str(tmp_path / "dark"))
    app.cal_vars["flat"].set(str(tmp_path / "flat"))
    assert len(app.calibration().dark) == 5
    app.workers_var.set(2)
    app._start(trial=False)
    assert app._busy()
    _wait(app)
    assert app.result is not None, app.log_lines[-3:]
    assert app.result.qc["frames_integrated"] == 8
    assert all(f.status == ACCEPTED for f in app.result.project.frames)
    assert "stack" in app.previews
    assert app.metrics["integrated"].value.cget("text") == "8"
    rows = [app.tree.item(i, "values") for i in app.tree.get_children()]
    assert len(rows) == 8 and any(r[7] == "參考 frame" for r in rows)
    # 換語言：重建介面，結果保留
    app.lang_var.set("en")
    app._change_language()
    rows = [app.tree.item(i, "values") for i in app.tree.get_children()]
    assert any(r[7] == "Reference frame" for r in rows)
    assert app.metrics["integrated"].value.cget("text") == "8"
    assert app.stack_btn.cget("text") == "Stack"


def test_trial_and_cancel(app, tmp_path):
    light, _cal = _make(tmp_path, with_cal=False)
    app.add_light_dir(light)
    app.output_var.set(str(tmp_path / "t" / "m.fits"))
    app.preview_n_var.set(5)
    app.workers_var.set(1)
    app._start(trial=True)
    _wait(app)
    assert app.result.output.name == "m_trial.fits"
    assert app.result.qc["frames_integrated"] == 5
    # 停止：在第一個檢查點就結束
    app._start(trial=False)
    app._stack_or_stop()
    _wait(app)
    assert app.result is None
    assert app.status_var.get() in ("已停止", "Stopped")
