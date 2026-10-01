"""視窗流程測試：不點滑鼠，直接呼叫方法。

Tk 在同一個行程只建一次（APU Pick 踩過：反覆建立、關閉 Tk 在 Windows 偶爾失敗、Mac 雲端機會卡死）。
"""

import gc
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
    a = gui.PhotonsView(tk_root, tk_root)
    a.pack(fill="both", expand=True)
    yield a
    a.close()
    a.destroy()
    gc.collect()
    set_language("zh")


def _wait(app, timeout=120):
    end = time.time() + timeout
    while app.is_busy() and time.time() < end:
        time.sleep(0.05)
    app.drain()
    assert not app.is_busy()


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
    app.add_light_folder(light)
    assert app.output_var.get().endswith(".fits")
    assert len(app.tree.get_children()) == 8          # 還沒跑：列出檔案
    app.output_var.set(str(tmp_path / "out" / "m.fits"))
    app.cal_vars["dark"].set(str(tmp_path / "dark"))
    app.cal_vars["flat"].set(str(tmp_path / "flat"))
    assert len(app.calibration().dark) == 5
    app.workers_var.set(2)
    app._start(trial=False)
    assert app.is_busy()
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
    app._language_clicked()
    rows = [app.tree.item(i, "values") for i in app.tree.get_children()]
    assert any(r[7] == "Reference frame" for r in rows)
    assert app.metrics["integrated"].value.cget("text") == "8"
    assert app.stack_btn.cget("text") == "Stack"


def test_trial_and_cancel(app, tmp_path):
    light, _cal = _make(tmp_path, with_cal=False)
    app.add_light_folder(light)
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


def test_gpu_switch_shows_off_without_gpu(app):
    app.gpu_var.set(True)
    app._gpu_checked, app.gpu_name = True, None
    app._update_gpu_view()
    sw = app.gpu_toggle.switch
    assert not sw.enabled and sw.forced_off
    assert app.gpu_var.get()  # 設定保留，換到有 GPU 的電腦照樣會用
    # 2× 強制 drizzle：停用但真的是開著，照實畫成開
    app.scale_var.set("2")
    assert not app.drizzle_toggle.switch.enabled and not app.drizzle_toggle.switch.forced_off


def test_panel_scroll_follows_delta(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app.panel_canvas, "yview", lambda *a: (0.2, 0.6) if not a else None)
    monkeypatch.setattr(app.panel_canvas, "yview_scroll", lambda n, what: calls.append(n))
    event = type("E", (), {"widget": app.panel_canvas})()
    monkeypatch.setattr(gui, "IS_MAC", True)
    for delta in (1, -1, 6):
        event.delta = delta
        app._scroll_panel(event)
    monkeypatch.setattr(gui, "IS_MAC", False)
    for delta in (120, -240):
        event.delta = delta
        app._scroll_panel(event)
    assert calls == [-1, 1, -6, -3, 6]


def test_view_is_embeddable(tk_root, tmp_path, monkeypatch):
    """整合版的用法：兩個畫面放在同一個視窗，事件各管各的；關掉一個不會動到另一個。"""
    monkeypatch.setenv("APU_PHOTONS_SETTINGS", str(tmp_path / "settings.json"))
    other = tk.Frame(tk_root)
    other.pack()
    marker = tk.Label(other, text="別的分頁")
    marker.pack()
    a = gui.PhotonsView(tk_root, tk_root)
    b = gui.PhotonsView(tk_root, tk_root, show_language=False)
    try:
        # 事件綁在各自的 tag，不是整個程式共用的 bind_all
        for seq in ("<Button-1>", "<Escape>", "<MouseWheel>"):
            assert not tk_root.bind_all(seq)
        assert a._tag != b._tag and a._tag in a.stack_btn.bindtags() and a._tag not in b.stack_btn.bindtags()
        assert a._tag not in marker.bindtags()
        # 說明氣泡：b 的 Esc 不會關掉 a 的
        a.show_popover(a.stack_btn, "a")
        b.show_popover(b.stack_btn, "b")
        b.close_popover()
        assert a.popover is not None
        # 外面接管語言切換
        asked = []
        c = gui.PhotonsView(tk_root, tk_root, on_language=asked.append)
        c.lang_var.set("en")
        c._language_clicked()
        assert asked == ["en"] and gui.get_language() == "zh"
        c.close()
        c.destroy()
        # 重建與關閉只動自己
        a.rebuild()
        a.close()
        a.destroy()
        assert marker.winfo_exists() and b.winfo_exists() and b.stack_btn.winfo_exists()
    finally:
        for v in (a, b):
            if v.winfo_exists():
                v.close()
                v.destroy()
        other.destroy()
        gc.collect()


def test_add_light_folder_entry(app, tmp_path):
    light, _cal = _make(tmp_path, with_cal=False)
    app.add_light_folder(str(light))   # 字串路徑也可以
    app.add_light_folder(light)        # 重複的不加
    assert app.light_dirs == [light]
    assert not app.is_busy()
