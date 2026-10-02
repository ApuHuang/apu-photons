"""視窗流程測試：不點滑鼠，直接呼叫方法。

Tk 在同一個行程只建一次（APU Pick 踩過：反覆建立、關閉 Tk 在 Windows 偶爾失敗、Mac 雲端機會卡死）。
"""

import gc
import time
import tkinter as tk
from pathlib import Path

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


def _file_rows(app):
    return [app.tree.item(i, "values") for i in app._all_items(app.tree) if i.startswith("f:")]


def test_stack_flow_and_language(app, tmp_path):
    light, inputs = _make(tmp_path)
    app.add_files(inputs)                              # light 與校正檔一起加入
    assert Path(app.output_dir_var.get()).name == "APU Photons"
    app.wait_plan()
    assert len(app.plan.lights) == 8 and len(app.plan.calibration_sets) == 2
    entry = next(iter(app.plan.calib_groups.values()))
    assert entry["dark"] and entry["flat"]             # 依資料夾名稱判斷類型、自動配對
    assert len(_file_rows(app)) == 8 + 10             # 依組顯示：light 與校正檔
    assert len(app.cal_tree.get_children()) == 1
    app.list_mode_var.set("all")
    assert len(app.tree.get_children()) == 18          # 全部列表：沒有分組節點
    app.output_dir_var.set(str(tmp_path / "out"))
    app.workers_var.set(2)
    app._start(trial=False)
    assert app.is_busy()
    _wait(app)
    assert app.result is not None, app.log_lines[-3:]
    assert app.result.qc["frames_integrated"] == 8
    assert all(f.status == ACCEPTED for f in app.result.project.lights)
    assert app.image_canvas.path == app.result.output and app.image_canvas.data is not None
    assert app.metrics["integrated"].value.cget("text") == "8"
    rows = _file_rows(app)
    assert any(r[-1] == "參考 frame" for r in rows)
    # 換語言：重建介面，結果保留
    app.lang_var.set("en")
    app._language_clicked()
    assert any(r[-1] == "Reference frame" for r in _file_rows(app))
    assert app.metrics["integrated"].value.cget("text") == "8"
    assert app.stack_btn.cget("text") == "Stack"
    assert app.image_canvas.path == app.result.output


def test_trial_and_cancel(app, tmp_path):
    light, _inputs = _make(tmp_path, with_cal=False)
    app.add_light_folder(light)
    app.wait_plan()
    app.output_dir_var.set(str(tmp_path / "t"))
    app.preview_n_var.set(5)
    app.workers_var.set(1)
    app._start(trial=True)
    _wait(app)
    assert app.result.output == tmp_path / "t" / "trial" / "light.fits"
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
    light, _inputs = _make(tmp_path, with_cal=False)
    app.add_light_folder(str(light))   # 字串路徑也可以
    app.add_light_folder(light)        # 重複的不加
    assert app.light_dirs == [light] and len(app.files) == 8
    assert not app.is_busy()


def test_unknown_kind_and_calibration_override(app, tmp_path):
    """判斷不出類型的檔案由使用者指定；校正配對可以逐組改。"""
    import shutil

    light, inputs = _make(tmp_path)
    misc = tmp_path / "M31"
    misc.mkdir()
    for f in sorted((tmp_path / "dark").iterdir())[:3]:
        shutil.copy(f, misc / f"x_{f.name}")
    app.add_files(inputs + [misc])
    app.wait_plan()
    assert len(app.plan.unknown) == 3
    unknown_rows = [i for i in app._all_items(app.tree) if i.startswith("f:") and "M31" in i]
    app.tree.selection_set(unknown_rows)
    app._set_kind("dark")
    app.wait_plan()
    assert not app.plan.unknown and len(app.plan.calibration_sets) == 2  # 跟原本的 dark 同條件，併成同一套
    # 不用 flat
    gk = next(iter(app.plan.calib_groups))
    app.cal_tree.selection_set(f"g:{gk}")
    app._fill_cal_editor()
    app.cal_combos["flat"].set(gui.tr("gui.cal.none_option"))
    app._cal_combo_selected("flat")
    app.wait_plan()
    g = app.plan.calib_groups[gk]
    assert g["flat"] is None and g["source"] == "user"
    assert app.settings().calib_overrides == {gk: {"flat": None}}
    app.cal_tree.selection_set(f"g:{gk}")
    app._cal_reset()
    app.wait_plan()
    assert app.plan.calib_groups[gk]["flat"] is not None


def test_zoom_view(tk_root, tmp_path):
    """放大檢視：倍率限制、以滑鼠位置為定點縮放、換到 2× 的圖時同一塊天空。"""
    import numpy as np
    from astropy.io import fits

    from apu_photons.zoomview import ZoomView

    a = np.random.default_rng(0).normal(100, 5, (100, 160)).astype(np.float32)
    fits.PrimaryHDU(a).writeto(tmp_path / "a.fits")
    fits.PrimaryHDU(np.kron(a, np.ones((2, 2), np.float32))).writeto(tmp_path / "b.fits")
    v = ZoomView(tk_root)
    v.place(x=0, y=0, width=400, height=300)
    tk_root.update()
    try:
        v.set_image(tmp_path / "a.fits")
        assert v.fit and v.size == (160, 100)
        v.zoom_to(2.0, anchor=(200, 150))
        v.cx, v.cy = 40.0, 30.0
        v.set_image(tmp_path / "b.fits")  # drizzle 2×：中心與天空倍率保持
        assert (v.cx, v.cy, v.zoom) == (80.0, 60.0, 1.0)
        v.zoom_to(99)
        assert v.zoom == 4.0
        v._draw()
        assert v._photo is not None
    finally:
        v.close()
        v.destroy()


def test_load_recipe(app, tmp_path):
    from apu_photons.engine import Settings, run

    light, inputs = _make(tmp_path, with_cal=False)
    res = run(inputs, tmp_path / "out", Settings(crop_common=False, low=3.5, target="R"), echo=None)
    app.load_recipe_file(res.recipe, reproduce=False)
    assert app.low_var.get() == 3.5 and not app.files and not app.crop_var.get()
    app.load_recipe_file(res.recipe, reproduce=True)
    assert len(app.files) == 8 and app.target_var.get() == "R"
    app.wait_plan()
    assert app._reference == [res.project.align_groups[0].reference.name]
