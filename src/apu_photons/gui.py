"""視窗介面，與 APU Processing、APU Pick 同一套介面語言（APU Astro 系列）：暗房深色主題、頂部列、右側可收合參數面板、底部狀態列。

流程（0.2）：加入檔案（light 與校正檔一起）→ 自動分類、分組、配對校正檔（可以改）→ 試跑或疊圖 →
在「結果」分頁逐組放大檢視。
啟動：python -m apu_photons.gui [檔案或資料夾...]
介面文字都在 i18n.py，可以切換繁體中文 / English。

暗房元件（Darkroom、PanelGroup、ParameterSlider…）在 darkroom.py，三套軟體共用同一份。
"""

from __future__ import annotations

import gc
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, backend
from .darkroom import (Darkroom, Fonts, MetricRow, PanelGroup, ParameterSlider, ParameterToggle, Segmented, Tooltip,
                       _short_path, dark_title_bar, enable_dpi_awareness, setup_style)
from .engine import ALIGN_OUTPUT_KEYS, Cancelled, Result, Settings, run
from .i18n import APP_NAME, LANGUAGES, get_language, set_language, tr
from .imageio import IMAGE_SUFFIXES
from .ingest import expand_inputs, filter_names, ingest
from .model import KINDS, UNKNOWN, Project
from .recipe import RecipeError, load_recipe
from .settings import load_settings, save_settings
from .zoomview import ZoomView

ASSETS = Path(__file__).parent / "assets"
ICON = ASSETS / "app.ico"
IS_MAC = sys.platform == "darwin"
OPEN_SHORTCUT = "⌘O" if IS_MAC else "Ctrl+O"
PREVIEW_FRAMES = 20
ALL_ALIGN = ""  # 輸出設定套用到「所有對齊組」

# 檔案清單欄位：(key, 翻譯代號, 寬度 px, 對齊)；第 0 欄（檔名或分組）另外設定
COLUMNS = [
    ("kind", "gui.col.kind", 76, "w"),
    ("filter", "gui.col.filter", 60, "w"),
    ("exp", "gui.col.exp", 56, "e"),
    ("gain", "gui.col.gain", 50, "e"),
    ("temp", "gui.col.temp", 50, "e"),
    ("night", "gui.col.night", 88, "w"),
    ("result", "gui.col.result", 62, "center"),
    ("fwhm", "gui.col.fwhm", 56, "e"),
    ("stars", "gui.col.stars", 50, "e"),
    ("weight", "gui.col.weight", 56, "e"),
    ("rejected", "gui.col.rejected", 66, "e"),
    ("reason", "gui.col.reason", 220, "w"),
]


def _fmt(v, nd: int = 0) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{f:.{nd}f}" if nd else f"{f:g}"


# ---------------------------------------------------------------------- 主畫面


class PhotonsView(tk.Frame):
    """疊圖的主畫面：可以放進任何視窗（單獨的 APU Photons 視窗，或整合版的分頁）。

    不建立 tk.Tk、不碰整個視窗：視窗標題與大小、選單列、快捷鍵、關閉前的詢問都由 main() 負責。
    對外：add_files(paths)、ask_add_files()、add_light_folder(path)、ask_add_light()、load_recipe_file(path)、
    ask_load_recipe()、is_busy()、rebuild()、close()、copy_selection()、select_all()。
    """

    def __init__(self, parent: tk.Misc, root: tk.Tk, *, light_dirs: list[str] | None = None,
                 on_language: Callable[[str], None] | None = None, show_language: bool = True):
        """light_dirs：一開始就加入的檔案或資料夾。on_language：按下頂部列的語言切換時呼叫（由外面換語言、
        重建 View 與選單列）；沒給就自己換語言並 rebuild()。show_language=False 時頂部列不顯示語言切換。"""
        super().__init__(parent, bg=Darkroom.canvas)
        self.root = root
        self.on_language = on_language
        self.show_language = show_language
        self.files: list[Path] = []
        self.kinds: dict[str, str] = {}             # 使用者指定的類型
        self.calib_overrides: dict[str, dict] = {}  # 使用者改的校正配對
        self.align_output: dict[str, dict] = {}     # 每個對齊組自己的輸出設定
        self.plan: Project | None = None
        self.plan_warnings: list = []
        self._plan_token = 0
        self._planning = False
        self.result: Result | None = None
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.log_lines: list[str] = []
        self.popover: tk.Toplevel | None = None
        self.popover_owner: tk.Widget | None = None
        self.gpu_name: str | None = None
        self._gpu_checked = False
        self._stage: tuple[str, int, int] | None = None
        self._stage_started = 0.0
        self._started = 0.0
        self._elapsed = 0.0
        self._loading_output = False
        self._reference = None                      # 載入 recipe「完全重現」時的參考 frame
        self._seen_nodes: set[str] = set()          # 檔案清單裡出現過的分組（展開狀態從第二次起照使用者的）
        self._status_fn: Callable[[], str] = lambda: tr("gui.status.start")

        st = load_settings()
        p = st.get("params", {})
        self.filter_aliases: dict[str, str] = dict(st.get("filter_aliases", {}))
        self.panel_state: dict[str, bool] = dict(st.get("panel", {}))
        self.lang_var = tk.StringVar(value=get_language())
        self.output_dir_var = tk.StringVar(value="")
        self.target_var = tk.StringVar(value="")
        self.rejection_var = tk.StringVar(value=p.get("rejection", "winsorized"))
        self.low_var = tk.DoubleVar(value=p.get("low", 4.0))
        self.high_var = tk.DoubleVar(value=p.get("high", 3.0))
        self.weights_var = tk.BooleanVar(value=p.get("weights", True))
        self.pick_boost_var = tk.BooleanVar(value=p.get("pick_boost", False))
        self.split_nights_var = tk.BooleanVar(value=p.get("split_nights", True))
        self.flat_any_night_var = tk.BooleanVar(value=p.get("flat_any_night", False))
        self.temp_tol_var = tk.DoubleVar(value=p.get("temp_tolerance", 2.0))
        # 輸出設定：面板上的控制項顯示「目前選的對齊組」的值；global_output 是所有對齊組的預設
        self.global_output = {"scale": p.get("scale", "1"), "drizzle": p.get("drizzle", False),
                              "pixfrac": p.get("pixfrac", 0.9), "fill_holes": p.get("fill_holes", True),
                              "crop": p.get("crop", True), "min_coverage": p.get("min_coverage", 0.9)}
        self.align_target_var = tk.StringVar(value=ALL_ALIGN)
        self.scale_var = tk.StringVar(value=self.global_output["scale"])
        self.drizzle_var = tk.BooleanVar(value=self.global_output["drizzle"])
        self.pixfrac_var = tk.DoubleVar(value=self.global_output["pixfrac"])
        self.fill_holes_var = tk.BooleanVar(value=self.global_output["fill_holes"])
        self.crop_var = tk.BooleanVar(value=self.global_output["crop"])
        self.min_cov_var = tk.DoubleVar(value=self.global_output["min_coverage"])
        self.merge_var = tk.BooleanVar(value=False)
        self.grid_var = tk.StringVar(value="")
        self.bits_var = tk.StringVar(value=p.get("bits", "32"))
        self.gpu_var = tk.BooleanVar(value=p.get("gpu", True))
        self.workers_var = tk.IntVar(value=p.get("workers", backend.default_workers()))
        self.memory_var = tk.IntVar(value=p.get("memory_gb", 2))
        self.preview_n_var = tk.IntVar(value=p.get("preview_n", PREVIEW_FRAMES))
        self.list_mode_var = tk.StringVar(value=st.get("list_mode", "groups"))
        self.view_var = tk.StringVar(value="drizzle")
        self.result_group_var = tk.StringVar(value="")
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.zoom_var = tk.StringVar(value="")

        self._scale = root.winfo_fpixels("1i") / 96.0
        self.fonts = Fonts(root)
        setup_style(root, self.px, self.fonts)
        _extra_style(root, self.px, self.fonts)
        # 這個 View 專用的事件標籤：加在自己底下每個元件上，不用 bind_all，
        # 跟別的畫面放在同一個視窗時（整合版的分頁）不會互相搶事件
        self._tag = f"ApuPhotonsView{id(self)}"
        self.bind_class(self._tag, "<Button-1>", self._maybe_close_popover, add="+")
        self.bind_class(self._tag, "<Escape>", lambda _e: self.close_popover())
        self.bind_class(self._tag, "<MouseWheel>", self._scroll_panel, add="+")
        self._build()

        for var in (self.rejection_var, self.low_var, self.high_var, self.weights_var, self.pick_boost_var,
                    self.bits_var, self.gpu_var, self.workers_var, self.memory_var, self.preview_n_var):
            var.trace_add("write", lambda *_: self._params_changed())
        for var in (self.scale_var, self.drizzle_var, self.pixfrac_var, self.fill_holes_var, self.crop_var,
                    self.min_cov_var):
            var.trace_add("write", lambda *_: self._output_changed())
        for var in (self.split_nights_var, self.flat_any_night_var, self.temp_tol_var, self.merge_var,
                    self.grid_var):
            var.trace_add("write", lambda *_: (self._params_changed(), self._schedule_plan()))
        self.target_var.trace_add("write", lambda *_: self._refresh_title())
        self.align_target_var.trace_add("write", lambda *_: self._load_output_controls())
        self.view_var.trace_add("write", lambda *_: self._show_preview())
        self.result_group_var.trace_add("write", lambda *_: self._show_preview())
        self.list_mode_var.trace_add("write", lambda *_: (save_settings(list_mode=self.list_mode_var.get()),
                                                          self._fill_files()))
        self.lang_var.trace_add("write", lambda *_: self.after_idle(self._language_clicked))
        threading.Thread(target=self._detect_gpu, daemon=True).start()
        if light_dirs:
            self.add_files([Path(d) for d in light_dirs])
        self._refresh_all()
        self._poll_job: str | None = self.after(100, self._poll)

    def px(self, v: float) -> int:
        return int(round(v * self._scale))

    def last_dir(self) -> str:
        return str(self.files[-1].parent) if self.files else load_settings().get("last_dir", "")

    # ------------------------------------------------------------------ 版面

    def _build(self) -> None:
        self._build_top_bar()
        self._build_status_bar()
        body = tk.Frame(self, bg=Darkroom.canvas)
        body.pack(fill="both", expand=True)
        self._build_panel(body)
        tk.Frame(body, bg=Darkroom.separator, width=1).pack(side="right", fill="y")
        self._build_canvas_area(body)
        self._tag_widgets(self)

    def _tag_widgets(self, widget: tk.Misc) -> None:
        """把這個 View 的事件標籤加到自己和底下每個元件（放在 'all' 之前；重建介面後要再做一次）。"""
        tags = list(widget.bindtags())
        if self._tag not in tags:
            tags.insert(max(0, len(tags) - 1), self._tag)
            widget.bindtags(tuple(tags))
        for child in widget.winfo_children():
            if not isinstance(child, tk.Toplevel):
                self._tag_widgets(child)

    def _build_top_bar(self) -> None:
        D = Darkroom
        bar = tk.Frame(self, bg=D.chrome, height=self.px(D.top_bar_height))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(self, bg=D.separator, height=1).pack(fill="x")

        identity = tk.Frame(bar, bg=D.chrome)
        identity.pack(side="left", padx=(self.px(14), 0))
        tk.Label(identity, text=APP_NAME, font=self.fonts.title, fg=D.label, bg=D.chrome).pack(side="left")
        badge = tk.Frame(identity, bg=D.beta, padx=1, pady=1)
        badge.pack(side="left", padx=(self.px(7), 0))
        tk.Label(badge, text=f"v{__version__}", font=self.fonts.badge, fg=D.beta, bg=D.chrome,
                 padx=self.px(4)).pack()

        actions = tk.Frame(bar, bg=D.chrome)
        actions.pack(side="right", padx=(0, self.px(14)))
        if self.show_language:
            Segmented(actions, self, [("zh", "繁中"), ("en", "EN")], self.lang_var).pack(side="left")
            tk.Frame(actions, bg=D.separator, width=1, height=self.px(18)).pack(side="left", padx=self.px(10))
        self.add_btn = ttk.Button(actions, text=tr("gui.btn.add_files"), style="Dark.TButton",
                                  command=self.ask_add_files)
        self.add_btn.pack(side="left")
        Tooltip(self.add_btn, tr("gui.btn.add_files.help", shortcut=OPEN_SHORTCUT), self)
        self.trial_btn = ttk.Button(actions, text=tr("gui.btn.trial"), style="Dark.TButton",
                                    command=lambda: self._start(trial=True))
        self.trial_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.trial_btn, tr("gui.btn.trial.help"), self)
        self.stack_btn = ttk.Button(actions, style="Prominent.TButton", command=self._stack_or_stop)
        self.stack_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.stack_btn, tr("gui.btn.stack.help"), self)

        # 中間：目標名稱（輸出檔名的開頭），對應 APU Processing 的文件名稱
        self.doc_title = tk.Label(bar, font=self.fonts.ui, bg=D.chrome)
        self.doc_title.pack(side="left", expand=True, padx=self.px(16))

    def _build_status_bar(self) -> None:
        D = Darkroom
        bar = tk.Frame(self, bg=D.chrome, height=self.px(D.status_bar_height))
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        tk.Frame(self, bg=D.separator, height=1).pack(side="bottom", fill="x")
        self.indicator = tk.Frame(bar, bg=D.chrome)
        self.indicator.pack(side="left", padx=(self.px(14), 0))
        self.status_dot = tk.Canvas(self.indicator, width=self.px(7), height=self.px(7), bg=D.chrome,
                                    highlightthickness=0)
        self.progress = ttk.Progressbar(self.indicator, style="Dark.Horizontal.TProgressbar", length=self.px(120),
                                        mode="determinate")
        tk.Label(bar, textvariable=self.summary_var, font=self.fonts.small, fg=D.label, bg=D.chrome
                 ).pack(side="right", padx=self.px(14))
        tk.Label(bar, textvariable=self.status_var, font=self.fonts.small, fg=D.secondary, bg=D.chrome,
                 anchor="w").pack(side="left", fill="x", expand=True, padx=self.px(8))

    def _build_panel(self, body: tk.Frame) -> None:
        D = Darkroom
        outer = tk.Frame(body, bg=D.panel, width=self.px(D.panel_width))
        outer.pack(side="right", fill="y")
        outer.pack_propagate(False)
        canvas = tk.Canvas(outer, bg=D.panel, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        panel = tk.Frame(canvas, bg=D.panel)
        window = canvas.create_window((0, 0), window=panel, anchor="nw")
        self.panel_canvas = canvas

        def relayout(_event: object = None) -> None:
            canvas.itemconfigure(window, width=canvas.winfo_width())
            canvas.configure(scrollregion=(0, 0, 0, panel.winfo_reqheight()))
            if panel.winfo_reqheight() > canvas.winfo_height() > 1:
                scrollbar.pack(side="right", fill="y", before=canvas)
            else:
                scrollbar.pack_forget()
                canvas.yview_moveto(0)

        panel.bind("<Configure>", relayout)
        canvas.bind("<Configure>", relayout)
        self._relayout_panel = relayout
        sec = lambda v: tr("gui.value.sigma", v=v)  # noqa: E731

        group = PanelGroup(panel, self, "result", tr("gui.group.result"), tr("gui.group.result.info"))
        self.metrics = {key: MetricRow(group.body, self, tr(f"gui.metric.{key}"))
                        for key in ("frames", "integrated", "rejected", "outputs", "fwhm", "holes", "elapsed")}

        group = PanelGroup(panel, self, "calibration", tr("gui.group.calibration"), tr("gui.group.calibration.info"))
        self.cal_summary = tk.Label(group.body, font=self.fonts.small, fg=D.label, bg=D.panel, anchor="w",
                                    justify="left")
        self.cal_summary.pack(fill="x", pady=(0, self.px(6)))
        ParameterSlider(group.body, self, tr("gui.slider.temp_tol"), self.temp_tol_var, 0.5, 10,
                        lambda v: f"±{v:.1f}°C", 0.5).pack(fill="x", pady=(0, self.px(6)))
        ParameterToggle(group.body, self, tr("gui.toggle.split_nights"), self.split_nights_var).pack(
            fill="x", pady=self.px(2))
        ParameterToggle(group.body, self, tr("gui.toggle.flat_any_night"), self.flat_any_night_var).pack(
            fill="x", pady=self.px(2))
        self.cal_btn = ttk.Button(group.body, text=tr("gui.btn.show_calibration"), style="Dark.TButton",
                                  command=lambda: self.notebook.select(1))
        self.cal_btn.pack(anchor="w", pady=(self.px(6), 0))

        group = PanelGroup(panel, self, "integration", tr("gui.group.integration"), tr("gui.group.integration.info"))
        Segmented(group.body, self, [("winsorized", "Winsorized"), ("sigma", "Sigma"),
                                     ("average", tr("gui.rej.average")), ("median", tr("gui.rej.median"))],
                  self.rejection_var, stretch=True).pack(fill="x", pady=(0, self.px(8)))
        self.low_slider = ParameterSlider(group.body, self, tr("gui.slider.low"), self.low_var, 1.5, 8, sec, 0.1)
        self.low_slider.pack(fill="x", pady=(0, self.px(4)))
        self.high_slider = ParameterSlider(group.body, self, tr("gui.slider.high"), self.high_var, 1.5, 8, sec, 0.1)
        self.high_slider.pack(fill="x", pady=(0, self.px(6)))
        ParameterToggle(group.body, self, tr("gui.toggle.weights"), self.weights_var).pack(fill="x", pady=self.px(2))
        ParameterToggle(group.body, self, tr("gui.toggle.pick_boost"), self.pick_boost_var).pack(fill="x", pady=self.px(2))

        group = PanelGroup(panel, self, "output", tr("gui.group.output"), tr("gui.group.output.info"))
        self.align_row = tk.Frame(group.body, bg=D.panel)
        tk.Label(self.align_row, text=tr("gui.output.apply_to"), font=self.fonts.small, fg=D.secondary,
                 bg=D.panel).pack(anchor="w")
        self.align_combo = ttk.Combobox(self.align_row, state="readonly", style="Dark.TCombobox",
                                        font=self.fonts.small)
        self.align_combo.pack(fill="x", pady=(self.px(3), self.px(8)))
        self.align_combo.bind("<<ComboboxSelected>>", self._align_combo_selected)
        self.align_anchor = tk.Frame(group.body, bg=D.panel)
        self.align_anchor.pack(fill="x")
        tk.Label(group.body, text=tr("gui.output.scale"), font=self.fonts.small, fg=D.secondary,
                 bg=D.panel).pack(anchor="w")
        Segmented(group.body, self, [("0.5", "0.5×"), ("1", "1×"), ("2", "2×")], self.scale_var,
                  stretch=True).pack(fill="x", pady=(self.px(3), self.px(8)))
        self.drizzle_toggle = ParameterToggle(group.body, self, tr("gui.toggle.drizzle"), self.drizzle_var)
        self.drizzle_toggle.pack(fill="x", pady=self.px(2))
        self.pixfrac_slider = ParameterSlider(group.body, self, tr("gui.slider.pixfrac"), self.pixfrac_var, 0.5, 1.0,
                                              lambda v: f"{v:.2f}", 0.05)
        self.pixfrac_slider.pack(fill="x", pady=(self.px(4), self.px(4)))
        self.fill_toggle = ParameterToggle(group.body, self, tr("gui.toggle.fill_holes"), self.fill_holes_var)
        self.fill_toggle.pack(fill="x", pady=self.px(2))
        ParameterToggle(group.body, self, tr("gui.toggle.crop"), self.crop_var).pack(fill="x", pady=self.px(2))
        self.cov_slider = ParameterSlider(group.body, self, tr("gui.slider.min_coverage"), self.min_cov_var, 0.5, 1.0,
                                          lambda v: f"{v:.0%}", 0.05)
        self.cov_slider.pack(fill="x", pady=(self.px(4), self.px(4)))
        self.merge_frame = tk.Frame(group.body, bg=D.panel)
        self.merge_frame.pack(fill="x")
        self.merge_toggle = ParameterToggle(self.merge_frame, self, tr("gui.toggle.merge"), self.merge_var)
        self.grid_row = tk.Frame(self.merge_frame, bg=D.panel)
        tk.Label(self.grid_row, text=tr("gui.output.grid"), font=self.fonts.small, fg=D.secondary,
                 bg=D.panel).pack(anchor="w")
        self.grid_combo = ttk.Combobox(self.grid_row, state="readonly", style="Dark.TCombobox", font=self.fonts.small,
                                       textvariable=self.grid_var)
        self.grid_combo.pack(fill="x", pady=(self.px(3), 0))
        row = tk.Frame(group.body, bg=D.panel)
        row.pack(fill="x", pady=(self.px(6), 0))
        tk.Label(row, text=tr("gui.output.bits"), font=self.fonts.ui, fg=D.label, bg=D.panel).pack(side="left")
        Segmented(row, self, [("32", "32-bit"), ("16", "16-bit")], self.bits_var).pack(side="right")
        tk.Label(group.body, text=tr("gui.output.target"), font=self.fonts.small, fg=D.secondary,
                 bg=D.panel).pack(anchor="w", pady=(self.px(10), 0))
        self.target_entry = ttk.Entry(group.body, textvariable=self.target_var, style="Dark.TEntry",
                                      font=self.fonts.ui)
        self.target_entry.pack(fill="x", pady=(self.px(3), 0))
        tk.Label(group.body, text=tr("gui.output.folder"), font=self.fonts.small, fg=D.secondary,
                 bg=D.panel).pack(anchor="w", pady=(self.px(10), 0))
        self.output_label = tk.Label(group.body, font=self.fonts.small, fg=D.label, bg=D.panel, anchor="w",
                                     justify="left")
        self.output_label.pack(fill="x", pady=(self.px(2), self.px(6)))
        row = tk.Frame(group.body, bg=D.panel)
        row.pack(fill="x")
        self.output_btn = ttk.Button(row, text=tr("gui.btn.change"), style="Dark.TButton", command=self._choose_output)
        self.output_btn.pack(side="left")
        self.reveal_btn = ttk.Button(row, text=tr("gui.btn.reveal"), style="Dark.TButton", command=self._reveal_output)
        self.reveal_btn.pack(side="left", padx=(self.px(6), 0))

        group = PanelGroup(panel, self, "performance", tr("gui.group.performance"), tr("gui.group.performance.info"))
        self.gpu_toggle = ParameterToggle(group.body, self, tr("gui.toggle.gpu"), self.gpu_var)
        self.gpu_toggle.pack(fill="x", pady=self.px(2))
        self.gpu_label = tk.Label(group.body, font=self.fonts.small, fg=D.secondary, bg=D.panel, anchor="w")
        self.gpu_label.pack(fill="x", pady=(0, self.px(6)))
        ParameterSlider(group.body, self, tr("gui.slider.workers"), self.workers_var, 1,
                        max(2, os.cpu_count() or 2), lambda v: f"{v:.0f}").pack(fill="x", pady=(0, self.px(4)))
        ParameterSlider(group.body, self, tr("gui.slider.memory"), self.memory_var, 1, 32,
                        lambda v: f"{v:.0f} GB").pack(fill="x", pady=(0, self.px(4)))
        ParameterSlider(group.body, self, tr("gui.slider.trial"), self.preview_n_var, 5, 60,
                        lambda v: tr("gui.value.frames", n=int(v))).pack(fill="x")

    def _scrolled(self, parent: tk.Misc, tree: ttk.Treeview) -> None:
        """清單加上直、橫捲軸（欄位比視窗寬時可以往右捲）。"""
        ys = ttk.Scrollbar(parent, orient="vertical", command=tree.yview, style="Dark.Vertical.TScrollbar")
        xs = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview, style="Dark.Horizontal.TScrollbar")
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

    def _toolbar(self, parent) -> tk.Frame:
        bar = tk.Frame(parent, bg=Darkroom.chrome)
        bar.pack(fill="x")
        return bar

    def _build_canvas_area(self, body: tk.Frame) -> None:
        D = Darkroom
        area = tk.Frame(body, bg=D.canvas)
        area.pack(side="left", fill="both", expand=True)
        nb = ttk.Notebook(area, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)
        self.notebook = nb

        # ---- 檔案分頁：分組顯示或全部列表
        files = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(files, text=tr("gui.tab.files"))
        top = self._toolbar(files)
        Segmented(top, self, [("groups", tr("gui.files.by_group")), ("all", tr("gui.files.all"))],
                  self.list_mode_var).pack(side="left", padx=self.px(12), pady=self.px(6))
        self.remove_btn = ttk.Button(top, text=tr("gui.btn.remove"), style="Dark.TButton", command=self._remove_files)
        self.remove_btn.pack(side="right", padx=(0, self.px(12)))
        self.kind_btn = ttk.Menubutton(top, text=tr("gui.btn.set_kind"), style="Dark.TMenubutton")
        self.kind_btn.pack(side="right", padx=(0, self.px(6)))
        menu = tk.Menu(self.kind_btn, tearoff=False, bg=D.group_header, fg=D.label, activebackground=D.prominent,
                       activeforeground="white", bd=0, font=self.fonts.ui)
        for kind in KINDS:
            menu.add_command(label=tr(f"kind.{kind}"), command=lambda k=kind: self._set_kind(k))
        menu.add_separator()
        menu.add_command(label=tr("gui.kind.auto"), command=lambda: self._set_kind(None))
        self.kind_btn["menu"] = menu
        self.filters_btn = ttk.Button(top, text=tr("gui.btn.filters"), style="Dark.TButton",
                                      command=self._edit_filters)
        self.filters_btn.pack(side="right", padx=(0, self.px(6)))
        if not IS_MAC:  # Mac 在選單列「檔案」裡；Windows 不放選單列（原生選單列沒辦法變深色）
            self.recipe_btn = ttk.Button(top, text=tr("gui.btn.load_recipe"), style="Dark.TButton",
                                         command=self.ask_load_recipe)
            self.recipe_btn.pack(side="right", padx=(0, self.px(6)))
        ttk.Button(top, text=tr("gui.btn.add"), style="Dark.TButton", command=self.ask_add_files).pack(
            side="right", padx=(0, self.px(6)))

        table = tk.Frame(files, bg=D.canvas)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=[c[0] for c in COLUMNS], show="tree headings", style="Dark.Treeview",
                                 selectmode="extended")
        self.tree.heading("#0", text=tr("gui.col.file"), anchor="w")
        self.tree.column("#0", width=self.px(290), minwidth=self.px(160), anchor="w", stretch=False)
        for key, label, width, anchor in COLUMNS:
            self.tree.heading(key, text=tr(label), anchor=anchor)
            self.tree.column(key, width=self.px(width), minwidth=self.px(40), anchor=anchor, stretch=key == "reason")
        self.tree.tag_configure("rejected", foreground=D.reject)
        self.tree.tag_configure("pending", foreground=D.secondary)
        self.tree.tag_configure("group", foreground=D.accent)
        self.tree.tag_configure("unknown", foreground=D.beta)
        self._scrolled(table, self.tree)
        # 提示放在清單上面，底色要跟清單一樣
        self.empty = tk.Frame(table, bg=D.list_bg)
        tk.Label(self.empty, text=APP_NAME, font=self.fonts.hero, fg=D.label, bg=D.list_bg).pack()
        tk.Label(self.empty, text=tr("gui.empty.hint", shortcut=OPEN_SHORTCUT), font=self.fonts.hero_sub,
                 fg=D.secondary, bg=D.list_bg, justify="center").pack(pady=(self.px(8), 0))

        # ---- 校正分頁：每個校正組配到的校正檔（可以改）＋ 所有校正檔
        cal = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(cal, text=tr("gui.tab.calibration"))
        top = self._toolbar(cal)
        tk.Label(top, text=tr("gui.cal.groups"), font=self.fonts.bold, fg=D.label, bg=D.chrome).pack(
            side="left", padx=self.px(12), pady=self.px(6))
        pane = tk.PanedWindow(cal, orient="vertical", bg=D.separator, sashwidth=self.px(4), bd=0)
        pane.pack(fill="both", expand=True)
        upper = tk.Frame(pane, bg=D.canvas)
        lower = tk.Frame(pane, bg=D.canvas)
        pane.add(upper, stretch="always")
        pane.add(lower, stretch="always")
        cols = [("n", "gui.col.count", 46, "e"), ("dark", "kind.dark", 180, "w"), ("bias", "kind.bias", 140, "w"),
                ("source", "gui.col.source", 56, "center"), ("flat", "kind.flat", 200, "w")]
        cal_holder = tk.Frame(upper, bg=D.canvas)
        cal_holder.pack(fill="both", expand=True)
        self.cal_tree = ttk.Treeview(cal_holder, columns=[c[0] for c in cols], show="tree headings",
                                     style="Dark.Treeview", selectmode="browse", height=6)
        self.cal_tree.heading("#0", text=tr("gui.col.cal_group"), anchor="w")
        self.cal_tree.column("#0", width=self.px(430), minwidth=self.px(200), anchor="w", stretch=False)
        for key, label, width, anchor in cols:
            self.cal_tree.heading(key, text=tr(label), anchor=anchor)
            self.cal_tree.column(key, width=self.px(width), minwidth=self.px(40), anchor=anchor,
                                 stretch=key == "flat")
        self.cal_tree.tag_configure("missing", foreground=D.beta)
        self.cal_tree.tag_configure("user", foreground=D.accent)
        self._scrolled(cal_holder, self.cal_tree)
        self.cal_tree.bind("<<TreeviewSelect>>", lambda _e: self._fill_cal_editor())
        editor = tk.Frame(upper, bg=D.panel, padx=self.px(12), pady=self.px(8))
        editor.pack(fill="x")
        self.cal_editor_title = tk.Label(editor, font=self.fonts.small, fg=D.secondary, bg=D.panel, anchor="w")
        self.cal_editor_title.grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, self.px(4)))
        self.cal_combos: dict[str, ttk.Combobox] = {}
        for i, kind in enumerate(("dark", "bias", "flat")):
            tk.Label(editor, text=tr(f"kind.{kind}"), font=self.fonts.ui, fg=D.label, bg=D.panel).grid(
                row=1, column=2 * i, sticky="w", padx=(0 if i == 0 else self.px(12), self.px(4)))
            cb = ttk.Combobox(editor, state="readonly", style="Dark.TCombobox", font=self.fonts.small, width=24)
            cb.grid(row=1, column=2 * i + 1, sticky="ew")
            cb.bind("<<ComboboxSelected>>", lambda _e, k=kind: self._cal_combo_selected(k))
            self.cal_combos[kind] = cb
            editor.columnconfigure(2 * i + 1, weight=1)
        self.cal_reset_btn = ttk.Button(editor, text=tr("gui.btn.auto"), style="Dark.TButton",
                                        command=self._cal_reset)
        self.cal_reset_btn.grid(row=1, column=6, padx=(self.px(12), 0))
        top2 = self._toolbar(lower)
        tk.Label(top2, text=tr("gui.cal.sets"), font=self.fonts.bold, fg=D.label, bg=D.chrome).pack(
            side="left", padx=self.px(12), pady=self.px(6))
        cols = [("kind", "gui.col.kind", 76, "w"), ("n", "gui.col.count", 50, "e"),
                ("master", "gui.col.master", 64, "center"), ("status", "gui.col.status", 420, "w")]
        set_holder = tk.Frame(lower, bg=D.canvas)
        set_holder.pack(fill="both", expand=True)
        self.set_tree = ttk.Treeview(set_holder, columns=[c[0] for c in cols], show="tree headings",
                                     style="Dark.Treeview", selectmode="none")
        self.set_tree.heading("#0", text=tr("gui.col.cal_set"), anchor="w")
        self.set_tree.column("#0", width=self.px(260), minwidth=self.px(160), anchor="w", stretch=False)
        for key, label, width, anchor in cols:
            self.set_tree.heading(key, text=tr(label), anchor=anchor)
            self.set_tree.column(key, width=self.px(width), minwidth=self.px(40), anchor=anchor,
                                 stretch=key == "status")
        self.set_tree.tag_configure("unused", foreground=D.secondary)
        self._scrolled(set_holder, self.set_tree)

        # ---- 結果分頁：可放大的預覽 + QC 摘要
        result = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(result, text=tr("gui.tab.result"))
        head = self._toolbar(result)
        self.group_combo = ttk.Combobox(head, state="readonly", style="Dark.TCombobox", font=self.fonts.small,
                                        width=28)
        self.group_combo.pack(side="left", padx=(self.px(12), 0), pady=self.px(6))
        self.group_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self.result_group_var.set(str(self.group_combo.current())))
        self.view_seg_holder = tk.Frame(head, bg=D.chrome)
        self.view_seg_holder.pack(side="left", padx=self.px(12))
        ttk.Button(head, text=tr("gui.zoom.fit"), style="Dark.TButton",
                   command=lambda: self.image_canvas.zoom_fit()).pack(side="left")
        ttk.Button(head, text="100%", style="Dark.TButton",
                   command=lambda: self.image_canvas.zoom_to(1.0)).pack(side="left", padx=(self.px(6), 0))
        tk.Label(head, textvariable=self.zoom_var, font=self.fonts.mono, fg=D.label, bg=D.chrome, width=6).pack(
            side="left", padx=self.px(8))
        tk.Label(head, text=tr("gui.result.stretch_note"), font=self.fonts.small, fg=D.secondary,
                 bg=D.chrome).pack(side="right", padx=self.px(12))
        self.image_canvas = ZoomView(result, tr("gui.result.empty"), self.fonts.hero_sub,
                                     on_zoom=lambda z: self.zoom_var.set("" if z is None else f"{z:.0%}"))
        self.image_canvas.pack(fill="both", expand=True)
        self.qc_text = tk.Text(result, height=6, bg=D.list_bg, fg=D.label, font=self.fonts.small, relief="flat",
                               wrap="word", padx=self.px(12), pady=self.px(8), highlightthickness=0)
        self.qc_text.pack(fill="x")
        self.qc_text.configure(state="disabled")

        # ---- 紀錄分頁
        logf = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(logf, text=tr("gui.tab.log"))
        self.log_text = tk.Text(logf, bg=D.list_bg, fg=D.label, font=self.fonts.mono, relief="flat", wrap="none",
                                padx=self.px(12), pady=self.px(8), highlightthickness=0, insertbackground=D.label)
        ls = ttk.Scrollbar(logf, orient="vertical", command=self.log_text.yview, style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=ls.set)
        ls.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "\n".join(self.log_lines) + ("\n" if self.log_lines else ""))
        self.log_text.configure(state="disabled")

    def _scroll_panel(self, event: tk.Event) -> None:
        w = event.widget
        try:
            inside = str(w).startswith(str(self.panel_canvas))
        except (tk.TclError, AttributeError):
            return
        if not inside or not self.panel_canvas.winfo_exists():
            return
        first, last = self.panel_canvas.yview()
        if first <= 0 and last >= 1:
            return
        # macOS 的 delta 是 ±1 起跳、隨觸控板速度變大的數字，要照大小捲（同 Pick）；Windows 一格是 120
        if IS_MAC:
            steps = -event.delta
        else:
            steps = -3 * (int(event.delta / 120) if abs(event.delta) >= 120 else (1 if event.delta > 0 else -1))
        if steps:
            self.panel_canvas.yview_scroll(steps, "units")

    # ------------------------------------------------------------------ 說明氣泡

    def show_popover(self, owner: tk.Widget, text: str) -> None:
        D = Darkroom
        self.close_popover()
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.configure(bg=D.separator)
        tk.Label(top, text=text, font=self.fonts.ui, fg=D.label, bg=D.group_header, justify="left",
                 wraplength=self.px(360), padx=self.px(16), pady=self.px(14)).pack(padx=1, pady=1)
        top.update_idletasks()
        x = owner.winfo_rootx() - top.winfo_reqwidth() - self.px(8)
        y = owner.winfo_rooty() - self.px(6)
        top.geometry(f"+{max(x, self.root.winfo_rootx())}+{y}")
        self.popover, self.popover_owner = top, owner

    def close_popover(self) -> None:
        if self.popover is not None:
            try:
                self.popover.destroy()
            except tk.TclError:
                pass
        self.popover, self.popover_owner = None, None

    def _maybe_close_popover(self, event: tk.Event) -> None:
        if self.popover is None or event.widget is self.popover_owner:
            return
        if str(event.widget).startswith(str(self.popover)):
            return
        self.close_popover()

    # ------------------------------------------------------------------ 語言

    def _language_clicked(self) -> None:
        lang = self.lang_var.get()
        if lang not in LANGUAGES or lang == get_language():
            return
        if self.is_busy():  # 執行中不換：重建介面會打斷進度顯示
            self.lang_var.set(get_language())
            return
        if self.on_language is not None:
            self.on_language(lang)
        else:
            set_language(lang)
            save_settings(language=lang)
            self.rebuild()

    def rebuild(self) -> None:
        """照目前語言重建介面；檔案、設定、結果都保留。"""
        if self.lang_var.get() != get_language():
            self.lang_var.set(get_language())
        self.close_popover()
        if hasattr(self, "image_canvas"):
            self.image_canvas.close()
        for child in self.winfo_children():
            child.destroy()
        self._build()
        self._refresh_all()
        self.update_idletasks()
        self._relayout_panel()

    # ------------------------------------------------------------------ 檔案

    @property
    def light_dirs(self) -> list[Path]:
        """0.1 相容：目前加入的檔案所在的資料夾。"""
        out: list[Path] = []
        for p in self.files:
            if p.parent not in out:
                out.append(p.parent)
        return out

    def add_files(self, paths) -> None:
        """加入檔案（資料夾等同其中的影像，不往子資料夾找）；light 與校正檔都從這裡加入。"""
        new = [p for p in expand_inputs([Path(x) for x in paths]) if p not in self.files]
        if not new:
            return
        if not self.files:
            save_settings(last_dir=str(new[0].parent))
        self.files.extend(new)
        self._reference = None
        if not self.output_dir_var.get():
            base = new[0].parent.parent if new[0].parent.parent != new[0].parent else new[0].parent
            self.output_dir_var.set(str(base / "APU Photons"))
        self.result = None
        self._schedule_plan()
        self._refresh_all()

    def add_light_folder(self, folder: str | Path) -> None:
        """0.1 相容（整合版、Dock 拖放也呼叫這個）：加入一個資料夾裡的影像。"""
        self.add_files([folder])

    def ask_add_files(self) -> None:
        """選擇檔案後加入（選單「加入檔案…」與 ⌘O／Ctrl+O 呼叫這個）。"""
        if self.is_busy():
            return
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_SUFFIXES)) + " " + \
            " ".join(f"*{e.upper()}" for e in sorted(IMAGE_SUFFIXES))
        paths = filedialog.askopenfilenames(parent=self.root, title=tr("gui.btn.add_files"),
                                            initialdir=self.last_dir(),
                                            filetypes=[(tr("gui.filetype.images"), exts), ("*", "*")])
        if paths:
            self.add_files(list(paths))

    def ask_add_light(self) -> None:
        """0.1 相容的名稱。"""
        self.ask_add_files()

    def copy_selection(self) -> None:
        """選單「拷貝」：拷貝紀錄分頁選取的文字。"""
        self.log_text.event_generate("<<Copy>>")

    def select_all(self) -> None:
        """選單「全選」：選取紀錄分頁的全部文字。"""
        self.log_text.event_generate("<<SelectAll>>")

    def _selected_files(self) -> list[Path]:
        chosen: list[Path] = []
        for iid in self.tree.selection():
            if iid.startswith("f:"):
                chosen.append(Path(iid[2:]))
            else:  # 選了分組：底下的檔案全部算
                stack = list(self.tree.get_children(iid))
                while stack:
                    c = stack.pop()
                    if c.startswith("f:"):
                        chosen.append(Path(c[2:]))
                    stack.extend(self.tree.get_children(c))
        return chosen

    def _remove_files(self) -> None:
        if self.is_busy():
            return
        chosen = {str(p) for p in self._selected_files()}
        if not chosen:
            return
        self.files = [p for p in self.files if str(p) not in chosen]
        self._reference = None
        self.result = None
        self._schedule_plan()
        self._refresh_all()

    def _set_kind(self, kind: str | None) -> None:
        if self.is_busy():
            return
        for p in self._selected_files():
            if kind is None:
                self.kinds.pop(str(p), None)
            else:
                self.kinds[str(p)] = kind
        self.result = None
        self._schedule_plan()

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title=tr("gui.output.folder"),
                                       initialdir=self.output_dir_var.get() or self.last_dir())
        if path:
            self.output_dir_var.set(path)
            self._refresh_all()

    def _reveal_output(self) -> None:
        folder = Path(self.output_dir_var.get())
        if not folder.is_dir():
            return
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606
        elif IS_MAC:
            import subprocess
            subprocess.run(["open", str(folder)], check=False)

    # ------------------------------------------------------------------ 濾鏡名稱對應

    def _edit_filters(self) -> None:
        """濾鏡名稱：列出 header 裡出現過的寫法與自動歸併的結果，可以手動指定合併到哪個名稱。"""
        if self.plan is None:
            return
        D = Darkroom
        names = filter_names(self.plan)
        spellings = sorted({s for v in names.values() for s in v} | set(self.filter_aliases))
        if not spellings:
            messagebox.showinfo(APP_NAME, tr("gui.filters.none"), parent=self.root)
            return
        top = tk.Toplevel(self.root)
        top.title(tr("gui.filters.title"))
        top.configure(bg=D.panel)
        top.transient(self.root)
        tk.Label(top, text=tr("gui.filters.help"), font=self.fonts.small, fg=D.secondary, bg=D.panel, justify="left",
                 wraplength=self.px(420)).grid(row=0, column=0, columnspan=2, sticky="w", padx=self.px(14),
                                               pady=(self.px(12), self.px(8)))
        tk.Label(top, text=tr("gui.filters.header"), font=self.fonts.bold, fg=D.label, bg=D.panel).grid(
            row=1, column=0, sticky="w", padx=self.px(14))
        tk.Label(top, text=tr("gui.filters.target"), font=self.fonts.bold, fg=D.label, bg=D.panel).grid(
            row=1, column=1, sticky="w", padx=self.px(14))
        targets = sorted(names)
        auto = tr("gui.filters.auto")
        combos = {}
        for i, sp in enumerate(spellings, start=2):
            tk.Label(top, text=sp, font=self.fonts.ui, fg=D.label, bg=D.panel).grid(
                row=i, column=0, sticky="w", padx=self.px(14), pady=self.px(2))
            cb = ttk.Combobox(top, values=[auto] + targets, style="Dark.TCombobox", font=self.fonts.small, width=18)
            cb.set(self.filter_aliases.get(sp, auto))
            cb.grid(row=i, column=1, sticky="ew", padx=self.px(14), pady=self.px(2))
            combos[sp] = cb

        def save() -> None:
            aliases = {}
            for sp, cb in combos.items():
                v = cb.get().strip()
                if v and v != auto and v != sp:
                    aliases[sp] = v
            self.filter_aliases = aliases
            save_settings(filter_aliases=aliases)  # 記住：下次相同名稱自動套用
            top.destroy()
            self.result = None
            self._schedule_plan()

        row = tk.Frame(top, bg=D.panel)
        row.grid(row=len(spellings) + 2, column=0, columnspan=2, sticky="e", padx=self.px(14), pady=self.px(12))
        ttk.Button(row, text=tr("gui.btn.cancel"), style="Dark.TButton", command=top.destroy).pack(side="right")
        ttk.Button(row, text=tr("gui.btn.ok"), style="Prominent.TButton", command=save).pack(
            side="right", padx=(0, self.px(6)))
        dark_title_bar(top)

    # ------------------------------------------------------------------ 分組（背景讀 header）

    def _plan_args(self) -> dict:
        return dict(kinds=dict(self.kinds), filter_aliases=dict(self.filter_aliases),
                    temp_tolerance=float(self.temp_tol_var.get()), split_nights=bool(self.split_nights_var.get()),
                    flat_any_night=bool(self.flat_any_night_var.get()),
                    overrides=json.loads(json.dumps(self.calib_overrides)), merge_trains=self._merge_trains(),
                    name=None)

    def _merge_trains(self) -> list[list[str]]:
        if not self.merge_var.get() or self.plan is None or len(self.plan.trains) < 2:
            return []
        return [list(self.plan.trains)]

    def _schedule_plan(self) -> None:
        """檔案或分組設定改了：在背景重新分類、分組、配對（RAW 要讀整個檔案才有 header）。"""
        self._plan_token += 1
        token = self._plan_token
        if not self.files:
            self.plan, self.plan_warnings, self._planning = None, [], False
            self._refresh_all()
            return
        self._planning = True
        files, args = list(self.files), self._plan_args()

        def work() -> None:
            warnings: list = []
            try:
                project = ingest(files, warnings, **args)
                self.events.put(("plan", token, project, warnings))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("plan", token, None, [str(exc)]))
        threading.Thread(target=work, daemon=True).start()
        self._status(lambda: tr("gui.status.reading"))

    def wait_plan(self, timeout: float = 60) -> None:
        """測試用：等背景分組做完並套用。"""
        end = time.time() + timeout
        while self._planning and time.time() < end:
            time.sleep(0.02)
            self.drain()

    # ------------------------------------------------------------------ 校正配對

    def _fill_cal_editor(self) -> None:
        sel = self.cal_tree.selection()
        project = self.plan
        gk = sel[0][2:] if sel and sel[0].startswith("g:") else None
        if project is None or gk not in project.calib_groups:
            self.cal_editor_title.configure(text=tr("gui.cal.select"))
            for cb in self.cal_combos.values():
                cb.set("")
                cb.state(["disabled"])
            self.cal_reset_btn.state(["disabled"])
            return
        g = project.calib_groups[gk]
        self.cal_editor_title.configure(text=gk)
        none = tr("gui.cal.none_option")
        for kind, cb in self.cal_combos.items():
            ids = [sid for sid, s in project.calibration_sets.items() if s.kind == kind]
            cb.configure(values=[none] + ids)
            cb.set(g.get(kind) or none)
            cb.state(["!disabled"] if not self.is_busy() else ["disabled"])
        self.cal_reset_btn.state(["!disabled"] if gk in self.calib_overrides and not self.is_busy() else ["disabled"])

    def _cal_combo_selected(self, kind: str) -> None:
        sel = self.cal_tree.selection()
        if not sel or not sel[0].startswith("g:"):
            return
        gk = sel[0][2:]
        value = self.cal_combos[kind].get()
        sid = None if value == tr("gui.cal.none_option") else value
        self.calib_overrides.setdefault(gk, {})[kind] = sid
        self.result = None
        self._schedule_plan()

    def _cal_reset(self) -> None:
        sel = self.cal_tree.selection()
        if sel and sel[0].startswith("g:"):
            self.calib_overrides.pop(sel[0][2:], None)
            self.result = None
            self._schedule_plan()

    # ------------------------------------------------------------------ 設定 → 引擎

    def _drizzle_of(self, values: dict) -> tuple[int, float]:
        scale = values["scale"]
        drizzle = 2 if scale == "2" else (1 if scale == "1" and values["drizzle"] else 0)
        return drizzle, 0.5 if scale == "0.5" else 1.0

    def _engine_output(self, values: dict) -> dict:
        drizzle, downsample = self._drizzle_of(values)
        return {"drizzle": drizzle, "downsample": downsample, "pixfrac": round(float(values["pixfrac"]), 2),
                "fill_holes": bool(values["fill_holes"]), "crop_common": bool(values["crop"]),
                "crop_min_coverage": round(float(values["min_coverage"]), 2)}

    def settings(self, trial: bool = False) -> Settings:
        g = self._engine_output(self.global_output)
        align_output = {aid: self._engine_output({**self.global_output, **vals})
                        for aid, vals in self.align_output.items()}
        merge = self._merge_trains()
        grid = {}
        if merge and self.grid_var.get() in merge[0]:
            grid = {" + ".join(merge[0]): self.grid_var.get()}
        return Settings(
            rejection=self.rejection_var.get(), low=round(float(self.low_var.get()), 2),
            high=round(float(self.high_var.get()), 2),
            weighting="snr2_over_fwhm2" if self.weights_var.get() else "none",
            pick_boost=bool(self.pick_boost_var.get()), downsample=g["downsample"],
            crop_common=g["crop_common"], crop_min_coverage=g["crop_min_coverage"],
            output_bits=int(self.bits_var.get()), memory_mb=int(self.memory_var.get()) * 1024,
            split_nights=bool(self.split_nights_var.get()), flat_any_night=bool(self.flat_any_night_var.get()),
            preview=int(self.preview_n_var.get()) if trial else 0,
            drizzle=g["drizzle"], pixfrac=g["pixfrac"], fill_holes=g["fill_holes"],
            gpu="auto" if self.gpu_var.get() else "cpu", workers=int(self.workers_var.get()),
            target=self.target_var.get().strip() or None, temp_tolerance=float(self.temp_tol_var.get()),
            kinds=dict(self.kinds), filter_aliases=dict(self.filter_aliases),
            calib_overrides=json.loads(json.dumps(self.calib_overrides)), merge_trains=merge,
            grid_train=grid, align_output=align_output)

    def output_dir(self, trial: bool = False) -> Path:
        out = Path(self.output_dir_var.get())
        return out / "trial" if trial else out

    def _params_changed(self) -> None:
        self._update_output_rules()
        save_settings(params={
            "rejection": self.rejection_var.get(), "low": float(self.low_var.get()),
            "high": float(self.high_var.get()), "weights": bool(self.weights_var.get()),
            "pick_boost": bool(self.pick_boost_var.get()), "split_nights": bool(self.split_nights_var.get()),
            "flat_any_night": bool(self.flat_any_night_var.get()),
            "temp_tolerance": float(self.temp_tol_var.get()),
            **{k: v for k, v in self.global_output.items()},
            "bits": self.bits_var.get(), "gpu": bool(self.gpu_var.get()),
            "workers": int(self.workers_var.get()), "memory_gb": int(self.memory_var.get()),
            "preview_n": int(self.preview_n_var.get())})

    def _output_values(self) -> dict:
        return {"scale": self.scale_var.get(), "drizzle": bool(self.drizzle_var.get()),
                "pixfrac": float(self.pixfrac_var.get()), "fill_holes": bool(self.fill_holes_var.get()),
                "crop": bool(self.crop_var.get()), "min_coverage": float(self.min_cov_var.get())}

    def _output_changed(self) -> None:
        """輸出控制項改了：存到目前選的對齊組（或所有對齊組的預設）。"""
        if self._loading_output:
            return
        self._update_output_rules()
        target = self.align_target_var.get()
        if target == ALL_ALIGN:
            self.global_output = self._output_values()
            self._params_changed()
        else:
            self.align_output[target] = self._output_values()

    def _load_output_controls(self) -> None:
        target = self.align_target_var.get()
        values = self.global_output if target == ALL_ALIGN else {**self.global_output,
                                                                **self.align_output.get(target, {})}
        self._loading_output = True
        try:
            self.scale_var.set(values["scale"])
            self.drizzle_var.set(values["drizzle"])
            self.pixfrac_var.set(values["pixfrac"])
            self.fill_holes_var.set(values["fill_holes"])
            self.crop_var.set(values["crop"])
            self.min_cov_var.set(values["min_coverage"])
        finally:
            self._loading_output = False
        self._update_output_rules()

    def _align_combo_selected(self, _event=None) -> None:
        i = self.align_combo.current()
        ids = [ALL_ALIGN] + ([a.id for a in self.plan.align_groups] if self.plan else [])
        self.align_target_var.set(ids[i] if 0 <= i < len(ids) else ALL_ALIGN)

    def _update_output_rules(self) -> None:
        """2× 一定要 drizzle；0.5× 不能 drizzle；1× 可選。rejection 是平均 / 中位數時沒有 σ。"""
        scale = self.scale_var.get()
        if scale == "2" and not self.drizzle_var.get():
            self.drizzle_var.set(True)
        elif scale == "0.5" and self.drizzle_var.get():
            self.drizzle_var.set(False)
        if not hasattr(self, "drizzle_toggle") or not self.drizzle_toggle.winfo_exists():
            return
        self.drizzle_toggle.set_enabled(scale == "1")
        on = bool(self.drizzle_var.get())
        self.pixfrac_slider.set_enabled(on)
        self.fill_toggle.set_enabled(on)
        self.cov_slider.set_enabled(bool(self.crop_var.get()))
        clip = self.rejection_var.get() in ("winsorized", "sigma")
        self.low_slider.set_enabled(clip)
        self.high_slider.set_enabled(clip)

    # ------------------------------------------------------------------ Recipe

    def ask_load_recipe(self) -> None:
        """選單「載入 Recipe…」。"""
        if self.is_busy():
            return
        path = filedialog.askopenfilename(parent=self.root, title=tr("gui.recipe.title"),
                                          initialdir=self.output_dir_var.get() or self.last_dir(),
                                          filetypes=[("Recipe", "*.recipe.json *.json"), ("*", "*")])
        if not path:
            return
        mode = messagebox.askyesnocancel(APP_NAME, tr("gui.recipe.ask"), parent=self.root)
        if mode is None:
            return
        self.load_recipe_file(Path(path), reproduce=bool(mode))

    def load_recipe_file(self, path: Path, reproduce: bool) -> list[str]:
        """reproduce=True：完全重現（加入 recipe 的檔案、分類、配對、參考 frame）；False：只套用設定。
        回傳要提醒使用者的事（找不到或內容不同的檔案）。"""
        try:
            rec = load_recipe(path, reproduce=reproduce, base=self.settings())
        except RecipeError as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            return []
        s = rec.settings
        self.rejection_var.set(s.rejection)
        self.low_var.set(s.low)
        self.high_var.set(s.high)
        self.weights_var.set(s.weighting != "none")
        self.pick_boost_var.set(s.pick_boost)
        self.split_nights_var.set(s.split_nights)
        self.flat_any_night_var.set(s.flat_any_night)
        self.temp_tol_var.set(s.temp_tolerance)
        self.bits_var.set(str(s.output_bits))
        scale = "2" if s.drizzle == 2 else ("0.5" if s.downsample == 0.5 else "1")
        self.global_output = {"scale": scale, "drizzle": s.drizzle >= 1, "pixfrac": s.pixfrac,
                              "fill_holes": s.fill_holes, "crop": s.crop_common, "min_coverage": s.crop_min_coverage}
        self.align_output = {}
        for aid, o in s.align_output.items():
            sc = "2" if o.get("drizzle") == 2 else ("0.5" if o.get("downsample") == 0.5 else "1")
            self.align_output[aid] = {"scale": sc, "drizzle": (o.get("drizzle") or 0) >= 1,
                                      "pixfrac": o.get("pixfrac", s.pixfrac),
                                      "fill_holes": o.get("fill_holes", s.fill_holes),
                                      "crop": o.get("crop_common", s.crop_common),
                                      "min_coverage": o.get("crop_min_coverage", s.crop_min_coverage)}
        self.align_target_var.set(ALL_ALIGN)
        self._load_output_controls()
        self.filter_aliases = dict(s.filter_aliases)
        self.merge_var.set(bool(s.merge_trains))
        if s.grid_train:
            self.grid_var.set(next(iter(s.grid_train.values())))
        if reproduce:
            self.files = list(rec.files)
            self.kinds = dict(s.kinds)
            self.calib_overrides = dict(s.calib_overrides)
            self.target_var.set(s.target or "")
            self._reference = s.reference
        self.result = None
        self._schedule_plan()
        self._refresh_all()
        for w in rec.warnings:
            self._append_log(f"⚠ {w}")
        if rec.warnings:
            messagebox.showwarning(APP_NAME, "\n".join(str(w) for w in rec.warnings[:12]), parent=self.root)
        return [str(w) for w in rec.warnings]

    # ------------------------------------------------------------------ GPU

    def _detect_gpu(self) -> None:
        be = backend.select("auto")
        self.events.put(("gpu", be.device if be.gpu else None))

    def _update_gpu_view(self) -> None:
        if not self.gpu_label.winfo_exists():
            return
        if not self._gpu_checked:
            self.gpu_label.configure(text=tr("gui.gpu.checking"))
            self.gpu_toggle.set_enabled(False)
        elif self.gpu_name:
            self.gpu_label.configure(text=self.gpu_name)
            self.gpu_toggle.set_enabled(True)
        else:
            self.gpu_label.configure(text=tr("gui.gpu.none"))
            self.gpu_toggle.set_enabled(False, forced_off=True)

    # ------------------------------------------------------------------ 執行

    def is_busy(self) -> bool:
        """有疊圖在跑（關閉視窗前要詢問）。"""
        return self.worker is not None and self.worker.is_alive()

    def _stack_or_stop(self) -> None:
        if self.is_busy():
            self.cancel.set()
            self._status(lambda: tr("gui.status.stopping"))
        else:
            self._start(trial=False)

    def _start(self, trial: bool) -> None:
        if self.is_busy() or not self.files:
            return
        if not self.output_dir_var.get():
            self._choose_output()
            if not self.output_dir_var.get():
                return
        self.cancel.clear()
        self.result = None
        self.image_canvas.set_image(None)
        self.log_lines = []
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._started = time.time()
        self._stage = None
        settings = self.settings(trial)
        settings.reference = getattr(self, "_reference", None)
        args = (list(self.files), self.output_dir(trial), settings)
        self.worker = threading.Thread(target=self._work, args=args, daemon=True)
        self.worker.start()
        self._status(lambda: tr("gui.status.starting"))
        self._refresh_all()

    def _work(self, files, output_dir, settings) -> None:
        """背景執行緒：跑引擎，把進度、紀錄、結果丟進佇列給主執行緒。"""
        put = self.events.put
        try:
            res = run(files, output_dir, settings, echo=lambda line: put(("log", line)),
                      progress=lambda stage, d, t: put(("progress", stage, d, t)), cancel=self.cancel)
            put(("done", res))
        except Cancelled:
            put(("cancelled",))
        except Exception as exc:  # noqa: BLE001  顯示給使用者，不讓執行緒默默死掉
            put(("error", str(exc) or type(exc).__name__, traceback.format_exc()))

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    def drain(self) -> None:
        """測試用：把佇列裡的事件全部處理掉。"""
        while True:
            try:
                self._handle(self.events.get_nowait())
            except queue.Empty:
                return

    def _append_log(self, line: str) -> None:
        self.log_lines.append(line)
        if self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "log":
            self._append_log(event[1])
        elif kind == "plan":
            _, token, project, warnings = event
            if token != self._plan_token:
                return  # 已經有更新的分組在跑
            self._planning = False
            self.plan, self.plan_warnings = project, warnings
            if not self.is_busy():
                self._status(lambda: tr("gui.status.start"))
            self._refresh_all()
        elif kind == "progress":
            _, stage, done, total = event
            if self._stage is None or self._stage[0] != stage:
                self._stage_started = time.time()
            self._stage = (stage, done, total)
            self._status(self._progress_text)
            self._update_progress()
        elif kind == "gpu":
            self._gpu_checked, self.gpu_name = True, event[1]
            self._update_gpu_view()
        elif kind == "done":
            self.result = event[1]
            self.view_var.set("drizzle" if any(o.stack for o in self.result.outputs) else "stack")
            self.result_group_var.set("0")
            elapsed = self._elapsed = time.time() - self._started
            self._status(lambda: tr("gui.status.done", seconds=elapsed))
            self.worker = None
            self._refresh_all()
            self.notebook.select(2)
        elif kind == "cancelled":
            self.worker = None
            self._status(lambda: tr("gui.status.cancelled"))
            self._refresh_all()
        elif kind == "error":
            self.worker = None
            self._append_log(event[2])
            message = event[1]
            self._status(lambda: tr("gui.status.error", message=message))
            self._refresh_all()
            messagebox.showerror(APP_NAME, message, parent=self.root)

    def _progress_text(self) -> str:
        if self._stage is None:
            return tr("gui.status.starting")
        stage, done, total = self._stage
        name = tr(f"gui.stage.{stage}")
        if not total:
            return name
        text = tr("gui.status.progress", stage=name, done=done, total=total)
        spent = time.time() - self._stage_started
        if 0 < done < total and spent > 3:
            text += tr("gui.status.eta", seconds=spent / done * (total - done))
        return text

    # ------------------------------------------------------------------ 畫面更新

    def _status(self, fn: Callable[[], str]) -> None:
        self._status_fn = fn
        self.status_var.set(fn())

    def _update_progress(self) -> None:
        D = Darkroom
        if not self.indicator.winfo_exists():
            return
        if self.is_busy():
            self.status_dot.pack_forget()
            self.progress.pack(side="left")
            if self._stage and self._stage[2]:
                self.progress.configure(mode="determinate", maximum=self._stage[2], value=self._stage[1])
            else:
                self.progress.configure(mode="determinate", maximum=1, value=0)
        else:
            self.progress.pack_forget()
            self.status_dot.pack(side="left")
            self.status_dot.delete("all")
            color = D.accent if self.result is not None else D.secondary
            self.status_dot.create_oval(0, 0, self.px(6), self.px(6), fill=color, outline=color)

    def _shown_project(self) -> Project | None:
        """跑完顯示結果（含每張的狀態）；還沒跑顯示分組。"""
        return self.result.project if self.result is not None else self.plan

    def _refresh_title(self) -> None:
        D = Darkroom
        if not hasattr(self, "doc_title") or not self.doc_title.winfo_exists():
            return
        name = self.target_var.get().strip() or (self.plan.name if self.plan is not None else "")
        if name:
            self.doc_title.configure(text=name, fg=D.label)
        else:
            self.doc_title.configure(text=tr("gui.no_project"), fg=D.secondary)

    def _refresh_all(self) -> None:
        busy = self.is_busy()
        self.status_var.set(self._status_fn())
        self._update_progress()
        self._update_output_rules()
        self._update_gpu_view()
        self._refresh_title()
        self.stack_btn.configure(text=tr("gui.btn.stop") if busy else tr("gui.btn.stack"))
        ready = bool(self.plan is not None and self.plan.lights and not self._planning)
        self.stack_btn.state(["!disabled"] if (ready or busy) else ["disabled"])
        self.trial_btn.state(["!disabled"] if ready and not busy else ["disabled"])
        for btn in (self.add_btn, self.output_btn):
            btn.state(["disabled"] if busy else ["!disabled"])
        for btn in (self.remove_btn, self.filters_btn):
            btn.state(["disabled"] if busy or not self.files else ["!disabled"])
        self.kind_btn.state(["disabled"] if busy or not self.files else ["!disabled"])
        out = self.output_dir_var.get()
        self.reveal_btn.state(["!disabled"] if out and Path(out).is_dir() else ["disabled"])
        self.output_label.configure(text=_short_path(out, 44) if out else tr("gui.output.unset"))
        self._fill_align_controls()
        self._fill_files()
        self._fill_calibration()
        self._fill_metrics()
        self._fill_result()
        p = self.plan
        if p is not None:
            n_cal = sum(len(s.frames) for s in p.calibration_sets.values())
            self.summary_var.set(tr("gui.summary2", files=len(p.files), lights=len(p.lights), cal=n_cal,
                                    unknown=len(p.unknown)))
        else:
            self.summary_var.set(tr("gui.summary_files", files=len(self.files)) if self.files else "")

    def _fill_align_controls(self) -> None:
        p = self.plan
        aligns = p.align_groups if p is not None else []
        if len(aligns) > 1:
            values = [tr("gui.output.all_align")] + [a.id for a in aligns]
            self.align_combo.configure(values=values)
            target = self.align_target_var.get()
            ids = [ALL_ALIGN] + [a.id for a in aligns]
            self.align_combo.current(ids.index(target) if target in ids else 0)
            self.align_row.pack(fill="x", before=self.align_anchor)
        else:
            self.align_row.pack_forget()
            if self.align_target_var.get() != ALL_ALIGN:
                self.align_target_var.set(ALL_ALIGN)
        trains = list(p.trains) if p is not None else []
        if len(trains) > 1:
            self.merge_toggle.pack(fill="x", pady=self.px(2))
            self.grid_combo.configure(values=trains)
            if self.merge_var.get():
                self.grid_row.pack(fill="x", pady=(self.px(2), self.px(4)))
            else:
                self.grid_row.pack_forget()
        else:
            self.merge_toggle.pack_forget()
            self.grid_row.pack_forget()

    def _file_values(self, f, accepted_n: int, refs: set) -> tuple[tuple, tuple]:
        """檔案清單一列的值與標籤。"""
        kind = tr(f"kind.{f.kind}") + (" ★" if f.is_master else "")
        exp = _fmt(f.exposure)
        gain = _fmt(f.gain)
        temp = _fmt(f.temp, 1) if f.temp is not None else ""
        night = f.night if f.night not in (None, "unknown", "all") else ""
        if self.result is None or f.kind != "light":
            tags = ("unknown",) if f.kind == UNKNOWN else (("rejected",) if not f.accepted else ())
            reason = tr(f"reason.{f.reject_reason}") if f.reject_reason else (
                tr("gui.kind.unknown_hint") if f.kind == UNKNOWN else "")
            return (kind, f.filter or "", exp, gain, temp, night, "", "", "", "", "", reason), tags
        ok = f.accepted
        fwhm = f"{f.stars.fwhm:.2f}" if f.stars is not None and f.stars.fwhm == f.stars.fwhm else ""
        stars = str(len(f.stars)) if f.stars is not None else ""
        weight = f"{f.weight * accepted_n:.2f}" if ok and f.weight else ""
        rej = f"{f.rejected_pixel_fraction:.2%}" if ok and f.rejected_pixel_fraction is not None else ""
        reason = tr(f"reason.{f.reject_reason}") if f.reject_reason else ""
        if id(f) in refs:
            reason = tr("gui.reference")
        skipped = f.reject_reason == "not_in_preview"  # 試跑沒抽到：不是問題，用灰色
        result = tr("gui.result.ok") if ok else ("—" if skipped else tr("gui.result.rejected"))
        tags = () if ok else (("pending",) if skipped else ("rejected",))
        return (kind, f.filter or "", exp, gain, temp, night, result, fwhm, stars, weight, rej, reason), tags

    def _fill_files(self) -> None:
        if not hasattr(self, "tree") or not self.tree.winfo_exists():
            return
        items = self._all_items(self.tree)
        open_nodes = {i for i in items if self.tree.item(i, "open")}
        self._seen_nodes.update(i for i in items if not i.startswith("f:"))
        self.tree.delete(*self.tree.get_children())

        def is_open(iid: str, default: bool) -> bool:
            """第一次出現的分組照預設；之後保持使用者展開或收合的狀態。"""
            return iid in open_nodes if iid in self._seen_nodes else default
        if not self.files:
            self.empty.place(relx=0.5, rely=0.42, anchor="center")
            return
        self.empty.place_forget()
        p = self._shown_project()
        if p is None:
            for path in self.files:
                self.tree.insert("", "end", iid=f"f:{path}", text=path.name,
                                 values=("…",) + ("",) * (len(COLUMNS) - 1), tags=("pending",))
            return
        refs = {id(a.reference) for a in p.align_groups if a.reference is not None}
        accepted_n = len(p.accepted) or 1
        blank = ("",) * len(COLUMNS)

        def add(parent, f):
            values, tags = self._file_values(f, accepted_n, refs)
            self.tree.insert(parent, "end", iid=f"f:{f.path}", text=f.name, values=values, tags=tags)

        if self.list_mode_var.get() == "all":
            for f in sorted(p.files, key=lambda f: f.name.casefold()):
                add("", f)
            return
        placed: set[int] = set()
        for a in p.align_groups:
            node = self.tree.insert("", "end", iid=f"a:{a.id}", text=a.id, values=blank, tags=("group",), open=True)
            for i, g in enumerate(a.groups):
                gnode = self.tree.insert(node, "end", iid=f"i:{a.id}:{i}",
                                         text=tr("gui.files.group", filter=g.label, n=len(g.frames)),
                                         values=blank, tags=("group",), open=is_open(f"i:{a.id}:{i}", True))
                for f in g.frames:
                    add(gnode, f)
                    placed.add(id(f))
        if p.calibration_sets:
            cnode = self.tree.insert("", "end", iid="c:", text=tr("gui.files.calibration"), values=blank,
                                     tags=("group",), open=is_open("c:", True))
            for sid, s in p.calibration_sets.items():
                snode = self.tree.insert(cnode, "end", iid=f"s:{sid}",
                                         text=f"{sid}（{len(s.frames)}）" if get_language() == "zh"
                                         else f"{sid} ({len(s.frames)})",
                                         values=blank, tags=("group",) if s.used_by else ("pending",),
                                         open=is_open(f"s:{sid}", False))
                for f in s.frames:
                    add(snode, f)
                    placed.add(id(f))
        rest = [f for f in p.files if id(f) not in placed]
        if rest:
            unk = [f for f in rest if f.kind == UNKNOWN]
            node = self.tree.insert("", "end", iid="u:", text=tr("gui.files.unused", n=len(rest)), values=blank,
                                    tags=("unknown",) if unk else ("pending",), open=True)
            for f in rest:
                add(node, f)

    def _all_items(self, tree: ttk.Treeview, parent: str = "") -> list[str]:
        out = []
        for i in tree.get_children(parent):
            out.append(i)
            out.extend(self._all_items(tree, i))
        return out

    def _fill_calibration(self) -> None:
        if not hasattr(self, "cal_tree") or not self.cal_tree.winfo_exists():
            return
        sel = self.cal_tree.selection()
        self.cal_tree.delete(*self.cal_tree.get_children())
        self.set_tree.delete(*self.set_tree.get_children())
        p = self.plan
        D = Darkroom
        if p is None:
            self.cal_summary.configure(text=tr("gui.cal.empty"), fg=D.secondary)
            self._fill_cal_editor()
            return
        missing_flat = 0
        for gk, g in p.calib_groups.items():
            src = tr("gui.cal.source.user") if g["source"] == "user" else tr("gui.cal.source.auto")
            none = tr("gui.cal.none_short")
            flat = g["flat"] or (tr("gui.cal.flat_other") if g["flat_other_nights"] else none)
            if not g["flat"]:
                missing_flat += 1
            bias = g["bias"] or (tr("gui.cal.bias_not_needed") if g["dark"] else none)
            tags = ("user",) if g["source"] == "user" else (("missing",) if not g["flat"] or not g["dark"] else ())
            self.cal_tree.insert("", "end", iid=f"g:{gk}", text=gk,
                                 values=(len(g["frames"]), g["dark"] or none, bias, src, flat), tags=tags)
        for sid, s in p.calibration_sets.items():
            status = tr("gui.cal.used", n=len(s.used_by)) if s.used_by else tr(f"msg.unused.{s.unused_reason}")
            self.set_tree.insert("", "end", text=sid, values=(tr(f"kind.{s.kind}"), len(s.frames),
                                                              "★" if s.is_master else "", status),
                                 tags=() if s.used_by else ("unused",))
        if sel and self.cal_tree.exists(sel[0]):
            self.cal_tree.selection_set(sel[0])
        lines = [tr("gui.cal.summary", groups=len(p.calib_groups), sets=len(p.calibration_sets))]
        if missing_flat:
            lines.append(tr("gui.cal.missing_flat", n=missing_flat))
        unused = sum(1 for s in p.calibration_sets.values() if not s.used_by)
        if unused:
            lines.append(tr("gui.cal.unused_sets", n=unused))
        self.cal_summary.configure(text="\n".join(lines), fg=D.label)
        self._fill_cal_editor()

    def _fill_metrics(self) -> None:
        m = self.metrics
        if self.result is None:
            for row in m.values():
                row.set("—")
            return
        q = self.result.qc
        m["frames"].set(str(q["frames_total"]))
        m["integrated"].set(str(q["frames_integrated"]))
        m["rejected"].set(str(sum(v for k, v in q["rejected_by_reason"].items() if k != "not_in_preview")))
        m["outputs"].set(str(len(self.result.outputs)))
        fw = [g["fwhm_px"]["median"] for g in q["groups"] if g.get("fwhm_px")]
        m["fwhm"].set(" / ".join(f"{v:.2f}" for v in fw) + " px" if fw else "—")
        holes = [v for g in q["groups"] for v in g.get("drizzle", {}).get("holes_fraction", [])]
        m["holes"].set(f"{max(holes):.1%}" if holes else "—")
        m["elapsed"].set(tr("gui.value.seconds", v=self._elapsed) if self._elapsed else "—")

    def _result_label(self, o) -> str:
        multi = len({x.align for x in self.result.outputs}) > 1
        name = o.filter or tr("gui.result.no_filter")
        return f"{o.align} / {name}" if multi else name

    def _fill_result(self) -> None:
        for child in self.view_seg_holder.winfo_children():
            child.destroy()
        outs = self.result.outputs if self.result is not None else []
        self.group_combo.configure(values=[self._result_label(o) for o in outs])
        if outs:
            try:
                i = int(self.result_group_var.get() or 0)
            except ValueError:
                i = 0
            self.group_combo.current(min(max(i, 0), len(outs) - 1))
            self.group_combo.state(["!disabled"])
        else:
            self.group_combo.set("")
            self.group_combo.state(["disabled"])
        if any(o.stack for o in outs):
            Segmented(self.view_seg_holder, self, [("drizzle", "Drizzle"), ("stack", tr("gui.result.stack"))],
                      self.view_var).pack(side="left")
            self._tag_widgets(self.view_seg_holder)
        self.qc_text.configure(state="normal")
        self.qc_text.delete("1.0", "end")
        if self.result is not None:
            q = self.result.qc
            lines = [tr("gui.qc.output", path=str(self.result.output_dir))]
            for g in q["groups"]:
                lines.append(tr("gui.qc.group", name=g["output"], used=g["frames_integrated"],
                                total=g["frames_total"],
                                drizzle=tr("gui.qc.drizzle_ok") if g["drizzle_suitable"] else tr("gui.qc.drizzle_no")))
            if q.get("suspicious_frames"):
                lines.append(tr("gui.qc.suspicious", files="、".join(q["suspicious_frames"])))
            lines += [f"⚠ {w}" for w in self.result.warnings]
            self.qc_text.insert("end", "\n".join(lines))
        self.qc_text.configure(state="disabled")
        self._show_preview()

    def _current_output(self):
        if self.result is None or not self.result.outputs:
            return None
        try:
            i = int(self.result_group_var.get() or 0)
        except ValueError:
            i = 0
        return self.result.outputs[min(max(i, 0), len(self.result.outputs) - 1)]

    def _show_preview(self) -> None:
        if not hasattr(self, "image_canvas") or not self.image_canvas.winfo_exists():
            return
        o = self._current_output()
        path = None
        if o is not None:
            path = o.stack if (self.view_var.get() == "stack" and o.stack) else o.output
        if path != self.image_canvas.path:
            self.image_canvas.set_image(path)

    # ------------------------------------------------------------------ 關閉

    def close(self) -> None:
        """取消背景工作、停掉排程、關掉說明氣泡。呼叫端接著 destroy() 這個 View；
        關掉之後請在主執行緒 gc.collect()，不然 View 的 Tk 變數可能在背景執行緒被循環回收釋放。
        疊圖的 worker 收到取消後，會在下一個檢查點結束。"""
        self.cancel.set()
        self._plan_token += 1
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except (tk.TclError, ValueError):
                pass
            self._poll_job = None
        self.close_popover()
        if hasattr(self, "image_canvas"):
            self.image_canvas.close()


def _extra_style(root: tk.Misc, px: Callable[[float], int], fonts: Fonts) -> None:
    """Photons 另外用到的 ttk 樣式（下拉選單、輸入框、選單按鈕）；共用的在 darkroom.setup_style。"""
    D = Darkroom
    style = ttk.Style(root)
    style.configure("Dark.TCombobox", fieldbackground=D.control, background=D.control, foreground=D.label,
                    arrowcolor=D.label, bordercolor=D.separator, lightcolor=D.control, darkcolor=D.control,
                    selectbackground=D.control, selectforeground=D.label, padding=(px(6), px(2)))
    style.map("Dark.TCombobox", fieldbackground=[("readonly", D.control), ("disabled", D.panel)],
              foreground=[("disabled", "#6b6b6b")], selectbackground=[("readonly", D.control)])
    style.configure("Dark.TEntry", fieldbackground=D.control, foreground=D.label, bordercolor=D.separator,
                    lightcolor=D.control, darkcolor=D.control, insertcolor=D.label, padding=(px(6), px(3)))
    style.configure("Dark.TMenubutton", background=D.control, foreground=D.label, arrowcolor=D.label,
                    bordercolor=D.control, lightcolor=D.control, darkcolor=D.control, padding=(px(10), px(4)),
                    font=fonts.ui)
    style.map("Dark.TMenubutton", background=[("active", D.hover), ("disabled", D.panel)],
              foreground=[("disabled", "#6b6b6b")])
    # 下拉清單本身是 Tk 的 Listbox
    root.option_add("*TCombobox*Listbox.background", D.group_header)
    root.option_add("*TCombobox*Listbox.foreground", D.label)
    root.option_add("*TCombobox*Listbox.selectBackground", D.prominent)
    root.option_add("*TCombobox*Listbox.selectForeground", "white")


# ---------------------------------------------------------------------- 視窗


def _build_menubar(root: tk.Tk, view: PhotonsView) -> None:
    """Mac 的選單列（Windows 沿用原本的做法，不放選單列）。"""
    menubar = tk.Menu(root)
    app_menu = tk.Menu(menubar, name="apple", tearoff=False)
    app_menu.add_command(label=tr("gui.menu.about"), command=lambda: root.tk.call("::tk::mac::standardAboutPanel"))
    app_menu.add_separator()
    menubar.add_cascade(menu=app_menu)
    file_menu = tk.Menu(menubar, tearoff=False)
    file_menu.add_command(label=tr("gui.menu.add_files"), accelerator="Command-O", command=view.ask_add_files)
    file_menu.add_command(label=tr("gui.menu.load_recipe"), command=view.ask_load_recipe)
    menubar.add_cascade(label=tr("gui.menu.file"), menu=file_menu)
    edit_menu = tk.Menu(menubar, tearoff=False)
    edit_menu.add_command(label=tr("gui.menu.copy"), accelerator="Command-C", command=view.copy_selection)
    edit_menu.add_command(label=tr("gui.menu.select_all"), accelerator="Command-A", command=view.select_all)
    menubar.add_cascade(label=tr("gui.menu.edit"), menu=edit_menu)
    menubar.add_cascade(label=tr("gui.menu.window"), menu=tk.Menu(menubar, name="window"))
    root.configure(menu=menubar)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    lang = load_settings().get("language")
    if lang in LANGUAGES:
        set_language(lang)
    enable_dpi_awareness()
    root = tk.Tk()
    if IS_MAC:
        root.tk.call("tk", "scaling", 96 / 72)
    if ICON.is_file():
        try:
            root.iconbitmap(default=str(ICON))
        except tk.TclError:
            pass
    scale = root.winfo_fpixels("1i") / 96.0
    width = min(int(round(1360 * scale)), root.winfo_screenwidth() - int(round(40 * scale)))
    height = min(int(round(900 * scale)), root.winfo_screenheight() - int(round(110 * scale)))
    root.geometry(f"{width}x{height}")
    root.minsize(int(round(1080 * scale)), int(round(720 * scale)))
    root.configure(bg=Darkroom.canvas)
    root.title(f"{APP_NAME} {__version__}")

    def menus() -> None:
        if IS_MAC:
            _build_menubar(root, view)

    def change_language(lang: str) -> None:
        set_language(lang)
        save_settings(language=lang)
        view.rebuild()
        menus()  # 選單文字跟著換

    view = PhotonsView(root, root, light_dirs=[a for a in argv if not a.startswith("-")],
                       on_language=change_language)
    view.pack(fill="both", expand=True)
    menus()
    if not IS_MAC:
        root.bind_all("<Control-o>", lambda _e: view.ask_add_files())

    def on_close() -> None:
        busy = view.is_busy()
        if busy and not messagebox.askyesno(APP_NAME, tr("gui.close.confirm"), parent=root):
            return
        view.close()
        if busy:
            # 行程要等 worker 做完手上這一段才會結束；先把視窗收起來，不然 Mac 上視窗會停在畫面上十幾秒
            root.withdraw()
            root.update()
        view.destroy()
        gc.collect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    if IS_MAC:
        root.createcommand("::tk::mac::Quit", on_close)
        # 把檔案或資料夾拖到 Dock 圖示上：加入
        root.createcommand("::tk::mac::OpenDocument", lambda *paths: view.add_files(list(paths)))
    dark_title_bar(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
