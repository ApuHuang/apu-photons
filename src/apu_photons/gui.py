"""視窗介面，與 APU Astro、APU Pick 同一套介面語言：暗房深色主題、頂部列、右側可收合參數面板、底部狀態列。

流程：加入 Light 資料夾 →（選擇 calibration）→ 試跑或疊圖 → 在「結果」分頁看成品。
啟動：python -m apu_photons.gui [Light 資料夾...]
介面文字都在 i18n.py，可以切換繁體中文 / English。

暗房元件（Darkroom、PanelGroup、ParameterSlider…）在 darkroom.py，三套軟體共用同一份；
這裡只留 Photons 獨有的 FolderRow 與清單欄位 COLUMNS。
"""

from __future__ import annotations

import gc
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

from PIL import Image, ImageTk

from . import __version__, backend
from .darkroom import (Darkroom, Fonts, MetricRow, PanelGroup, ParameterSlider, ParameterToggle, Segmented, Tooltip,
                       _short_path, _trace, dark_title_bar, enable_dpi_awareness, setup_style)
from .engine import Cancelled, Result, Settings, run
from .i18n import APP_NAME, LANGUAGES, get_language, set_language, tr
from .imageio import list_images
from .model import Calibration
from .preview import load_preview
from .settings import load_settings, save_settings

ASSETS = Path(__file__).parent / "assets"
ICON = ASSETS / "app.ico"
IS_MAC = sys.platform == "darwin"
OPEN_SHORTCUT = "⌘O" if IS_MAC else "Ctrl+O"
CAL_KINDS = ("bias", "dark", "flat", "flat_dark")
PREVIEW_FRAMES = 20


# 清單欄位：(key, 翻譯代號, 寬度 px, 對齊)
COLUMNS = [
    ("file", "gui.col.file", 300, "w"),
    ("session", "gui.col.session", 95, "w"),
    ("result", "gui.col.result", 70, "center"),
    ("fwhm", "gui.col.fwhm", 60, "e"),
    ("stars", "gui.col.stars", 60, "e"),
    ("weight", "gui.col.weight", 70, "e"),
    ("rejected", "gui.col.rejected", 80, "e"),
    ("reason", "gui.col.reason", 260, "w"),
]


class FolderRow(tk.Frame):
    """calibration 資料夾一列：名稱、張數或「未使用」、選擇 / 清除。"""

    def __init__(self, master: tk.Widget, app: PhotonsView, kind: str, variable: tk.StringVar):
        D = Darkroom
        super().__init__(master, bg=D.panel)
        self.app, self.kind, self.var = app, kind, variable
        head = tk.Frame(self, bg=D.panel)
        head.pack(fill="x")
        tk.Label(head, text=tr(f"gui.cal.{kind}"), font=app.fonts.ui, fg=D.label, bg=D.panel).pack(side="left")
        self.clear_btn = ttk.Button(head, text=tr("gui.btn.clear"), style="Dark.TButton",
                                    command=lambda: variable.set(""))
        self.clear_btn.pack(side="right")
        self.pick_btn = ttk.Button(head, text=tr("gui.btn.choose"), style="Dark.TButton", command=self._choose)
        self.pick_btn.pack(side="right", padx=(0, app.px(4)))
        self.info = tk.Label(self, font=app.fonts.small, fg=D.secondary, bg=D.panel, anchor="w", justify="left")
        self.info.pack(fill="x", pady=(app.px(2), 0))
        _trace(self, variable, self._refresh)
        self._refresh()

    def _choose(self) -> None:
        path = filedialog.askdirectory(parent=self.app.root, title=tr(f"gui.cal.{self.kind}"),
                                       initialdir=self.var.get() or self.app.last_dir())
        if path:
            self.var.set(path)

    def _refresh(self) -> None:
        path = self.var.get()
        if not path:
            self.info.configure(text=tr("gui.cal.none"), fg=Darkroom.secondary)
            self.clear_btn.state(["disabled"])
            return
        self.clear_btn.state(["!disabled"])
        try:
            n = len(list_images(Path(path)))
        except OSError:
            n = 0
        self.info.configure(text=tr("gui.cal.count", n=n, path=_short_path(path, 36)),
                            fg=Darkroom.label if n else Darkroom.reject)


# ---------------------------------------------------------------------- 主視窗


class PhotonsView(tk.Frame):
    """疊圖的主畫面：可以放進任何視窗（單獨的 APU Photons 視窗，或整合版的分頁）。

    不建立 tk.Tk、不碰整個視窗：視窗標題與大小、選單列、快捷鍵、關閉前的詢問都由 main() 負責。
    對外：add_light_folder(path)、ask_add_light()、is_busy()、rebuild()、close()。
    """

    def __init__(self, parent: tk.Misc, root: tk.Tk, *, light_dirs: list[str] | None = None,
                 on_language: Callable[[str], None] | None = None, show_language: bool = True):
        """on_language：按下頂部列的語言切換時呼叫（由外面換語言、重建 View 與選單列）；
        沒給就自己換語言並 rebuild()。show_language=False 時頂部列不顯示語言切換。"""
        super().__init__(parent, bg=Darkroom.canvas)
        self.root = root
        self.on_language = on_language
        self.show_language = show_language
        self.light_dirs: list[Path] = []
        self.result: Result | None = None
        self.previews: dict[str, Image.Image] = {}
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
        self._photo: ImageTk.PhotoImage | None = None
        self._status_fn: Callable[[], str] = lambda: tr("gui.status.start")

        st = load_settings()
        p = st.get("params", {})
        self.panel_state: dict[str, bool] = dict(st.get("panel", {}))
        self.lang_var = tk.StringVar(value=get_language())
        self.cal_vars = {k: tk.StringVar(value="") for k in CAL_KINDS}
        self.output_var = tk.StringVar(value="")
        self.rejection_var = tk.StringVar(value=p.get("rejection", "winsorized"))
        self.low_var = tk.DoubleVar(value=p.get("low", 4.0))
        self.high_var = tk.DoubleVar(value=p.get("high", 3.0))
        self.weights_var = tk.BooleanVar(value=p.get("weights", True))
        self.pick_boost_var = tk.BooleanVar(value=p.get("pick_boost", False))
        self.split_nights_var = tk.BooleanVar(value=p.get("split_nights", True))
        self.scale_var = tk.StringVar(value=p.get("scale", "1"))
        self.drizzle_var = tk.BooleanVar(value=p.get("drizzle", False))
        self.pixfrac_var = tk.DoubleVar(value=p.get("pixfrac", 0.9))
        self.fill_holes_var = tk.BooleanVar(value=p.get("fill_holes", True))
        self.crop_var = tk.BooleanVar(value=p.get("crop", True))
        self.bits_var = tk.StringVar(value=p.get("bits", "32"))
        self.gpu_var = tk.BooleanVar(value=p.get("gpu", True))
        self.workers_var = tk.IntVar(value=p.get("workers", backend.default_workers()))
        self.memory_var = tk.IntVar(value=p.get("memory_gb", 2))
        self.preview_n_var = tk.IntVar(value=p.get("preview_n", PREVIEW_FRAMES))
        self.view_var = tk.StringVar(value="drizzle")
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()

        self._scale = root.winfo_fpixels("1i") / 96.0
        self.fonts = Fonts(root)
        setup_style(root, self.px, self.fonts)
        # 這個 View 專用的事件標籤：加在自己底下每個元件上，不用 bind_all，
        # 跟別的畫面放在同一個視窗時（整合版的分頁）不會互相搶事件
        self._tag = f"ApuPhotonsView{id(self)}"
        self.bind_class(self._tag, "<Button-1>", self._maybe_close_popover, add="+")
        self.bind_class(self._tag, "<Escape>", lambda _e: self.close_popover())
        self.bind_class(self._tag, "<MouseWheel>", self._scroll_panel, add="+")
        self._build()

        for var in (self.rejection_var, self.low_var, self.high_var, self.weights_var, self.pick_boost_var,
                    self.split_nights_var, self.scale_var, self.drizzle_var, self.pixfrac_var, self.fill_holes_var,
                    self.crop_var, self.bits_var, self.gpu_var, self.workers_var, self.memory_var,
                    self.preview_n_var):
            var.trace_add("write", lambda *_: self._params_changed())
        self.view_var.trace_add("write", lambda *_: self._show_preview())
        self.lang_var.trace_add("write", lambda *_: self.after_idle(self._language_clicked))
        threading.Thread(target=self._detect_gpu, daemon=True).start()
        for d in light_dirs or []:
            self.add_light_folder(Path(d))
        self._refresh_all()
        self._poll_job: str | None = self.after(100, self._poll)

    def px(self, v: float) -> int:
        return int(round(v * self._scale))

    def last_dir(self) -> str:
        return str(self.light_dirs[-1].parent) if self.light_dirs else load_settings().get("last_dir", "")

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
        self.add_btn = ttk.Button(actions, text=tr("gui.btn.add_light"), style="Dark.TButton",
                                  command=self.ask_add_light)
        self.add_btn.pack(side="left")
        Tooltip(self.add_btn, tr("gui.btn.add_light.help", shortcut=OPEN_SHORTCUT), self)
        self.trial_btn = ttk.Button(actions, text=tr("gui.btn.trial"), style="Dark.TButton",
                                    command=lambda: self._start(trial=True))
        self.trial_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.trial_btn, tr("gui.btn.trial.help"), self)
        self.stack_btn = ttk.Button(actions, style="Prominent.TButton", command=self._stack_or_stop)
        self.stack_btn.pack(side="left", padx=(self.px(6), 0))
        Tooltip(self.stack_btn, tr("gui.btn.stack.help"), self)

        # 中間：專案名稱（輸出檔名），對應 APU Astro 的文件名稱
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
                        for key in ("frames", "integrated", "rejected", "reference", "fwhm", "holes", "elapsed")}

        group = PanelGroup(panel, self, "calibration", tr("gui.group.calibration"), tr("gui.group.calibration.info"))
        for i, kind in enumerate(CAL_KINDS):
            FolderRow(group.body, self, kind, self.cal_vars[kind]).pack(fill="x", pady=(0 if i == 0 else self.px(8), 0))

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
        ParameterToggle(group.body, self, tr("gui.toggle.split_nights"), self.split_nights_var).pack(
            fill="x", pady=self.px(2))

        group = PanelGroup(panel, self, "output", tr("gui.group.output"), tr("gui.group.output.info"))
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
        row = tk.Frame(group.body, bg=D.panel)
        row.pack(fill="x", pady=(self.px(6), 0))
        tk.Label(row, text=tr("gui.output.bits"), font=self.fonts.ui, fg=D.label, bg=D.panel).pack(side="left")
        Segmented(row, self, [("32", "32-bit"), ("16", "16-bit")], self.bits_var).pack(side="right")
        tk.Label(group.body, text=tr("gui.output.file"), font=self.fonts.small, fg=D.secondary,
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

    def _build_canvas_area(self, body: tk.Frame) -> None:
        D = Darkroom
        area = tk.Frame(body, bg=D.canvas)
        area.pack(side="left", fill="both", expand=True)
        nb = ttk.Notebook(area, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)
        self.notebook = nb

        # ---- Light 分頁：資料夾清單 + 每張的結果
        light = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(light, text=tr("gui.tab.light"))
        top = tk.Frame(light, bg=D.chrome)
        top.pack(fill="x")
        tk.Label(top, text=tr("gui.light.folders"), font=self.fonts.bold, fg=D.label, bg=D.chrome).pack(
            side="left", padx=self.px(12), pady=self.px(6))
        self.remove_btn = ttk.Button(top, text=tr("gui.btn.remove"), style="Dark.TButton", command=self._remove_light)
        self.remove_btn.pack(side="right", padx=(0, self.px(12)))
        ttk.Button(top, text=tr("gui.btn.add"), style="Dark.TButton", command=self.ask_add_light).pack(
            side="right", padx=(0, self.px(6)))
        self.dir_tree = ttk.Treeview(light, columns=("path", "n"), show="headings", height=4, style="Dark.Treeview",
                                     selectmode="extended")
        self.dir_tree.heading("path", text=tr("gui.col.folder"), anchor="w")
        self.dir_tree.heading("n", text=tr("gui.col.count"), anchor="e")
        self.dir_tree.column("path", width=self.px(600), anchor="w")
        self.dir_tree.column("n", width=self.px(80), anchor="e", stretch=False)
        self.dir_tree.pack(fill="x")
        tk.Frame(light, bg=D.separator, height=1).pack(fill="x")

        table = tk.Frame(light, bg=D.canvas)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=[c[0] for c in COLUMNS], show="headings", style="Dark.Treeview")
        for key, label, width, anchor in COLUMNS:
            self.tree.heading(key, text=tr(label), anchor=anchor)
            self.tree.column(key, width=self.px(width), anchor=anchor, stretch=key in ("file", "reason"))
        self.tree.tag_configure("rejected", foreground=D.reject)
        self.tree.tag_configure("pending", foreground=D.secondary)
        ys = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview, style="Dark.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        # 提示放在清單上面，底色要跟清單一樣
        self.empty = tk.Frame(table, bg=D.list_bg)
        tk.Label(self.empty, text=APP_NAME, font=self.fonts.hero, fg=D.label, bg=D.list_bg).pack()
        tk.Label(self.empty, text=tr("gui.empty.hint", shortcut=OPEN_SHORTCUT), font=self.fonts.hero_sub,
                 fg=D.secondary, bg=D.list_bg, justify="center").pack(pady=(self.px(8), 0))

        # ---- 結果分頁：自動拉伸的預覽 + QC 摘要
        result = ttk.Frame(nb, style="Canvas.TFrame")
        nb.add(result, text=tr("gui.tab.result"))
        head = tk.Frame(result, bg=D.chrome)
        head.pack(fill="x")
        self.view_seg_holder = tk.Frame(head, bg=D.chrome)
        self.view_seg_holder.pack(side="left", padx=self.px(12), pady=self.px(6))
        tk.Label(head, text=tr("gui.result.stretch_note"), font=self.fonts.small, fg=D.secondary,
                 bg=D.chrome).pack(side="right", padx=self.px(12))
        self.image_canvas = tk.Canvas(result, bg=D.canvas, highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True)
        self.image_canvas.bind("<Configure>", lambda _e: self._show_preview())
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
        """照目前語言重建介面；資料夾、設定、結果都保留。"""
        if self.lang_var.get() != get_language():
            self.lang_var.set(get_language())
        self.close_popover()
        for child in self.winfo_children():
            child.destroy()
        self._build()
        self._refresh_all()
        self.update_idletasks()
        self._relayout_panel()

    # ------------------------------------------------------------------ 資料夾

    def add_light_folder(self, folder: str | Path) -> None:
        """加入一個 Light 資料夾（整合版也呼叫這個）。"""
        folder = Path(folder)
        if not folder.is_dir() or folder in self.light_dirs:
            return
        self.light_dirs.append(folder)
        save_settings(last_dir=str(folder.parent))
        if not self.output_var.get():
            self.output_var.set(str(folder.parent / "APU Photons" / f"{folder.parent.name or 'master'}.fits"))
        self.result = None
        self._refresh_all()

    def ask_add_light(self) -> None:
        """選擇資料夾後加入（選單「加入 Light 資料夾…」與 ⌘O／Ctrl+O 呼叫這個）。"""
        if self.is_busy():
            return
        path = filedialog.askdirectory(parent=self.root, title=tr("gui.btn.add_light"), initialdir=self.last_dir())
        if path:
            self.add_light_folder(path)

    def copy_selection(self) -> None:
        """選單「拷貝」：拷貝紀錄分頁選取的文字。"""
        self.log_text.event_generate("<<Copy>>")

    def select_all(self) -> None:
        """選單「全選」：選取紀錄分頁的全部文字。"""
        self.log_text.event_generate("<<SelectAll>>")

    def _remove_light(self) -> None:
        if self.is_busy():
            return
        chosen = {self.dir_tree.index(i) for i in self.dir_tree.selection()}
        self.light_dirs = [d for i, d in enumerate(self.light_dirs) if i not in chosen]
        self.result = None
        self._refresh_all()

    def _choose_output(self) -> None:
        current = Path(self.output_var.get()) if self.output_var.get() else None
        path = filedialog.asksaveasfilename(
            parent=self.root, title=tr("gui.output.file"), defaultextension=".fits",
            filetypes=[("FITS", "*.fits *.fit")], initialdir=str(current.parent) if current else self.last_dir(),
            initialfile=current.name if current else "master.fits")
        if path:
            self.output_var.set(path)
            self._refresh_all()

    def _reveal_output(self) -> None:
        path = Path(self.output_var.get())
        folder = path.parent if path.parent.is_dir() else None
        if folder is None:
            return
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606
        elif IS_MAC:
            import subprocess
            subprocess.run(["open", str(folder)], check=False)

    # ------------------------------------------------------------------ 設定 → 引擎

    def settings(self, trial: bool = False) -> Settings:
        scale = self.scale_var.get()
        drizzle = 2 if scale == "2" else (1 if scale == "1" and self.drizzle_var.get() else 0)
        return Settings(
            rejection=self.rejection_var.get(), low=round(float(self.low_var.get()), 2),
            high=round(float(self.high_var.get()), 2),
            weighting="snr2_over_fwhm2" if self.weights_var.get() else "none",
            pick_boost=bool(self.pick_boost_var.get()), downsample=0.5 if scale == "0.5" else 1.0,
            crop_common=bool(self.crop_var.get()), output_bits=int(self.bits_var.get()),
            memory_mb=int(self.memory_var.get()) * 1024, split_nights=bool(self.split_nights_var.get()),
            preview=int(self.preview_n_var.get()) if trial else 0, drizzle=drizzle,
            pixfrac=round(float(self.pixfrac_var.get()), 2), fill_holes=bool(self.fill_holes_var.get()),
            gpu="auto" if self.gpu_var.get() else "cpu", workers=int(self.workers_var.get()))

    def calibration(self) -> Calibration:
        files = {k: list_images(Path(v.get())) if v.get() and Path(v.get()).is_dir() else []
                 for k, v in self.cal_vars.items()}
        return Calibration(bias=files["bias"], dark=files["dark"], flat=files["flat"], flat_dark=files["flat_dark"])

    def output_path(self, trial: bool = False) -> Path:
        out = Path(self.output_var.get())
        return out.with_name(out.stem + "_trial" + out.suffix) if trial else out

    def _params_changed(self) -> None:
        self._update_output_rules()
        save_settings(params={
            "rejection": self.rejection_var.get(), "low": float(self.low_var.get()),
            "high": float(self.high_var.get()), "weights": bool(self.weights_var.get()),
            "pick_boost": bool(self.pick_boost_var.get()), "split_nights": bool(self.split_nights_var.get()),
            "scale": self.scale_var.get(), "drizzle": bool(self.drizzle_var.get()),
            "pixfrac": float(self.pixfrac_var.get()), "fill_holes": bool(self.fill_holes_var.get()),
            "crop": bool(self.crop_var.get()), "bits": self.bits_var.get(), "gpu": bool(self.gpu_var.get()),
            "workers": int(self.workers_var.get()), "memory_gb": int(self.memory_var.get()),
            "preview_n": int(self.preview_n_var.get())})

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
        clip = self.rejection_var.get() in ("winsorized", "sigma")
        self.low_slider.set_enabled(clip)
        self.high_slider.set_enabled(clip)

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
        if self.is_busy() or not self.light_dirs:
            return
        if not self.output_var.get():
            self._choose_output()
            if not self.output_var.get():
                return
        self.cancel.clear()
        self.result = None
        self.previews = {}
        self.log_lines = []
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._started = time.time()
        self._stage = None
        args = (list(self.light_dirs), self.calibration(), self.output_path(trial), self.settings(trial))
        self.worker = threading.Thread(target=self._work, args=args, daemon=True)
        self.worker.start()
        self._status(lambda: tr("gui.status.starting"))
        self._refresh_all()

    def _work(self, light_dirs, cal, output, settings) -> None:
        """背景執行緒：跑引擎，把進度、紀錄、結果丟進佇列給主執行緒。"""
        put = self.events.put
        try:
            res = run(light_dirs, cal, output, settings, echo=lambda line: put(("log", line)),
                      progress=lambda stage, d, t: put(("progress", stage, d, t)), cancel=self.cancel)
            previews = {}
            outs = [("drizzle", output), ("stack", output.with_name(output.stem + "_stack" + output.suffix))] \
                if settings.drizzle else [("stack", output)]
            for key, path in outs:
                if path.is_file():
                    previews[key] = load_preview(path)
            put(("done", res, previews))
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

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "log":
            self.log_lines.append(event[1])
            if self.log_text.winfo_exists():
                self.log_text.configure(state="normal")
                self.log_text.insert("end", event[1] + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
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
            self.result, self.previews = event[1], event[2]
            self.view_var.set("drizzle" if "drizzle" in self.previews else "stack")
            elapsed = self._elapsed = time.time() - self._started
            self._status(lambda: tr("gui.status.done", seconds=elapsed))
            self.worker = None
            self._refresh_all()
            self.notebook.select(1)
        elif kind == "cancelled":
            self.worker = None
            self._status(lambda: tr("gui.status.cancelled"))
            self._refresh_all()
        elif kind == "error":
            self.worker = None
            self.log_lines.append(event[2])
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

    def _refresh_all(self) -> None:
        D = Darkroom
        busy = self.is_busy()
        self.status_var.set(self._status_fn())
        self._update_progress()
        self._update_output_rules()
        self._update_gpu_view()
        # 頂部列
        out = Path(self.output_var.get()) if self.output_var.get() else None
        if out is not None:
            self.doc_title.configure(text=out.stem, fg=D.label)
        else:
            self.doc_title.configure(text=tr("gui.no_project"), fg=D.secondary)
        self.stack_btn.configure(text=tr("gui.btn.stop") if busy else tr("gui.btn.stack"))
        ready = bool(self.light_dirs)
        self.stack_btn.state(["!disabled"] if (ready or busy) else ["disabled"])
        for btn in (self.trial_btn,):
            btn.state(["!disabled"] if ready and not busy else ["disabled"])
        self.add_btn.state(["disabled"] if busy else ["!disabled"])
        self.output_btn.state(["disabled"] if busy else ["!disabled"])
        self.reveal_btn.state(["!disabled"] if out is not None and out.parent.is_dir() else ["disabled"])
        self.output_label.configure(text=_short_path(str(out), 44) if out else tr("gui.output.unset"))
        # Light 資料夾
        self.dir_tree.delete(*self.dir_tree.get_children())
        total = 0
        for d in self.light_dirs:
            try:
                n = len(list_images(d))
            except OSError:
                n = 0
            total += n
            self.dir_tree.insert("", "end", values=(str(d), n))
        self.remove_btn.state(["disabled"] if busy or not self.light_dirs else ["!disabled"])
        self._fill_table()
        self._fill_metrics()
        self._fill_result()
        self.summary_var.set(tr("gui.summary", dirs=len(self.light_dirs), frames=total) if self.light_dirs else "")

    def _fill_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if not self.light_dirs:
            self.empty.place(relx=0.5, rely=0.42, anchor="center")
            return
        self.empty.place_forget()
        if self.result is None:
            for d in self.light_dirs:
                for p in list_images(d):
                    self.tree.insert("", "end", values=(p.name, "", "—", "", "", "", "", ""), tags=("pending",))
            return
        session_of = {id(f): s.id for s in self.result.project.sessions for f in s.frames}
        for f in self.result.project.frames:
            ok = f.accepted
            fwhm = f"{f.stars.fwhm:.2f}" if f.stars is not None and f.stars.fwhm == f.stars.fwhm else ""
            stars = str(len(f.stars)) if f.stars is not None else ""
            weight = f"{f.weight * len(self.result.project.accepted):.2f}" if ok and f.weight else ""
            rej = f"{f.rejected_pixel_fraction:.2%}" if ok and f.rejected_pixel_fraction is not None else ""
            reason = tr(f"reason.{f.reject_reason}") if f.reject_reason else ""
            if f is self.result.project.reference:
                reason = tr("gui.reference")
            skipped = f.reject_reason == "not_in_preview"  # 試跑沒抽到：不是問題，用灰色
            result = tr("gui.result.ok") if ok else ("—" if skipped else tr("gui.result.rejected"))
            self.tree.insert("", "end", values=(
                f.name, session_of.get(id(f), ""), result, fwhm, stars, weight, rej, reason),
                tags=() if ok else (("pending",) if skipped else ("rejected",)))

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
        m["reference"].set(_short_path(self.result.project.reference.name, 24))
        m["fwhm"].set(f"{q['fwhm_px']['median']:.2f} px" if q.get("fwhm_px") else "—")
        holes = q.get("drizzle", {}).get("holes_fraction")
        m["holes"].set(" / ".join(f"{v:.1%}" for v in holes) if holes else "—")
        m["elapsed"].set(tr("gui.value.seconds", v=self._elapsed) if self._elapsed else "—")

    def _fill_result(self) -> None:
        for child in self.view_seg_holder.winfo_children():
            child.destroy()
        if len(self.previews) > 1:
            Segmented(self.view_seg_holder, self, [("drizzle", "Drizzle"), ("stack", tr("gui.result.stack"))],
                      self.view_var).pack(side="left")
        self.qc_text.configure(state="normal")
        self.qc_text.delete("1.0", "end")
        if self.result is not None:
            q = self.result.qc
            lines = [tr("gui.qc.output", path=str(self.result.output))]
            if q.get("suspicious_frames"):
                lines.append(tr("gui.qc.suspicious", files="、".join(q["suspicious_frames"])))
            lines.append(tr("gui.qc.drizzle_ok") if q.get("drizzle_suitable") else tr("gui.qc.drizzle_no"))
            lines += [f"⚠ {w}" for w in self.result.warnings]
            self.qc_text.insert("end", "\n".join(lines))
        self.qc_text.configure(state="disabled")
        self._show_preview()

    def _show_preview(self) -> None:
        c = self.image_canvas
        if not c.winfo_exists():
            return
        c.delete("all")
        img = self.previews.get(self.view_var.get()) or next(iter(self.previews.values()), None)
        w, h = c.winfo_width(), c.winfo_height()
        if img is None:
            c.create_text(w / 2, h / 2, text=tr("gui.result.empty"), fill=Darkroom.secondary, font=self.fonts.hero_sub)
            return
        if w < 10 or h < 10:
            return
        shown = img.copy()
        shown.thumbnail((w - self.px(16), h - self.px(16)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(shown, master=c)
        c.create_image(w / 2, h / 2, image=self._photo)

    # ------------------------------------------------------------------ 關閉

    def close(self) -> None:
        """取消背景工作、停掉排程、關掉說明氣泡。呼叫端接著 destroy() 這個 View；
        關掉之後請在主執行緒 gc.collect()，不然 View 的 Tk 變數可能在背景執行緒被循環回收釋放。
        疊圖的 worker 收到取消後，會在下一個檢查點結束。"""
        self.cancel.set()
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except (tk.TclError, ValueError):
                pass
            self._poll_job = None
        self.close_popover()


# ---------------------------------------------------------------------- 視窗


def _build_menubar(root: tk.Tk, view: PhotonsView) -> None:
    """Mac 的選單列（Windows 沿用原本的做法，不放選單列）。"""
    menubar = tk.Menu(root)
    app_menu = tk.Menu(menubar, name="apple", tearoff=False)
    app_menu.add_command(label=tr("gui.menu.about"), command=lambda: root.tk.call("::tk::mac::standardAboutPanel"))
    app_menu.add_separator()
    menubar.add_cascade(menu=app_menu)
    file_menu = tk.Menu(menubar, tearoff=False)
    file_menu.add_command(label=tr("gui.menu.add_light"), accelerator="Command-O", command=view.ask_add_light)
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
    width = min(int(round(1320 * scale)), root.winfo_screenwidth() - int(round(40 * scale)))
    height = min(int(round(900 * scale)), root.winfo_screenheight() - int(round(110 * scale)))
    root.geometry(f"{width}x{height}")
    root.minsize(int(round(1080 * scale)), int(round(720 * scale)))
    root.configure(bg=Darkroom.canvas)
    root.title(f"{APP_NAME} {__version__}")

    def change_language(lang: str) -> None:
        set_language(lang)
        save_settings(language=lang)
        view.rebuild()
        if IS_MAC:
            _build_menubar(root, view)  # 選單文字跟著換

    view = PhotonsView(root, root, light_dirs=[a for a in argv if not a.startswith("-")],
                       on_language=change_language)
    view.pack(fill="both", expand=True)
    if IS_MAC:
        _build_menubar(root, view)
    else:
        root.bind_all("<Control-o>", lambda _e: view.ask_add_light())

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
        # 把資料夾拖到 Dock 圖示上：當成 Light 資料夾加入
        root.createcommand("::tk::mac::OpenDocument", lambda *paths: [view.add_light_folder(p) for p in paths])
    dark_title_bar(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
