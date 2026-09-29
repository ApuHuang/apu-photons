"""視窗介面，與 APU Astro、APU Pick 同一套介面語言：暗房深色主題、頂部列、右側可收合參數面板、底部狀態列。

流程：加入 Light 資料夾 →（選擇 calibration）→ 試跑或疊圖 → 在「結果」分頁看成品。
啟動：python -m apu_photons.gui [Light 資料夾...]
介面文字都在 i18n.py，可以切換繁體中文 / English。

暗房元件（Darkroom、PanelGroup、ParameterSlider…）目前是從 APU Pick 複製過來的同一套，
兩邊穩定後再抽成共用套件（見 SPEC §5.4）。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk

from PIL import Image, ImageTk

from . import __version__, backend
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


class Darkroom:
    """暗房介面的色票與尺寸，跟 APU Astro 的 DarkroomTheme、APU Pick 同一套。

    固定深色、不跟系統設定：星空影像的背景亮度、色偏跟周圍環境有關，淺色視窗會拉走對比的判斷。
    """
    canvas = "#000000"
    chrome = "#1c1c1c"
    panel = "#252525"
    group_header = "#2f2f2f"
    control = "#3d3d3d"
    separator = "#454545"
    label = "#ebebeb"
    secondary = "#949494"
    accent = "#5cadff"
    beta = "#ffa842"
    prominent = "#0a84ff"
    segment_on = "#636366"
    hover = "#4a4a4a"
    list_bg = "#141414"
    reject = "#ff6961"
    top_bar_height = 48
    status_bar_height = 28
    panel_width = 320


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


def _trace(widget: tk.Misc, variable: tk.Variable, callback: Callable[[], None]) -> None:
    """變數改了就呼叫 callback；元件被刪掉時一起拿掉，切換語言重建介面時才不會呼叫到已刪除的元件。"""
    name = variable.trace_add("write", lambda *_: callback())

    def remove(event: tk.Event) -> None:
        if event.widget is widget:
            try:
                variable.trace_remove("write", name)
            except tk.TclError:
                pass

    widget.bind("<Destroy>", remove, add="+")


def _short_path(path: str, limit: int = 40) -> str:
    if len(path) <= limit:
        return path
    head = limit // 3
    return f"{path[:head]}…{path[-(limit - head - 1):]}"


class Fonts:
    def __init__(self, root: tk.Tk):
        families = set(font.families(root))
        ui = next((f for f in ("Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Helvetica Neue")
                   if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "Menlo", "DejaVu Sans Mono") if f in families), "TkFixedFont")
        self.title = font.Font(root, family=ui, size=11, weight="bold")
        self.badge = font.Font(root, family=ui, size=7, weight="bold")
        self.ui = font.Font(root, family=ui, size=9)
        self.small = font.Font(root, family=ui, size=9)
        self.bold = font.Font(root, family=ui, size=9, weight="bold")
        self.mono = font.Font(root, family=mono, size=9)
        self.hero = font.Font(root, family=ui, size=24, weight="bold")
        self.hero_sub = font.Font(root, family=ui, size=11)
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            font.nametofont(name).configure(family=ui, size=9)


# ---------------------------------------------------------------------- 元件（與 APU Pick 相同）


class Tooltip:
    """滑鼠停一下才出現的說明，對應 APU Astro 按鈕的 .help。"""

    def __init__(self, widget: tk.Widget, text: str, app: App):
        self.widget, self.text, self.app = widget, text, app
        self._job: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: object = None) -> None:
        self._job = self.widget.after(600, self._show)

    def _show(self) -> None:
        if not self.widget.winfo_exists():
            return
        D = Darkroom
        self._tip = tip = tk.Toplevel(self.widget)
        tip.overrideredirect(True)
        tip.configure(bg=D.separator)
        tk.Label(tip, text=self.text, font=self.app.fonts.small, fg=D.label, bg=D.group_header,
                 justify="left", wraplength=self.app.px(300), padx=self.app.px(8), pady=self.app.px(4)
                 ).pack(padx=1, pady=1)
        tip.update_idletasks()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() - tip.winfo_reqwidth()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + self.app.px(4)
        tip.geometry(f"+{max(x, 0)}+{y}")

    def _hide(self, _event: object = None) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class Switch(tk.Canvas):
    """開關：對應 APU Astro 的 ParameterToggle。停用時畫成暗色、點了沒反應。"""

    def __init__(self, master: tk.Widget, app: App, variable: tk.BooleanVar, bg: str):
        self.w, self.h = app.px(30), app.px(17)
        super().__init__(master, width=self.w, height=self.h, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.var = variable
        self.enabled = True
        self.bind("<Button-1>", lambda _e: self.enabled and variable.set(not variable.get()))
        _trace(self, variable, self._draw)
        self._draw()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        on, w, h = bool(self.var.get()), self.w, self.h
        fill = (Darkroom.prominent if on else Darkroom.control) if self.enabled else (
            "#1d3550" if on else Darkroom.group_header)
        self.create_oval(0, 0, h - 1, h - 1, fill=fill, outline=fill)
        self.create_oval(w - h, 0, w - 1, h - 1, fill=fill, outline=fill)
        self.create_rectangle(h / 2, 0, w - h / 2, h - 1, fill=fill, outline=fill)
        pad = max(2, round(h * 0.12))
        d = h - 2 * pad - 1
        x = w - pad - d - 1 if on else pad
        knob = "white" if self.enabled else "#8a8a8a"
        self.create_oval(x, pad, x + d, pad + d, fill=knob, outline=knob)


class Segmented(tk.Frame):
    """分段切換：對應 APU Astro 頂部列的「繁中｜EN」。"""

    def __init__(self, master: tk.Widget, app: App, options: list[tuple[str, str]], variable: tk.StringVar,
                 stretch: bool = False):
        super().__init__(master, bg=Darkroom.control, padx=2, pady=2)
        self.var = variable
        self.labels: dict[str, tk.Label] = {}
        for i, (value, text) in enumerate(options):
            lb = tk.Label(self, text=text, font=app.fonts.small, cursor="hand2",
                          padx=app.px(10), pady=app.px(1))
            if stretch:
                lb.grid(row=0, column=i, sticky="ew", padx=1)
                self.columnconfigure(i, weight=1, uniform="seg")
            else:
                lb.pack(side="left", padx=1)
            lb.bind("<Button-1>", lambda _e, v=value: variable.set(v))
            self.labels[value] = lb
        _trace(self, variable, self._draw)
        self._draw()

    def _draw(self) -> None:
        current = self.var.get()
        for value, lb in self.labels.items():
            if value == current:
                lb.configure(bg=Darkroom.segment_on, fg="white")
            else:
                lb.configure(bg=Darkroom.control, fg=Darkroom.label)


class PanelGroup(tk.Frame):
    """右側面板裡可收合的一組，對應 APU Astro 的 PanelGroup。收合狀態會記住。"""

    def __init__(self, master: tk.Widget, app: App, key: str, title: str, info: str | None = None):
        D = Darkroom
        super().__init__(master, bg=D.panel)
        self.app, self.key = app, key
        header = tk.Frame(self, bg=D.group_header, height=app.px(34), cursor="hand2")
        header.pack(fill="x")
        header.pack_propagate(False)
        self.chevron = tk.Label(header, font=app.fonts.small, fg=D.secondary, bg=D.group_header, width=2)
        self.chevron.pack(side="left", padx=(app.px(8), 0))
        title_label = tk.Label(header, text=title, font=app.fonts.bold, fg=D.label, bg=D.group_header)
        title_label.pack(side="left")
        if info:
            InfoButton(header, app, info).pack(side="right", padx=app.px(10))
        for w in (header, self.chevron, title_label):
            w.bind("<Button-1>", self.toggle)
        self.body = tk.Frame(self, bg=D.panel, padx=app.px(14), pady=app.px(10))
        self.sep = tk.Frame(self, bg=D.separator, height=1)
        self.sep.pack(fill="x", side="bottom")
        self.expanded = app.panel_state.get(key, True)
        self._layout()
        self.pack(fill="x")

    def _layout(self) -> None:
        self.chevron.configure(text="▾" if self.expanded else "▸")
        if self.expanded:
            self.body.pack(fill="x", before=self.sep)
        else:
            self.body.pack_forget()

    def toggle(self, _event: object = None) -> None:
        self.expanded = not self.expanded
        self._layout()
        self.app.panel_state[self.key] = self.expanded
        save_settings(panel=self.app.panel_state)


class InfoButton(tk.Label):
    """分組標題上的 ⓘ，點了才顯示說明，對應 APU Astro 的 InfoButton。"""

    def __init__(self, master: tk.Widget, app: App, text: str):
        super().__init__(master, text="ⓘ", font=app.fonts.ui, fg=Darkroom.secondary, bg=Darkroom.group_header,
                         cursor="hand2")
        self.app, self.text = app, text
        self.bind("<Button-1>", self._toggle)
        Tooltip(self, tr("gui.details"), app)

    def _toggle(self, _event: object = None) -> str:
        if self.app.popover_owner is self:
            self.app.close_popover()
        else:
            self.app.show_popover(self, self.text)
        return "break"  # 不要連帶收合分組


class ParameterSlider(tk.Frame):
    """滑桿列：標題、目前的值（等寬數字）、滑桿，對應 APU Astro 的 ParameterSlider。"""

    def __init__(self, master: tk.Widget, app: App, title: str, variable: tk.Variable, lo: float, hi: float,
                 fmt: Callable[[float], str], step: float = 1.0):
        D = Darkroom
        super().__init__(master, bg=D.panel)
        self.var, self.fmt, self.step = variable, fmt, step
        head = tk.Frame(self, bg=D.panel)
        head.pack(fill="x")
        self.title = tk.Label(head, text=title, font=app.fonts.small, fg=D.secondary, bg=D.panel)
        self.title.pack(side="left")
        self.value = tk.Label(head, font=app.fonts.mono, fg=D.label, bg=D.panel)
        self.value.pack(side="right")
        self.scale = ttk.Scale(self, from_=lo, to=hi, variable=variable, style="Dark.Horizontal.TScale",
                               command=self._moved)
        self.scale.pack(fill="x", pady=(app.px(3), 0))
        _trace(self, variable, self._refresh)
        self._refresh()

    def set_enabled(self, enabled: bool) -> None:
        self.scale.state(["!disabled"] if enabled else ["disabled"])
        self.value.configure(fg=Darkroom.label if enabled else "#6b6b6b")

    def _moved(self, value: str) -> None:
        snapped = round(float(value) / self.step) * self.step
        if abs(snapped - float(value)) > 1e-9:
            self.var.set(snapped)

    def _refresh(self) -> None:
        try:
            self.value.configure(text=self.fmt(float(self.var.get())))
        except (tk.TclError, ValueError):
            self.value.configure(text="—")


class ParameterToggle(tk.Frame):
    """開關列：標題填滿寬度，開關在右邊。"""

    def __init__(self, master: tk.Widget, app: App, title: str, variable: tk.BooleanVar, bg: str = Darkroom.panel):
        super().__init__(master, bg=bg)
        self.label = tk.Label(self, text=title, font=app.fonts.ui, fg=Darkroom.label, bg=bg, cursor="hand2")
        self.label.pack(side="left")
        self.switch = Switch(self, app, variable, bg)
        self.switch.pack(side="right")
        self.label.bind("<Button-1>", lambda _e: self.switch.enabled and variable.set(not variable.get()))

    def set_enabled(self, enabled: bool) -> None:
        self.switch.set_enabled(enabled)
        self.label.configure(fg=Darkroom.label if enabled else "#6b6b6b")


class MetricRow(tk.Frame):
    """唯讀的數值列，對應 APU Astro 的 MetricRow。"""

    def __init__(self, master: tk.Widget, app: App, title: str):
        D = Darkroom
        super().__init__(master, bg=D.panel)
        tk.Label(self, text=title, font=app.fonts.small, fg=D.secondary, bg=D.panel).pack(side="left")
        self.value = tk.Label(self, text="—", font=app.fonts.mono, fg=D.label, bg=D.panel)
        self.value.pack(side="right")
        self.pack(fill="x", pady=app.px(1))

    def set(self, text: str) -> None:
        self.value.configure(text=text)


class FolderRow(tk.Frame):
    """calibration 資料夾一列：名稱、張數或「未使用」、選擇 / 清除。"""

    def __init__(self, master: tk.Widget, app: App, kind: str, variable: tk.StringVar):
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


class App:
    def __init__(self, root: tk.Tk, light_dirs: list[str] | None = None):
        self.root = root
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
        root.title(f"{APP_NAME} {__version__}")
        width = min(self.px(1320), root.winfo_screenwidth() - self.px(40))
        height = min(self.px(900), root.winfo_screenheight() - self.px(110))
        root.geometry(f"{width}x{height}")
        root.minsize(self.px(1080), self.px(720))
        root.configure(bg=Darkroom.canvas)
        self._setup_style()
        self._build()

        for var in (self.rejection_var, self.low_var, self.high_var, self.weights_var, self.pick_boost_var,
                    self.split_nights_var, self.scale_var, self.drizzle_var, self.pixfrac_var, self.fill_holes_var,
                    self.crop_var, self.bits_var, self.gpu_var, self.workers_var, self.memory_var,
                    self.preview_n_var):
            var.trace_add("write", lambda *_: self._params_changed())
        self.view_var.trace_add("write", lambda *_: self._show_preview())
        self.lang_var.trace_add("write", lambda *_: root.after_idle(self._change_language))
        if not IS_MAC:
            root.bind_all("<Control-o>", lambda _e: self._add_light())
        root.bind_all("<Button-1>", self._maybe_close_popover, add="+")
        root.bind_all("<Escape>", lambda _e: self.close_popover())
        root.bind_all("<MouseWheel>", self._scroll_panel, add="+")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if IS_MAC:
            root.createcommand("::tk::mac::Quit", self._on_close)
        threading.Thread(target=self._detect_gpu, daemon=True).start()
        for d in light_dirs or []:
            self.add_light_dir(Path(d))
        self._refresh_all()
        self._poll_job: str | None = root.after(100, self._poll)

    def px(self, v: float) -> int:
        return int(round(v * self._scale))

    def last_dir(self) -> str:
        return str(self.light_dirs[-1].parent) if self.light_dirs else load_settings().get("last_dir", "")

    # ------------------------------------------------------------------ 樣式與版面

    def _setup_style(self) -> None:
        D = Darkroom
        style = ttk.Style(self.root)
        style.theme_use("clam")
        flat = {"bordercolor": D.separator, "focuscolor": D.control}
        style.configure("Dark.TButton", background=D.control, foreground=D.label, lightcolor=D.control,
                        darkcolor=D.control, padding=(self.px(10), self.px(2)), font=self.fonts.ui, **flat)
        style.map("Dark.TButton",
                  background=[("disabled", D.group_header), ("pressed", D.segment_on), ("active", D.hover)],
                  lightcolor=[("active", D.hover)], darkcolor=[("active", D.hover)],
                  foreground=[("disabled", "#6b6b6b")])
        style.configure("Prominent.TButton", background=D.prominent, foreground="white", lightcolor=D.prominent,
                        darkcolor=D.prominent, bordercolor=D.prominent, focuscolor=D.prominent,
                        padding=(self.px(10), self.px(2)), font=self.fonts.ui)
        style.map("Prominent.TButton",
                  background=[("disabled", "#1d3550"), ("pressed", "#0064d2"), ("active", "#2a95ff")],
                  lightcolor=[("disabled", "#1d3550"), ("active", "#2a95ff")],
                  darkcolor=[("disabled", "#1d3550"), ("active", "#2a95ff")],
                  bordercolor=[("disabled", "#1d3550")],
                  foreground=[("disabled", "#7f93a8")])
        style.layout("Dark.Horizontal.TScale", [("Horizontal.Scale.trough", {"sticky": "nswe", "children": [
            ("Horizontal.Scale.slider", {"side": "left", "sticky": ""})]})])
        style.configure("Dark.Horizontal.TScale", background=D.label, troughcolor=D.control, bordercolor=D.control,
                        lightcolor=D.control, darkcolor=D.control, gripcount=0)
        style.map("Dark.Horizontal.TScale", background=[("disabled", "#6b6b6b"), ("active", "white")])
        style.configure("Dark.Horizontal.TProgressbar", background=D.accent, troughcolor=D.control,
                        bordercolor=D.chrome, lightcolor=D.accent, darkcolor=D.accent)
        style.configure("Dark.TNotebook", background=D.canvas, borderwidth=0, tabmargins=(0, 0, 0, 0),
                        bordercolor=D.canvas, lightcolor=D.canvas, darkcolor=D.canvas)
        style.configure("Dark.TNotebook.Tab", background=D.chrome, foreground=D.secondary, bordercolor=D.separator,
                        lightcolor=D.chrome, darkcolor=D.chrome, padding=(self.px(14), self.px(4)),
                        font=self.fonts.ui)
        style.map("Dark.TNotebook.Tab", background=[("selected", D.group_header)],
                  foreground=[("selected", D.label)], lightcolor=[("selected", D.group_header)])
        style.configure("Canvas.TFrame", background=D.canvas)
        style.configure("Dark.Treeview", background=D.list_bg, fieldbackground=D.list_bg, foreground=D.label,
                        bordercolor=D.chrome, lightcolor=D.list_bg, darkcolor=D.list_bg,
                        rowheight=self.px(22), font=self.fonts.ui)
        style.map("Dark.Treeview", background=[("selected", "#1f3d63")], foreground=[("selected", D.label)])
        style.configure("Dark.Treeview.Heading", background=D.group_header, foreground=D.label,
                        bordercolor=D.separator, lightcolor=D.group_header, darkcolor=D.group_header,
                        relief="flat", font=self.fonts.bold)
        style.map("Dark.Treeview.Heading", background=[("active", D.control)])
        for orient in ("Vertical", "Horizontal"):
            style.configure(f"Dark.{orient}.TScrollbar", background=D.control, troughcolor=D.chrome,
                            bordercolor=D.chrome, arrowcolor=D.secondary, lightcolor=D.control, darkcolor=D.control)
            style.map(f"Dark.{orient}.TScrollbar", background=[("active", D.hover)])

    def _build(self) -> None:
        if IS_MAC:
            self._build_menubar()
        self._build_top_bar()
        self._build_status_bar()
        body = tk.Frame(self.root, bg=Darkroom.canvas)
        body.pack(fill="both", expand=True)
        self._build_panel(body)
        tk.Frame(body, bg=Darkroom.separator, width=1).pack(side="right", fill="y")
        self._build_canvas_area(body)

    def _build_menubar(self) -> None:
        root = self.root
        menubar = tk.Menu(root)
        app_menu = tk.Menu(menubar, name="apple", tearoff=False)
        app_menu.add_command(label=tr("gui.menu.about"),
                             command=lambda: root.tk.call("::tk::mac::standardAboutPanel"))
        app_menu.add_separator()
        menubar.add_cascade(menu=app_menu)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label=tr("gui.menu.add_light"), accelerator="Command-O", command=self._add_light)
        menubar.add_cascade(label=tr("gui.menu.file"), menu=file_menu)
        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label=tr("gui.menu.copy"), accelerator="Command-C",
                              command=lambda: self.log_text.event_generate("<<Copy>>"))
        edit_menu.add_command(label=tr("gui.menu.select_all"), accelerator="Command-A",
                              command=lambda: self.log_text.event_generate("<<SelectAll>>"))
        menubar.add_cascade(label=tr("gui.menu.edit"), menu=edit_menu)
        menubar.add_cascade(label=tr("gui.menu.window"), menu=tk.Menu(menubar, name="window"))
        root.configure(menu=menubar)

    def _build_top_bar(self) -> None:
        D = Darkroom
        bar = tk.Frame(self.root, bg=D.chrome, height=self.px(D.top_bar_height))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(self.root, bg=D.separator, height=1).pack(fill="x")

        identity = tk.Frame(bar, bg=D.chrome)
        identity.pack(side="left", padx=(self.px(14), 0))
        tk.Label(identity, text=APP_NAME, font=self.fonts.title, fg=D.label, bg=D.chrome).pack(side="left")
        badge = tk.Frame(identity, bg=D.beta, padx=1, pady=1)
        badge.pack(side="left", padx=(self.px(7), 0))
        tk.Label(badge, text=f"v{__version__}", font=self.fonts.badge, fg=D.beta, bg=D.chrome,
                 padx=self.px(4)).pack()

        actions = tk.Frame(bar, bg=D.chrome)
        actions.pack(side="right", padx=(0, self.px(14)))
        Segmented(actions, self, [("zh", "繁中"), ("en", "EN")], self.lang_var).pack(side="left")
        tk.Frame(actions, bg=D.separator, width=1, height=self.px(18)).pack(side="left", padx=self.px(10))
        self.add_btn = ttk.Button(actions, text=tr("gui.btn.add_light"), style="Dark.TButton", command=self._add_light)
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
        bar = tk.Frame(self.root, bg=D.chrome, height=self.px(D.status_bar_height))
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        tk.Frame(self.root, bg=D.separator, height=1).pack(side="bottom", fill="x")
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
        ttk.Button(top, text=tr("gui.btn.add"), style="Dark.TButton", command=self._add_light).pack(
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
        delta = -1 if event.delta > 0 else 1
        self.panel_canvas.yview_scroll(delta * (1 if IS_MAC else 3), "units")

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

    def _change_language(self) -> None:
        lang = self.lang_var.get()
        if lang not in LANGUAGES or lang == get_language():
            return
        if self._busy():  # 執行中不換：重建介面會打斷進度顯示
            self.lang_var.set(get_language())
            return
        set_language(lang)
        save_settings(language=lang)
        self._rebuild()

    def _rebuild(self) -> None:
        """換語言：整個重建介面，資料夾、設定、結果都保留。"""
        self.close_popover()
        for child in self.root.winfo_children():
            child.destroy()
        self._build()
        self._refresh_all()
        self.root.update_idletasks()
        self._relayout_panel()

    # ------------------------------------------------------------------ 資料夾

    def add_light_dir(self, folder: Path) -> None:
        folder = Path(folder)
        if not folder.is_dir() or folder in self.light_dirs:
            return
        self.light_dirs.append(folder)
        save_settings(last_dir=str(folder.parent))
        if not self.output_var.get():
            self.output_var.set(str(folder.parent / "APU Photons" / f"{folder.parent.name or 'master'}.fits"))
        self.result = None
        self._refresh_all()

    def _add_light(self) -> None:
        if self._busy():
            return
        path = filedialog.askdirectory(parent=self.root, title=tr("gui.btn.add_light"), initialdir=self.last_dir())
        if path:
            self.add_light_dir(Path(path))

    def _remove_light(self) -> None:
        if self._busy():
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
            self.gpu_toggle.set_enabled(False)

    # ------------------------------------------------------------------ 執行

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _stack_or_stop(self) -> None:
        if self._busy():
            self.cancel.set()
            self._status(lambda: tr("gui.status.stopping"))
        else:
            self._start(trial=False)

    def _start(self, trial: bool) -> None:
        if self._busy() or not self.light_dirs:
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
        self._poll_job = self.root.after(100, self._poll)

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
        if self._busy():
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
        busy = self._busy()
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

    def _on_close(self) -> None:
        if self._busy():
            if not messagebox.askyesno(APP_NAME, tr("gui.close.confirm"), parent=self.root):
                return
            self.cancel.set()
        self.close()
        self.root.destroy()

    def close(self) -> None:
        """停掉計時器、解除全域綁定、清掉畫面；之後 root 可以直接關掉，或拿來開新的 App。"""
        self.cancel.set()
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        self.close_popover()
        for sequence in ("<Control-o>", "<Button-1>", "<Escape>", "<MouseWheel>"):
            self.root.unbind_all(sequence)
        for child in self.root.winfo_children():
            child.destroy()


def _enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def _dark_title_bar(root: tk.Tk) -> None:
    """系統標題列也用深色（Windows 10 20H1 以後 / macOS），整個視窗才是一致的暗房。"""
    if sys.platform == "win32":
        try:
            import ctypes
            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            value = ctypes.c_int(1)
            for attribute in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value),
                                                              ctypes.sizeof(value)) == 0:
                    break
        except (AttributeError, OSError):
            pass
    elif IS_MAC:
        try:
            root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", root, "darkaqua")
        except tk.TclError:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    lang = load_settings().get("language")
    if lang in LANGUAGES:
        set_language(lang)
    _enable_dpi_awareness()
    root = tk.Tk()
    if IS_MAC:
        root.tk.call("tk", "scaling", 96 / 72)
    if ICON.is_file():
        try:
            root.iconbitmap(default=str(ICON))
        except tk.TclError:
            pass
    App(root, argv)
    _dark_title_bar(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
