"""結果預覽的放大檢視（SPEC §6.5）：滾輪縮放 25%～400%、拖曳平移、「適合視窗」與 100%。

只讀畫面上看得到的區域（FITS 用 memmap 開）；拉伸參數在載入時用整張圖算一次後固定，縮放、平移時亮度不跳。
換另一張（同一對齊組的其他濾鏡、或一般疊圖／Drizzle）時保持同一塊天空與同樣的天空放大倍率。
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk

from .darkroom import Darkroom
from .imageio import TOP_DOWN, row_order
from .preview import apply_stretch, stretch_params

ZOOMS = (0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 4.0)
MIN_ZOOM, MAX_ZOOM = 0.02, 4.0


class ZoomView(tk.Canvas):
    def __init__(self, master: tk.Misc, empty_text: str = "", font=None, on_zoom=None):
        super().__init__(master, bg=Darkroom.canvas, highlightthickness=0, cursor="fleur")
        self.empty_text, self.font, self.on_zoom = empty_text, font, on_zoom
        self.path: Path | None = None
        self._hdul = None
        self.data: np.ndarray | None = None    # (C, H, W) 或 (H, W)，已翻成由上往下
        self.params: list = []
        self.zoom = 1.0
        self.fit = True
        self.cx = self.cy = 0.0                # 畫面中心對到的影像座標（像素）
        self._photo: ImageTk.PhotoImage | None = None
        self._drag: tuple[int, int, float, float] | None = None
        self._pending = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._wheel(e, 120))   # Linux
        self.bind("<Button-5>", lambda e: self._wheel(e, -120))
        self.bind("<Double-Button-1>", lambda _e: self.zoom_fit())

    # ---------- 影像 ----------

    @property
    def size(self) -> tuple[int, int] | None:
        return None if self.data is None else (self.data.shape[-1], self.data.shape[-2])

    def set_image(self, path: Path | None, keep_view: bool = True) -> None:
        old_size, old = self.size, (self.cx, self.cy, self.zoom, self.fit)
        self.close()
        self.path = Path(path) if path else None
        if self.path is None or not self.path.is_file():
            self.path = None
            self.redraw()
            return
        self._hdul = fits.open(self.path, memmap=True)
        data, header = self._hdul[0].data, self._hdul[0].header
        if row_order(header) != TOP_DOWN:
            data = data[..., ::-1, :]  # bottom-up：第 0 列是畫面最下面，翻成由上往下顯示
        self.data = data
        h, w = data.shape[-2:]
        step = max(1, max(h, w) // 1000)
        chans = data if data.ndim == 3 else data[None]
        self.params = [stretch_params(np.asarray(c[::step, ::step], np.float32)) for c in chans]
        if keep_view and old_size is not None and not old[3]:
            k = w / old_size[0]  # drizzle 2× 與一般疊圖：同一塊天空、同樣的天空放大倍率
            self.cx, self.cy, self.zoom, self.fit = old[0] * k, old[1] * k, old[2] / k, False
        else:
            self.cx, self.cy, self.fit = w / 2, h / 2, True
        self.redraw()

    def close(self) -> None:
        self.data = None
        if self._hdul is not None:
            try:
                self._hdul.close()
            except OSError:
                pass
            self._hdul = None

    # ---------- 縮放 ----------

    def _fit_zoom(self) -> float:
        if self.data is None:
            return 1.0
        w, h = self.winfo_width(), self.winfo_height()
        iw, ih = self.size
        return max(MIN_ZOOM, min((w - 16) / iw, (h - 16) / ih))

    def effective_zoom(self) -> float:
        return self._fit_zoom() if self.fit else self.zoom

    def zoom_fit(self) -> None:
        if self.data is not None:
            iw, ih = self.size
            self.cx, self.cy, self.fit = iw / 2, ih / 2, True
        self.redraw()

    def zoom_to(self, z: float, anchor: tuple[float, float] | None = None) -> None:
        """z：新的倍率；anchor：畫面上的定點（滑鼠位置），縮放時那一點不動。"""
        if self.data is None:
            return
        old = self.effective_zoom()
        z = float(np.clip(z, min(MIN_ZOOM, self._fit_zoom()), MAX_ZOOM))
        if anchor is not None:
            ax, ay = anchor
            w, h = self.winfo_width(), self.winfo_height()
            ix = self.cx + (ax - w / 2) / old
            iy = self.cy + (ay - h / 2) / old
            self.cx = ix - (ax - w / 2) / z
            self.cy = iy - (ay - h / 2) / z
        self.zoom, self.fit = z, False
        self.redraw()

    def step_zoom(self, direction: int, anchor=None) -> None:
        cur = self.effective_zoom()
        if direction > 0:
            nxt = next((z for z in ZOOMS if z > cur * 1.01), MAX_ZOOM)
        else:
            nxt = next((z for z in reversed(ZOOMS) if z < cur * 0.99), self._fit_zoom())
        self.zoom_to(nxt, anchor)

    def _wheel(self, event: tk.Event, delta: int | None = None) -> None:
        d = delta if delta is not None else event.delta
        if d:
            self.step_zoom(1 if d > 0 else -1, (event.x, event.y))

    def _press(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y, self.cx, self.cy)

    def _move(self, event: tk.Event) -> None:
        if self._drag is None or self.data is None:
            return
        x0, y0, cx, cy = self._drag
        z = self.effective_zoom()
        if self.fit:
            self.zoom, self.fit = z, False
        self.cx, self.cy = cx - (event.x - x0) / z, cy - (event.y - y0) / z
        self.redraw()

    # ---------- 繪製 ----------

    def redraw(self) -> None:
        if self._pending is None:
            self._pending = self.after_idle(self._draw)

    def _draw(self) -> None:
        self._pending = None
        if not self.winfo_exists():
            return
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if self.data is None:
            self.create_text(w / 2, h / 2, text=self.empty_text, fill=Darkroom.secondary, font=self.font)
            self._report()
            return
        if w < 10 or h < 10:
            return
        iw, ih = self.size
        z = self.effective_zoom()
        # 不要拖到影像外面：中心限制在影像範圍內
        self.cx = float(np.clip(self.cx, 0, iw))
        self.cy = float(np.clip(self.cy, 0, ih))
        x0 = max(0, int(np.floor(self.cx - w / 2 / z)))
        y0 = max(0, int(np.floor(self.cy - h / 2 / z)))
        x1 = min(iw, int(np.ceil(self.cx + w / 2 / z)) + 1)
        y1 = min(ih, int(np.ceil(self.cy + h / 2 / z)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        step = max(1, int(1 / z))  # 縮小時跳著讀，只讀需要的像素
        region = np.asarray(self.data[..., y0:y1:step, x0:x1:step], np.float32)
        chans = region if region.ndim == 3 else region[None]
        out = [apply_stretch(c, p) for c, p in zip(chans, self.params)]
        if len(out) == 3:
            img = Image.fromarray((np.dstack(out) * 255).astype(np.uint8), "RGB")
        else:
            img = Image.fromarray((out[0] * 255).astype(np.uint8), "L")
        tw = max(1, int(round((x1 - x0) * z)))
        th = max(1, int(round((y1 - y0) * z)))
        img = img.resize((tw, th), Image.NEAREST if z >= 1 else Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img, master=self)
        sx = w / 2 - (self.cx - x0) * z
        sy = h / 2 - (self.cy - y0) * z
        self.create_image(sx, sy, image=self._photo, anchor="nw")
        self._report()

    def _report(self) -> None:
        if self.on_zoom is not None:
            self.on_zoom(None if self.data is None else self.effective_zoom())
