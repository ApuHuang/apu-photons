"""SPEC §4 的資料結構：Project / Session / Frame。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ACCEPTED, REJECTED, PENDING = "accepted", "rejected", "pending"


@dataclass
class PickMetrics:
    """APU Pick sidecar 裡的一筆（SPEC §5.1）。"""
    fwhm_px: float | None = None
    fwhm_arcsec: float | None = None
    eccentricity: float | None = None
    star_count: int | None = None
    background: float | None = None
    snr: float | None = None
    score: float | None = None
    verdict: str | None = None
    group: str | None = None


@dataclass
class Stars:
    """參考座標之前、frame 自己座標下的星點（原始像素座標）。"""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    fwhm: float        # 中位數（像素）
    background: float  # 背景中位數（ADU）
    noise: float       # 背景雜訊（ADU）
    snr: float         # 亮星 峰值/雜訊 中位數

    def __len__(self) -> int:
        return len(self.x)


KINDS = ("light", "dark", "bias", "flat", "flat_dark")
UNKNOWN = "unknown"


@dataclass
class Frame:
    path: Path
    header: dict = field(default_factory=dict)
    exposure: float | None = None
    date_obs: str | None = None
    bayer: str | None = None
    shape: tuple[int, int] | None = None
    hash: str | None = None
    kind: str = UNKNOWN                       # light / dark / bias / flat / flat_dark / unknown（SPEC §3 Stage 0）
    kind_source: str | None = None            # header / filename / folder / user
    is_master: bool = False                   # 做好的 master 校正檔
    filter: str | None = None                 # 歸併後的濾鏡名稱
    camera: str | None = None                 # 相機的比對鍵（OpticalTrain.camera_key）
    train: str | None = None                  # 光學系統 id
    night: str | None = None
    calib: dict | None = None                 # light：{"dark": set id, "bias": …, "flat": …, "source": "auto"/"user"}
    pick_metrics: PickMetrics | None = None
    pick_group: str | None = None
    calibrated_path: Path | None = None
    stars: Stars | None = None
    transform: np.ndarray | None = None       # 3×3，frame 座標 → 參考座標
    registration_residual_px: float | None = None
    norm: dict | None = None                  # {"scale": [...], "offset": [...]}，每通道一個
    weight: float | None = None
    rejected_pixel_fraction: float | None = None
    status: str = PENDING
    reject_reason: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def accepted(self) -> bool:
        return self.status != REJECTED

    def reject(self, reason: str) -> None:
        self.status, self.reject_reason = REJECTED, reason

    @property
    def gain(self):
        return self.header.get("GAIN", self.header.get("ISO"))

    @property
    def offset(self):
        return self.header.get("OFFSET")

    @property
    def temp(self) -> float | None:
        try:
            return float(self.header["CCD-TEMP"])
        except (KeyError, TypeError, ValueError):
            return None

    def summary(self) -> dict:
        try:
            size = self.path.stat().st_size
        except OSError:
            size = None
        return {
            "file": str(self.path), "size": size, "sha256": self.hash, "kind": self.kind,
            "kind_source": self.kind_source,
            "train": self.train, "filter": self.filter, "night": self.night, "calibration": self.calib,
            "status": self.status, "reject_reason": self.reject_reason, "weight": self.weight,
            "fwhm_px": None if self.stars is None else round(self.stars.fwhm, 3),
            "n_stars": None if self.stars is None else len(self.stars),
            "transform": None if self.transform is None else np.round(self.transform, 6).tolist(),
            "registration_residual_px": self.registration_residual_px,
            "norm": self.norm, "rejected_pixel_fraction": self.rejected_pixel_fraction,
            "pick": None if self.pick_metrics is None else asdict(self.pick_metrics),
        }


@dataclass
class OpticalTrain:
    """光學系統：同一台相機＋同一個焦距（SPEC §3.1.1）。"""
    id: str
    camera: str                     # 顯示用的相機名稱
    camera_key: str                 # 比對用（型號、像素、尺寸、Bayer、binning）
    focal_mm: float | None
    lens: str | None
    pixel_um: float | None
    shape: tuple[int, int] | None
    bayer: str | None
    pixel_scale_arcsec: float | None

    def summary(self) -> dict:
        return {"id": self.id, "camera": self.camera, "focal_mm": self.focal_mm, "lens": self.lens,
                "pixel_um": self.pixel_um, "shape": self.shape, "bayer": self.bayer,
                "pixel_scale_arcsec": None if self.pixel_scale_arcsec is None else round(self.pixel_scale_arcsec, 4)}


@dataclass
class CalibrationSet:
    """同類型、同條件的一套校正檔（SPEC §4.3）；做好的 master 自己一套。"""
    id: str
    kind: str
    frames: list[Frame]
    is_master: bool
    conditions: dict
    used_by: set = field(default_factory=set)   # 用到它的 light 校正組
    unused_reason: str | None = None
    unused_detail: dict = field(default_factory=dict)  # 原因的參數（例如沒有哪個濾鏡的 light）

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self.frames]

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "is_master": self.is_master, "conditions": self.conditions,
                "files": [{"file": str(f.path), "sha256": f.hash} for f in self.frames],
                "used": bool(self.used_by), "unused_reason": self.unused_reason, "unused_detail": self.unused_detail}


@dataclass
class IntegrationGroup:
    """平均在一起、輸出一個 master 的 light（對齊組 × 濾鏡）。"""
    align: str
    filter: str | None
    frames: list[Frame]
    output_name: str = ""

    @property
    def label(self) -> str:
        return self.filter or "—"


@dataclass
class AlignGroup:
    """對齊組：共用一張參考 frame 與同一個輸出網格（SPEC §3.1）。"""
    id: str
    trains: list[str]
    groups: list[IntegrationGroup]
    reference: Frame | None = None

    @property
    def frames(self) -> list[Frame]:
        return [f for g in self.groups for f in g.frames]


@dataclass
class Calibration:
    """0.1 的舊介面：各類型直接指定檔案（命令列的 --dark 等仍用它強制類型）。"""
    bias: list[Path] = field(default_factory=list)
    dark: list[Path] = field(default_factory=list)
    flat: list[Path] = field(default_factory=list)
    flat_dark: list[Path] = field(default_factory=list)


@dataclass
class Project:
    """Stage 0 的結果：所有加入的檔案、分類、分組與校正配對。"""
    name: str
    files: list[Frame]
    trains: dict[str, OpticalTrain] = field(default_factory=dict)
    align_groups: list[AlignGroup] = field(default_factory=list)
    calibration_sets: dict[str, CalibrationSet] = field(default_factory=dict)
    calib_groups: dict[str, dict] = field(default_factory=dict)   # 校正組 key → 配到的套（SPEC Stage 1）
    filter_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def lights(self) -> list[Frame]:
        return [f for f in self.files if f.kind == "light"]

    @property
    def frames(self) -> list[Frame]:
        """0.1 相容：所有 light。"""
        return self.lights

    @property
    def accepted(self) -> list[Frame]:
        return [f for f in self.lights if f.accepted]

    @property
    def unknown(self) -> list[Frame]:
        return [f for f in self.files if f.kind == UNKNOWN]

    @property
    def groups(self) -> list[IntegrationGroup]:
        return [g for a in self.align_groups for g in a.groups]
