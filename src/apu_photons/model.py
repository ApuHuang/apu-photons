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


@dataclass
class Frame:
    path: Path
    header: dict = field(default_factory=dict)
    exposure: float | None = None
    date_obs: str | None = None
    bayer: str | None = None
    shape: tuple[int, int] | None = None
    hash: str | None = None
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

    def summary(self) -> dict:
        return {
            "file": str(self.path), "sha256": self.hash, "status": self.status,
            "reject_reason": self.reject_reason, "weight": self.weight,
            "fwhm_px": None if self.stars is None else round(self.stars.fwhm, 3),
            "n_stars": None if self.stars is None else len(self.stars),
            "transform": None if self.transform is None else np.round(self.transform, 6).tolist(),
            "registration_residual_px": self.registration_residual_px,
            "norm": self.norm, "rejected_pixel_fraction": self.rejected_pixel_fraction,
            "pick": None if self.pick_metrics is None else asdict(self.pick_metrics),
        }


@dataclass
class Calibration:
    bias: list[Path] = field(default_factory=list)
    dark: list[Path] = field(default_factory=list)
    flat: list[Path] = field(default_factory=list)
    flat_dark: list[Path] = field(default_factory=list)
    masters: dict[str, Path] = field(default_factory=dict)


@dataclass
class Session:
    id: str
    frames: list[Frame]
    calibration: Calibration = field(default_factory=Calibration)
    pixel_scale_arcsec: float | None = None


@dataclass
class Project:
    name: str
    sessions: list[Session]
    reference: Frame | None = None

    @property
    def frames(self) -> list[Frame]:
        return [f for s in self.sessions for f in s.frames]

    @property
    def accepted(self) -> list[Frame]:
        return [f for f in self.frames if f.accepted]
