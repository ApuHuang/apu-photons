"""運算後端：CPU（NumPy / SciPy）或 NVIDIA GPU（CuPy）。

整合與 drizzle 的程式碼只寫一份，透過 `xp`（陣列模組）與 `ndi`（ndimage）切換。
沒有 CuPy 或沒有 NVIDIA 顯示卡時自動用 CPU；Mac 一律 CPU。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import scipy.ndimage as _scipy_ndi

from .i18n import Msg


@dataclass(frozen=True)
class Backend:
    name: str            # "cpu" 或 "cuda"
    xp: ModuleType
    ndi: ModuleType
    device: str = ""

    @property
    def gpu(self) -> bool:
        return self.name == "cuda"

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        return a.get() if self.gpu else np.asarray(a)

    def free(self) -> None:
        if self.gpu:
            self.xp.get_default_memory_pool().free_all_blocks()

    def memory_bytes(self) -> int | None:
        """GPU 目前可用的記憶體。"""
        if not self.gpu:
            return None
        free, _total = self.xp.cuda.runtime.memGetInfo()
        return int(free)


CPU = Backend("cpu", np, _scipy_ndi)
# 最近一次 GPU 偵測失敗的原因（給錯誤訊息與介面顯示）
last_error: str | Msg | None = None


def _try_cuda() -> Backend | None:
    global last_error
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 用 pip 的 CUDA 函式庫時 CuPy 會抱怨找不到 CUDA_PATH，不影響
            import cupy as cp
            import cupyx.scipy.ndimage as cndi

        if cp.cuda.runtime.getDeviceCount() < 1:
            last_error = Msg("msg.no_gpu")
            return None
        props = cp.cuda.runtime.getDeviceProperties(0)
        cp.asarray([1.0]).sum()  # 真的能跑才算
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return Backend("cuda", cp, cndi, name)
    except Exception as exc:  # noqa: BLE001  CuPy 沒裝、驅動不合、CUDA 函式庫缺，一律退回 CPU
        last_error = f"{type(exc).__name__}: {exc}"
        return None


def select(mode: str = "auto") -> Backend:
    """mode：auto（有 GPU 就用）、gpu（沒有就報錯）、cpu。環境變數 APU_PHOTONS_GPU=0 可強制關閉。"""
    if mode == "cpu" or os.environ.get("APU_PHOTONS_GPU") == "0":
        return CPU
    be = _try_cuda()
    if be is None and mode == "gpu":
        raise RuntimeError(Msg("msg.gpu_required", error=last_error))
    return be or CPU


def default_workers() -> int:
    """CPU 平行處理數：保留一個核心給系統與介面。"""
    return max(1, (os.cpu_count() or 2) - 1)
