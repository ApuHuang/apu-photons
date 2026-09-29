"""打包成可直接執行的程式，再壓成 zip 方便分享（做法與 APU Pick 相同）。

    pip install -e .[exe]
    python packaging/build_exe.py          # 標準版：CPU 多行程
    python packaging/build_exe.py --gpu    # GPU 版：另外包進 CuPy 與 CUDA runtime / NVRTC（需要 NVIDIA 驅動）

- Windows：dist/APUPhotons/APUPhotons.exe → dist/APUPhotons-<版本>-win64.zip（GPU 版為 -win64-gpu.zip）

打包完會用合成星場實際跑一次打包好的程式（多行程、CFA Drizzle 2×、預覽；GPU 版再用 GPU 跑一次），
都正常才算成功。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from apu_photons import __version__  # noqa: E402

NAME = "APUPhotons"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
ASSETS = ROOT / "src" / "apu_photons" / "assets"
IS_MAC = sys.platform == "darwin"
EXCLUDES = ["pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6", "notebook", "sphinx", "matplotlib"]
GPU_MODULES = ["cupy", "cupyx", "cupy_backends", "fastrlock", "cuda", "cuda_pathfinder"]
# CuPy 實際用到的 CUDA 函式庫：runtime 與執行時編譯 kernel 的 NVRTC。cuBLAS、cuFFT… 用不到，不包（省 2 GB）
GPU_CUDA_LIBS = ["cuda_runtime", "cuda_nvrtc"]
UNUSED_CUDA = ("cublas", "cufft", "curand", "cusolver", "cusparse", "nvjitlink")


def platform_tag(gpu: bool) -> str:
    tag = "win64" if sys.platform == "win32" else f"{sys.platform}-{platform.machine()}"
    return tag + ("-gpu" if gpu else "")


def build(gpu: bool) -> Path:
    import PyInstaller.__main__

    args = [
        str(ROOT / "packaging" / "gui_entry.py"),
        "--name", NAME,
        "--windowed",
        "--noconfirm", "--clean",
        "--icon", str(ASSETS / "app.ico"),
        "--paths", str(ROOT / "src"),
        "--add-data", f"{ASSETS}{os.pathsep}apu_photons/assets",
        "--collect-submodules", "photutils",
        "--collect-submodules", "apu_photons",
        "--collect-all", "rawpy",
        "--copy-metadata", "photutils",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
    ]
    if gpu:
        import nvidia

        args += ["--collect-all", "cupy", "--collect-all", "cupyx", "--collect-all", "cupy_backends",
                 "--collect-all", "cuda_pathfinder", "--hidden-import", "graphlib", "--copy-metadata", "cupy-cuda12x"]
        base = Path(list(nvidia.__path__)[0])
        for lib in GPU_CUDA_LIBS:
            for dll in (base / lib).rglob("*"):
                if dll.suffix.lower() in (".dll", ".so") or ".so." in dll.name:
                    rel = dll.parent.relative_to(base.parent)
                    args += ["--add-binary", f"{dll}{os.pathsep}{rel.as_posix()}"]
            args += ["--copy-metadata", f"nvidia-{lib.replace('_', '-')}-cu12"]
    else:
        for mod in GPU_MODULES:
            args += ["--exclude-module", mod]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]
    PyInstaller.__main__.run(args)
    if gpu:
        prune_cuda(DIST / NAME)
    return DIST / NAME / f"{NAME}.exe"


def prune_cuda(bundle: Path) -> None:
    """PyInstaller 順著 CuPy 各模組的相依性把 cuBLAS、cuFFT… 都收進來（約 1.8 GB），但用不到：
    CuPy 只在第一次呼叫這些函式庫時才載入。刪掉後由冒煙測試確認 GPU 疊圖仍正常。"""
    removed = 0
    for dll in (bundle / "_internal").glob("*.dll"):
        if dll.name.lower().startswith(UNUSED_CUDA):
            removed += dll.stat().st_size
            dll.unlink()
    print(f"移除用不到的 CUDA 函式庫 {removed / 2**20:.0f} MB")


def smoke_test(exe: Path, gpu: bool) -> None:
    from tests.test_drizzle import _dithered
    from tests.test_pipeline import _make

    with tempfile.TemporaryDirectory() as tmp:
        light, _cal = _make(Path(tmp), bayer="RGGB", with_cal=False, shifts=_dithered(10))
        out = Path(tmp) / "smoke.txt"
        cmd = [str(exe), "--smoke-test", str(light), str(out)] + (["gpu"] if gpu else [])
        proc = subprocess.run(cmd, timeout=900)
        text = out.read_text(encoding="utf-8") if out.exists() else "(沒有輸出)"
        if proc.returncode != 0 or not text.startswith("ok ") or "frames=10" not in text:
            raise SystemExit(f"打包好的程式測試失敗（exit {proc.returncode}）：\n{text}")
        print(f"打包好的程式測試通過：{text.strip()}")


def archive(gpu: bool) -> tuple[Path, Path]:
    base = DIST / f"{NAME}-{__version__}-{platform_tag(gpu)}"
    return DIST / NAME, Path(shutil.make_archive(str(base), "zip", root_dir=DIST, base_dir=NAME))


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def main() -> None:
    gpu = "--gpu" in sys.argv
    exe = build(gpu)
    smoke_test(exe, gpu)
    bundle, zip_path = archive(gpu)
    print(f"\n程式：{bundle}（{folder_size(bundle) / 2**20:.0f} MB）")
    print(f"分享用：{zip_path}（{zip_path.stat().st_size / 2**20:.0f} MB）")


if __name__ == "__main__":
    main()
