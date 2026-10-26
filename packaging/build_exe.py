"""打包成可直接執行的程式，再壓成 zip 方便分享（做法與 APU Pick 相同）。

    pip install -e .[exe,gpu]
    python packaging/build_exe.py

- Windows：dist/APUPhotons/APUPhotons.exe → dist/APUPhotons-<版本>-win64.zip
- macOS：  dist/APUPhotons.app            → dist/APUPhotons-<版本>-macos-<arm64|x86_64>.zip
  （PyInstaller 不能跨平台，Mac 版要在 Mac 上打包，平常由 GitHub Actions 的雲端 Mac 負責）

每個平台只有一個版本。Windows 一律包進 CuPy 與 CUDA runtime / NVRTC，執行時自動偵測，
有 NVIDIA 顯示卡就用 GPU，沒有就用 CPU 多行程。Mac 沒有 CUDA，只用 CPU 多行程（pip install -e .[exe] 即可）。

打包完會用合成星場實際跑一次打包好的程式（多行程、CFA Drizzle 2×、預覽；這台有 GPU 時再用 GPU 跑一次），
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


def platform_tag() -> str:
    if sys.platform == "win32":
        return "win64"
    if IS_MAC:
        return f"macos-{platform.machine()}"
    return f"{sys.platform}-{platform.machine()}"


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
        # photutils 3.1 起匯入時用 astropy 的 minversion 檢查 scipy／numpy 版本，要讀套件資訊；沒包進去會 KeyError: 'scipy'
        "--copy-metadata", "scipy",
        "--copy-metadata", "numpy",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
    ]
    if IS_MAC:
        args += ["--osx-bundle-identifier", "tw.apu-astrophotography.apu-photons"]
    if gpu:
        try:
            import cupy  # noqa: F401
            import nvidia
        except ImportError:
            raise SystemExit("打包環境沒有 CuPy：先執行 pip install -e .[gpu]") from None

        args += ["--collect-all", "cupy", "--collect-all", "cupyx", "--collect-all", "cupy_backends",
                 "--collect-all", "cuda_pathfinder", "--hidden-import", "graphlib", "--copy-metadata", "cupy-cuda12x"]
        base = Path(list(nvidia.__path__)[0])
        for lib in GPU_CUDA_LIBS:
            for dll in (base / lib).rglob("*"):
                if dll.suffix.lower() in (".dll", ".so") or ".so." in dll.name:
                    rel = dll.parent.relative_to(base.parent)
                    args += ["--add-binary", f"{dll}{os.pathsep}{rel.as_posix()}"]
            args += ["--copy-metadata", f"nvidia-{lib.replace('_', '-')}-cu12"]
        # CuPy 用 NVRTC 編譯 kernel 要 CUDA 標頭檔，且必須與 NVRTC 同版。沒包的話會去找使用者自己裝的 CUDA Toolkit：
        # 比 NVRTC 舊就編譯失敗（__nv_fp8_e8m0 incomplete type），沒裝的話找的過程會逾時失敗（2026-10-26，同 APU Astro 0.1.4）。
        # gui_entry.py 會把 CUDA_PATH 指過來
        args += ["--add-data", f"{base / 'cuda_runtime' / 'include'}{os.pathsep}nvidia/cuda_runtime/include"]
    else:
        for mod in GPU_MODULES:
            args += ["--exclude-module", mod]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]
    PyInstaller.__main__.run(args)
    if IS_MAC:
        app = DIST / f"{NAME}.app"
        set_bundle_info(app)
        return app / "Contents" / "MacOS" / NAME
    if gpu:
        prune_cuda(DIST / NAME)
    return DIST / NAME / f"{NAME}.exe"


def set_bundle_info(app: Path) -> None:
    """補齊 Info.plist 後重新簽章（與 APU Pick 相同）。

    - 版本：命令列打包的 PyInstaller 一律填 0.0.0
    - 系統元件跟著系統語言：沒宣告的話，選資料夾視窗、確認對話框、選單的「隱藏」「結束」都是英文
    - 改了 Info.plist 原本的簽章就失效，Apple 晶片的 Mac 會說 app「已損毀」，所以要重新做 ad-hoc 簽章
    """
    import plistlib

    plist = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["CFBundleShortVersionString"] = __version__
    info["CFBundleVersion"] = __version__
    info["CFBundleAllowMixedLocalizations"] = True
    info["CFBundleLocalizations"] = ["en", "zh-Hant"]
    plist.write_bytes(plistlib.dumps(info))
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def prune_cuda(bundle: Path) -> None:
    """PyInstaller 順著 CuPy 各模組的相依性把 cuBLAS、cuFFT… 都收進來（約 1.8 GB），但用不到：
    CuPy 只在第一次呼叫這些函式庫時才載入。刪掉後由冒煙測試確認 GPU 疊圖仍正常。"""
    removed = 0
    for dll in (bundle / "_internal").glob("*.dll"):
        if dll.name.lower().startswith(UNUSED_CUDA):
            removed += dll.stat().st_size
            dll.unlink()
    print(f"移除用不到的 CUDA 函式庫 {removed / 2**20:.0f} MB")


def isolated_cuda_env(tmp: Path) -> dict:
    """模擬沒有裝 CUDA Toolkit 的電腦：CUDA_PATH 指到空資料夾、PATH 拿掉 CUDA、kernel 快取用全新的資料夾
    （不然會沿用以前編好的 kernel，根本不會重新編譯）。打包版必須只靠包進去的 DLL 與標頭檔就能在 GPU 上跑。"""
    env = dict(os.environ)
    for key in [k for k in env if k.upper().startswith("CUDA_PATH") or k.upper() == "CUDA_HOME"]:
        env.pop(key)
    (tmp / "no_cuda").mkdir()
    (tmp / "cupy_cache").mkdir()
    env["CUDA_PATH"] = str(tmp / "no_cuda")
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                  if "cuda" not in p.lower() and "nvidia gpu computing toolkit" not in p.lower())
    env["CUPY_CACHE_DIR"] = str(tmp / "cupy_cache")
    return env


def smoke_test(exe: Path, gpu: bool) -> None:
    from tests.test_drizzle import _dithered
    from tests.test_pipeline import _make

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 含 dark 與 flat：校正檔要建 master（GPU 那次在顯示卡上建，2026-10-26 的問題就出在這一步）
        _light, inputs = _make(tmp / "data", bayer="RGGB", with_cal=True, shifts=_dithered(10))
        out = tmp / "smoke.txt"
        cmd = [str(exe), "--smoke-test", str(out)] + [str(p) for p in inputs] + (["--expect-gpu"] if gpu else [])
        env = isolated_cuda_env(tmp)
        proc = subprocess.run(cmd, timeout=900, env=env)
        text = out.read_text(encoding="utf-8") if out.exists() else "(沒有輸出)"
        if proc.returncode != 0 or not text.startswith("ok ") or "frames=10" not in text or "masters=0" in text:
            raise SystemExit(f"打包好的程式測試失敗（exit {proc.returncode}）：\n{text}")
        if "gpu: frames" in text:
            kernels = [p for p in Path(env["CUPY_CACHE_DIR"]).rglob("*") if p.is_file()]
            if not kernels:
                raise SystemExit("GPU 疊圖通過，但 kernel 快取是空的：沒有真的在沒有 CUDA Toolkit 的條件下重新編譯")
            text = f"{text.strip()}; 沒有 CUDA Toolkit 時新編 kernel {len(kernels)} 個"
        print(f"打包好的程式測試通過：{text.strip()}")


def archive() -> tuple[Path, Path]:
    """回傳 (打包好的資料夾或 .app, zip)。"""
    base = DIST / f"{NAME}-{__version__}-{platform_tag()}"
    if IS_MAC:
        app = DIST / f"{NAME}.app"
        # 用 ditto 壓縮才會保留 .app 裡的符號連結和執行權限
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), f"{base}.zip"],
                       check=True)
        return app, Path(f"{base}.zip")
    return DIST / NAME, Path(shutil.make_archive(str(base), "zip", root_dir=DIST, base_dir=NAME))


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def main() -> None:
    gpu = not IS_MAC
    exe = build(gpu)
    smoke_test(exe, gpu)
    bundle, zip_path = archive()
    print(f"\n程式：{bundle}（{folder_size(bundle) / 2**20:.0f} MB）")
    print(f"分享用：{zip_path}（{zip_path.stat().st_size / 2**20:.0f} MB）")


if __name__ == "__main__":
    main()
