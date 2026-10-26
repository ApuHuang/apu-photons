"""APU Photons 打包用的進入點（PyInstaller），做法與 APU Pick 相同。

multiprocessing.freeze_support() 一定要最先呼叫：打包後校正、整合、drizzle 的子行程跑的也是這個 exe，
少了它每個子行程都會再開一個視窗。

--smoke-test <結果檔> <輸入…> [--expect-gpu]：確認打包好的程式能完整疊圖（含檔案分類、用 dark／flat 建 master、
多行程、CFA Drizzle、預覽、相機 RAW 函式庫），build_exe.py 打包完會自動跑。這台電腦有 NVIDIA GPU 時再用 GPU 跑一次
（含 GPU 建 master：CuPy 第一次編譯 kernel 要用到包進來的 CUDA 標頭檔）；
有包 GPU（--expect-gpu，Windows）但這台沒有顯示卡時，至少確認 CuPy 有打包進來（偵測失敗的原因不能是找不到模組）。

--cli <apu-photons 的參數…>：用打包好的程式跑命令列（測試打包版用；視窗版沒有主控台，結果看輸出資料夾的紀錄檔）。
"""

import multiprocessing
import sys


def use_bundled_cuda_headers() -> None:
    """CuPy 編譯 kernel 用包進來、與 NVRTC 同版的 CUDA 標頭檔，不用使用者自己裝的 CUDA Toolkit（見 build_exe.py）。
    只影響這個程式與它開的子行程。（與 APU Astro 整合版相同）"""
    import os

    cuda = os.path.join(getattr(sys, "_MEIPASS", ""), "nvidia", "cuda_runtime")
    if getattr(sys, "frozen", False) and os.path.isfile(os.path.join(cuda, "include", "cuda_fp8.h")):
        os.environ["CUDA_PATH"] = cuda
        os.environ.pop("CUDA_HOME", None)


def smoke_test(out: str, inputs: list[str], expect_gpu: bool) -> int:
    import traceback
    from pathlib import Path

    result = Path(out)
    try:
        import rawpy

        from apu_photons import backend, gui, recipe, zoomview  # noqa: F401  視窗介面用的東西都要有打包進來
        from apu_photons.engine import Settings, run
        from apu_photons.preview import load_preview

        assert rawpy.libraw_version
        lines = []
        has_gpu = backend.select("auto").gpu
        if not has_gpu and expect_gpu:
            # 沒有 GPU 的電腦：CuPy 要能載入，只是找不到顯示卡或驅動
            assert backend.last_error and "ModuleNotFound" not in backend.last_error                 and "ImportError" not in backend.last_error, f"CuPy 沒有打包好：{backend.last_error}"
            lines.append(f"gpu: none ({backend.last_error})")
        for mode in ["cpu"] + (["gpu"] if has_gpu else []):
            out_dir = result.parent / f"smoke_{mode}"
            res = run([Path(p) for p in inputs], out_dir,
                      Settings(drizzle=2, workers=2, gpu=mode, crop_common=False), echo=None)
            load_preview(res.output)
            holes = max(res.qc["groups"][0]["drizzle"]["holes_fraction"])
            # 有給 dark／flat 時，master 要真的建出來（GPU 那次就是在顯示卡上建，會編譯 kernel）
            masters = len(list((out_dir / ".photons_cache" / "masters").glob("master_*.fits")))
            lines.append(f"{mode}: frames={res.qc['frames_integrated']} holes={holes:.4f} masters={masters}")
        result.write_text("ok " + "; ".join(lines) + "\n", encoding="utf-8")
        return 0
    except Exception:  # noqa: BLE001
        result.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    use_bundled_cuda_headers()
    if len(sys.argv) >= 4 and sys.argv[1] == "--smoke-test":
        args = [a for a in sys.argv[3:] if a != "--expect-gpu"]
        sys.exit(smoke_test(sys.argv[2], args, "--expect-gpu" in sys.argv[3:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--cli":
        from apu_photons.cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    from apu_photons.gui import main

    sys.exit(main())
