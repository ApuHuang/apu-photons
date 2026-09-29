"""APU Photons 打包用的進入點（PyInstaller），做法與 APU Pick 相同。

multiprocessing.freeze_support() 一定要最先呼叫：打包後校正、整合、drizzle 的子行程跑的也是這個 exe，
少了它每個子行程都會再開一個視窗。

--smoke-test <Light 資料夾> <結果檔> [gpu]：確認打包好的 exe 能完整疊圖（含多行程、CFA Drizzle、預覽、
相機 RAW 函式庫），build_exe.py 打包完會自動跑。加上 gpu 時另外用 GPU 跑一次，確認 CuPy 有打包進來。
"""

import multiprocessing
import sys


def smoke_test(folder: str, out: str, gpu: bool) -> int:
    import traceback
    from pathlib import Path

    result = Path(out)
    try:
        import rawpy

        from apu_photons import backend, gui  # noqa: F401  視窗介面用的東西都要有打包進來
        from apu_photons.engine import Settings, run
        from apu_photons.model import Calibration
        from apu_photons.preview import load_preview

        assert rawpy.libraw_version
        lines = []
        modes = ["cpu"] + (["gpu"] if gpu else [])
        for mode in modes:
            if mode == "gpu":
                be = backend.select("gpu")
                assert be.gpu, "沒有偵測到 GPU"
            master = result.parent / f"smoke_{mode}.fits"
            res = run([Path(folder)], Calibration(), master,
                      Settings(drizzle=2, workers=2, gpu=mode, crop_common=False), echo=None)
            load_preview(master)
            holes = max(res.qc["drizzle"]["holes_fraction"])
            lines.append(f"{mode}: frames={res.qc['frames_integrated']} holes={holes:.4f}")
        result.write_text("ok " + "; ".join(lines) + "\n", encoding="utf-8")
        return 0
    except Exception:  # noqa: BLE001
        result.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 4 and sys.argv[1] == "--smoke-test":
        sys.exit(smoke_test(sys.argv[2], sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "gpu"))
    from apu_photons.gui import main

    sys.exit(main())
