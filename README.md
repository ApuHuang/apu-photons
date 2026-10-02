# APU Photons

Astrophotography Photons Utility：天文攝影的校正、對齊、疊圖。

工作流：**APU Pick**（挑片）→ **APU Photons**（疊圖）→ **APU Astro**（後製）

設計規格見 [SPEC.md](SPEC.md)。目前是開發中的 MVP。

## 視窗介面

```sh
.venv\Scripts\apu-photons-gui
```

與 APU Astro、APU Pick 同一套暗房介面：頂部列（加入 Light、試跑、疊圖／停止、繁中｜EN）、
右側可收合面板（結果、校正、疊圖、輸出、效能，說明在 ⓘ 裡）、底部狀態列（進度與剩餘時間）。
分頁：Light（資料夾與每張 frame 的結果）、結果（自動拉伸預覽、Drizzle／一般疊圖切換、警告）、紀錄。
參數、語言、面板收合狀態會記住。

## 已完成

| Stage | 內容 |
|---|---|
| 0 Ingest | FITS 與相機 RAW（2×2 Bayer）；依觀測夜分 session；尺寸／Bayer 不一致的自動排除；讀 APU Pick sidecar |
| 1 Master | bias / dark / flat / flat-dark，winsorized sigma clip 整合，快取 |
| 2 校正 | 在 CFA 原始資料上校正；flat 各色分別正規化；熱像素（master dark + 單張集中度偵測）與冷像素修正 |
| 3 對齊 | 星點偵測（CFA 用 super-pixel）、三角形配對 + RANSAC 相似變換、中天翻轉、自動參考 frame |
| 4 正規化 | 以配對星點的亮度比當 scale、背景中位數當 offset |
| 5 權重 | SNR² / FWHM²（SNR 以正規化後的雜訊計）；APU Pick 的 reject 當篩選，分數加成可選 |
| 6 整合 | 分段（band）串流，不一次載入全部；winsorized / sigma / average / median；保存每張的剔除遮罩 |
| 7 Drizzle | 1× / 2×；OSC 用 CFA drizzle（不解馬賽克、各色像素直接投進 R/G/B 網格）；沿用 Stage 4~6 的正規化、權重、剔除遮罩；輸出權重圖、零星空洞自動補；張數或 dither 不足只提醒 |
| 8 輸出 | 裁切共同區域、Downsample 0.5×、32-bit / 16-bit FITS、`recipe.json`、log |
| QC | 剔除原因統計、FWHM 分布、dither 分析與 drizzle 適用性、剔除像素特別多的 frame（衛星、飛機、雲） |

**尚未完成**：local normalization。

### 效能

- 校正與找星：多行程平行（預設核心數 − 1，`--workers`）
- 整合與 Drizzle：有 NVIDIA 顯示卡就自動用 GPU（原始碼執行時需要 `pip install -e ".[gpu]"`；打包版已內含）；否則用 CPU 多行程。`--gpu cpu` 可強制用 CPU
- GPU 與 CPU 結果一致（只有極少數剛好落在剔除門檻上的像素因浮點誤差判斷不同）

實測（Sony ARW 6024×4024，7 張，CFA Drizzle 2×，Ryzen 5 5600X + RTX 3090）：

| | 優化前 | CPU 11 行程 | GPU |
|---|---|---|---|
| 校正與找星 | 35 秒 | 5 秒 | 9 秒* |
| 整合 | 104 秒 | 42 秒 | 8 秒 |
| Drizzle | 253 秒 | 103 秒 | 8 秒 |
| 總計 | 392 秒 | 173 秒 | 48 秒 |

\* 校正與找星一律用 CPU；GPU 那次是第一次跑、沒有快取，CPU 那次沿用了校正快取。

Drizzle：`--drizzle 2`（或 1）、`--pixfrac 0.9`。輸出 `master.fits`（drizzle）、`master_stack.fits`（一般疊圖）、`master_weight.fits`（每個通道的覆蓋權重）。

## 使用

```sh
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\apu-photons stack Light夜1 Light夜2 --dark Dark --flat Flat --flat-dark FlatDark -o out\master.fits
```

常用選項：`--lang en`（英文）、`--preview 20`（抽樣試跑）、`--downsample`、`--rejection sigma`、`--reference 檔名`、`--memory 4096`、`--bits 16`。
`apu-photons stack -h` 看全部。

校正後的中介檔、master 與剔除遮罩放在輸出旁的 `.photons_cache/`，重跑會沿用；可以直接刪掉。

## 打包

```sh
.venv\Scripts\pip install -e ".[exe,gpu]" matplotlib   # matplotlib 只給 astropy 的打包 hook 掃描用，不會打包進去
.venv\Scripts\python packaging\build_exe.py            # → dist\APUPhotons-<版本>-win64.zip（約 243 MB）
```

只有一個版本：執行時自動偵測，有 NVIDIA 顯示卡（與新版驅動）就用 GPU，沒有就用 CPU 多行程，使用者不用選、也不用另外安裝 CUDA。
包進去的只有 CUDA runtime 與 NVRTC；cuBLAS、cuFFT 等用不到的函式庫（約 1.9 GB）會自動移除。

打包完會自動用合成星場跑一次打包好的 exe（多行程、CFA Drizzle 2×、預覽；打包的電腦有 GPU 時再用 GPU 跑一次），
失敗就不產生 zip。圖示由 `packaging\make_assets.py` 產生。

Mac 版（Apple Silicon 與 Intel）由 GitHub Actions 的雲端 Mac 打包（`.github/workflows/build-macos.yml`）：
發布 Release 時自動執行並把 zip 附到同一個 Release，也可以在 Actions 頁面手動執行。Mac 沒有 CUDA，只用 CPU 多行程。

Windows 版也可以由雲端 Windows 打包（`.github/workflows/build-windows.yml`），觸發方式相同。
雲端機沒有 NVIDIA 顯示卡，打包後的測試只跑 CPU 並確認 CuPy 有包進去；GPU 疊圖要在有顯示卡的電腦上實測。

## 測試

```sh
.venv\Scripts\python -m pytest -q
```

用合成星場驗證：已知位移／旋轉的對齊精度、中天翻轉、衛星軌跡剔除、熱像素與暗角修正、APU Pick sidecar 篩選、Preview 與 Downsample。

## 授權

[MIT License](LICENSE)
