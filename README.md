# APU Photons

Astrophotography Photons Utility：天文攝影的校正、對齊、疊圖。

工作流：**APU Pick**（挑片）→ **APU Photons**（疊圖）→ **APU Processing**（後製），同屬 APU Astro 系列。

設計規格見 [SPEC.md](SPEC.md)。目前開發 0.2：多組資料（濾鏡、相機、光學系統）、校正檔自動配對、檔案輸入。

## 視窗介面

```sh
.venv\Scripts\apu-photons-gui
```

與 APU Processing、APU Pick 同一套暗房介面：頂部列（加入檔案、試跑、疊圖／停止、繁中｜EN）、
右側可收合面板（結果、校正、疊圖、輸出、效能，說明在 ⓘ 裡）、底部狀態列（進度與剩餘時間）。

- **檔案**：light 與 dark、flat、bias 一起加入（可以多選、分幾次加入），類型自動判斷；
  可切換「依組顯示」（光學系統 → 濾鏡 → 檔案，校正檔依套分）或「全部列表」；判斷不出來的用「設定類型」指定；
  「濾鏡名稱…」手動合併拼法不同的濾鏡（會記住）；「指定校正檔…」讓選取的 light 改用指定的校正檔
- **校正**：每個校正組自動配到的 dark / bias / flat / flat 扣除，可以逐組改；所有校正檔與沒用到的原因；
  某晚沒有 flat 時，可打開「沒有 flat 時用最近一晚的」
- **結果**：每個整合組各一個 master，下拉選單切換；滾輪縮放 25%～400%、拖曳平移、適合視窗／100%，
  換濾鏡或 Drizzle／一般疊圖時保持同一塊天空
- **紀錄**

參數、語言、面板收合狀態、濾鏡名稱對應會記住。Mac 的選單列「檔案」與 Windows 檔案分頁的按鈕可以載入 Recipe：
完全重現（同樣的檔案、分類、配對、參考 frame）或只套用設定。

## 已完成

| Stage | 內容 |
|---|---|
| 0 Ingest | FITS 與相機 RAW（2×2 Bayer）；輸入是檔案，類型依 `IMAGETYP`、檔名、資料夾名稱判斷；分光學系統（相機＋焦距）、濾鏡（大小寫與空白自動歸併）、觀測夜；讀 APU Pick sidecar |
| 1 Master | 校正檔依條件分套，自動配對：dark（相機、增益、offset、曝光、溫度 ±2°C）、bias、flat（濾鏡、同一晚）、flat-dark；做好的 master 直接用；winsorized sigma clip 整合，快取 |
| 2 校正 | 在 CFA 原始資料上校正；flat 各色分別正規化；熱像素（master dark 的局部偵測 + 單張集中度偵測）與冷像素修正 |
| 3 對齊 | 星點偵測（CFA 用 super-pixel）、三角形配對 + RANSAC 相似變換、中天翻轉；每個對齊組一張參考 frame，組內各濾鏡都對齊到它 |
| 4 正規化 | 整合組內，以配對星點的亮度比當 scale、背景中位數當 offset |
| 5 權重 | SNR² / FWHM²（SNR 以正規化後的雜訊計）；APU Pick 的 reject 當篩選 |
| 6 整合 | 每個整合組（光學系統 × 濾鏡）各自整合；分段（band）串流，不一次載入全部；winsorized / sigma / average / median；保存每張的剔除遮罩 |
| 7 Drizzle | 1× / 2×（對齊組統一）；OSC 用 CFA drizzle（不解馬賽克、各色像素直接投進 R/G/B 網格）；沿用 Stage 4~6 的正規化、權重、剔除遮罩；輸出權重圖、零星空洞自動補；張數或 dither 不足只提醒 |
| 8 輸出 | 每組一個 master（`<目標>_<濾鏡>.fits`）與覆蓋率圖；對齊組共用裁切範圍（預設至少 90% 的 frame 覆蓋）；Downsample 0.5×、32-bit / 16-bit FITS、專案一份 `recipe.json` 與 log |
| QC | 剔除原因統計、FWHM 分布、dither 分析與 drizzle 適用性、剔除像素特別多的 frame（衛星、飛機、雲）、沒用到的校正檔與原因 |

可選：不同光學系統拍同一個濾鏡時合併成一個 master（重新取樣到像素最細的網格）。

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

Drizzle：`--drizzle 2`（或 1）、`--pixfrac 0.9`。輸出 `<目標>_<濾鏡>.fits`（drizzle）、`…_stack.fits`（一般疊圖）、`…_weight.fits`（每個通道的覆蓋權重）。

## 使用

```sh
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\apu-photons stack M31\LIGHT M31\DARK M31\FLAT M31\BIAS -o M31\out
```

light 與校正檔一起給（檔案或資料夾；資料夾等同其中的影像，不往子資料夾找），類型自動判斷。
判斷不出來的（例如相機 RAW 放在以目標命名的資料夾）用 `--light`、`--dark`、`--bias`、`--flat`、`--flat-dark` 指定
（這些選項後面接多個路徑，請放在最後）。輸出到 `-o` 資料夾：每個整合組一個 master，例如 `M31_Ha.fits`、`M31_OIII.fits`。

常用選項：`--lang en`（英文）、`--preview 20`（每組抽樣試跑）、`--target 名稱`、`--temp-tolerance 2`、`--flat-any-night`（某晚沒有 flat 時用最近一晚的）、
`--min-coverage 0.9`、`--no-crop`、`--filter-alias H-alpha=Ha`、`--merge "相機A @ 400mm,相機B @ 400mm"`、
`--downsample`、`--rejection sigma`、`--reference 檔名`、`--memory 4096`、`--bits 16`。`apu-photons stack -h` 看全部。

校正後的中介檔、master 與剔除遮罩放在輸出資料夾的 `.photons_cache/`，重跑會沿用；可以直接刪掉。

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

用合成星場驗證：已知位移／旋轉的對齊精度、中天翻轉、衛星軌跡剔除、熱像素與暗角修正、APU Pick sidecar 篩選、Preview 與 Downsample；
0.2 另有檔案分類、多濾鏡分組與像素對齊、dark 配對忽略濾鏡、flat 依觀測夜、做好的 master、合併光學系統、載入 Recipe、放大檢視。

## 授權

[MIT License](LICENSE)
