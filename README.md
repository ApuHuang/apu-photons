# APU Photons

Astrophotography Photons Utility：天文攝影的校正、對齊、疊圖。免費、開放原始碼（MIT）。

工作流：**[APU Pick](https://github.com/ApuHuang/apu-pick)**（挑片）→ **APU Photons**（疊圖）→
**[APU Processing](https://github.com/ApuHuang/apu-processing)**（後製），同屬 APU Astro 系列。

目前版本 **0.2.2**：light 與 dark、flat、bias 整包加入，自動分類、自動配對校正檔；多個濾鏡一次疊完，彼此像素對齊；
彩色相機用 CFA Drizzle；有 NVIDIA 顯示卡自動用 GPU 加速。設計規格見 [SPEC.md](SPEC.md)。

## 下載與安裝

到 [Releases](https://github.com/ApuHuang/apu-photons/releases/latest) 下載最新版，不用安裝 Python：

| 系統 | 檔案 |
|---|---|
| Windows 10 / 11（64 位元） | `APUPhotons-<版本>-win64.zip` |
| Mac（Apple M 系列） | `APUPhotons-<版本>-macos-arm64.zip` |
| Mac（Intel） | `APUPhotons-<版本>-macos-x86_64.zip` |

- **Windows**：解壓縮後雙擊 `APUPhotons.exe`。第一次開啟若出現「Windows 已保護您的電腦」，按「其他資訊」→「仍要執行」
  （程式沒有購買數位簽章）。
- **Mac**：解壓縮後把 `APUPhotons.app` 拖到「應用程式」。第一次開啟被擋下時，到「系統設定」→「隱私權與安全性」按「強制打開」
  （沒有 Apple 付費簽章）。
- **GPU**：Windows 有 NVIDIA 顯示卡（與新版驅動）時自動用 GPU，不用另外安裝 CUDA；沒有就用 CPU 多核心。Mac 一律用 CPU。
- 建議先用 [APU Pick](https://github.com/ApuHuang/apu-pick) 挑掉模糊、拖線、過雲的片，再交給 Photons 疊圖；Pick 淘汰的片 Photons 會自動跳過。

以下是從原始碼執行與開發的說明。

## 視窗介面

```sh
.venv\Scripts\apu-photons-gui
```

與 APU Processing、APU Pick 同一套暗房介面：頂部列（加入檔案、載入 Recipe、試跑、疊圖／停止、繁中｜EN）、
右側可收合面板（結果、校正、疊圖、輸出、效能，說明在 ⓘ 裡）、底部狀態列（進度與剩餘時間）。

- **檔案**：light 與 dark、flat、bias 一起加入（可以多選、分幾次加入），類型自動判斷；
  可切換「依組顯示」（光學系統 → 濾鏡 → 檔案，校正檔依套分）或「全部列表」；判斷不出來的用「設定類型」指定；
  「濾鏡名稱…」手動合併拼法不同的濾鏡（會記住）；「指定校正檔…」讓選取的 light 改用指定的校正檔
- **校正**：每個校正組自動配到的 dark / bias / flat / flat 扣除，可以逐組改；所有校正檔與沒用到的原因；
  某晚沒有 flat 時，可打開「沒有 flat 時用最近一晚的」
- **結果**：每個整合組各一個 master，下拉選單切換；滾輪縮放 25%～400%、拖曳平移、適合視窗／100%，
  換濾鏡或 Drizzle／一般疊圖時保持同一塊天空。下方會給建議：剔除特別多的片是天光變化（天亮、月光）、雲，
  還是衛星與飛機，該怎麼處理；也會提示適合 Drizzle，或裁切掉太多時可以調低覆蓋門檻
- **紀錄**

參數、語言、面板收合狀態、濾鏡名稱對應會記住。「載入 Recipe…」（Windows 在上方標題列，Mac 在選單列「檔案」）：
完全重現（同樣的檔案、分類、配對、參考 frame）或只套用設定。

## 功能

| Stage | 內容 |
|---|---|
| 0 Ingest | FITS 與相機 RAW（2×2 Bayer）；輸入是檔案，類型依 `IMAGETYP`、檔名、資料夾名稱判斷；已經疊好的成品（Seestar 的 `Stacked_*.fit` 等）自動跳過；分光學系統（相機＋焦距）、濾鏡（大小寫與空白自動歸併）、觀測夜；讀 APU Pick sidecar |
| 1 Master | 校正檔依條件分套，自動配對：dark（相機、增益、offset、曝光、溫度 ±2°C）、bias、flat（濾鏡、同一晚）、flat-dark；做好的 master 直接用；winsorized sigma clip 整合（GPU 或多行程），快取 |
| 2 校正 | 在 CFA 原始資料上校正；flat 各色分別正規化；熱像素（master dark 的局部偵測 + 單張集中度偵測）與冷像素修正 |
| 3 對齊 | 星點偵測（CFA 用 super-pixel）、三角形配對 + RANSAC 相似變換、中天翻轉；每個對齊組一張參考 frame，組內各濾鏡都對齊到它；縮放不合理或與參考重疊太少的判為對齊失敗 |
| 4 正規化 | 整合組內，以配對星點的亮度比當 scale、背景中位數當 offset；可選「局部正規化」：offset 改成平滑的面，各張的光害梯度對齊到參考 frame |
| 5 權重 | SNR² / FWHM²（SNR 以正規化後的雜訊計）；APU Pick 的 reject 當篩選；權重算不出來的 frame 排除 |
| 6 整合 | 每個整合組（光學系統 × 濾鏡）各自整合；分段（band）串流，不一次載入全部；winsorized / sigma / average / median；保存每張的剔除遮罩；沒有有效像素的組不輸出 |
| 7 Drizzle | 1× / 2×（對齊組統一）；OSC 用 CFA drizzle（不解馬賽克、各色像素直接投進 R/G/B 網格）；沿用 Stage 4~6 的正規化、權重、剔除遮罩；輸出權重圖、零星空洞自動補；張數或 dither 不足只提醒 |
| 8 輸出 | 每組一個 master（`<目標>_<濾鏡>.fits`）與覆蓋率圖；彩色相機＋雙窄帶濾鏡（Seestar 的 LP、L-eXtreme 等）另存 Ha、OIII 兩個單色 master；對齊組共用裁切範圍（預設至少 90% 的 frame 覆蓋）；Downsample 0.5×、32-bit / 16-bit FITS、專案一份 `recipe.json` 與 log |
| QC | 剔除原因統計、FWHM 分布、dither 分析與 drizzle 適用性、剔除像素特別多的 frame 分原因給建議（天光變化或梯度、雲、衛星與飛機）、沒用到的校正檔與原因 |

可選：不同光學系統拍同一個濾鏡時合併成一個 master（重新取樣到像素最細的網格）。

局部正規化（`--local-norm`，預設關閉）：多晚拍攝、各晚光害梯度不同時使用，剔除才不會把梯度差當成離群值；
成品保留參考 frame 的梯度，交給後製去除。過雲的片救不了。

### 效能

- 校正與找星：多行程平行（預設核心數 − 1，`--workers`）
- 建 master 校正檔、整合與 Drizzle：有 NVIDIA 顯示卡就自動用 GPU（原始碼執行時需要 `pip install -e ".[gpu]"`；打包版已內含）；否則用 CPU 多行程。`--gpu cpu` 可強制用 CPU
- GPU 與 CPU 結果一致（只有極少數剛好落在剔除門檻上的像素因浮點誤差判斷不同）

實測（Sony ARW 6024×4024，7 張，CFA Drizzle 2×，Ryzen 5 5600X + RTX 3090）：

| | 優化前 | CPU 11 行程 | GPU |
|---|---|---|---|
| 校正與找星 | 35 秒 | 5 秒 | 9 秒* |
| 整合 | 104 秒 | 42 秒 | 8 秒 |
| Drizzle | 253 秒 | 103 秒 | 8 秒 |
| 總計 | 392 秒 | 173 秒 | 48 秒 |

\* 校正與找星一律用 CPU；GPU 那次是第一次跑、沒有快取，CPU 那次沿用了校正快取。

建 master 校正檔（QHY183M 300 s dark × 40 張）：單執行緒約 80 秒 → CPU 多行程約 37 秒、GPU 約 6 秒，結果完全相同。

Seestar S50（1080×1920，240 張 20 秒，Drizzle 1×）：123 秒 → 53 秒，結果完全相同（對齊改多行程、解馬賽克與熱像素修正加速；
沒有 dark 時不再依溫度把 light 拆成好幾個校正組）。

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
`--downsample`、`--rejection sigma`、`--local-norm`（局部正規化）、`--no-dual-band`（雙窄帶不另存 Ha／OIII）、`--reference 檔名`、`--memory 4096`、`--bits 16`。`apu-photons stack -h` 看全部。

校正後的中介檔、master 與剔除遮罩放在輸出資料夾的 `.photons_cache/`，重跑會沿用；可以直接刪掉。

## 打包

```sh
.venv\Scripts\pip install -e ".[exe,gpu]" matplotlib   # matplotlib 只給 astropy 的打包 hook 掃描用，不會打包進去
.venv\Scripts\python packaging\build_exe.py            # → dist\APUPhotons-<版本>-win64.zip（約 243 MB）
```

只有一個版本：執行時自動偵測，有 NVIDIA 顯示卡（與新版驅動）就用 GPU，沒有就用 CPU 多行程，使用者不用選、也不用另外安裝 CUDA。
包進去的只有 CUDA runtime 與 NVRTC；cuBLAS、cuFFT 等用不到的函式庫（約 1.9 GB）會自動移除。

打包完會自動用合成星場（含 dark、flat）跑一次打包好的 exe（建 master、多行程、CFA Drizzle 2×、預覽；打包的電腦有 GPU 時
再用 GPU 跑一次），而且是在模擬「沒有裝 CUDA Toolkit」的條件下跑（藏起 `CUDA_PATH` 與 PATH 裡的 CUDA、kernel 快取用全新的資料夾），
確認打包版只靠包進去的 CUDA DLL 與標頭檔就能在 GPU 上編譯 kernel。失敗就不產生 zip。圖示由 `packaging\make_assets.py` 產生。

Mac 版（Apple Silicon 與 Intel）由 GitHub Actions 的雲端 Mac 打包（`.github/workflows/build-macos.yml`）：
發布 Release 時自動執行並把 zip 附到同一個 Release，也可以在 Actions 頁面手動執行。Mac 沒有 CUDA，只用 CPU 多行程。

Windows 版也可以由雲端 Windows 打包（`.github/workflows/build-windows.yml`），觸發方式相同。
雲端機沒有 NVIDIA 顯示卡，打包後的測試只跑 CPU 並確認 CuPy 有包進去；GPU 疊圖要在有顯示卡的電腦上實測。

## 測試

```sh
.venv\Scripts\python -m pytest -q
```

用合成星場驗證：已知位移／旋轉的對齊精度、中天翻轉、衛星軌跡剔除、熱像素與暗角修正、APU Pick sidecar 篩選、Preview 與 Downsample；
0.2 另有檔案分類、多濾鏡分組與像素對齊、dark 配對忽略濾鏡、flat 依觀測夜、做好的 master、合併光學系統、載入 Recipe、放大檢視、
壞 frame（權重算不出來、對齊到別處）不拖垮 master、master 校正檔多行程與 GPU 結果一致、局部正規化對齊各張的梯度。

## English

APU Photons (Astrophotography Photons Utility) calibrates, registers and stacks astrophotography frames. It is the
stacking step of the APU Astro series: [APU Pick](https://github.com/ApuHuang/apu-pick) (frame selection) →
APU Photons (stacking) → [APU Processing](https://github.com/ApuHuang/apu-processing) (post-processing).
Free and open source (MIT); the interface is available in English and Traditional Chinese.

- Add lights together with darks, flats and bias; types are detected and calibration frames are matched automatically
- Every filter is stacked in one run, and the masters are pixel aligned with each other (ready for HOO / SHO / LRGB)
- CFA Drizzle for colour cameras; dual-band filters (Seestar LP, L-eXtreme, …) also give Ha and OIII mono masters
- Advice after stacking (dawn or moonlight, cloud, satellites; Drizzle and crop hints); optional local normalization
- NVIDIA GPU acceleration on Windows without installing CUDA; multi-core CPU otherwise

**Download** the latest version from [Releases](https://github.com/ApuHuang/apu-photons/releases/latest)
(Windows 10 / 11 64-bit, Mac Apple silicon and Intel) — no Python needed. On Windows, unzip and run `APUPhotons.exe`
(if "Windows protected your PC" appears, choose "More info" → "Run anyway"). On Mac, drag `APUPhotons.app` to
Applications; if the first launch is blocked, open System Settings → Privacy & Security and choose "Open Anyway".

## 授權

[MIT License](LICENSE)
