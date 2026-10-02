# APU Photons — Technical Specification (v0.2)

> Astrophotography Photons Utility
> 工作流：**APU Pick → APU Photons → APU Processing**（APU Astro 系列；後製軟體原名 APU Astro，2026-10 改名）
> Pick the frames. Collect the photons. Process the image.

本文件依據 2026-09-28 Claude 與 ChatGPT 的三輪審查整理出 MVP（0.1），2026-10-01 依試跑後的討論定案 0.2。
範圍為處理流程、資料結構與 0.2 的介面決定；0.2 的決策理由見 §10。

---

## 1. 產品定位與原則

| 原則 | 說明 |
|---|---|
| 不做另一個 PixInsight | 讓「挑好的幾百張 → 丟進去 → 得到可靠 Master」變簡單 |
| 提醒但不阻擋 | 條件不足（沒 calibration、drizzle 張數不夠）時顯示警告，仍允許執行 |
| 自動判斷，讓使用者決定 | 分類、配對、門檻都先自動處理並顯示結果，使用者可以改；不確定時不靜默套用 |
| 原始檔唯讀 | 永不改寫使用者的 light / calibration FITS |
| 可重現 | 每次輸出附帶 recipe，可重新載入並得到相同結果 |
| 不鎖定 | Pick 的輸出格式公開、可被其他軟體讀取 |

### 1.1 Pick 與 Photons 的分工（核心架構決定）

- **APU Pick**：frame 層級的決策 —「這張要不要留？」輸出原始品質指標。
- **APU Photons**：pixel 層級的決策 — 權重、正規化、剔除、整合。
- Pick 分數 **≠** 疊圖權重。Pick 分數只做 gate（篩選）或加成；權重由 Photons 以物理量計算。

---

## 2. 範圍

### MVP（0.1，`v0.1.0` 已發布）
- CFA-aware calibration（Bias / Dark / Flat / Flat-dark），皆為 optional
- Cosmetic correction（熱／冷像素）
- 星點偵測 + registration（similarity），含中天翻轉
- 自動參考 frame 選擇（可手動覆寫）
- 全域 normalization（scale + offset）
- 權重計算（物理量為主，Pick 為輔）
- Pixel-level rejection（sigma clip / winsorized sigma clip）
- 標準疊圖（Average / Median / 加權平均）
- CFA (Bayer) Drizzle 1× / 2×
- Downsample 0.5×
- Multi-session：依觀測夜分 session
- 32-bit float 內部流程、分塊串流整合；CPU 多行程與 NVIDIA GPU
- QC 摘要 + Preview（小樣本試跑）
- Recipe / Log 輸出
- 讀取 APU Pick sidecar
- 暗房風格 GUI（繁中／英文）、Windows 與 macOS 打包

### 0.2（2026-10-01 定案）
1. **輸入改成檔案**：light 與校正檔一起加入，自動分類（§3 Stage 0）
2. **多組資料**：光學系統／整合組／校正組三種分組，濾鏡名稱歸併（§3.1）
3. **校正檔自動配對**：可手動修改，支援做好的 master（§3 Stage 1）
4. **合併不同光學系統**（可選）：同濾鏡疊成一個 master（§3.1.3，原列 V2）
5. **預覽放大**（§6.5）
6. **自動裁切改版**與覆蓋率圖（§3 Stage 8）
7. **載入 Recipe**（MVP 遺留項目，§7.2）

### V1
- Local normalization（處理 Bortle 6 各晚梯度差異）
- 3× Drizzle（僅在欠取樣時開放）
- 更多 rejection 演算法（linear fit clipping、ESD 等）
- 進階權重模型
- Channel Separation（RGB → R/G/B，彩色相機＋雙窄帶濾鏡的 Ha/OIII extraction）
- Dark scaling 進階選項
- 跨專案的校正檔庫

### V2
- 不同光學系統之間的 master 對齊輸出（目前交給 APU Processing）
- Mosaic

### 暫不做
- 自創 calibration 演算法
- 非星點式對齊（行星、月面）
- 後製（屬於 APU Processing）：RGB / HOO / SHO 合成、去梯度、拉伸

---

## 3. Processing Pipeline

```
Stage 0 Ingest（分類、分組）
Stage 1 Master Calibration（自動配對）
Stage 2 Calibrate + Cosmetic
Stage 3 Star Detection + Registration（每個對齊組一張參考）
Stage 4 Normalize（整合組內）
Stage 5 Weight（整合組內）
Stage 6 Integrate (+ rejection metadata)
Stage 7 Drizzle (optional，對齊組統一設定)
Stage 8 Post / Output（對齊組共用裁切範圍）
```

每個 Stage 是**純函數**：輸入 Frame 狀態 + 參數 → 輸出新欄位。結果以 cache key 快取，修改某個參數只重跑其下游 Stage。

### 3.1 分組模型（0.2）

| 分組 | 作用 | 由什麼決定 |
|---|---|---|
| **光學系統** | 同一套相機＋光學；資料的基本單位 | 相機（`INSTRUME`、像素大小 `XPIXSZ`、尺寸、Bayer 排列）＋焦距 `FOCALLEN` |
| **對齊組** | 組內所有 frame 對齊到同一張參考 frame，各 master 像素完全對齊；drizzle、downsample、裁切統一設定 | 預設＝一個光學系統；可選合併多個光學系統（§3.1.3） |
| **整合組** | 平均在一起、輸出一個 master | 對齊組 × 濾鏡（歸併後，§3.1.2） |
| **校正組** | 共用同一套 dark / bias / flat / flat-dark | 依 Stage 1 的配對條件 |
| Session（觀測夜） | flat 配對與 QC 的時間單位 | 以當地正午切分（`SITELONG`），與 0.1 相同 |

- 同一個整合組裡，不同觀測夜、曝光、增益／ISO、offset、溫度的片**合併**整合：
  - 曝光、增益不同 → 由正規化（星點亮度比例＋背景偏移）對齊亮度，SNR² 權重讓訊號好的片占比較多；
  - 校正時各用對應條件的 dark / bias；飽和門檻每張各自判斷。
- 正規化、權重、剔除都在同一整合組內進行；跨組不比較亮度。
- 同一對齊組的各濾鏡 master 像素完全對齊，APU Processing 合成時不必再對齊；**不同對齊組之間不對齊**，由 APU Processing 處理。

#### 3.1.1 光學系統的判斷
- FITS：`INSTRUME` + `XPIXSZ` + 影像尺寸 + `BAYERPAT` + `FOCALLEN`。
  **不使用 `TELESCOP`**：常寫的是赤道儀名稱（例如 NGC1499 資料的 `TELESCOP=AM3`）。
- 相機 RAW：rawpy 讀得到焦距與鏡頭型號（`lens.model`），讀不到機身型號 → 以影像尺寸＋Bayer 排列區分相機。
- 缺 `FOCALLEN` 時，同一台相機的片視為同一光學系統。
- 名稱顯示：`<相機> @ <焦距>mm`，例如 `QHY183M @ 250mm`；RAW 用鏡頭型號。

#### 3.1.2 濾鏡名稱歸併
- **自動歸併**：只忽略大小寫與前後空白，例如 `Ha`、`HA`、` ha ` 視為同一個；顯示名稱用出現最多的寫法。
- **手動對應**：拼法不同（`H-alpha`、`Halpha`、`H_alpha`）要使用者指定才合併，避免把不同濾鏡誤合（例如 `L` 與 `L-Pro`）。
  手動對應記在設定裡，下次相同名稱自動套用，也寫進 recipe。
- 沒有 `FILTER` 的片（例如彩色相機 RAW）歸成一組，輸出檔名不加濾鏡。

#### 3.1.3 合併不同光學系統（可選）
- 預設不合併：每個光學系統各自一個對齊組，各自輸出 master。
- 使用者可以把多個光學系統合併成一個對齊組；合併後同濾鏡的片疊成一個 master。適合「兩套相同器材同時拍」。
- 合併時：
  - 參考網格預設用**像素尺度最細**的光學系統（不損失解析度），可以改；
  - 每張以相似變換（含縮放）重新取樣到參考網格；
  - 權重的 FWHM 換算成參考像素（乘上變換的縮放倍率），避免短焦大視野的片被誤判較銳利；
  - 只輸出各光學系統都覆蓋的重疊區域；
  - CFA Drizzle 每張依自己的 Bayer 排列與縮放投射到同一輸出網格；
  - 不同彩色相機色彩響應不同，介面提醒可能偏色。

### Stage 0 — Ingest
- **輸入是檔案**（0.2）：
  - GUI：「加入檔案」可多選、可分多次加入（例如每晚一次）；light 與校正檔一起加入。Tk 沒有原生拖放，0.2 不做拖放
    （Mac 可以把檔案或資料夾拖到 Dock 圖示上）。
  - CLI：直接給檔案（可用萬用字元）；為了相容，給資料夾時等同該資料夾內的所有影像（不往子資料夾找）。
  - 不再掃描資料夾，所以 Pick 的 `rejected/` 等子資料夾不會被誤讀。
- **類型判斷**（light / dark / bias / flat / flat-dark），依序：
  1. FITS `IMAGETYP`（`LIGHT`、`DARK`、`BIAS`、`FLAT`、`DARKFLAT`／`FLATDARK`、`MASTER DARK` 等常見寫法）；
  2. 檔名關鍵字；
  3. 上層資料夾名稱關鍵字（`light(s)`、`dark(s)`、`bias`、`flat(s)`、`flatdark`／`darkflat`／`flat-dark`／`dark flat`，不分大小寫）；
  4. 都判斷不出來 → 標成「未知」，請使用者指定，不參與疊圖。
  - 例外：header 寫 `LIGHT`、但檔名或資料夾名稱明確寫了 dark / flat / bias / flat-dark 時，**以名稱為準**。
    `LIGHT` 是多數拍攝軟體的預設值，用一般序列拍的 flat 也會寫 `LIGHT`（NGC2244 實拍：FLAT 資料夾 280 張都是 `IMAGETYP=LIGHT`）。
- **做好的 master**：
  - 檔名或 `IMAGETYP` 含 `master`（不分大小寫）→ master；
  - 判斷不出來、但同一套校正檔（同類型＋同配對條件）只有一張 → 視為 master；
  - master 直接使用、不再整合；0～1 的浮點 master（Siril、PixInsight 常見）自動 ×65535 換算成 ADU；
  - **做好的 master flat 視為已扣過 flat-dark / bias**，不再扣。
- 讀 header：`IMAGETYP`、`EXPTIME`、`GAIN`、`OFFSET`、`CCD-TEMP`、`XBINNING/YBINNING`、`BAYERPAT`（及 `XBAYROFF/YBAYROFF`）、`DATE-OBS`、`FILTER`、`INSTRUME`、`FOCALLEN`、`XPIXSZ`、`PIERSIDE`、`SITELONG`、`OBJECT`；相機 RAW 讀 ISO、快門、時間、焦距、鏡頭型號。
- 分組：光學系統 → 濾鏡（歸併後）→ 觀測夜；校正組在 Stage 1 配對。
- 讀 Pick sidecar（見 §5），在每個 light 檔案所在的資料夾找；以檔案 hash 對應 frame；hash 不符時警告並忽略該筆 metric。
- 像素尺度（arcsec/px）＝ 206.265 × XPIXSZ(µm) / FOCALLEN(mm)。
  **不乘 binning**：依 SBFITSEXT 慣例（NINA、SGP、ASCOM 相機），`XPIXSZ` 已經是 binning 後的像素大小。
- 驗證：與所屬光學系統的尺寸、binning、Bayer pattern 不一致，或列順序（`ROWORDER`）與多數相反的 frame 標記 `rejected(reason=incompatible)`。

### Stage 1 — Master Calibration
- Master Bias / Dark / Flat-dark：sigma-clipped average（做好的 master 直接使用）。
- Master Flat：每張 flat 減 flat-dark（或 bias），正規化後 sigma-clipped 整合；在 **CFA 資料上以各色通道分別正規化**。
- **自動配對**（0.2），每張 light 依條件找校正檔；條件相同的 light 共用一個校正組：

| 校正檔 | 必須相同 | 容許差距（可調） | 多個候選時 |
|---|---|---|---|
| dark | 相機、增益／ISO、offset、曝光 | 溫度 ±2°C | 溫度最接近，再看日期最接近 |
| bias | 相機、增益／ISO、offset | — | 日期最接近 |
| flat | 相機、光學系統、濾鏡、**同一觀測夜** | — | — |
| flat-dark | 相機、增益、offset、與 flat 相同的曝光 | 溫度 ±2°C | 先找 flat-dark，沒有再找曝光相同的 dark，都沒有用 bias |

- 校正檔先依類型與上表的條件分成「套」（溫度取整數度）；同一套整合成一個 master。
  flat 不依曝光分套（天光 flat 常常每張曝光不同）；同一套 flat 的曝光不一致時，flat-dark 無從配對，改扣 bias。
- 有 dark 時 light 不扣 bias（dark 已含偏壓），配到的 bias 列為「未使用：有 dark，不需要 bias」；使用者手動指定時照指定的。
- header 缺某個條件（例如做好的 master 少了 `GAIN`）時視為未知：仍可配對，但排在條件完全相同的候選之後。

- **dark、bias、flat-dark 的配對忽略 `FILTER`**：拍校正檔時濾鏡輪停在哪一格會寫進 header（NGC1499 的 dark / bias 都寫 `FILTER=Ha`），與校正無關。
- **別晚的 flat 預設不自動代用**：某一晚沒有對應的 flat 時，該校正組顯示「沒有 flat」，由使用者選擇改用另一晚的 flat，或不用。
  使用者可以打開「沒有 flat 時用最近一晚的」（命令列 `--flat-any-night`，預設關閉）：缺 flat 的組一律改用同濾鏡、同光學系統、
  日期最近那晚的 flat，並逐組提醒用了哪一晚（2026-10-02 加入；NGC2244 實拍的 flat 都在另一晚）。
- 配不到、或超出容許差距時提醒，不靜默套用；缺的項目跳過並在結果裡提醒（同 0.1）。
- 沒有被任何 light 用到的校正檔列為「未使用」並說明原因（例如 NGC1499 的 0.2 s、1 s dark：沒有 flat，所以用不到）。
- 使用者可以在介面上逐一修改每個校正組配到的校正檔；修改結果寫進 recipe。
- 不做 dark scaling（同 0.1）。

### Stage 2 — Calibrate + Cosmetic
- `cal = (light − dark) / normalized_flat`（無 dark 時減 bias）。
- **全程在未 debayer 的 CFA 資料上進行。**
- Cosmetic correction：由 master dark 找熱像素 + 局部 sigma 偵測，以同色鄰近像素中位數取代（CFA-aware）。
  - master dark 的熱像素：比**同色鄰近像素的中位數**高 5σ（0.2；0.1 用整張的中位數，amp glow 那一側整片會被當成壞像素，
    QHY183M 右側 17% 的像素被換掉）。
- 輸出：32-bit float 的 CFA 中介檔（cache）。

### Stage 3 — Star Detection + Registration
- 亮度代理圖：CFA 2×2 superpixel（僅供偵測，不作為輸出）；單色相機直接使用原圖。
- 星點偵測：背景估計 → 閾值 → centroid，並記錄 FWHM、flux、偏心率。
- 配對：三角形不變量匹配 + RANSAC。
- 變換模型：預設 **similarity（平移＋旋轉＋等比縮放）**；affine 為選用，且需通過殘差檢查。錯配的風險大於對不上，配對失敗的 frame 標記 `rejected(reason=registration_failed)`。
- 縮放倍率檢查：同一套器材的變換縮放必須在 1 ± 10%；合併不同器材時以兩者像素尺度的比例為準（不知道時只擋 0.2～5 倍以外的）。
  縮放接近 0 時所有星點會擠到同一顆星附近、殘差反而很小，不擋的話會在整合時出現無法反轉的變換（NGC2244 實拍：flat 被當成 light）。
- 中天翻轉：允許 ~180° 旋轉解。
- **參考 frame：每個對齊組一張**（0.2）。自動選擇時在整個對齊組（所有濾鏡）裡，以 FWHM 低、偏心率低、星點數多、背景低綜合評分；可手動指定。不同濾鏡的片都對齊到這張。
- Dither 分析：統計 transform 平移分量的小數部分分布，供 drizzle 適用性判斷（每個整合組各自判斷）。
- 覆蓋率：每個整合組計算每個像素被幾張 frame 覆蓋（Stage 8 裁切與覆蓋率圖使用）。

### Stage 4 — Normalize
- 在**整合組內**對正規化參考求每張的 `scale`、`offset`（additive + multiplicative）。
  正規化參考：對齊參考在該整合組時用它，否則用該組裡依 Stage 3 綜合評分最高的 frame（不同濾鏡的亮度不能互相比）。
- **估計時只使用未被污染的像素**：排除星點（依 Stage 3 星表遮罩）、飽和像素、以及 robust 統計下的離群像素（衛星、雲）。
- 使用 robust 估計量（median / MAD 或 biweight）。

### Stage 5 — Weight
- 主權重由物理量計算，預設：`w ∝ SNR² / FWHM²`（SNR 由背景雜訊與星點 flux 估計；FWHM 用參考像素，合併光學系統時換算，§3.1.3）。
- Pick 的作用：
  - Gate：Pick 標記為淘汰的 frame 直接排除（`rejected(reason=pick)`）。
  - 可選加成：`w' = w × f(pick_score)`，預設關閉。
- 權重在整合組內正規化到總和為 1，記錄至 Frame。

### Stage 6 — Integrate
- 分塊（tile / row-band）串流讀取，記憶體上限可設定，禁止一次載入全部 frame。
- 非 drizzle 路徑：各 frame debayer（bilinear）→ 套 transform（內插，clamp）→ normalize → 加權 rejection 整合。
- Rejection：sigma clip / winsorized sigma clip，低／高各自的 σ 可設。
- 每個整合組各自整合，輸出網格相同（同一對齊組的參考網格）。可用的 frame 少於 2 張的組不輸出並提醒。
- 輸出：
  - Master image（32-bit float）
  - **Per-frame pixel validity / rejection metadata**：遮罩 + 剔除類型（low / high / saturated / out-of-bounds），以參考座標存放（見 §4.4）。
  - 統計：每張 frame 被剔除的像素比例（供 QC 判讀衛星、雲）。

### Stage 7 — Drizzle（選用）
- CFA (Bayer) Drizzle：**不 debayer**，每個 CFA 像素依其顏色投入 R / G / B 輸出網格；單色相機為一般 drizzle。
- 參數：scale（1× / 2×）、drop shrink（pixfrac，預設 0.9）。
- **對齊組統一設定**（0.2）：同一對齊組的所有整合組用同一個輸出倍率與 drizzle 開關，各濾鏡 master 才能像素對齊。
- **消費** Stage 4 的 normalization、Stage 5 的權重、Stage 6 的 rejection metadata；Drizzle 不自行做 rejection，也不和 Stage 6 的內部實作耦合。
- 適用性檢查（只警告，**每個整合組各自判斷**，由使用者決定是否仍開啟）：
  - frame 數 < 20
  - dither 不足（次像素位移分布過於集中）
  - 2× 時 FWHM > ~3 px（過取樣，drizzle 效益低）
- 輸出 master + 各通道 weight map（檢查破洞與覆蓋率）。

### Stage 8 — Post / Output
- Downsample 0.5×：對 master 做 2×2 平均（**整合後**才做；UI 名稱為 *Downsample*，不叫 binning）；對齊組統一設定。
- **裁切**（0.2）：
  - 預設裁到「至少 90% 的 frame 覆蓋」的範圍（門檻可調；0.1 是幾乎 100%，多晚有旋轉或 dither 大時會裁掉很多視野）；
  - 找範圍的方法：從整張開始，每次把四條邊裡達標比例最低的那條往內收一格，直到四條邊都有 99.5% 以上的像素達標
    （0.1 分別看每一列、每一行是否 98% 達標；中天翻轉後上下各缺一條、邊緣又略斜時，幾乎每一行都差一點，會裁掉大半張）；
  - 同一對齊組的所有整合組**用同一個裁切範圍**（各整合組範圍的交集），各濾鏡 master 才能像素對齊；
  - 可關閉；需要完整視野（例如之後的 mosaic）時，交給 APU Processing 的非破壞裁切決定。
  - 保留自動裁切的理由：邊緣只被少數 frame 覆蓋，雜訊高、可能有對齊邊界痕跡，會干擾 APU Processing 的去光害梯度與自動拉伸。
- **覆蓋率圖**（0.2）：每個 master 一律另外輸出 `_coverage.fits`（每個像素被幾張覆蓋，裁切後與 master 同尺寸）。
- 輸出格式：32-bit float FITS（預設）、16-bit FITS（選用）。
- 輸出的 FITS（master、`_stack`、`_weight`、`_coverage`）一律寫 `ROWORDER`：照參考 frame 的列順序（FITS 沒寫就是 `BOTTOM-UP`，相機 RAW 是 `TOP-DOWN`）。資料不翻轉，只標明方向。
- **輸出位置與檔名**（0.2）：使用者選一個輸出資料夾，每個整合組一個 master：
  - `<目標>_<濾鏡>[_<光學系統>].fits`，例如 `NGC1499_Ha.fits`、`NGC1499_OIII.fits`；
  - `<目標>` 取 light 的 `OBJECT`，沒有時用第一個 light 檔案所在的資料夾名稱；可在介面上修改；
  - 只有一個對齊組時不加光學系統；沒有濾鏡時不加濾鏡；
  - drizzle 時另有 `_weight.fits`；
  - 整個專案一份 `<目標>.recipe.json` 與 `<目標>.log`。

---

## 4. 資料結構

以下為邏輯模型。

### 4.1 Project
```
Project {
  name: string                 // 輸出用的目標名稱，例如 "NGC1499"
  files: InputFile[]           // 使用者加入的所有檔案（light 與校正檔）
  trains: OpticalTrain[]
  align_groups: AlignGroup[]
  calibration_sets: CalibrationSet[]
  filter_aliases: { [name]: string }   // 手動對應，例如 {"H-alpha": "Ha"}
  settings: PipelineSettings
  recipe_version: string
}
```

### 4.2 OpticalTrain、AlignGroup、IntegrationGroup
```
OpticalTrain {
  id: string                   // 例如 "QHY183M @ 250mm"
  camera: string, focal_mm?: float, lens?: string
  pixel_um?: float, shape: [h, w], bayer?: "RGGB"|"BGGR"|"GRBG"|"GBRG"
  pixel_scale_arcsec?: float
}

AlignGroup {
  id: string
  trains: OpticalTrain[]       // 預設一個；合併光學系統時多個
  reference_frame_id?: string  // null = 自動
  grid_train?: string          // 合併時的參考網格，預設像素尺度最細者
  output: { drizzle: 0|1|2, pixfrac: float, downsample: bool,
            crop: { enabled: bool, min_coverage: float } }   // 預設 0.9
  groups: IntegrationGroup[]
}

IntegrationGroup {
  filter?: string              // 歸併後的濾鏡名稱；無濾鏡為 null
  frames: Frame[]              // 可跨觀測夜、曝光、增益
  output_name: string
}
```

### 4.3 CalibrationSet
```
CalibrationSet {
  id: string
  kind: "dark" | "bias" | "flat" | "flat_dark"
  is_master: bool              // 做好的 master（§3 Stage 0）
  files: FrameRef[]
  conditions: { camera, gain, offset, exptime, temp, filter?, night?, train? }
  master_path?: Path           // cache 中的 master
  used_by: string[]            // 哪些校正組用到；空 = 未使用（附原因）
}
```
- Light 的校正組＝ `{dark?, bias?, flat?, flat_dark?}` 四個 CalibrationSet 的組合；自動配對或使用者指定（記錄來源 `auto` / `user`）。

### 4.4 Frame
```
Frame {
  id: string
  path: Path
  hash: string            // sha256，用於 Pick 對應、cache、Load Recipe 比對
  kind: "light" | "dark" | "bias" | "flat" | "flat_dark" | "unknown"
  kind_source: "header" | "filename" | "folder" | "user"
  header: { exptime, gain, offset, temp, binning, filter, date_obs, pier_side, object, ... }
  train: string
  night: string
  bayer?: string
  pick_metrics?: {        // 來自 APU Pick sidecar，可能缺
    fwhm_px, fwhm_arcsec, eccentricity, star_count, background, snr, score, verdict, group
  }
  calibration?: { dark?, bias?, flat?, flat_dark?, source: "auto" | "user" }
  stars?: Star[]          // x, y, flux, fwhm, ecc
  transform?: Mat3x3      // frame → 對齊組的參考
  registration_residual_px?: float
  norm?: { scale: float, offset: float }  // per channel for OSC
  weight?: float
  rejection_ref?: Path    // Stage 6 的 per-frame validity metadata
  rejected_pixel_fraction?: float
  status: "pending" | "accepted" | "rejected"
  reject_reason?: "pick" | "incompatible" | "registration_failed" | "user" | "qc"
}
```

### 4.5 Rejection Metadata
- 每張 frame 一個檔案，在**參考座標**下記錄每個像素的狀態碼：
  `0 = valid, 1 = low, 2 = high, 3 = saturated, 4 = out_of_bounds, 5 = cosmetic`
- 附帶描述：產生它的 rejection 演算法名稱、版本、參數。
- Drizzle 以反向變換查詢；未來更換 rejection 演算法時，Drizzle 不需改寫。

### 4.6 Cache
- Cache key = `hash(stage_name, algorithm_version, params, upstream_keys, input_file_hash)`。
- **演算法版本必須包含在 key 中**；升級演算法後，舊 cache 自動失效。
- 位置：使用者的 cache 資料夾，可一鍵清除；同一套條件的 master 跨次執行共用。

---

## 5. APU Pick → Photons 資料交換

> **狀態（2026-10-01）**：Photons 讀取端已完成並保留；**Pick 端暫不輸出，優先度低**。
> 理由：sidecar 不提升畫質也不提升速度（Photons 仍會自己找星、量測，被淘汰的片用 Pick 搬到 `rejected/` 效果相同），
> 好處只在流程（不用搬檔、可反悔）與紀錄。0.2 改成輸入檔案後，使用者直接不加入被淘汰的片即可。

- 格式：每個 light 資料夾一份 sidecar，與 light frames 同目錄；同一資料夾的多個分組寫在同一份，以 `group` 區分。
  - `apupick.json`：完整結構（主要來源）
  - `apupick.csv`：扁平版本，給人看或給其他軟體讀；沒有 JSON 時才讀
- 不寫入 FITS header（原始檔唯讀）。

### 5.1 `apupick.json` schema（v1）
```json
{
  "schema": "apupick/1",
  "pick_version": "x.y.z",
  "created": "2026-09-28T21:00:00+08:00",
  "metric_definitions": {
    "fwhm_px": "median star FWHM in pixels",
    "fwhm_arcsec": "fwhm_px × pixel_scale",
    "eccentricity": "median, 0 = round",
    "background": "median ADU per second",
    "snr": "estimated star SNR",
    "score": "0–100, group-relative"
  },
  "pixel_scale_arcsec": 1.23,
  "frames": [
    {
      "file": "Light_M31_300s_0001.fits",
      "sha256": "…",
      "group": "Ha 300s",
      "fwhm_px": 2.31, "fwhm_arcsec": 2.84,
      "eccentricity": 0.41, "star_count": 812,
      "background": 0.62, "snr": 18.7,
      "score": 92.4, "verdict": "keep"
    }
  ]
}
```
- `schema` 必須完全等於 `"apupick/1"`，否則整份略過並提醒。

### 5.2 CSV
```
file,sha256,group,fwhm_px,fwhm_arcsec,eccentricity,star_count,background,snr,score,verdict
```

### 5.3 規則
- `sha256` 是唯一的對應依據（整個檔案的 SHA-256，小寫十六進位）；沒有 `sha256` 的列略過。
- `score` 只在 Pick 的同一分組內有意義；**跨組比較一律使用物理量**（`fwhm_arcsec`、`background`（ADU/s）、`snr`）。
- `verdict ∈ {keep, reject}`：Photons 把 `reject` 當作 gate；必須反映 Pick 的手動覆寫。
- 缺少 sidecar 時，Photons 仍可獨立運作，由 Stage 3 / 5 自行計算指標。
- **列出哪些 frame**：只列出輸出當下仍在 light 資料夾內的 frame。已被 Pick 搬到 `rejected/` 的不列出。
- **像素尺度**：與 Photons 相同，不乘 binning（§3 Stage 0）；header 缺 `XPIXSZ` 或 `FOCALLEN` 時 `pixel_scale_arcsec` 與 `fwhm_arcsec` 留空（JSON 為 `null`，CSV 為空白）。

### 5.4 Photons 端的整合掛勾
- Pick 的分組（濾鏡、曝光、增益）是為了在同條件下比較品質；Photons 的整合組（濾鏡）是為了把相同訊號疊在一起，不同曝光會合併。
  兩邊分組不必相同；sidecar 的分組只存成 `Frame.pick_metrics.group`，不拿來決定 Photons 的分組或校正配對。
  兩邊只需對齊**濾鏡名稱的比對規則**（§3.1.2）。
- 讀取介面集中在 `apu_photons/integrations/pick.py`：
  `load_pick_sidecar(folder) -> dict[sha256, (file, PickMetrics)] | None`；找不到 sidecar、schema 不認得、hash 對不上時一律回傳 `None` 或略過該筆，並寫入 QC 警告，不中斷流程。
- 使用點只有兩處：Stage 0 填入 `Frame.pick_metrics`、Stage 5 套用 gate（`verdict == "reject"`）與可選加成。其他 Stage 不得直接讀 Pick 的資料。
- 將來可共用的程式碼（FITS/RAW 讀取、CFA super-pixel、星點量測、暗房介面元件）先各自實作、介面保持相容；等兩邊穩定後再考慮抽成共用套件。

---

## 6. QC、Preview 與介面

### 6.1 執行前摘要
- 加入的檔案依類型分類的結果（含「未知」）
- 光學系統、對齊組、整合組與各組張數
- 濾鏡名稱：自動歸併了哪些、有哪些名稱可能需要手動對應
- 各校正組配到的校正檔、缺哪些、超出容許差距的、未使用的校正檔
- 相容性問題、Pick 淘汰數

### 6.2 Registration 後
- 對齊成功數／失敗數（附原因），每個對齊組的參考 frame
- FWHM（arcsec）分布
- Dither 評估、drizzle 適用性（每個整合組）
- 覆蓋率與預計裁切範圍

### 6.3 整合後
- 每張 frame 的剔除像素比例（異常高 → 可能有衛星或雲）
- Drizzle weight map 覆蓋率

### 6.4 試跑（Preview run）
- 每個整合組抽樣 N 張（預設 20，依權重分層抽樣）跑 Stage 0–6，快速產生預覽 master。
- 與完整執行共用同一套 cache。

### 6.5 結果預覽放大（0.2）
- 比照 APU Processing：滾輪縮放 25%～400%、拖曳平移、一鍵「適合視窗」與 100%。
- 放大時才從輸出 FITS（memmap）讀出可見區域；拉伸參數用整張圖算一次後固定，縮放時亮度不跳。
- 「一般疊圖／Drizzle」切換時保持同一位置與倍率，直接比較細節。
- 用下拉選單切換各整合組的 master；同一對齊組切換時保持位置（像素對齊，可直接比較 Ha / OIII）。

### 6.6 介面（0.2）
- **檔案分頁**（取代 0.1 的 Light 分頁）：
  - 「加入檔案」（多選，可分多次）、移除選取；
  - 顯示方式可切換：**依組顯示**（光學系統 → 整合組 → 檔案；校正檔依類型與條件）或**全部列表**；
  - 每個檔案顯示類型（可改）、濾鏡、曝光、增益、溫度、觀測夜；類型「未知」的醒目標示。
- **校正分頁**（取代 0.1 的四個資料夾欄位；右側面板太窄，放不下校正組的條件）：
  - 上半：每個校正組自動配到的 dark / bias / flat，選一組後在下方的下拉選單改，或「還原自動」；
    缺 flat 的觀測夜在這裡選要用哪一晚的 flat 或不用；
  - 下半：所有校正檔（每套的類型、張數、是否 master、使用中或沒用到的原因）。
  - 右側面板的「校正」只放摘要、溫度容許差距與「依觀測夜分組」。
- **濾鏡名稱對應**：檔案分頁的「濾鏡名稱…」，列出 header 裡出現過的寫法，可手動指定合併到哪個名稱（會記住）。
- **輸出設定**：右側面板「輸出」。有多個對齊組時上方多一個「設定套用到」（所有對齊組／某一個），
  drizzle、pixfrac、downsample、裁切與覆蓋門檻可以每組不同；只有一個對齊組時與 0.1 外觀相同。
  有兩個以上光學系統時多一個「合併不同光學系統」開關與參考網格選擇。
- **設定**：溫度容許差距（預設 ±2°C）、裁切覆蓋門檻（預設 90%）、輸出資料夾與目標名稱。
- **載入 Recipe**：Mac 在選單列「檔案 → 載入 Recipe…」；Windows 不放原生選單列（沒辦法變成深色），按鈕在檔案分頁。
- 結果分頁：整合組下拉選單、Drizzle／一般疊圖、適合視窗／100%、預覽放大（§6.5）。

---

## 7. Recipe

### 7.1 格式（`apuphotons-recipe/2`，0.2）
整個專案一份 `<目標>.recipe.json`，記錄重現所有輸出所需的全部資訊：

```json
{
  "schema": "apuphotons-recipe/2",
  "photons_version": "0.2.0",
  "created": "…",
  "target": "NGC1499",
  "settings": { "rejection": "winsorized", "low": 4.0, "high": 3.0, "weighting": true,
                "pick_boost": false, "temp_tolerance_c": 2.0, "bits": 32 },
  "filter_aliases": { "H-alpha": "Ha" },
  "optical_trains": [
    { "id": "QHY183M @ 250mm", "camera": "QHY183M", "focal_mm": 250.0, "pixel_um": 2.4,
      "bayer": null, "pixel_scale_arcsec": 1.98 }
  ],
  "calibration_sets": [
    { "id": "dark-300s-g20-o40", "kind": "dark", "is_master": false,
      "conditions": { "exptime": 300.0, "gain": 20, "offset": 40, "temp": -10.0 },
      "files": [{ "file": "…", "sha256": "…" }] }
  ],
  "align_groups": [
    { "id": "QHY183M @ 250mm", "trains": ["QHY183M @ 250mm"], "reference_frame": "…",
      "output": { "drizzle": 0, "pixfrac": 0.9, "downsample": false,
                  "crop": { "enabled": true, "min_coverage": 0.9, "box": [0, 0, 5544, 3684] } } }
  ],
  "inputs": [
    { "file": "…", "sha256": "…", "kind": "light", "kind_source": "header",
      "train": "QHY183M @ 250mm", "filter": "Ha", "night": "2024-11-09",
      "calibration": { "dark": "dark-300s-g20-o40", "bias": "…", "flat": null, "flat_dark": null, "source": "auto" },
      "status": "accepted", "weight": 0.051 }
  ],
  "stages": {
    "registration": { "algorithm": "triangle-ransac", "version": "1", "model": "similarity" },
    "normalize":    { "algorithm": "robust-scale-offset", "version": "1" },
    "weight":       { "formula": "snr2_over_fwhm2", "pick_boost": false },
    "integrate":    { "method": "weighted_average", "rejection": "winsorized_sigma", "low": 4.0, "high": 3.0 },
    "drizzle":      { "version": "drizzle-1" }
  },
  "outputs": [
    { "align_group": "QHY183M @ 250mm", "filter": "Ha", "frames": 20,
      "file": "NGC1499_Ha.fits", "coverage": "NGC1499_Ha_coverage.fits" }
  ],
  "warnings": ["…"]
}
```

### 7.2 載入 Recipe（0.2）
- **套用設定**：把設定、濾鏡名稱對應、各對齊組的輸出設定套用到目前加入的檔案；校正配對依目前的檔案重新自動配對。
- **完全重現**：依 `inputs` 的路徑重新加入檔案；以檔案大小比對，有記錄 sha256 時（讀過 Pick sidecar 的 light）再比 sha256；
  找不到或內容不同的檔案列出來，其餘照 recipe 的類型、校正配對、參考 frame 執行。
  裁切範圍不另外保存：同樣的資料與設定會算出同樣的範圍。
- 0.1 的 recipe（`apuphotons-recipe/1`）只能「套用設定」。

---

## 8. 非功能需求

| 項目 | 要求 |
|---|---|
| 數值精度 | 內部全程 32-bit float；只有最終輸出可降為 16-bit |
| 記憶體 | 分塊串流整合；可設定上限；300 張 6248×4176 OSC 需能在 16 GB RAM 完成 |
| 效能 | Stage 2~3 多行程平行；Stage 6、7 依列分段，CPU 多行程或 NVIDIA GPU（CuPy，可選，沒有就自動退回 CPU）。同一份程式碼透過 `backend.py` 切換 numpy / cupy；Mac 的 Metal 後端暫不做。實測（7 張 6024×4024 ARW、2× CFA Drizzle）：優化前 392 s、CPU 11 行程 173 s、RTX 3090 48 s |
| 穩定性 | 單張 frame 失敗不中斷整體流程，標記後繼續 |
| 可中斷 | 中斷後可由 cache 續跑 |
| 可測試 | 每個 Stage 有合成資料單元測試（已知位移、已知熱像素、人工衛星軌跡）；0.2 另加多濾鏡、多曝光、多光學系統、多晚 flat 的合成資料測試，以及 NGC1499 實拍驗收（§10.3） |

---

## 9. 待決事項

1. ~~實作語言與框架~~ **已定**：Python ≥ 3.10，與 APU Pick 相同的套件（numpy、scipy、astropy、photutils、rawpy），GUI 沿用 Pick 的 Tk 暗房主題
2. Debayer 演算法選擇（bilinear；VNG / AHD 列 V1）
3. ~~RAW 是否納入 MVP~~ **已定**：納入，讀取方式比照 APU Pick 的 `rawfile.py`（rawpy，扣黑位後當 CFA）
4. Rejection metadata 的具體壓縮格式
5. Seestar 資料的特殊處理（已內建疊圖、檔案格式差異）
6. ~~Multi-session 的 Flat 共用規則~~ **已定（2026-10-01）**：不自動跨夜代用；缺 flat 的觀測夜由使用者選擇改用哪一晚的 flat 或不用
7. 相機 RAW 的機身型號：rawpy 不提供，0.2 以影像尺寸＋Bayer 區分；需要時再加 EXIF 套件

---

## 10. 決策紀錄

### 10.1 0.1 試跑後的討論（2026-09-29）
- 預覽要能放大 → §6.5
- 單色相機＋濾鏡（LRGB、Ha / OIII / SII）會被疊在一起 → §3.1
- 自動裁切裁掉太多視野 → §3 Stage 8

### 10.2 0.2 定案（2026-10-01）
| 項目 | 決定 | 理由 |
|---|---|---|
| 分組 | 光學系統／對齊組／整合組／校正組（§3.1） | 「哪些可以平均」（訊號）與「用哪套校正檔」（感光條件）是兩件事；同濾鏡不同曝光、增益可以疊在一起，但各用對應的 dark |
| 不同光學系統、同濾鏡 | 預設各自輸出 master；可選合併 | 合併要重新取樣、換算 FWHM、只留重疊區，彩色相機還會偏色；預設保守 |
| 對齊參考 | **每個對齊組一張**，組內各濾鏡共用 | 同一套器材的各濾鏡 master 像素對齊，APU Processing 直接合成；不同器材若硬對齊到同一張參考，像素較細的一組會損失解析度 |
| Drizzle | 對齊組統一設定，適用性每個整合組各自提醒、使用者決定 | 同組各濾鏡 drizzle 倍率不同會對不齊 |
| 裁切 | 對齊組共用裁切範圍，預設 90% 覆蓋、可調、可關，一律輸出覆蓋率圖 | 保留視野，又不讓邊緣的高雜訊干擾後製；共用範圍才能對齊 |
| 輸入 | 檔案，不掃資料夾 | 使用者直接控制要疊哪些片；Pick 的 `rejected/` 不會被誤讀 |
| 校正檔 | 一起加入、自動配對、可手動修改 | 多組資料時每組手動指定太繁瑣 |
| 做好的 master | 名稱辨識；辨識不出來但只有一張就是 master | 不必每次附全部原始校正檔 |
| 濾鏡名稱 | 大小寫與前後空白自動歸併，其他手動對應並記住 | 避免把 `L` 與 `L-Pro` 這類不同濾鏡誤合 |
| 別晚的 flat | 讓使用者選擇 | 相機轉動、灰塵移動與否只有使用者知道 |
| 門檻（溫度、覆蓋率） | 可調，預設 ±2°C、90% | 讓使用者決定 |
| 介面 | 依組顯示／全部列表可切換 | 讓使用者選擇 |
| 輸出 | 一個資料夾，`<目標>_<濾鏡>[_<光學系統>].fits`，專案一份 recipe | |
| Pick sidecar | 讀取端保留，Pick 端暫不輸出 | 不提升畫質與速度（§5） |

### 10.3 0.2 實拍驗收資料：`E:\AstroPhotography\NGC1499`
- QHY183M 單色、焦距 250 mm、像素 2.4 µm（1.98″/px）、5544×3684；同一晚（2024-11-09）。
- LIGHT 32 張：**Ha 20、OIII 12**，皆 300 s、增益 20、offset 40、−10°C。
- DARK 120 張：300 s、0.2 s、1 s 各 40；BIAS 40 張。**沒有 flat**（使用者忘了拍，純測試用）。
- 預期結果：
  - 一個光學系統、一個對齊組、兩個整合組（Ha 20、OIII 12）；
  - Ha、OIII 都配到 300 s dark（雖然校正檔 header 寫 `FILTER=Ha`，OIII 照樣配得到）；bias 列為未使用（有 dark）；
  - 0.2 s、1 s dark 列為未使用（沒有 flat 可配）；提醒沒有 flat；
  - OIII 若開 drizzle，提醒張數 < 20；
  - 輸出 `NGC1499_Ha.fits`、`NGC1499_OIII.fits` 與各自的覆蓋率圖，兩者尺寸相同、像素對齊。
- 實測（2026-10-02，0.2.0.dev0，RTX 3090）：以上全部符合；兩個 master 都是 5450×3597，互相對齊的殘差 0.06 px、旋轉 0.0004°；熱像素 69,591 個（0.1 的整張門檻是 478,282 個，多的是 amp glow）；全程 167 秒。試跑（每組 5 張，跨中天翻轉）時發現 0.1 的裁切法只留 2236 px 寬，改成往內收邊後是 5470 px。
- flat、多晚、多光學系統、彩色相機 RAW 混入等情況用合成資料測試。

### 10.4 0.2.0.dev0 實測問題（2026-10-02，`G:\NGC2244`，SHO：H 24、O 56、S 54，dark 313，flat 280，無 bias）
- 使用者實測時整合出錯（`Singular matrix`），master 先退回 0.1.0，0.2 在 `v0.2` 分支修正：
  - FLAT 資料夾的檔案 header 都是 `IMAGETYP=LIGHT` → 被當成 light 疊進去 → 改成名稱優先（Stage 0 例外規則）；
  - 某張 flat「對齊」成功但縮放接近 0 → 整合時反轉變換失敗 → 加上縮放倍率檢查（Stage 3）；
  - 視窗只顯示一行錯誤 → 完整的錯誤經過寫進紀錄檔與紀錄分頁。
- 修正後同一組資料（使用者先用 APU Pick 淘汰 14 張，剩 120 張）：H 21、O 51、S 44 張整合，3 個 master 5368×3141，367 秒；
  flat 是 2/15 拍的、light 是 2/12～2/13，依規則不自動使用，提醒可手動選。
