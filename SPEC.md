# APU Photons — Technical Specification (v0.1 draft)

> Astrophotography Photons Utility
> 工作流：**APU Pick → APU Photons → APU Astro**
> Pick the frames. Collect the photons. Process the image.

本文件依據 2026-09-28 Claude 與 ChatGPT 的三輪審查整理，範圍為 MVP 的處理流程與資料結構，不含 UI 設計。

---

## 1. 產品定位與原則

| 原則 | 說明 |
|---|---|
| 不做另一個 PixInsight | 讓「挑好的幾百張 → 丟進去 → 得到可靠 Master」變簡單 |
| 提醒但不阻擋 | 條件不足（沒 calibration、drizzle 張數不夠）時顯示警告，仍允許執行 |
| 原始檔唯讀 | 永不改寫使用者的 light / calibration FITS |
| 可重現 | 每次輸出附帶 recipe，可重新載入並得到相同結果 |
| 不鎖定 | Pick 的輸出格式公開、可被其他軟體讀取 |

### 1.1 Pick 與 Photons 的分工（核心架構決定）

- **APU Pick**：frame 層級的決策 —「這張要不要留？」輸出原始品質指標。
- **APU Photons**：pixel 層級的決策 — 權重、正規化、剔除、整合。
- Pick 分數 **≠** 疊圖權重。Pick 分數只做 gate（篩選）或加成；權重由 Photons 以物理量計算。

---

## 2. 範圍

### MVP
- CFA-aware calibration（Bias / Dark / Flat / Flat-dark），皆為 optional
- Cosmetic correction（熱／冷像素）
- 星點偵測 + registration（translation + rotation 為主），含中天翻轉
- 自動參考 frame 選擇（可手動覆寫）
- 全域 normalization（scale + offset）
- 權重計算（物理量為主，Pick 為輔）
- Pixel-level rejection（sigma clip / winsorized sigma clip）
- 標準疊圖（Average / Median / 加權平均）
- CFA (Bayer) Drizzle 1× / 2×
- Downsample 0.5×
- Multi-session：各 session 各自校正 + 全域 normalize
- 32-bit float 內部流程、分塊串流整合
- QC 摘要 + Preview（小樣本試跑）
- Recipe / Log 輸出與載入
- 讀取 APU Pick sidecar

### V1
- Local normalization（處理 Bortle 6 各晚梯度差異）
- 3× Drizzle（僅在欠取樣時開放）
- 更多 rejection 演算法（linear fit clipping、ESD 等）
- 進階權重模型
- Channel Separation（RGB → R/G/B，Ha/OIII extraction）
- Dark scaling 進階選項

### V2
- 雙鏡筒 / 多儀器資料融合（解析度匹配、不同焦段合併）
- Mosaic

### 暫不做
- 自創 calibration 演算法
- 非星點式對齊（行星、月面）
- 後製（屬於 APU Astro）

---

## 3. Processing Pipeline

```
Stage 0 Ingest
Stage 1 Master Calibration
Stage 2 Calibrate + Cosmetic
Stage 3 Star Detection + Registration
Stage 4 Normalize
Stage 5 Weight
Stage 6 Integrate (+ rejection metadata)
Stage 7 Drizzle (optional)
Stage 8 Post / Output
```

每個 Stage 是**純函數**：輸入 Frame 狀態 + 參數 → 輸出新欄位。結果以 cache key 快取，修改某個參數只重跑其下游 Stage。

### Stage 0 — Ingest
- 讀 FITS header：`EXPTIME`、`GAIN`、`OFFSET`、`CCD-TEMP`、`XBINNING/YBINNING`、`BAYERPAT`（及 `XBAYROFF/YBAYROFF`）、`DATE-OBS`、`FILTER`、`INSTRUME`、`TELESCOP`、`FOCALLEN`、`XPIXSZ`、`PIERSIDE`。
- 分群：Session → Filter → Exposure/Gain/Temp。
- 讀 Pick sidecar（見 §5），以檔案 hash 對應 frame；hash 不符時警告並忽略該筆 metric。
- 計算像素尺度（arcsec/px）＝ 206.265 × XPIXSZ(µm) / FOCALLEN(mm) × binning，供跨 session 物理量比較。
- 驗證：尺寸、binning、Bayer pattern 不一致的 frame 標記 `rejected(reason=incompatible)`。

### Stage 1 — Master Calibration
- Master Bias / Dark / Flat-dark：sigma-clipped average。
- Master Flat：每張 flat 減 flat-dark（或 bias），正規化後 sigma-clipped 整合；在 **CFA 資料上以各色通道分別正規化**。
- Dark 匹配規則（MVP）：優先 exposure + gain + temp（±2°C）完全匹配；不匹配時警告，不做 scaling。
- 缺任何 calibration frame → 跳過該項並顯示提示。

### Stage 2 — Calibrate + Cosmetic
- `cal = (light − dark) / normalized_flat`（無 dark 時減 bias）。
- **全程在未 debayer 的 CFA 資料上進行。**
- Cosmetic correction：由 master dark 找熱像素 + 局部 sigma 偵測，以同色鄰近像素中位數取代（CFA-aware）。
- 輸出：32-bit float 的 CFA 中介 FITS。

### Stage 3 — Star Detection + Registration
- 亮度代理圖：CFA 2×2 superpixel（僅供偵測，不作為輸出）。
- 星點偵測：背景估計 → 閾值 → centroid，並記錄 FWHM、flux、偏心率。
- 配對：三角形不變量匹配 + RANSAC。
- 變換模型：MVP 預設 **similarity（平移＋旋轉＋等比縮放）**；affine 為選用，且需通過殘差檢查。錯配的風險大於對不上，配對失敗的 frame 標記 `rejected(reason=registration_failed)`。
- 中天翻轉：允許 ~180° 旋轉解。
- 參考 frame：以 FWHM 低、偏心率低、星點數多、背景低綜合評分自動選擇；可手動指定。
- Dither 分析：統計 transform 平移分量的小數部分分布，供 drizzle 適用性判斷。
- Common area：所有已接受 frame 變換後的交集。

### Stage 4 — Normalize
- 對參考 frame 求每張的 `scale`、`offset`（additive + multiplicative）。
- **估計時只使用未被污染的像素**：排除星點（依 Stage 3 星表遮罩）、飽和像素、以及 robust 統計下的離群像素（衛星、雲）。
- 使用 robust 估計量（median / MAD 或 biweight）。

### Stage 5 — Weight
- 主權重由物理量計算，預設：`w ∝ SNR² / FWHM_arcsec²`（SNR 由背景雜訊與星點 flux 估計）。
- Pick 的作用：
  - Gate：Pick 標記為淘汰的 frame 直接排除（`rejected(reason=pick)`）。
  - 可選加成：`w' = w × f(pick_score)`，預設關閉。
- 權重正規化到總和為 1，記錄至 Frame。

### Stage 6 — Integrate
- 分塊（tile / row-band）串流讀取，記憶體上限可設定，禁止一次載入全部 frame。
- 非 drizzle 路徑：各 frame debayer（MVP：bilinear 或 VNG）→ 套 transform（Lanczos-3 內插，clamp）→ normalize → 加權 rejection 整合。
- Rejection：sigma clip / winsorized sigma clip（MVP），低／高各自的 σ 可設。
- 輸出：
  - Master image（32-bit float）
  - **Per-frame pixel validity / rejection metadata**：遮罩 + 剔除類型（low / high / saturated / out-of-bounds），以 frame 座標或參考座標存放（見 §4.4）。
  - 統計：每張 frame 被剔除的像素比例（供 QC 判讀衛星、雲）。

### Stage 7 — Drizzle（選用）
- CFA (Bayer) Drizzle：**不 debayer**，每個 CFA 像素依其顏色投入 R / G / B 輸出網格。
- 參數：scale（1× / 2×）、drop shrink（pixfrac，預設 0.9）。
- **消費** Stage 4 的 normalization、Stage 5 的權重、Stage 6 的 rejection metadata；Drizzle 不自行做 rejection，也不和 Stage 6 的內部實作耦合。
- 適用性檢查（只警告）：
  - frame 數 < 20
  - dither 不足（次像素位移分布過於集中）
  - 2× 時 FWHM > ~3 px（過取樣，drizzle 效益低）
- 輸出 RGB master + 各通道 weight map（檢查破洞與覆蓋率）。

### Stage 8 — Post / Output
- Downsample 0.5×：對 master 做 2×2 平均（**整合後**才做；UI 名稱為 *Downsample*，不叫 binning）。
- 裁切到 common area（可選）。
- 輸出：32-bit float FITS（預設）、16-bit TIFF / FITS（選用）。
- 寫出 `recipe.json` 與 `photons.log`。

---

## 4. 資料結構

以下為邏輯模型，實作語言另行決定。

### 4.1 Project
```
Project {
  id: string
  name: string            // e.g. "M31"
  target?: string
  sessions: Session[]
  settings: PipelineSettings
  reference_frame_id?: string   // null = auto
  recipe_version: string
}
```

### 4.2 Session
```
Session {
  id: string
  date: date              // 觀測夜
  instrument: { telescope, camera, focal_mm, pixel_um, bayer?: "RGGB"|"BGGR"|"GRBG"|"GBRG" }
  pixel_scale_arcsec: float
  calibration: {
    bias?: FrameRef[], dark?: FrameRef[], flat?: FrameRef[], flat_dark?: FrameRef[]
    masters: { bias?: Path, dark?: Path, flat?: Path, flat_dark?: Path }
  }
  frames: Frame[]
}
```

### 4.3 Frame
```
Frame {
  id: string
  path: Path
  hash: string            // sha256，用於 Pick 對應與 cache
  header: { exptime, gain, offset, temp, binning, filter, date_obs, pier_side, ... }
  bayer?: string
  pick_metrics?: {        // 來自 APU Pick sidecar，可能缺
    fwhm_px, fwhm_arcsec, eccentricity, star_count, background, snr, score, verdict
  }
  calibrated_path?: Path
  stars?: Star[]          // x, y, flux, fwhm, ecc
  transform?: Mat3x3      // frame → reference
  registration_residual_px?: float
  norm?: { scale: float, offset: float }  // per channel for OSC
  weight?: float
  rejection_ref?: Path    // Stage 6 的 per-frame validity metadata
  rejected_pixel_fraction?: float
  status: "pending" | "accepted" | "rejected"
  reject_reason?: "pick" | "incompatible" | "registration_failed" | "user" | "qc"
}
```

### 4.4 Rejection Metadata
- 每張 frame 一個壓縮檔（例如 RLE 或 bit-packed），在**參考座標**下記錄每個像素的狀態碼：
  `0 = valid, 1 = low, 2 = high, 3 = saturated, 4 = out_of_bounds, 5 = cosmetic`
- 附帶描述：產生它的 rejection 演算法名稱、版本、參數。
- Drizzle 以反向變換查詢；未來更換 rejection 演算法時，Drizzle 不需改寫。

### 4.5 Cache
- Cache key = `hash(stage_name, algorithm_version, params, upstream_keys, input_file_hash)`。
- **演算法版本必須包含在 key 中**；升級演算法後，舊 cache 自動失效。
- 位置：`<project>/.photons_cache/`，可一鍵清除。

---

## 5. APU Pick → Photons 資料交換

- 格式：每個 session 一份 sidecar，與 light frames 同目錄。
  - `apupick.json`：完整結構（主要來源）
  - `apupick.csv`：扁平版本，給人看或給其他軟體讀
- 不寫入 FITS header（原始檔唯讀）。Pick 可選擇另外輸出「帶 header 的副本」，但這不是 Photons 依賴的介面。

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
    "score": "0–100, session-relative"
  },
  "pixel_scale_arcsec": 1.23,
  "frames": [
    {
      "file": "Light_M31_300s_0001.fits",
      "sha256": "…",
      "fwhm_px": 2.31, "fwhm_arcsec": 2.84,
      "eccentricity": 0.41, "star_count": 812,
      "background": 0.62, "snr": 18.7,
      "score": 92.4, "verdict": "keep"
    }
  ]
}
```

### 5.2 CSV
```
file,sha256,fwhm_px,fwhm_arcsec,eccentricity,star_count,background,snr,score,verdict
```

### 5.3 規則
- `score` 只在同一 session 內有意義；**跨 session 比較一律使用物理量**（`fwhm_arcsec`、`background`（ADU/s）、`snr`）。
- `verdict ∈ {keep, reject}`：Photons 把 `reject` 當作 gate。
- 缺少 sidecar 時，Photons 仍可獨立運作，由 Stage 3 / 5 自行計算指標。
- **Session 的定義**：以 APU Pick 的分組為準（濾鏡 + 曝光時間）；`score` 只在同一組內有意義。
- **列出哪些 frame**：只列出輸出當下仍在 light 資料夾內的 frame。已被 Pick 搬到 `rejected/` 的不列出。
  使用者只輸出 sidecar、沒有搬檔時，淘汰的 frame 仍在資料夾內，照樣列出並標 `verdict: "reject"`。
- **像素尺度缺值**：header 缺 `XPIXSZ` 或 `FOCALLEN` 時，`pixel_scale_arcsec` 與 `fwhm_arcsec` 留空
  （JSON 為 `null`，CSV 為空白）；Photons 此時不做跨 session 的角秒比較，只用像素單位。

### 5.4 Photons 端的整合掛勾（MVP 先預留，Pick 輸出 sidecar 後再接上）
- Photons 的 **Session（觀測夜 + 光路）** 與 Pick 的 **分組（濾鏡 + 曝光）** 不是同一個概念：
  sidecar 的分組只存成 `Frame.pick_group`，不拿來決定 Photons 的 session 或 calibration 配對。
- 讀取介面集中在 `apu_photons/integrations/pick.py`：
  `load_pick_sidecar(folder) -> dict[sha256, PickMetrics] | None`；找不到 sidecar、schema 不認得、hash 對不上時一律回傳 `None` 或略過該筆，並寫入 QC 警告，不中斷流程。
- 使用點只有兩處：Stage 0 填入 `Frame.pick_metrics`、Stage 5 套用 gate（`verdict == "reject"`）與可選加成。其他 Stage 不得直接讀 Pick 的資料。
- 在 Pick 開始輸出 sidecar 之前，這個模組只會回傳 `None`，Photons 靠自己在 Stage 3 算出的星點指標運作。
- 將來可共用的程式碼（FITS/RAW 讀取、CFA super-pixel、星點量測）先在 Photons 內各自實作、介面保持相容；等兩邊穩定後再考慮抽成共用套件。

---

## 6. QC 與 Preview

### 6.1 執行前摘要
- 偵測到的 frame 數／各 session 分布
- 相容性問題、Pick 淘汰數
- Calibration 狀態（缺哪些、dark 是否匹配）

### 6.2 Registration 後
- 對齊成功數／失敗數（附原因）
- FWHM（arcsec）分布、參考 frame
- Dither 評估、drizzle 適用性
- Common area 比例

### 6.3 整合後
- 每張 frame 的剔除像素比例（異常高 → 可能有衛星或雲）
- Drizzle weight map 覆蓋率

### 6.4 Preview
- 抽樣 N 張（預設 20，依權重分層抽樣）跑 Stage 0–6，快速產生預覽 master。
- 與完整執行共用同一套 cache。

---

## 7. Recipe

`recipe.json` 記錄重現一次輸出所需的全部資訊：

```json
{
  "schema": "apuphotons-recipe/1",
  "photons_version": "x.y.z",
  "created": "…",
  "inputs": [{ "session": "…", "file": "…", "sha256": "…", "status": "accepted", "weight": 0.0031 }],
  "calibration": { "bias": "…", "dark": "…", "flat": "…", "flat_dark": "…" },
  "stages": {
    "registration": { "algorithm": "triangle-ransac", "version": "1", "model": "similarity" },
    "normalize":    { "algorithm": "robust-scale-offset", "version": "1" },
    "weight":       { "formula": "snr2_over_fwhm2", "pick_boost": false },
    "integrate":    { "method": "weighted_average", "rejection": "winsorized_sigma", "low": 4.0, "high": 3.0 },
    "drizzle":      { "enabled": true, "scale": 2, "pixfrac": 0.9, "mode": "cfa" },
    "post":         { "downsample": 1.0, "crop_common": true }
  },
  "reference_frame": "…",
  "outputs": ["M31_master.fits"]
}
```

- 支援 **Load Recipe**：套用相同設定到新資料，或完全重現。

---

## 8. 非功能需求

| 項目 | 要求 |
|---|---|
| 數值精度 | 內部全程 32-bit float；只有最終輸出可降為 16-bit |
| 記憶體 | 分塊串流整合；可設定上限；300 張 6248×4176 OSC 需能在 16 GB RAM 完成 |
| 效能 | Stage 2~3 多行程平行；Stage 6、7 依列分段，CPU 多行程或 NVIDIA GPU（CuPy，可選，沒有就自動退回 CPU）。同一份程式碼透過 `backend.py` 切換 numpy / cupy；Mac 的 Metal 後端暫不做 |
| 穩定性 | 單張 frame 失敗不中斷整體流程，標記後繼續 |
| 可中斷 | 中斷後可由 cache 續跑 |
| 可測試 | 每個 Stage 有合成資料單元測試（已知位移、已知熱像素、人工衛星軌跡） |

---

## 9. 待決事項

1. ~~實作語言與框架~~ **已定**：Python ≥ 3.10，與 APU Pick 相同的套件（numpy、scipy、astropy、photutils、rawpy），日後 GUI 沿用 Pick 的 Tk 暗房主題；先做核心引擎與 CLI
2. Debayer 演算法選擇（MVP 先用 bilinear；VNG / AHD 列 V1）
3. ~~RAW 是否納入 MVP~~ **已定**：納入，讀取方式比照 APU Pick 的 `rawfile.py`（rawpy，扣黑位後當 CFA）
4. Rejection metadata 的具體壓縮格式
5. Seestar 資料的特殊處理（已內建疊圖、檔案格式差異）
6. Multi-session 的 Flat 共用規則（同一光路跨夜是否允許共用 master flat）
