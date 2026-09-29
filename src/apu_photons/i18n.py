"""介面文字：繁體中文（預設）與英文，用詞比照 APU Astro 與 APU Pick。

程式裡只放代號，顯示時才用 tr() 依目前語言轉成文字。兩份的代號必須一致（tests/test_gui.py 會檢查）。
疊圖引擎的紀錄與警告目前只有繁體中文（與 APU Pick 的命令列相同）。
"""

from __future__ import annotations

APP_NAME = "APU Photons"
APP_SUBTITLE = "Astrophotography Photons Utility"

LANGUAGES = {"zh": "繁體中文", "en": "English"}
DEFAULT_LANGUAGE = "zh"
_language = DEFAULT_LANGUAGE


def set_language(lang: str) -> None:
    global _language
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang}")
    _language = lang


def get_language() -> str:
    return _language


def tr(key: str, **kw: object) -> str:
    text = _CATALOG[_language].get(key)
    if text is None:
        text = _CATALOG[DEFAULT_LANGUAGE][key]
    return text.format(**kw) if kw else text


_ZH: dict[str, str] = {
    # 頂部列與選單
    "gui.btn.add_light": "加入 Light",
    "gui.btn.add_light.help": "加入一個 light frame 資料夾（{shortcut}）；不同晚的資料可以分別加入",
    "gui.btn.trial": "試跑",
    "gui.btn.trial.help": "只抽樣部分 frame 快速跑一次，先確認設定與結果",
    "gui.btn.stack": "疊圖",
    "gui.btn.stack.help": "校正、對齊並整合全部 frame",
    "gui.btn.stop": "停止",
    "gui.menu.about": "關於 APU Photons",
    "gui.menu.file": "檔案",
    "gui.menu.add_light": "加入 Light 資料夾…",
    "gui.menu.edit": "編輯",
    "gui.menu.copy": "拷貝",
    "gui.menu.select_all": "全選",
    "gui.menu.window": "視窗",
    "gui.no_project": "尚未加入 Light",
    "gui.details": "說明",
    # 分頁
    "gui.tab.light": "Light",
    "gui.tab.result": "結果",
    "gui.tab.log": "紀錄",
    "gui.light.folders": "Light 資料夾",
    "gui.btn.add": "加入…",
    "gui.btn.remove": "移除",
    "gui.col.folder": "資料夾",
    "gui.col.count": "張數",
    "gui.col.file": "檔案",
    "gui.col.session": "觀測夜",
    "gui.col.result": "結果",
    "gui.col.fwhm": "FWHM",
    "gui.col.stars": "星點數",
    "gui.col.weight": "權重",
    "gui.col.rejected": "剔除像素",
    "gui.col.reason": "說明",
    "gui.empty.hint": "加入 Light 資料夾開始（{shortcut}）\n挑過片的資料夾會自動讀取 APU Pick 的結果",
    "gui.result.ok": "使用",
    "gui.result.rejected": "未使用",
    "gui.reference": "參考 frame",
    "gui.result.stack": "一般疊圖",
    "gui.result.empty": "疊圖完成後在這裡預覽",
    "gui.result.stretch_note": "預覽為自動拉伸，輸出的 FITS 仍是線性資料",
    "gui.qc.output": "輸出：{path}",
    "gui.qc.suspicious": "剔除像素特別多（可能有衛星、飛機或雲）：{files}",
    "gui.qc.drizzle_ok": "張數與 dither 足夠，適合 Drizzle",
    "gui.qc.drizzle_no": "張數或 dither 不足，不建議 Drizzle",
    "reason.pick": "APU Pick 淘汰",
    "reason.incompatible": "尺寸或 Bayer 排列不同",
    "reason.registration_failed": "對齊失敗",
    "reason.too_few_stars": "星點太少",
    "reason.unreadable": "無法讀取或校正",
    "reason.not_in_preview": "試跑未抽到",
    # 面板
    "gui.group.result": "結果",
    "gui.group.result.info": "最近一次疊圖的摘要。每張 frame 的詳細結果在「Light」分頁，警告與輸出位置在「結果」分頁。",
    "gui.metric.frames": "Light 總數",
    "gui.metric.integrated": "整合張數",
    "gui.metric.rejected": "未使用",
    "gui.metric.reference": "參考 frame",
    "gui.metric.fwhm": "FWHM 中位數",
    "gui.metric.holes": "Drizzle 空洞",
    "gui.metric.elapsed": "耗時",
    "gui.group.calibration": "校正",
    "gui.group.calibration.info": "每一項都可以不選：沒有的就跳過，並在結果裡提醒。\n\n"
                                  "Dark 已含偏壓，有 dark 時不需要 bias。Flat 會先扣 flat-dark（沒有就扣 bias），"
                                  "彩色相機的 flat 各色分別正規化。\n\n所有校正都在解馬賽克之前的原始資料上進行。",
    "gui.cal.bias": "Bias",
    "gui.cal.dark": "Dark",
    "gui.cal.flat": "Flat",
    "gui.cal.flat_dark": "Flat-dark",
    "gui.cal.none": "未使用",
    "gui.cal.count": "{n} 張　{path}",
    "gui.btn.choose": "選擇…",
    "gui.btn.clear": "清除",
    "gui.group.integration": "疊圖",
    "gui.group.integration.info": "Winsorized（建議）：對離群值穩健，衛星、飛機、熱像素會被剔除。\n"
                                  "Sigma：傳統的反覆 σ 剔除。平均：不剔除。中位數：最穩但雜訊較多。\n\n"
                                  "加權：依每張的 SNR 與星點大小給權重，好的片貢獻較多。\n"
                                  "APU Pick 分數加成：預設 Pick 只負責篩選（淘汰的不用），開啟後分數也會影響權重。\n"
                                  "依觀測夜分組：同一晚的 frame 歸為同一個 session。",
    "gui.rej.average": "平均",
    "gui.rej.median": "中位數",
    "gui.slider.low": "低端剔除",
    "gui.slider.high": "高端剔除",
    "gui.value.sigma": "{v:.1f} σ",
    "gui.toggle.weights": "依品質加權",
    "gui.toggle.pick_boost": "APU Pick 分數加成",
    "gui.toggle.split_nights": "依觀測夜分組",
    "gui.group.output": "輸出",
    "gui.group.output.info": "輸出倍率：0.5× 在整合後縮小；2× 使用 Drizzle；1× 可選擇是否 Drizzle。\n\n"
                             "彩色相機自動使用 CFA Drizzle：不解馬賽克，每個像素直接放到自己的顏色，色彩更乾淨。"
                             "Drizzle 需要足夠張數（建議 20 張以上）與 dither，不足時仍會執行，但可能出現格紋或空洞。\n\n"
                             "Drizzle 時一般疊圖會另存為 *_stack.fits，覆蓋權重存為 *_weight.fits。",
    "gui.output.scale": "輸出倍率",
    "gui.toggle.drizzle": "Drizzle",
    "gui.slider.pixfrac": "Drop 大小（pixfrac）",
    "gui.toggle.fill_holes": "補上零星空洞",
    "gui.toggle.crop": "裁切到共同區域",
    "gui.output.bits": "位元深度",
    "gui.output.file": "輸出檔案",
    "gui.output.unset": "尚未設定",
    "gui.btn.change": "變更…",
    "gui.btn.reveal": "開啟資料夾",
    "gui.group.performance": "效能",
    "gui.group.performance.info": "GPU：有 NVIDIA 顯示卡時，整合與 Drizzle 在顯示卡上執行，通常快 5～30 倍。"
                                  "結果與 CPU 相同。\n\n平行處理數：校正與找星同時處理幾張。\n"
                                  "記憶體上限：整合時每次處理的資料量；越大越快，但不要超過實體記憶體。\n"
                                  "試跑張數：按「試跑」時抽樣的張數。",
    "gui.toggle.gpu": "使用 GPU",
    "gui.gpu.checking": "偵測顯示卡中…",
    "gui.gpu.none": "沒有可用的 NVIDIA GPU，使用 CPU",
    "gui.slider.workers": "平行處理數",
    "gui.slider.memory": "記憶體上限",
    "gui.slider.trial": "試跑張數",
    "gui.value.frames": "{n} 張",
    "gui.value.seconds": "{v:.0f} 秒",
    # 狀態列
    "gui.status.start": "加入 Light 資料夾，選擇校正檔，然後按「疊圖」",
    "gui.status.starting": "準備中…",
    "gui.status.stopping": "停止中…",
    "gui.status.progress": "{stage} {done}/{total}",
    "gui.status.eta": "　約剩 {seconds:.0f} 秒",
    "gui.status.done": "完成，耗時 {seconds:.0f} 秒",
    "gui.status.cancelled": "已停止",
    "gui.status.error": "失敗：{message}",
    "gui.summary": "{dirs} 個資料夾、{frames} 張",
    "gui.stage.ingest": "讀取檔案資訊",
    "gui.stage.masters": "建立 master 校正檔",
    "gui.stage.prepare": "校正與找星",
    "gui.stage.register": "對齊",
    "gui.stage.integrate": "整合（列）",
    "gui.stage.drizzle": "Drizzle（列）",
    "gui.stage.output": "輸出檔案",
    "gui.close.confirm": "疊圖還在進行，確定要停止並關閉嗎？",
}

_EN: dict[str, str] = {
    "gui.btn.add_light": "Add Lights",
    "gui.btn.add_light.help": "Add a folder of light frames ({shortcut}); add each night's folder separately",
    "gui.btn.trial": "Trial",
    "gui.btn.trial.help": "Stack a sample of the frames to check the settings quickly",
    "gui.btn.stack": "Stack",
    "gui.btn.stack.help": "Calibrate, register and integrate every frame",
    "gui.btn.stop": "Stop",
    "gui.menu.about": "About APU Photons",
    "gui.menu.file": "File",
    "gui.menu.add_light": "Add Light Folder…",
    "gui.menu.edit": "Edit",
    "gui.menu.copy": "Copy",
    "gui.menu.select_all": "Select All",
    "gui.menu.window": "Window",
    "gui.no_project": "No lights added",
    "gui.details": "Details",
    "gui.tab.light": "Lights",
    "gui.tab.result": "Result",
    "gui.tab.log": "Log",
    "gui.light.folders": "Light Folders",
    "gui.btn.add": "Add…",
    "gui.btn.remove": "Remove",
    "gui.col.folder": "Folder",
    "gui.col.count": "Frames",
    "gui.col.file": "File",
    "gui.col.session": "Night",
    "gui.col.result": "Result",
    "gui.col.fwhm": "FWHM",
    "gui.col.stars": "Stars",
    "gui.col.weight": "Weight",
    "gui.col.rejected": "Rejected",
    "gui.col.reason": "Notes",
    "gui.empty.hint": "Add a light folder to begin ({shortcut})\nAPU Pick results in the folder are read automatically",
    "gui.result.ok": "Used",
    "gui.result.rejected": "Unused",
    "gui.reference": "Reference frame",
    "gui.result.stack": "Stack",
    "gui.result.empty": "The result appears here after stacking",
    "gui.result.stretch_note": "Preview is auto-stretched; the FITS output stays linear",
    "gui.qc.output": "Output: {path}",
    "gui.qc.suspicious": "Many rejected pixels (satellite, plane or cloud?): {files}",
    "gui.qc.drizzle_ok": "Enough frames and dither for Drizzle",
    "gui.qc.drizzle_no": "Not enough frames or dither; Drizzle not recommended",
    "reason.pick": "Rejected by APU Pick",
    "reason.incompatible": "Different size or Bayer pattern",
    "reason.registration_failed": "Registration failed",
    "reason.too_few_stars": "Too few stars",
    "reason.unreadable": "Could not read or calibrate",
    "reason.not_in_preview": "Not sampled in trial",
    "gui.group.result": "Result",
    "gui.group.result.info": "Summary of the last stack. Per-frame results are on the Lights tab; warnings and "
                             "the output location are on the Result tab.",
    "gui.metric.frames": "Lights",
    "gui.metric.integrated": "Integrated",
    "gui.metric.rejected": "Unused",
    "gui.metric.reference": "Reference",
    "gui.metric.fwhm": "Median FWHM",
    "gui.metric.holes": "Drizzle Holes",
    "gui.metric.elapsed": "Time",
    "gui.group.calibration": "Calibration",
    "gui.group.calibration.info": "Every item is optional: anything missing is skipped and noted in the result.\n\n"
                                  "Darks include the bias, so bias is not needed with darks. Flats have the flat-dark "
                                  "(or bias) subtracted; colour flats are normalized per channel.\n\n"
                                  "All calibration happens on the raw CFA data, before debayering.",
    "gui.cal.bias": "Bias",
    "gui.cal.dark": "Dark",
    "gui.cal.flat": "Flat",
    "gui.cal.flat_dark": "Flat-dark",
    "gui.cal.none": "Not used",
    "gui.cal.count": "{n} frames  {path}",
    "gui.btn.choose": "Choose…",
    "gui.btn.clear": "Clear",
    "gui.group.integration": "Integration",
    "gui.group.integration.info": "Winsorized (recommended): robust to outliers; satellites, planes and hot pixels "
                                  "are rejected.\nSigma: classic iterative σ clipping. Average: no rejection. "
                                  "Median: most robust, but noisier.\n\nWeighting: better frames (higher SNR, "
                                  "smaller stars) contribute more.\nAPU Pick Score Boost: by default Pick only "
                                  "filters (rejected frames are skipped); with this on, the score also affects the "
                                  "weight.\nGroup By Night: frames from the same night form one session.",
    "gui.rej.average": "Average",
    "gui.rej.median": "Median",
    "gui.slider.low": "Low Rejection",
    "gui.slider.high": "High Rejection",
    "gui.value.sigma": "{v:.1f} σ",
    "gui.toggle.weights": "Weight By Quality",
    "gui.toggle.pick_boost": "APU Pick Score Boost",
    "gui.toggle.split_nights": "Group By Night",
    "gui.group.output": "Output",
    "gui.group.output.info": "Output scale: 0.5× downsamples after integration; 2× uses Drizzle; at 1× Drizzle is "
                             "optional.\n\nColour cameras use CFA Drizzle: no debayering, each pixel goes straight "
                             "to its own colour for cleaner colour. Drizzle needs enough frames (20+) and dither; "
                             "it still runs otherwise, but may show a grid pattern or holes.\n\nWith Drizzle the "
                             "regular stack is also saved as *_stack.fits and the coverage as *_weight.fits.",
    "gui.output.scale": "Output Scale",
    "gui.toggle.drizzle": "Drizzle",
    "gui.slider.pixfrac": "Drop Size (pixfrac)",
    "gui.toggle.fill_holes": "Fill Small Holes",
    "gui.toggle.crop": "Crop To Common Area",
    "gui.output.bits": "Bit Depth",
    "gui.output.file": "Output File",
    "gui.output.unset": "Not set",
    "gui.btn.change": "Change…",
    "gui.btn.reveal": "Show Folder",
    "gui.group.performance": "Performance",
    "gui.group.performance.info": "GPU: with an NVIDIA card, integration and Drizzle run on the GPU, usually 5–30× "
                                  "faster, with the same result as the CPU.\n\nWorkers: frames calibrated at once.\n"
                                  "Memory Limit: how much data integration handles at a time; larger is faster, but "
                                  "stay below your physical memory.\nTrial Frames: frames sampled by Trial.",
    "gui.toggle.gpu": "Use GPU",
    "gui.gpu.checking": "Detecting graphics card…",
    "gui.gpu.none": "No NVIDIA GPU available; using CPU",
    "gui.slider.workers": "Workers",
    "gui.slider.memory": "Memory Limit",
    "gui.slider.trial": "Trial Frames",
    "gui.value.frames": "{n} frames",
    "gui.value.seconds": "{v:.0f} s",
    "gui.status.start": "Add light folders, choose calibration, then press Stack",
    "gui.status.starting": "Preparing…",
    "gui.status.stopping": "Stopping…",
    "gui.status.progress": "{stage} {done}/{total}",
    "gui.status.eta": "  about {seconds:.0f} s left",
    "gui.status.done": "Done in {seconds:.0f} s",
    "gui.status.cancelled": "Stopped",
    "gui.status.error": "Failed: {message}",
    "gui.summary": "Folders: {dirs}  Frames: {frames}",
    "gui.stage.ingest": "Reading file info",
    "gui.stage.masters": "Building master calibration",
    "gui.stage.prepare": "Calibrating and finding stars",
    "gui.stage.register": "Registering",
    "gui.stage.integrate": "Integrating (rows)",
    "gui.stage.drizzle": "Drizzle (rows)",
    "gui.stage.output": "Writing output",
    "gui.close.confirm": "Stacking is still running. Stop and close?",
}

_CATALOG = {"zh": _ZH, "en": _EN}
