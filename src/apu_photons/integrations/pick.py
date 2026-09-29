"""APU Pick sidecar 讀取（SPEC §5、§5.4）。

Pick 還沒開始輸出 sidecar 時，這裡只會回傳 None；任何問題都只回報警告、不中斷流程。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..model import PickMetrics

SCHEMA = "apupick/1"
JSON_NAME = "apupick.json"
CSV_NAME = "apupick.csv"
_FIELDS = ("fwhm_px", "fwhm_arcsec", "eccentricity", "star_count", "background", "snr", "score")


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _entry(row: dict) -> PickMetrics:
    vals = {k: _num(row.get(k)) for k in _FIELDS}
    if vals["star_count"] is not None:
        vals["star_count"] = int(vals["star_count"])
    verdict = row.get("verdict")
    return PickMetrics(**vals, verdict=verdict if verdict in ("keep", "reject") else None,
                       group=row.get("group") or None)


def load_pick_sidecar(folder: Path, warnings: list[str] | None = None) -> dict[str, tuple[str, PickMetrics]] | None:
    """讀 folder 裡的 sidecar，回傳 {sha256: (檔名, 指標)}；沒有或讀不懂就回傳 None。"""
    warn = warnings.append if warnings is not None else (lambda _m: None)
    rows: list[dict] | None = None
    jpath, cpath = folder / JSON_NAME, folder / CSV_NAME
    try:
        if jpath.is_file():
            doc = json.loads(jpath.read_text(encoding="utf-8"))
            if doc.get("schema") != SCHEMA:
                warn(f"{jpath.name}: 不認得的 schema {doc.get('schema')!r}，略過 APU Pick 資料")
                return None
            rows = list(doc.get("frames", []))
        elif cpath.is_file():
            with cpath.open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
    except (OSError, ValueError) as exc:
        warn(f"APU Pick sidecar 讀取失敗：{exc}")
        return None
    if rows is None:
        return None
    out: dict[str, tuple[str, PickMetrics]] = {}
    for row in rows:
        sha = row.get("sha256")
        if sha:
            out[sha.lower()] = (row.get("file", ""), _entry(row))
    return out
