"""載入 Recipe（SPEC §7.2）。

- 套用設定：設定、濾鏡名稱對應、各對齊組的輸出設定，套用到目前加入的檔案。
- 完全重現：另外照 recipe 加入同樣的檔案、類型、校正配對與參考 frame；找不到或內容不同的檔案列出來。
0.1 的 recipe（apuphotons-recipe/1）只能套用設定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from .engine import RECIPE_SCHEMA, Settings
from .i18n import Msg
from .ingest import cached_sha256

SCHEMAS = (RECIPE_SCHEMA, "apuphotons-recipe/1")
# 跟這次的資料無關、不從 recipe 帶過來的設定
_NOT_APPLIED = ("preview", "gpu", "workers", "memory_mb")


class RecipeError(RuntimeError):
    pass


@dataclass
class LoadedRecipe:
    settings: Settings
    files: list[Path] = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reproducible: bool = True


def load_recipe(path: Path, reproduce: bool = False, base: Settings | None = None) -> LoadedRecipe:
    """base：目前的設定（效能相關的項目沿用它）。"""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecipeError(Msg("msg.recipe_unreadable", name=Path(path).name, error=exc)) from exc
    schema = doc.get("schema")
    if schema not in SCHEMAS:
        raise RecipeError(Msg("msg.recipe_schema", name=Path(path).name, schema=repr(schema)))
    base = base or Settings()
    known = {f.name for f in fields(Settings)}
    values = {k: v for k, v in (doc.get("settings") or {}).items() if k in known and k not in _NOT_APPLIED}
    if schema != RECIPE_SCHEMA:
        values = {k: v for k, v in values.items() if k not in ("reference",)}
    s = Settings(**{**{f.name: getattr(base, f.name) for f in fields(Settings)}, **values})
    if doc.get("filter_aliases"):
        s.filter_aliases = dict(doc["filter_aliases"])
    out = LoadedRecipe(settings=s, reproducible=schema == RECIPE_SCHEMA)
    if not reproduce:
        s.kinds, s.calib_overrides, s.reference = {}, {}, None
        return out
    if schema != RECIPE_SCHEMA:
        out.warnings.append(Msg("msg.recipe_v1_settings_only"))
        s.kinds, s.calib_overrides, s.reference = {}, {}, None
        return out
    for item in doc.get("inputs", []):
        p = Path(item["file"])
        if not p.is_file():
            out.warnings.append(Msg("msg.recipe_missing", name=p.name, path=p))
            continue
        size = item.get("size")
        sha = item.get("sha256")
        if (size is not None and p.stat().st_size != size) or (sha and cached_sha256(p) != sha):
            out.warnings.append(Msg("msg.recipe_changed", name=p.name))
        out.files.append(p)
    refs = [a.get("reference_frame") for a in doc.get("align_groups", []) if a.get("reference_frame")]
    s.reference = refs or None
    return out
