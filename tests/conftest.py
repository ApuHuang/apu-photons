from datetime import timedelta

import pytest


@pytest.fixture(autouse=True)
def _fixed_timezone(monkeypatch):
    """沒有經度時觀測夜依電腦時區；測試固定用 UTC，不依賴跑測試那台的時區。"""
    from apu_photons import ingest
    monkeypatch.setattr(ingest, "_local_offset", lambda t: timedelta(0))
