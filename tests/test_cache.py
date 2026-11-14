"""快取管理：大小、清除、中斷留下的暫存檔、磁碟空間檢查。"""

import os
import tempfile
import time

import pytest

from apu_photons import cache as cache_mod
from apu_photons import cli
from apu_photons.cache import cache_size, clean_stale, clear_cache, require_space
from apu_photons.engine import Settings, run

from .test_pipeline import _make


def test_size_and_clear_keep_foreign_files(tmp_path):
    cache = tmp_path / "cache"
    (cache / "calibrated").mkdir(parents=True)
    (cache / "groups" / "a" / "rejection").mkdir(parents=True)
    (cache / "calibrated" / "x.npy").write_bytes(b"0" * 1000)
    (cache / "groups" / "a" / "rejection" / "y.npy").write_bytes(b"0" * 500)
    (cache / "mine.txt").write_text("keep")           # --cache 指到使用者自己的資料夾：別的東西不能刪
    assert cache_size(cache) == 1500
    assert clear_cache(cache) == 1500
    assert cache_size(cache) == 0 and (cache / "mine.txt").is_file()
    assert not (cache / "calibrated").exists()
    assert cache_size(tmp_path / "missing") == 0 and clear_cache(tmp_path / "missing") == 0


def test_clear_removes_empty_cache_folder(tmp_path):
    cache = tmp_path / ".photons_cache"
    (cache / "masters").mkdir(parents=True)
    (cache / "masters" / "m.fits").write_bytes(b"0" * 10)
    clear_cache(cache)
    assert not cache.exists()


def test_clean_stale_only_old_files(tmp_path, monkeypatch):
    sys_tmp = tmp_path / "systmp"
    sys_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))
    cache = tmp_path / "cache"
    (cache / "tmp").mkdir(parents=True)
    old = time.time() - 2 * cache_mod.STALE_SECONDS
    files = {"old": cache / "tmp" / "combine_old.npy", "new": cache / "tmp" / "combine_new.npy",
             "legacy": sys_tmp / "apu_photons_123.npy", "other": sys_tmp / "something_else.npy"}
    for name, p in files.items():
        p.write_bytes(b"0")
        if name != "new":
            os.utime(p, (old, old))
    clean_stale(cache)
    assert not files["old"].exists() and not files["legacy"].exists()
    assert files["new"].exists() and files["other"].exists()


def test_require_space(tmp_path, monkeypatch):
    require_space(tmp_path / "not" / "yet", 1, "msg.space.calibrated")  # 資料夾還不存在：看上層
    monkeypatch.setattr(cache_mod, "free_bytes", lambda _p: 10 * 2 ** 20)
    with pytest.raises(RuntimeError) as err:
        require_space(tmp_path, 5 * 2 ** 30, "msg.space.calibrated")
    assert "5.0 GB" in str(err.value) and "10 MB" in str(err.value)
    require_space(tmp_path, 0, "msg.space.calibrated")


def test_run_keeps_temp_files_in_cache_and_cleans_up(tmp_path, monkeypatch):
    sys_tmp = tmp_path / "systmp"
    sys_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))
    _light, inputs = _make(tmp_path)
    out = tmp_path / "out"
    res = run(inputs, out, Settings(workers=1, gpu="cpu"), echo=None)
    cache = out / ".photons_cache"
    assert list((cache / "masters").glob("master_*.fits"))       # 有建 master
    assert not list((cache / "tmp").iterdir())                    # 暫存檔用完就刪
    assert not list(sys_tmp.glob("*.npy"))                        # 不放系統暫存資料夾
    assert cache_size(cache) > 0
    assert cli.main(["clear-cache", "-o", str(out)]) == 0
    assert not cache.exists() and res.output.is_file()


def test_run_stops_early_when_disk_is_full(tmp_path, monkeypatch):
    _light, inputs = _make(tmp_path, with_cal=False)
    monkeypatch.setattr(cache_mod, "free_bytes", lambda _p: 0)
    with pytest.raises(RuntimeError) as err:
        run(inputs, tmp_path / "out", Settings(workers=1, gpu="cpu"), echo=None)
    assert "磁碟空間不足" in str(err.value)
    assert not list((tmp_path / "out" / ".photons_cache" / "calibrated").glob("*.npy"))
