"""中英文：引擎的警告、紀錄、錯誤與命令列都要能切換。"""

import re

import pytest

from apu_photons import cli
from apu_photons.engine import Settings, run
from apu_photons.i18n import _EN, _ZH, Msg, set_language

from .test_pipeline import _make

CJK = re.compile(r"[一-鿿]")


@pytest.fixture(autouse=True)
def reset_language():
    yield
    set_language("zh")


def test_catalogs_complete():
    assert set(_ZH) == set(_EN)
    # 英文不能混進中文
    assert not [k for k, v in _EN.items() if CJK.search(v)]


def test_msg_follows_current_language():
    m = Msg("msg.reg_failed", name="a.fits", error=Msg("msg.reg.residual", rms=1.5))
    set_language("zh")
    assert str(m) == "a.fits: 對齊失敗（對齊殘差 1.50 px 過大）"
    set_language("en")
    assert str(m) == "a.fits: registration failed (residual 1.50 px is too large)"


def test_engine_in_english(tmp_path):
    light, inputs = _make(tmp_path, with_cal=False)
    (light / "broken.fits").write_bytes(b"not a fits file")
    set_language("en")
    res = run(inputs, tmp_path / "m", Settings(drizzle=2, workers=2), echo=None)
    texts = [str(w) for w in res.warnings]
    assert any("no flat" in t for t in texts) and any("could not be read" in t for t in texts)
    assert not [t for t in texts if CJK.search(t.replace("broken.fits", ""))]
    log = res.log.read_text(encoding="utf-8")
    assert "Reference frame for" in log and not CJK.search(log.split("\n", 1)[1])
    # 換回中文：已經產生的警告跟著換
    set_language("zh")
    assert any("沒有 flat" in str(w) for w in res.warnings)


def test_cli_english(tmp_path, capsys):
    light, _inputs = _make(tmp_path, with_cal=False)
    assert cli.main(["--lang", "en", "stack", str(light), "-o", str(tmp_path / "c"), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "Integrated 8/8 frames" in out and "Recipe:" in out
    assert not CJK.search(out.replace(str(tmp_path), ""))
