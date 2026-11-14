"""新版本提醒：版本比較、挑這台電腦對應的 zip、一天最多查一次。不連網路。"""

from apu_photons import update
from apu_photons.settings import save_settings

ZIPS = ("APUPhotons-{v}-win64.zip", "APUPhotons-{v}-macos-arm64.zip", "APUPhotons-{v}-macos-x86_64.zip")


def _release(tag, names=ZIPS):
    v = tag.lstrip("v")
    return {"tag_name": tag, "html_url": f"https://github.com/ApuHuang/apu-photons/releases/tag/{tag}",
            "assets": [{"name": n.format(v=v), "browser_download_url": f"https://github.com/dl/{n.format(v=v)}"}
                       for n in names]}


def test_parse_version():
    assert update.parse_version("v0.2.3") == (0, 2, 3) == update.parse_version(" 0.2.3 ")
    assert update.parse_version("0.3.0rc1") is None and update.parse_version("nightly") is None


def test_newer_picks_the_zip_for_this_computer(monkeypatch):
    monkeypatch.setattr(update, "platform_tag", lambda: "macos-arm64")
    got = update.newer(_release("v0.2.3"), "0.2.2")
    assert got == update.Update("0.2.3", "https://github.com/dl/APUPhotons-0.2.3-macos-arm64.zip",
                                "APUPhotons-0.2.3-macos-arm64.zip")
    assert update.newer(_release("v0.2.2"), "0.2.2") is None          # 一樣
    assert update.newer(_release("v0.2.1"), "0.2.2") is None          # 比較舊
    assert update.newer(_release("v0.10.0"), "0.9.9").version == "0.10.0"   # 照數字比，不是照字串比
    page_only = update.newer(_release("v0.3.0", names=()), "0.2.2")   # 還沒打包好：給 Release 頁面
    assert page_only.url.endswith("/releases/tag/v0.3.0") and page_only.file == ""
    for bad in (None, {}, {"tag_name": "nightly"}, {"tag_name": "v0.3.0rc1"}):
        assert update.newer(bad, "0.2.2") is None


def test_github_is_asked_at_most_once_a_day(tmp_path, monkeypatch):
    monkeypatch.setenv("APU_PHOTONS_SETTINGS", str(tmp_path / "settings.json"))
    assert update.enabled() and update.due(now=1000.0) and update.cached() is None
    update.remember(_release("v0.2.3") | {"body": "很長的說明"}, now=1000.0)
    assert not update.due(now=1000.0 + 3600)
    assert update.due(now=1000.0 + update.INTERVAL + 1)
    assert update.cached()["tag_name"] == "v0.2.3" and "body" not in update.cached()   # 只存需要的欄位
    save_settings(check_updates=False)
    assert not update.enabled()
