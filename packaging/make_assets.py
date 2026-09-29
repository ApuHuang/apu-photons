"""產生程式圖示與 Logo（與 APU Pick 同一家族：深藍圓角底、帶繞射芒的星點）：

- src/apu_photons/assets/app.ico      視窗與 exe 圖示：三層錯開的 frame 疊成一顆亮星
- src/apu_photons/assets/icon_128.png 視窗 Logo 用
- docs/logo.png                       README 最上方的 Logo 橫幅（APU Photons + 副標）

python packaging/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "apu_photons" / "assets"
DOCS = ROOT / "docs"
SIZE = 1024
NAVY = (16, 28, 58, 255)
NAME = "APU Photons"
SUBTITLE = "Astrophotography Photons Utility"


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE * 0.2, fill=NAVY)

    # 三張錯開的 frame（疊圖），由後往前越來越亮
    for i, (dx, dy, alpha) in enumerate(((-120, -120, 70), (-60, -60, 120), (0, 0, 190))):
        layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        x0, y0 = 0.30 * SIZE + dx, 0.30 * SIZE + dy
        x1, y1 = 0.80 * SIZE + dx, 0.80 * SIZE + dy
        ImageDraw.Draw(layer).rounded_rectangle((x0, y0, x1, y1), radius=SIZE * 0.05,
                                                outline=(150, 190, 255, alpha), width=int(SIZE * 0.018))
        img.alpha_composite(layer)

    c = 0.55 * SIZE  # 星點落在最前面那張 frame 的中心
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((c - 200, c - 200, c + 200, c + 200), fill=(120, 170, 255, 120))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(80)))

    spike = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spike)
    length, width = SIZE * 0.30, SIZE * 0.03
    sd.polygon([(c, c - length), (c + width, c), (c, c + length), (c - width, c)], fill=(235, 242, 255, 255))
    sd.polygon([(c - length, c), (c, c - width), (c + length, c), (c, c + width)], fill=(235, 242, 255, 255))
    img.alpha_composite(spike.filter(ImageFilter.GaussianBlur(3)))

    core = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(core).ellipse((c - 80, c - 80, c + 80, c + 80), fill=(255, 255, 255, 255))
    img.alpha_composite(core.filter(ImageFilter.GaussianBlur(10)))

    # 裁掉超出圓角底的部分
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=SIZE * 0.2, fill=255)
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (SIZE, SIZE), 0), mask))
    return img


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in (Path("C:/Windows/Fonts") / n for n in names):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def draw_logo(icon: Image.Image) -> Image.Image:
    w, h = 1480, 400
    scale = 2
    img = Image.new("RGBA", (w * scale, h * scale), NAVY)
    icon_size = 280 * scale
    img.alpha_composite(icon.resize((icon_size, icon_size), Image.LANCZOS), (60 * scale, 60 * scale))
    d = ImageDraw.Draw(img)
    title = _font(["segoeuib.ttf"], 150 * scale)
    sub = _font(["seguisb.ttf", "segoeui.ttf"], 58 * scale)
    x = (60 + 280 + 60) * scale
    d.text((x, 70 * scale), NAME, font=title, fill=(255, 255, 255, 255))
    d.text((x + 6 * scale, 262 * scale), SUBTITLE, font=sub, fill=(174, 187, 214, 255))
    return img.resize((w, h), Image.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    icon = draw_icon()
    icon.resize((256, 256), Image.LANCZOS).save(
        ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    icon.resize((128, 128), Image.LANCZOS).save(ASSETS / "icon_128.png", optimize=True)
    draw_logo(icon).convert("RGB").save(DOCS / "logo.png", optimize=True)
    print(f"已產生 {ASSETS / 'app.ico'}、{ASSETS / 'icon_128.png'}、{DOCS / 'logo.png'}")


if __name__ == "__main__":
    main()
