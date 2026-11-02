#!/usr/bin/env python3
"""Generator referentnog test dataseta (10 slika) za Fazu 1.

Slike su potpuno proceduralne i determinističke (fiksni seed-evi), pa dataset
moze da se regenerise bez mreze i bez autorskih prava. Svaka slika pokriva
jedan slucaj iz plana (portret, silueta, logo, tekst, line-art, kompleksna
pozadina, sitni detalji, niska/visoka rezolucija, transparentni PNG).

Pokretanje:
    uv run python scripts/generate_test_dataset.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "test_images"

FONT_CANDIDATES = [
    "/usr/share/fonts/gsfonts/NimbusRoman-Regular.otf",
    "/usr/share/fonts/gsfonts/NimbusRoman-Bold.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
]


# ---------------------------------------------------------------------------
# Pomocne funkcije
# ---------------------------------------------------------------------------

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Ucitava serifni font; ako ga nema, vraca PIL default font."""
    candidates = list(FONT_CANDIDATES)
    if bold:
        candidates = [c.replace("Regular", "Bold") for c in candidates] + candidates
    for path in candidates:
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def fractal_noise(w: int, h: int, *, seed: int, octaves: int = 5,
                  base: float = 6.0, persistence: float = 0.55) -> np.ndarray:
    """Fraktalni (Perlin-like) sum u opsegu [0, 1] na rezoluciji w x h."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp, total, scale = 1.0, 0.0, base
    for octave in range(octaves):
        gw = max(2, int(w / (scale * (2 ** octave))))
        gh = max(2, int(h / (scale * (2 ** octave))))
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        acc += layer * amp
        total += amp
        amp *= persistence
    return normalize(acc / total)


def add_grain(img: Image.Image, amount: float, seed: int) -> Image.Image:
    """Dodaje film-sko zrno na RGB slici."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, amount * 255.0, arr.shape[:2]).astype(np.float32)
    arr = np.clip(arr + noise[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def vignette(arr: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """Zatamni ivice (blagi studio/objektiv efekat)."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = np.clip(1.0 - strength * np.clip(r - 0.35, 0.0, None), 0.25, 1.0)
    return np.clip(arr.astype(np.float32) * mask[..., None], 0, 255)


def linear_gradient(w: int, h: int, top: tuple[int, int, int],
                    bottom: tuple[int, int, int], *, diagonal: bool = False) -> Image.Image:
    """Vertikalni (ili dijagonalni) gradijent."""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    if diagonal:
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        t = np.clip(0.6 * yy + 0.4 * xx, 0.0, 1.0)
    else:
        t = np.broadcast_to(yy, (h, w))
    top_a = np.array(top, dtype=np.float32)
    bottom_a = np.array(bottom, dtype=np.float32)
    arr = top_a[None, None, :] * (1 - t[..., None]) + bottom_a[None, None, :] * t[..., None]
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


class Canvas:
    """Supersampled crtacki canvas (koordinate u finalnim pikselima)."""

    def __init__(self, w: int, h: int, *, ss: int = 2,
                 bg: tuple[int, int, int] | Image.Image = (255, 255, 255)) -> None:
        self.w, self.h, self.ss = w, h, ss
        if isinstance(bg, Image.Image):
            base = bg.convert("RGB").resize((w * ss, h * ss), Image.LANCZOS)
        else:
            base = Image.new("RGB", (w * ss, h * ss), bg)
        self.base = base
        self.img = base.copy()
        self.draw = ImageDraw.Draw(self.img)

    def _s(self, v: float) -> float:
        return v * self.ss

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, *,
                fill=None, outline=None, width: float = 0) -> None:
        ss = self.ss
        self.draw.ellipse([(cx - rx) * ss, (cy - ry) * ss,
                           (cx + rx) * ss, (cy + ry) * ss],
                          fill=fill, outline=outline, width=int(round(width * ss)))

    def polygon(self, pts: list[tuple[float, float]], *, fill=None,
                outline=None, width: float = 0) -> None:
        ss = self.ss
        xy = [(x * ss, y * ss) for x, y in pts]
        self.draw.polygon(xy, fill=fill, outline=outline, width=int(round(width * ss)))

    def line(self, pts: list[tuple[float, float]], *, fill,
             width: float = 1.0) -> None:
        ss = self.ss
        xy = [(x * ss, y * ss) for x, y in pts]
        self.draw.line(xy, fill=fill, width=max(1, int(round(width * ss))), joint="curve")

    def text(self, xy: tuple[float, float], txt: str, font: ImageFont.FreeTypeFont,
             *, fill=(0, 0, 0), anchor: str | None = None) -> None:
        self.draw.text((xy[0] * self.ss, xy[1] * self.ss), txt, font=font,
                       fill=fill, anchor=anchor)

    def finish(self, *, blur: float = 0.0) -> Image.Image:
        img = self.img
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur * self.ss))
        return img.resize((self.w, self.h), Image.LANCZOS)


def rotated_ellipse(cx: float, cy: float, rx: float, ry: float,
                    angle_deg: float, n: int = 48) -> list[tuple[float, float]]:
    """Tacke rotirane elipse (za listove i latice)."""
    a = math.radians(angle_deg)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + x * math.cos(a) - y * math.sin(a),
                    cy + x * math.sin(a) + y * math.cos(a)))
    return pts


def bezier(p0, p1, p2, p3, n: int = 40) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------------------
# 1. Portret (fotografija)
# ---------------------------------------------------------------------------

def make_portrait(path: Path) -> None:
    w, h = 1200, 1500
    bg = linear_gradient(w, h, (52, 55, 66), (128, 126, 138), diagonal=True)
    noise = fractal_noise(w, h, seed=11, octaves=6, base=20)
    arr = np.asarray(bg, dtype=np.float32) + (noise[..., None] - 0.5) * 26.0
    bg = Image.fromarray(np.clip(vignette(arr, 0.5), 0, 255).astype(np.uint8), "RGB")

    c = Canvas(w, h, ss=2, bg=bg)
    skin = (221, 172, 138)
    hair = (46, 33, 27)
    jacket = (38, 44, 58)

    # ramena i trup
    c.polygon([(140, 1500), (210, 1235), (420, 1120), (780, 1120),
               (990, 1235), (1060, 1500)], fill=jacket)
    c.ellipse(600, 1520, 480, 300, fill=jacket)
    # vrat
    c.polygon([(505, 900), (695, 900), (720, 1150), (480, 1150)], fill=skin)
    c.ellipse(600, 1130, 130, 90, fill=(198, 148, 116))
    # kosa (iza glave)
    c.ellipse(600, 700, 275, 330, fill=hair)
    # glava
    c.ellipse(600, 760, 225, 290, fill=skin)
    # celo i vilica senke
    c.ellipse(600, 640, 205, 140, fill=(228, 180, 146))
    # usi
    c.ellipse(378, 790, 32, 55, fill=(204, 152, 120))
    c.ellipse(822, 790, 32, 55, fill=(204, 152, 120))
    # oci
    for sx in (505, 695):
        c.ellipse(sx, 760, 52, 30, fill=(246, 244, 240))
        c.ellipse(sx, 762, 26, 26, fill=(92, 118, 88))
        c.ellipse(sx, 762, 12, 13, fill=(20, 18, 16))
        c.ellipse(sx - 7, 753, 6, 6, fill=(250, 250, 250))
        c.line([(sx - 58, 722), (sx - 22, 708), (sx + 18, 712), (sx + 52, 728)],
               fill=(60, 42, 34), width=11)
    # nos
    c.line([(602, 780), (588, 842), (612, 858), (632, 850)], fill=(178, 126, 98), width=9)
    c.ellipse(578, 858, 10, 6, fill=(150, 100, 80))
    c.ellipse(624, 858, 10, 6, fill=(150, 100, 80))
    # usta
    c.ellipse(600, 928, 62, 16, fill=(150, 78, 78))
    c.ellipse(600, 918, 58, 12, fill=(176, 96, 92))
    # kosa preko cela (siske)
    c.polygon([(390, 640), (410, 470), (520, 430), (600, 470), (690, 425),
               (790, 480), (818, 650), (790, 600), (700, 590), (600, 610),
               (500, 595), (410, 610)], fill=hair)
    # brada senka
    c.ellipse(600, 1010, 120, 30, fill=(206, 156, 126))
    # jakna - reveri
    c.polygon([(500, 1140), (600, 1320), (700, 1140), (660, 1125),
               (600, 1180), (540, 1125)], fill=(58, 66, 86))

    img = c.finish(blur=1.1)
    img = add_grain(img, 0.022, seed=12)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 2. Silueta figure u pokretu
# ---------------------------------------------------------------------------

def make_silhouette(path: Path) -> None:
    w, h = 1000, 1400
    bg = linear_gradient(w, h, (247, 246, 242), (216, 214, 208))
    bg = Image.fromarray(np.clip(vignette(np.asarray(bg, dtype=np.float32), 0.2), 0, 255)
                         .astype(np.uint8), "RGB")
    c = Canvas(w, h, ss=3, bg=bg)
    ink = (18, 18, 20)

    # trkac: glava, trup, ruke, noge
    c.ellipse(505, 300, 88, 100, fill=ink)
    c.polygon([(430, 400), (575, 385), (625, 640), (560, 700), (470, 650)],
              fill=ink)
    # zadnja noga
    c.line([(500, 660), (395, 900), (300, 1040)], fill=ink, width=62)
    c.line([(300, 1040), (250, 1130)], fill=ink, width=40)
    # prednja noga
    c.line([(555, 670), (640, 900), (700, 1130)], fill=ink, width=64)
    c.line([(700, 1130), (760, 1180)], fill=ink, width=38)
    # zadnja ruka
    c.line([(470, 455), (330, 560), (250, 640)], fill=ink, width=46)
    # prednja ruka
    c.line([(570, 450), (720, 520), (800, 470)], fill=ink, width=48)
    # stopala
    c.ellipse(238, 1140, 40, 22, fill=ink)
    c.ellipse(772, 1185, 40, 22, fill=ink)
    # kosa/rep
    c.ellipse(455, 275, 46, 56, fill=ink)

    img = c.finish(blur=0.8)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 3. Logo (crno-beli znak, ostri uglovi)
# ---------------------------------------------------------------------------

def make_logo(path: Path) -> None:
    size = 900
    img = Image.new("RGB", (size, size), (255, 255, 255))
    c = Canvas(size, size, ss=3, bg=(255, 255, 255))
    cx = cy = size / 2
    ink = (10, 10, 10)

    # spoljni heksagon
    hexagon = [(cx + 330 * math.cos(math.radians(60 * i - 90)),
                cy + 330 * math.sin(math.radians(60 * i - 90))) for i in range(6)]
    c.polygon(hexagon, outline=ink, width=26)
    # unutrasnji heksagon
    hexagon2 = [(cx + 240 * math.cos(math.radians(60 * i - 90)),
                 cy + 240 * math.sin(math.radians(60 * i - 90))) for i in range(6)]
    c.polygon(hexagon2, outline=ink, width=10)
    # stilizovano slovo V
    c.polygon([(cx - 160, cy - 150), (cx - 70, cy - 150), (cx, cy + 40),
               (cx + 70, cy - 150), (cx + 160, cy - 150), (cx + 40, cy + 170),
               (cx - 40, cy + 170)], fill=ink)
    # strelica gore kroz V
    c.polygon([(cx - 46, cy + 130), (cx, cy + 250), (cx + 46, cy + 130)], fill=ink)
    c.polygon([(cx - 26, cy + 60), (cx + 26, cy + 60), (cx + 26, cy + 150),
               (cx - 26, cy + 150)], fill=ink)
    # tacke u uglovima (test ostrine malih elemenata)
    for ang in range(6):
        px = cx + 285 * math.cos(math.radians(60 * ang))
        py = cy + 285 * math.sin(math.radians(60 * ang))
        c.ellipse(px, py, 14, 14, fill=ink)

    c.finish().save(path, optimize=True)


# ---------------------------------------------------------------------------
# 4. Tekst / tipografija sa serifima
# ---------------------------------------------------------------------------

def make_text(path: Path) -> None:
    w, h = 1800, 700
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    font_big = load_font(230, bold=True)
    font_small = load_font(120)
    d.text((w / 2, 190), "GRAVURA 2026", font=font_big, fill=0, anchor="mm")
    d.text((w / 2, 450), "aegR8Q Oo Bb ffi", font=font_small, fill=0, anchor="mm")
    d.text((1500, 620), "Serif", font=load_font(64), fill=0, anchor="rm")
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.convert("RGB").save(path, optimize=True)


# ---------------------------------------------------------------------------
# 5. Line art (skica olovkom -> centerline)
# ---------------------------------------------------------------------------

def make_line_art(path: Path) -> None:
    w, h = 1200, 1200
    paper = fractal_noise(w, h, seed=31, octaves=6, base=3)
    base = np.stack([252 - paper * 12, 250 - paper * 12, 242 - paper * 16], axis=-1)
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    c = Canvas(w, h, ss=2, bg=bg)
    ink = (44, 40, 36)

    # rekurzivno drvo
    def branch(x: float, y: float, angle: float, length: float,
               width: float, depth: int) -> None:
        if depth == 0 or length < 22:
            return
        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)
        # blago zakrivljena grana
        mx = (x + x2) / 2 + length * 0.08 * math.sin(angle * 2.3 + depth)
        my = (y + y2) / 2 + length * 0.08 * math.cos(angle * 1.7 - depth)
        c.line([(x, y), (mx, my), (x2, y2)], fill=ink, width=max(1.6, width))
        rng = np.random.default_rng(depth * 97 + int(x))
        for _ in range(2):
            spread = rng.uniform(0.32, 0.62)
            branch(x2, y2, angle - spread, length * rng.uniform(0.62, 0.78),
                   width * 0.68, depth - 1)
            branch(x2, y2, angle + spread, length * rng.uniform(0.62, 0.78),
                   width * 0.68, depth - 1)

    branch(600, 1140, -math.pi / 2, 260, 11.0, 7)

    # par poteza olovke (tekstura)
    rng = np.random.default_rng(7)
    for _ in range(18):
        x = rng.uniform(60, 1140)
        y = rng.uniform(60, 1140)
        ln = rng.uniform(18, 46)
        a = rng.uniform(-0.4, 0.4)
        c.line([(x, y), (x + ln * math.cos(a), y + ln * math.sin(a))],
               fill=(96, 92, 86), width=1.4)

    img = c.finish(blur=0.5)
    img = add_grain(img, 0.012, seed=33)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 6. Kompleksna pozadina (objekat u prirodi)
# ---------------------------------------------------------------------------

def make_complex_scene(path: Path) -> None:
    w, h = 1400, 1000
    sky = linear_gradient(w, h, (96, 158, 214), (196, 222, 236))
    arr = np.asarray(sky, dtype=np.float32)
    # oblaci
    clouds = fractal_noise(w, h, seed=41, octaves=6, base=9)
    cloud_mask = np.clip((clouds - 0.52) * 5.0, 0, 1)
    arr = arr * (1 - cloud_mask[..., None] * 0.75) + 250 * cloud_mask[..., None] * 0.75
    # sunce
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sun = np.clip(1.0 - np.sqrt((xx - 1150) ** 2 + (yy - 180) ** 2) / 190.0, 0, 1)
    arr = arr * (1 - sun[..., None] * 0.5) + np.array([255, 246, 214]) * sun[..., None] * 0.5
    bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    c = Canvas(w, h, ss=2, bg=bg)
    # daleke planine
    c.polygon([(0, 620), (180, 380), (360, 560), (540, 330), (760, 590),
               (900, 430), (1080, 560), (1250, 360), (1400, 540), (1400, 700), (0, 700)],
              fill=(92, 110, 132))
    # sneg na vrhovima
    c.polygon([(480, 400), (540, 330), (600, 405), (560, 430), (540, 395), (510, 430)],
              fill=(240, 244, 250))
    # suma
    rng = np.random.default_rng(43)
    for x in range(-20, w + 40, 34):
        top = 640 + rng.integers(-30, 40)
        c.polygon([(x, 780), (x + 17, top), (x + 34, 780)], fill=(34, 62, 48))
    # jezero
    c.polygon([(0, 760), (w, 740), (w, 1000), (0, 1000)], fill=(56, 96, 118))
    for i in range(26):
        y = 770 + i * 9
        x0 = rng.uniform(-100, w * 0.6)
        c.line([(x0, y), (x0 + rng.uniform(120, 420), y)], fill=(120, 160, 180),
               width=1.6)
    # suma u prednjem planu (tamna)
    for x in range(-40, w + 60, 52):
        top = 700 + rng.integers(-40, 30)
        c.polygon([(x, 1000), (x + 26, top), (x + 52, 1000)], fill=(24, 40, 34))
    # crveni sator (subjekt)
    c.polygon([(560, 940), (700, 730), (840, 940)], fill=(178, 46, 38))
    c.polygon([(700, 730), (840, 940), (700, 940)], fill=(130, 32, 28))
    c.polygon([(680, 940), (700, 790), (720, 940)], fill=(54, 42, 36))
    # covek pored satora
    c.ellipse(900, 800, 14, 15, fill=(42, 36, 34))
    c.line([(900, 815), (900, 890)], fill=(42, 36, 34), width=16)
    c.line([(900, 835), (872, 872)], fill=(42, 36, 34), width=8)
    c.line([(900, 835), (928, 872)], fill=(42, 36, 34), width=8)
    c.line([(900, 888), (882, 940)], fill=(42, 36, 34), width=9)
    c.line([(900, 888), (920, 940)], fill=(42, 36, 34), width=9)
    # vatra
    c.ellipse(760, 920, 22, 10, fill=(235, 132, 40))

    img = c.finish(blur=0.7)
    img = add_grain(img, 0.018, seed=44)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 7. Sitni detalji (c lace / krosnja -> min_island_area filter)
# ---------------------------------------------------------------------------

def make_fine_details(path: Path) -> None:
    w, h = 1200, 1200
    c = Canvas(w, h, ss=2, bg=(255, 255, 255))
    ink = (16, 16, 16)
    rng = np.random.default_rng(51)

    # mreza cipke
    step = 46
    for x in range(40, w - 20, step):
        pts = [(x + 8 * math.sin(y / 60.0), y) for y in range(30, h - 10, 8)]
        c.line(pts, fill=ink, width=2)
    for y in range(40, h - 20, step):
        pts = [(x, y + 8 * math.cos(x / 70.0)) for x in range(20, w - 10, 8)]
        c.line(pts, fill=ink, width=2)
    # cvetni motivi
    for cy in range(140, h - 100, 230):
        for cx in range(140, w - 100, 230):
            for k in range(8):
                a = 2 * math.pi * k / 8
                px = cx + 34 * math.cos(a)
                py = cy + 34 * math.sin(a)
                c.ellipse(px, py, 11, 7, outline=ink, width=2.2)
            c.ellipse(cx, cy, 12, 12, outline=ink, width=2.2)
    # mnogo sitnih tacaka (treba ih filter ukloniti)
    for _ in range(420):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.8, 2.4)
        c.ellipse(x, y, r, r, fill=ink)

    img = c.finish(blur=0.4)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 8. Niska rezolucija (300x300)
# ---------------------------------------------------------------------------

def make_low_res(path: Path) -> None:
    size = 300
    c = Canvas(size, size, ss=2, bg=(250, 249, 246))
    ink = (20, 20, 22)
    cx = cy = size / 2
    c.polygon([(cx - 105, cy - 70), (cx + 105, cy - 70), (cx + 105, cy + 60),
               (cx, cy + 118), (cx - 105, cy + 60)], outline=ink, width=9)
    c.polygon([(cx - 58, cy - 22), (cx, cy + 30), (cx + 58, cy - 22),
               (cx + 58, cy - 2), (cx, cy + 62), (cx - 58, cy - 2)], fill=ink)
    c.ellipse(cx, cy - 34, 11, 11, fill=ink)
    img = c.finish(blur=0.6)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 9. Visoka rezolucija (4K)
# ---------------------------------------------------------------------------

def make_high_res(path: Path) -> None:
    w, h = 3840, 2160
    sky = linear_gradient(w, h, (20, 30, 58), (240, 148, 96))
    arr = np.asarray(sky, dtype=np.float32)
    # zvezde
    rng = np.random.default_rng(61)
    ys = rng.integers(0, h // 2, 900)
    xs = rng.integers(0, w, 900)
    arr[ys, xs] = 250
    # mesec
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    moon = np.clip(1.0 - np.sqrt((xx - 3000) ** 2 + (yy - 420) ** 2) / 150.0, 0, 1)
    arr = arr * (1 - moon[..., None] * 0.85) + np.array([250, 244, 220]) * moon[..., None] * 0.85
    bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    c = Canvas(w, h, ss=1, bg=bg)
    # planine u tri plana
    c.polygon([(0, 1500), (520, 980), (1080, 1420), (1600, 900), (2200, 1460),
               (2800, 1020), (3400, 1400), (3840, 1120), (3840, 2160), (0, 2160)],
              fill=(46, 48, 72))
    c.polygon([(0, 1720), (700, 1320), (1500, 1740), (2300, 1280), (3100, 1700),
               (3840, 1420), (3840, 2160), (0, 2160)], fill=(30, 30, 48))
    # linije polja / perspektiva
    for i in range(-20, 40):
        x0 = i * 160
        c.line([(x0, 2160), (x0 + 900, 1600)], fill=(58, 58, 84), width=2)
    for j in range(14):
        y = 1620 + j * (540 / 14)
        c.line([(0, y), (3840, y)], fill=(58, 58, 84), width=2)
    # detalji: kuca sa prozorima
    c.polygon([(2600, 1700), (2600, 1580), (2700, 1520), (2800, 1580),
               (2800, 1700)], fill=(18, 18, 26))
    for wx in range(2625, 2790, 45):
        for wy in range(1600, 1680, 34):
            c.ellipse(wx, wy, 8, 8, fill=(255, 208, 120))
    # ograda
    for x in range(0, w, 46):
        c.line([(x, 1830), (x, 1960)], fill=(22, 22, 34), width=5)
    c.line([(0, 1850), (3840, 1850)], fill=(22, 22, 34), width=6)

    img = c.finish(blur=0.4)
    img = add_grain(img, 0.006, seed=62)
    img.save(path, optimize=True)


# ---------------------------------------------------------------------------
# 10. Transparentni PNG (stiker sa alfa kanalom)
# ---------------------------------------------------------------------------

def make_transparent_sticker(path: Path) -> None:
    w = h = 1000
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    c = Canvas(w, h, ss=2, bg=(255, 255, 255))
    c.img = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
    c.draw = ImageDraw.Draw(c.img)
    cx = cy = 500
    ink = (20, 20, 24)
    accent = (214, 62, 48)

    # zvezdasti Bedz
    bad = [(cx + 330 * math.cos(math.radians(45 * i - 90)),
            cy + 330 * math.sin(math.radians(45 * i - 90))) for i in range(8)]
    c.polygon(bad, fill=(255, 255, 255), outline=ink, width=22)
    # unutrasnji prsten
    c.ellipse(cx, cy, 250, 250, outline=accent, width=16)
    c.ellipse(cx, cy, 205, 205, outline=ink, width=6)
    # natpis
    font = load_font(150, bold=True)
    c.text((cx, cy - 46), "LASER", font, fill=ink, anchor="mm")
    font2 = load_font(58)
    c.text((cx, cy + 92), "CUT  •  ENGRAVE", font2, fill=accent, anchor="mm")
    # zvezdice
    for ang in (0, 90, 180, 270):
        x = cx + 285 * math.cos(math.radians(ang))
        y = cy + 285 * math.sin(math.radians(ang))
        c.ellipse(x, y, 9, 9, fill=accent)

    rgb = c.finish(blur=0.5)
    # alfa kanal: oblik bedza (spolja providno)
    alpha = Image.new("L", (w * 2, h * 2), 0)
    ad = ImageDraw.Draw(alpha)
    ad.polygon([(x * 2, y * 2) for x, y in bad], fill=255)
    alpha = alpha.resize((w, h), Image.LANCZOS)
    sticker = rgb.convert("RGBA")
    sticker.putalpha(alpha)
    sticker.save(path, optimize=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CASES = [
    ("01_portret_foto.png", "Portret (foto)", make_portrait,
     "rembg maska + XDoG/threshold detalji lica"),
    ("02_silueta_figura.png", "Silueta", make_silhouette,
     "Stencil mod + tacnost Cut offseta"),
    ("03_logo_bw.png", "Logo (rasterski)", make_logo,
     "Ostrina uglova u vektorizaciji"),
    ("04_tekst_serif.png", "Tekst / tipografija", make_text,
     "Ocuvanje unutrasnjih supljina slova"),
    ("05_crtez_line_art.png", "Crtez (line art)", make_line_art,
     "Single-line skeletonizacija i spur pruning"),
    ("06_kompleksna_pozadina.png", "Kompleksna pozadina", make_complex_scene,
     "Otpornost rembg segmentacije"),
    ("07_sitni_detalji.png", "Sitni detalji", make_fine_details,
     "min_island_area filter (sprecavanje hiljada tackica)"),
    ("08_niska_rezolucija_300.png", "Niska rezolucija", make_low_res,
     "Upscale / bilateralno glacanje pre obrade"),
    ("09_visoka_rezolucija_4k.png", "Visoka rezolucija", make_high_res,
     "Automatski downscale i memorijska stabilnost"),
    ("10_transparentni_stiker.png", "Transparentni PNG", make_transparent_sticker,
     "Direktno citanje postojeceg alfa kanala"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, category, factory, purpose in CASES:
        target = OUT_DIR / filename
        factory(target)
        with Image.open(target) as im:
            manifest.append({
                "file": filename,
                "category": category,
                "purpose": purpose,
                "width": im.width,
                "height": im.height,
                "mode": im.mode,
                "size_kb": round(target.stat().st_size / 1024, 1),
            })
        print(f"  OK  {filename:32s} {manifest[-1]['width']}x{manifest[-1]['height']} "
              f"{manifest[-1]['size_kb']:8.1f} KB")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total_mb = sum(m["size_kb"] for m in manifest) / 1024
    print(f"\nUkupno: {len(manifest)} slika, {total_mb:.1f} MB -> {OUT_DIR}")


if __name__ == "__main__":
    main()
