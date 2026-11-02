"""Geometrijski model vektorskih putanja u milimetrima.

Sve koordinate u projektu su u milimetrima (1 SVG jedinica = 1 mm).
Rasterske koordinate (pikseli, y nadole) se skaliraju u mm na ulazu u model,
tako da je y osa i u SVG-u nadole - sto je identicno koordinatnom sistemu
LightBurn platna.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence, Union

# Metrika laser sapa (0.1 mm) - koristi se kao podrazumevana debljina linije.
LASER_STROKE_MM = 0.1

# LightBurn paleta slojeva
COLOR_ENGRAVE = "#000000"   # LightBurn C00
COLOR_DETAIL = "#0000FF"    # LightBurn C01
COLOR_CUT = "#FF0000"       # LightBurn C02


@dataclass(frozen=True, slots=True)
class Point:
    """Tacka u milimetrima."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "Point":
        return Point(self.x * factor, self.y * factor)

    def translated(self, dx: float, dy: float) -> "Point":
        return Point(self.x + dx, self.y + dy)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Line:
    """Pravolinijski segment."""

    start: Point
    end: Point

    def point_at(self, t: float) -> Point:
        return Point(
            self.start.x + (self.end.x - self.start.x) * t,
            self.start.y + (self.end.y - self.start.y) * t,
        )

    def flatten(self, tolerance: float = 0.05) -> list[Point]:  # noqa: ARG002
        return [self.start, self.end]

    def length(self) -> float:
        return self.start.distance_to(self.end)

    def bbox(self) -> "BBox":
        return BBox(
            min(self.start.x, self.end.x),
            min(self.start.y, self.end.y),
            max(self.start.x, self.end.x),
            max(self.start.y, self.end.y),
        )


@dataclass(frozen=True, slots=True)
class CubicBezier:
    """Kubna Bezier kriva."""

    start: Point
    control1: Point
    control2: Point
    end: Point

    def point_at(self, t: float) -> Point:
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3.0 * mt * mt * t
        c = 3.0 * mt * t * t
        d = t * t * t
        return Point(
            a * self.start.x + b * self.control1.x + c * self.control2.x + d * self.end.x,
            a * self.start.y + b * self.control1.y + c * self.control2.y + d * self.end.y,
        )

    def flatten(self, tolerance: float = 0.05) -> list[Point]:
        """Adaptivna de Casteljau rekurzija."""
        points: list[Point] = [self.start]
        _flatten_cubic(self, tolerance, points, depth=0)
        points.append(self.end)
        return points

    def length(self, tolerance: float = 0.01) -> float:
        pts = self.flatten(tolerance)
        return sum(a.distance_to(b) for a, b in zip(pts, pts[1:]))

    def bbox(self) -> "BBox":
        # Kontrolne tacke sadrze konveksnu ljusku krive, pa su sigurne granice.
        pts = [self.start, self.control1, self.control2, self.end]
        return BBox.from_points(pts)


Segment = Union[Line, CubicBezier]


@dataclass(frozen=True, slots=True)
class PathStyle:
    """Stil putanje koji se direktno preslikava u LightBurn sloj."""

    fill: str | None = None
    stroke: str | None = None
    stroke_width: float = LASER_STROKE_MM
    layer_name: str = "00_ENGRAVE"

    def with_color(self, color: str | None) -> "PathStyle":
        return PathStyle(fill=self.fill, stroke=color, stroke_width=self.stroke_width,
                         layer_name=self.layer_name)


# Gotovi stilovi po LightBurn konvenciji
STYLE_ENGRAVE_FILL = PathStyle(fill=COLOR_ENGRAVE, stroke=None, layer_name="00_ENGRAVE")
STYLE_ENGRAVE_LINE = PathStyle(fill=None, stroke=COLOR_ENGRAVE, layer_name="00_ENGRAVE")
STYLE_DETAIL = PathStyle(fill=None, stroke=COLOR_DETAIL, layer_name="01_DETAIL")
STYLE_CUT = PathStyle(fill=None, stroke=COLOR_CUT, layer_name="02_CUT")


@dataclass(frozen=True, slots=True)
class BBox:
    """Ogranicavajuci pravougaonik."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point:
        return Point((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def expanded(self, margin: float) -> "BBox":
        return BBox(self.min_x - margin, self.min_y - margin,
                    self.max_x + margin, self.max_y + margin)

    def contains(self, point: Point) -> bool:
        return (self.min_x <= point.x <= self.max_x
                and self.min_y <= point.y <= self.max_y)

    @classmethod
    def from_points(cls, points: Sequence[Point]) -> "BBox":
        if not points:
            raise ValueError("BBox.from_points: prazna lista tacaka")
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))


@dataclass
class VectorPath:
    """Jedna laserska putanja: niz segmenata u milimetrima."""

    segments: list[Segment] = field(default_factory=list)
    closed: bool = False
    style: PathStyle = STYLE_ENGRAVE_LINE
    meta: dict = field(default_factory=dict)

    # -- osnovne informacije -------------------------------------------------
    @property
    def start(self) -> Point:
        if not self.segments:
            raise ValueError("Prazna putanja nema pocetnu tacku")
        return self.segments[0].start

    @property
    def end(self) -> Point:
        if not self.segments:
            raise ValueError("Prazna putanja nema krajnju tacku")
        return self.segments[-1].end

    def is_empty(self) -> bool:
        return not self.segments

    def node_count(self) -> int:
        """Broj kontrolnih tacaka (cvorova) koje LightBurn prikazuje."""
        count = 1
        for seg in self.segments:
            count += 3 if isinstance(seg, CubicBezier) else 1
        return count

    # -- geometrija ----------------------------------------------------------
    def flatten(self, tolerance: float = 0.05) -> list[Point]:
        if not self.segments:
            return []
        points: list[Point] = [self.segments[0].start]
        for seg in self.segments:
            points.extend(seg.flatten(tolerance)[1:])
        return points

    def length(self, tolerance: float = 0.01) -> float:
        return sum(seg.length() if isinstance(seg, Line) else seg.length(tolerance)
                   for seg in self.segments)

    def bbox(self) -> BBox:
        if not self.segments:
            raise ValueError("Prazna putanja nema bbox")
        box = self.segments[0].bbox()
        for seg in self.segments[1:]:
            box = box.union(seg.bbox())
        return box

    def transformed(self, scale: float, dx: float = 0.0, dy: float = 0.0,
                    *, style: PathStyle | None = None) -> "VectorPath":
        """Uniformno skaliranje i translacija svih tacaka."""

        def tx(p: Point) -> Point:
            return Point(p.x * scale + dx, p.y * scale + dy)

        segments: list[Segment] = []
        for seg in self.segments:
            if isinstance(seg, Line):
                segments.append(Line(tx(seg.start), tx(seg.end)))
            else:
                segments.append(CubicBezier(tx(seg.start), tx(seg.control1),
                                            tx(seg.control2), tx(seg.end)))
        return VectorPath(segments, self.closed, style or self.style, dict(self.meta))

    def affine(self, matrix: tuple[float, float, float, float, float, float],
               *, style: PathStyle | None = None) -> "VectorPath":
        """Primena 2D afine transformacije (a, b, c, d, e, f)."""
        a, b, c, d, e, f = matrix

        def tx(p: Point) -> Point:
            return Point(a * p.x + c * p.y + e, b * p.x + d * p.y + f)

        segments: list[Segment] = []
        for seg in self.segments:
            if isinstance(seg, Line):
                segments.append(Line(tx(seg.start), tx(seg.end)))
            else:
                segments.append(CubicBezier(tx(seg.start), tx(seg.control1),
                                            tx(seg.control2), tx(seg.end)))
        return VectorPath(segments, self.closed, style or self.style, dict(self.meta))

    def reversed(self) -> "VectorPath":
        segments: list[Segment] = []
        for seg in reversed(self.segments):
            if isinstance(seg, Line):
                segments.append(Line(seg.end, seg.start))
            else:
                segments.append(CubicBezier(seg.end, seg.control2, seg.control1, seg.start))
        return VectorPath(segments, self.closed, self.style, dict(self.meta))

    def as_polyline(self, tolerance: float = 0.05) -> list[Point]:
        return self.flatten(tolerance)


# ---------------------------------------------------------------------------
# Konstruktori
# ---------------------------------------------------------------------------

def path_from_points(points: Iterable[Point], *, closed: bool = False,
                     style: PathStyle = STYLE_ENGRAVE_LINE,
                     meta: dict | None = None) -> VectorPath:
    pts = list(points)
    if len(pts) < 2:
        return VectorPath([], closed=False, style=style, meta=meta or {})
    segments: list[Segment] = [Line(a, b) for a, b in zip(pts, pts[1:])]
    if closed and pts[0].distance_to(pts[-1]) > 1e-9:
        segments.append(Line(pts[-1], pts[0]))
    return VectorPath(segments, closed=closed, style=style, meta=meta or {})


def path_from_coords(coords: Sequence[Sequence[float]], *, closed: bool = False,
                     style: PathStyle = STYLE_ENGRAVE_LINE,
                     meta: dict | None = None) -> VectorPath:
    """Iz niza (x, y) parova (npr. iz OpenCV konture)."""
    return path_from_points((Point(float(x), float(y)) for x, y in coords),
                            closed=closed, style=style, meta=meta)


# ---------------------------------------------------------------------------
# BBox i statistika
# ---------------------------------------------------------------------------

def paths_bbox(paths: Iterable[VectorPath]) -> BBox | None:
    box: BBox | None = None
    for path in paths:
        if path.is_empty():
            continue
        try:
            pb = path.bbox()
        except ValueError:
            continue
        box = pb if box is None else box.union(pb)
    return box


def paths_length(paths: Iterable[VectorPath], tolerance: float = 0.01) -> float:
    total = 0.0
    for path in paths:
        total += path.length(tolerance)
    return total


@dataclass
class PathStats:
    path_count: int = 0
    node_count: int = 0
    total_length_m: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    stroke_length_m: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path_count": self.path_count,
            "node_count": self.node_count,
            "total_length_m": round(self.total_length_m, 3),
            "width_mm": round(self.width_mm, 3),
            "height_mm": round(self.height_mm, 3),
            "stroke_length_m": {k: round(v, 3) for k, v in self.stroke_length_m.items()},
        }


def compute_stats(paths: Sequence[VectorPath], width_mm: float,
                  height_mm: float) -> PathStats:
    stats = PathStats(path_count=len(paths), width_mm=width_mm, height_mm=height_mm)
    for path in paths:
        stats.node_count += path.node_count()
        length_mm = path.length()
        stats.total_length_m += length_mm / 1000.0
        color = path.style.stroke or path.style.fill or "none"
        stats.stroke_length_m[color] = stats.stroke_length_m.get(color, 0.0) + length_mm / 1000.0
    return stats


# ---------------------------------------------------------------------------
# Formatiranje brojeva i SVG path "d" stringa
# ---------------------------------------------------------------------------

def format_number(value: float, decimals: int = 3) -> str:
    """Deterministicki zapis broja bez eksponenta i bez suvisnih nula."""
    if not math.isfinite(value):
        raise ValueError(f"Nekonacan broj u izlazu: {value!r}")
    if abs(value) < 0.5 * 10 ** (-decimals):
        return "0"
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _fmt_pair(point: Point, decimals: int) -> str:
    return f"{format_number(point.x, decimals)} {format_number(point.y, decimals)}"


def path_to_d(path: VectorPath, *, decimals: int = 3) -> str:
    """Serijalizacija putanje u SVG 'd' atribut (M/L/C/Z)."""
    if path.is_empty():
        return ""
    parts: list[str] = [f"M{_fmt_pair(path.start, decimals)}"]
    for seg in path.segments:
        if isinstance(seg, Line):
            parts.append(f"L{_fmt_pair(seg.end, decimals)}")
        else:
            parts.append(
                f"C{_fmt_pair(seg.control1, decimals)} "
                f"{_fmt_pair(seg.control2, decimals)} "
                f"{_fmt_pair(seg.end, decimals)}"
            )
    if path.closed:
        parts.append("Z")
    return "".join(parts)


def points_to_d(points: Sequence[Point], *, closed: bool = False,
                decimals: int = 3) -> str:
    if not points:
        return ""
    parts: list[str] = [f"M{_fmt_pair(points[0], decimals)}"]
    parts.extend(f"L{_fmt_pair(p, decimals)}" for p in points[1:])
    if closed:
        parts.append("Z")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Parser SVG "d" atributa (potreban za potrace izlaz)
# ---------------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_COMMAND_RE = re.compile(rf"([MmLlHhVvCcSsQqTtZz])|({_NUMBER})")


class SvgPathParseError(ValueError):
    """Neispravan SVG path string."""


def _tokenize_d(d: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(d):
        char = d[position]
        if char in " \t\r\n,":
            position += 1
            continue
        match = _COMMAND_RE.match(d, position)
        if not match:
            raise SvgPathParseError(f"Neocekivan karakter na poziciji {position}: {d[position:position + 20]!r}")
        kind = "cmd" if match.group(1) else "num"
        tokens.append((kind, match.group(1) or match.group(2)))
        position = match.end()
    return tokens


def parse_path_d(d: str) -> list[VectorPath]:
    """Parsira SVG path 'd' u listu putanja (apsolutne koordinate, y nadole).

    Podrzava M, L, H, V, C, S, Q, T, Z (apsolutne i relativne). Lucne
    komande (A) nisu podrzane jer ih potrace ne emituje.
    """
    tokens = _tokenize_d(d)
    index = 0
    paths: list[VectorPath] = []
    current = Point(0.0, 0.0)
    subpath_start = Point(0.0, 0.0)
    segments: list[Segment] = []
    command = ""
    last_control: Point | None = None
    last_quad: Point | None = None

    def finish_path(closed: bool) -> None:
        nonlocal segments
        if segments:
            paths.append(VectorPath(segments, closed=closed))
        segments = []

    def read_numbers(count: int) -> list[float]:
        nonlocal index
        values: list[float] = []
        for _ in range(count):
            if index >= len(tokens) or tokens[index][0] != "num":
                raise SvgPathParseError(f"Nedostaje broj u 'd' atributu (komanda {command!r})")
            values.append(float(tokens[index][1]))
            index += 1
        return values

    while index < len(tokens):
        kind, value = tokens[index]
        if kind == "cmd":
            command = value
            index += 1
            if command in "Zz":
                finish_path(closed=True)
                current = subpath_start
                last_control = None
                last_quad = None
                continue
        elif not command:
            raise SvgPathParseError("Path string ne pocinje komandom")

        absolute = command.isupper()
        cmd_lower = command.lower()
        if cmd_lower == "m":
            x, y = read_numbers(2)
            target = Point(x, y) if absolute else Point(current.x + x, current.y + y)
            if segments:
                finish_path(closed=False)
            current = target
            subpath_start = target
            last_control = last_quad = None
            command = "L" if absolute else "l"
        elif cmd_lower == "l":
            x, y = read_numbers(2)
            target = Point(x, y) if absolute else Point(current.x + x, current.y + y)
            segments.append(Line(current, target))
            current = target
            last_control = last_quad = None
        elif cmd_lower == "h":
            (x,) = read_numbers(1)
            target = Point(x, current.y) if absolute else Point(current.x + x, current.y)
            segments.append(Line(current, target))
            current = target
            last_control = last_quad = None
        elif cmd_lower == "v":
            (y,) = read_numbers(1)
            target = Point(current.x, y) if absolute else Point(current.x, current.y + y)
            segments.append(Line(current, target))
            current = target
            last_control = last_quad = None
        elif cmd_lower == "c":
            x1, y1, x2, y2, x, y = read_numbers(6)
            if absolute:
                c1, c2, target = Point(x1, y1), Point(x2, y2), Point(x, y)
            else:
                c1 = Point(current.x + x1, current.y + y1)
                c2 = Point(current.x + x2, current.y + y2)
                target = Point(current.x + x, current.y + y)
            segments.append(CubicBezier(current, c1, c2, target))
            last_control = c2
            last_quad = None
            current = target
        elif cmd_lower == "s":
            x2, y2, x, y = read_numbers(4)
            if absolute:
                c2, target = Point(x2, y2), Point(x, y)
            else:
                c2 = Point(current.x + x2, current.y + y2)
                target = Point(current.x + x, current.y + y)
            if segments and isinstance(segments[-1], CubicBezier) and last_control is not None:
                previous_end = segments[-1].end
                c1 = Point(2 * previous_end.x - last_control.x,
                           2 * previous_end.y - last_control.y)
            else:
                c1 = current
            segments.append(CubicBezier(current, c1, c2, target))
            last_control = c2
            last_quad = None
            current = target
        elif cmd_lower == "q":
            x1, y1, x, y = read_numbers(4)
            if absolute:
                control, target = Point(x1, y1), Point(x, y)
            else:
                control = Point(current.x + x1, current.y + y1)
                target = Point(current.x + x, current.y + y)
            c1 = Point(current.x + 2.0 / 3.0 * (control.x - current.x),
                       current.y + 2.0 / 3.0 * (control.y - current.y))
            c2 = Point(target.x + 2.0 / 3.0 * (control.x - target.x),
                       target.y + 2.0 / 3.0 * (control.y - target.y))
            segments.append(CubicBezier(current, c1, c2, target))
            last_quad = control
            last_control = None
            current = target
        elif cmd_lower == "t":
            x, y = read_numbers(2)
            target = Point(x, y) if absolute else Point(current.x + x, current.y + y)
            if last_quad is not None:
                control = Point(2 * current.x - last_quad.x, 2 * current.y - last_quad.y)
            else:
                control = current
            c1 = Point(current.x + 2.0 / 3.0 * (control.x - current.x),
                       current.y + 2.0 / 3.0 * (control.y - current.y))
            c2 = Point(target.x + 2.0 / 3.0 * (control.x - target.x),
                       target.y + 2.0 / 3.0 * (control.y - target.y))
            segments.append(CubicBezier(current, c1, c2, target))
            last_quad = control
            last_control = None
            current = target
        else:  # pragma: no cover - A/a i nepoznate komande
            raise SvgPathParseError(f"Nepodrzana SVG path komanda: {command!r}")

    finish_path(closed=False)
    return paths


def _flatten_cubic(curve: CubicBezier, tolerance: float, output: list[Point],
                   depth: int) -> None:
    """Rekurzivno deli krivu dok greska aproksimacije nije ispod tolerancije."""
    if depth >= 12:
        return
    # Maksimalna udaljenost kontrolnih tacaka od tetive je gornja granica greske.
    chord = curve.start.distance_to(curve.end)
    if chord < 1e-9:
        deviation = max(curve.control1.distance_to(curve.start),
                        curve.control2.distance_to(curve.start))
    else:
        deviation = max(_point_line_distance(curve.control1, curve.start, curve.end),
                        _point_line_distance(curve.control2, curve.start, curve.end))
    if deviation <= tolerance:
        return
    left, right = _split_cubic(curve, 0.5)
    _flatten_cubic(left, tolerance, output, depth + 1)
    output.append(left.end)
    _flatten_cubic(right, tolerance, output, depth + 1)


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end.x - start.x, end.y - start.y
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return point.distance_to(start)
    return abs(dy * (point.x - start.x) - dx * (point.y - start.y)) / norm


def _split_cubic(curve: CubicBezier, t: float) -> tuple[CubicBezier, CubicBezier]:
    p01 = _lerp(curve.start, curve.control1, t)
    p12 = _lerp(curve.control1, curve.control2, t)
    p23 = _lerp(curve.control2, curve.end, t)
    p012 = _lerp(p01, p12, t)
    p123 = _lerp(p12, p23, t)
    middle = _lerp(p012, p123, t)
    return (CubicBezier(curve.start, p01, p012, middle),
            CubicBezier(middle, p123, p23, curve.end))


def _lerp(a: Point, b: Point, t: float) -> Point:
    return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
