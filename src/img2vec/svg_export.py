"""Izvoz putanja u LightBurn-kompatibilan SVG sa milimetarskim koordinatama."""

from __future__ import annotations

import svgwrite

from .geometry import (LASER_STROKE_MM, PathStyle, VectorPath, format_number,
                       path_to_d)

SVG_NS = "http://www.w3.org/2000/svg"
GENERATOR = "img2vec-lightburn v2.0"


def _dimension(value: float) -> str:
    """Dimenzija u mm za root <svg> element (npr. '150mm')."""
    return f"{format_number(round(value, 3))}mm"


def paths_to_svg(paths: list[VectorPath], width_mm: float, height_mm: float,
                 *, title: str | None = None) -> str:
    """Serijalizuje putanje u SVG 1.1 dokument.

    Kljucna pravila za LightBurn:
      * width/height su eksplicitno u milimetrima,
      * viewBox je 0 0 W H (1 jedinica = 1 mm),
      * boje su tacno #000000 / #0000FF / #FF0000,
      * stroke-width je 0.1 (u mm, bez jedinice).
    """
    width = round(float(width_mm), 3)
    height = round(float(height_mm), 3)
    if width <= 0 or height <= 0:
        raise ValueError(f"Neispravne dimenzije platna: {width} x {height} mm")

    drawing = svgwrite.Drawing(
        size=(_dimension(width), _dimension(height)),
        profile="full",
        debug=False,
    )
    # viewBox se postavlja rucno: svgwrite bi upisao zapete ("0,0,150,150"),
    # a LightBurn ocekuje standardni oblik sa razmacima.
    drawing.attribs["viewBox"] = (f"0 0 {format_number(width)} "
                                  f"{format_number(height)}")
    drawing.attribs["version"] = "1.1"

    for group, style in _group_compound(paths):
        d = " ".join(filter(None, (path_to_d(path, decimals=3) for path in group)))
        if not d:
            continue
        attribs = {
            "d": d,
            "fill": style.fill or "none",
            "stroke": style.stroke or "none",
        }
        if style.fill:
            # compound putanje (spoljna kontura + rupe) zahtevaju evenodd pravilo
            attribs["fill-rule"] = "evenodd"
        if style.stroke:
            attribs["stroke-width"] = format_number(style.stroke_width or LASER_STROKE_MM)
            attribs["stroke-linecap"] = "round"
            attribs["stroke-linejoin"] = "round"
        drawing.add(svgwrite.path.Path(**attribs))

    return drawing.tostring()


def _group_compound(paths: list[VectorPath]):
    """Grupiše podputanje istog potrace elementa u jedan SVG <path> (rupe!).

    Potrace vraca spoljne konture i rupe kao podputanja jednog path elementa,
    obelezena sa meta["compound"]. Grupisanje cuva informaciju o rupama.
    """
    groups: list[tuple[list[VectorPath], PathStyle]] = []
    for path in paths:
        if path.is_empty():
            continue
        compound_id = path.meta.get("compound")
        if (compound_id is not None and groups
                and groups[-1][0][0].meta.get("compound") == compound_id):
            groups[-1][0].append(path)
        else:
            groups.append(([path], path.style))
    return groups


def paths_to_svg_bytes(paths: list[VectorPath], width_mm: float,
                       height_mm: float, *, title: str | None = None) -> bytes:
    return paths_to_svg(paths, width_mm, height_mm, title=title).encode("utf-8")


def write_svg(paths: list[VectorPath], width_mm: float, height_mm: float,
              target: str, *, title: str | None = None) -> None:
    from pathlib import Path
    Path(target).write_text(paths_to_svg(paths, width_mm, height_mm, title=title),
                            encoding="utf-8")