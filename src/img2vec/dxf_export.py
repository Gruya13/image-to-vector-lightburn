"""Izvoz putanja u DXF R2010 sa razdvojenim LaserBurn lejerima.

DXF ne podrzava Bezier krive direktno, pa se sve krive pre izvoza
flatten-uju u polilinije sa tolerancijom 0.05 mm (ispod rezolucije lasera).
"""

from __future__ import annotations

import io
from pathlib import Path

import ezdxf
from ezdxf import units

from .geometry import PathStyle, VectorPath

# Naziv DXF lejera -> (ACI boja, opis)
LAYERS: dict[str, tuple[int, str]] = {
    "00_ENGRAVE": (7, "Glavna gravura (LightBurn C00)"),
    "01_DETAIL": (5, "Pomocni detalji (LightBurn C01)"),
    "02_CUT": (1, "Spoljni obris za secenje (LightBurn C02)"),
}

FLATTEN_TOLERANCE_MM = 0.05


def _layer_for(style: PathStyle) -> str:
    name = style.layer_name
    return name if name in LAYERS else "00_ENGRAVE"


def paths_to_dxf(paths: list[VectorPath], width_mm: float, height_mm: float,
                 *, title: str | None = None) -> str:
    """Vraca ASCII DXF R2010 sadrzaj sa $INSUNITS = 4 (milimetri)."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = units.MM
    doc.header["$MEASUREMENT"] = 1
    if title:
        doc.header.custom_vars.append("IMG2VEC_TITLE", title)

    for name, (color, _description) in LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    msp = doc.modelspace()
    for path in paths:
        points = path.flatten(FLATTEN_TOLERANCE_MM)
        if len(points) < 2:
            continue
        # Ukloni ponovljene tacke (DXF ne voli duple vertekse)
        clean = [points[0]]
        for point in points[1:]:
            if point.distance_to(clean[-1]) > 1e-6:
                clean.append(point)
        if len(clean) < 2:
            continue
        is_closed = path.closed or clean[0].distance_to(clean[-1]) < 1e-6
        if is_closed and clean[0].distance_to(clean[-1]) <= 1e-6:
            clean.pop()
        msp.add_lwpolyline(
            [(round(p.x, 4), round(p.y, 4)) for p in clean],
            close=is_closed,
            dxfattribs={"layer": _layer_for(path.style)},
        )

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue()


def paths_to_dxf_bytes(paths: list[VectorPath], width_mm: float, height_mm: float,
                       *, title: str | None = None) -> bytes:
    return paths_to_dxf(paths, width_mm, height_mm, title=title).encode("utf-8")


def write_dxf(paths: list[VectorPath], width_mm: float, height_mm: float,
              target: str, *, title: str | None = None) -> None:
    Path(target).write_text(paths_to_dxf(paths, width_mm, height_mm, title=title),
                            encoding="utf-8")
