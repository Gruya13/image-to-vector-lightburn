"""Testovi LightBurn SVG/DXF izvoza."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import pytest

from img2vec.dxf_export import LAYERS, paths_to_dxf
from img2vec.geometry import (STYLE_CUT, STYLE_ENGRAVE_FILL, STYLE_ENGRAVE_LINE,
                              Point, VectorPath, path_from_coords)
from img2vec.svg_export import paths_to_svg


def _square(style=STYLE_ENGRAVE_FILL, size=10.0, x=5.0, y=5.0) -> VectorPath:
    return path_from_coords(
        [(x, y), (x + size, y), (x + size, y + size), (x, y + size)],
        closed=True, style=style)


def test_svg_milimetarske_dimenzije():
    svg = paths_to_svg([_square()], 150.0, 192.5)
    root = ET.fromstring(svg)
    assert root.attrib["width"] == "150mm"
    assert root.attrib["height"] == "192.5mm"
    assert root.attrib["viewBox"] == "0 0 150 192.5"


def test_svg_paleta_i_stilovi():
    svg = paths_to_svg([_square(STYLE_ENGRAVE_FILL),
                        _square(STYLE_ENGRAVE_LINE, size=3),
                        _square(STYLE_CUT, size=20)], 100.0, 100.0)
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(svg)
    paths = list(root.iter(ns + "path"))
    assert len(paths) == 3
    fills = {path.attrib.get("fill") for path in paths}
    strokes = {path.attrib.get("stroke") for path in paths}
    assert "#000000" in fills
    assert "#FF0000" in strokes
    assert "#000000" in strokes
    for path in paths:
        assert path.attrib.get("fill") in ("#000000", "none")
        assert path.attrib.get("stroke") in ("#000000", "#FF0000", "none")
        if path.attrib.get("stroke") != "none":
            assert path.attrib["stroke-width"] == "0.1"


def test_svg_compound_evenodd():
    outer = _square(STYLE_ENGRAVE_FILL, size=20, x=10, y=10)
    inner = _square(STYLE_ENGRAVE_FILL, size=6, x=17, y=17)
    for path in (outer, inner):
        path.meta["compound"] = 0
    svg = paths_to_svg([outer, inner], 100.0, 100.0)
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.fromstring(svg)
    paths = list(root.iter(ns + "path"))
    assert len(paths) == 1
    assert paths[0].attrib["fill-rule"] == "evenodd"
    assert paths[0].attrib["d"].count("M") == 2


def test_svg_viewbox_nema_zapeta():
    svg = paths_to_svg([_square()], 100.0, 50.5)
    assert 'viewBox="0 0 100 50.5"' in svg
    assert "," not in svg.split("viewBox=")[1].split('"')[1]


def test_svg_odbija_nevalidne_dimenzije():
    with pytest.raises(ValueError):
        paths_to_svg([_square()], 0.0, 100.0)


def _read_dxf_insunits(dxf: str) -> int:
    match = re.search(r"\$INSUNITS\s*\n\s*70\s*\n\s*(\d+)", dxf)
    assert match is not None, "DXF nema $INSUNITS"
    return int(match.group(1))


def test_dxf_jedinice_i_lejeri():
    dxf = paths_to_dxf([_square(STYLE_ENGRAVE_FILL),
                        _square(STYLE_CUT, size=30)], 100.0, 100.0)
    assert _read_dxf_insunits(dxf) == 4
    assert "00_ENGRAVE" in dxf
    assert "02_CUT" in dxf
    assert "LWPOLYLINE" in dxf
    for layer in LAYERS:
        assert layer in dxf


def test_dxf_zatvorena_polilinija():
    dxf = paths_to_dxf([_square()], 100.0, 100.0)
    # kod zatvorene LWPOLYLINE grupa 70 sadrzi bit 1
    assert re.search(r"\n\s*70\s*\n\s*1\b", dxf) is not None


def test_svg_bezier_krive():
    curve = VectorPath([
        __import__("img2vec.geometry", fromlist=["CubicBezier"]).CubicBezier(
            Point(0, 0), Point(10, 0), Point(20, 10), Point(30, 10))
    ], style=STYLE_ENGRAVE_LINE)
    svg = paths_to_svg([curve], 40.0, 20.0)
    assert "C10 0 20 10 30 10" in svg
