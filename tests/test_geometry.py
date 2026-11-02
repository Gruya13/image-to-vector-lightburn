"""Testovi geometrijskog jezgra."""

from __future__ import annotations

import math

import pytest

from img2vec.geometry import (CubicBezier, Line, Point, STYLE_CUT, VectorPath,
                              format_number, parse_path_d, path_from_coords,
                              path_to_d, paths_bbox)


def test_format_number_stabilan():
    assert format_number(0.1 + 0.2) == "0.3"
    assert format_number(150.0) == "150"
    assert format_number(192.5) == "192.5"
    assert format_number(-0.00001) == "0"
    assert format_number(1e-9) == "0"
    assert format_number(123.4567) == "123.457"


def test_format_number_odbija_nekonacno():
    with pytest.raises(ValueError):
        format_number(float("nan"))
    with pytest.raises(ValueError):
        format_number(float("inf"))


def test_parser_roundtrip():
    d = "M10 20L30 40C40 50 60 70 80 90Z"
    paths = parse_path_d(d)
    assert len(paths) == 1
    path = paths[0]
    assert path.closed
    assert path_to_d(path) == d


def test_parser_relative_komande():
    paths = parse_path_d("M10 10 l10 0 v10 h-10 z")
    assert len(paths) == 1
    points = paths[0].flatten(0.01)
    assert points[0] == Point(10, 10)
    assert Point(20, 20) in points


def test_parser_vise_podputanja():
    paths = parse_path_d("M0 0 L10 0 Z M20 20 L30 20 Z")
    assert len(paths) == 2
    assert paths[0].closed and paths[1].closed


def test_bezier_flatten_tacnost():
    curve = CubicBezier(Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0))
    points = curve.flatten(0.01)
    assert points[0] == Point(0, 0)
    assert points[-1] == Point(10, 0)
    # teme krive je na y=7.5 za simetricnu kubnu Bezier krivu
    assert max(point.y for point in points) == pytest.approx(7.5, abs=0.05)
    assert curve.length() == pytest.approx(20.0, abs=0.5)


def test_bbox_i_transformacije():
    path = path_from_coords([(0, 0), (10, 0), (10, 10)])
    box = path.bbox()
    assert (box.min_x, box.min_y, box.max_x, box.max_y) == (0, 0, 10, 10)
    scaled = path.transformed(2.0, 5.0, 3.0)
    assert scaled.bbox().min_x == 5.0
    assert scaled.bbox().max_x == 25.0


def test_duzina_putanje():
    path = path_from_coords([(0, 0), (30, 40)])
    assert path.length() == pytest.approx(50.0)


def test_closed_putanja_dodaje_zavrsnu_liniju():
    path = path_from_coords([(0, 0), (10, 0), (10, 10)], closed=True)
    assert path.closed
    assert path.segments[-1].end == Point(0, 0)


def test_cut_stil_boja():
    path = path_from_coords([(0, 0), (1, 0)], style=STYLE_CUT)
    assert path.style.stroke == "#FF0000"
    assert path.style.fill is None


def test_paths_bbox_union():
    a = path_from_coords([(0, 0), (10, 10)])
    b = path_from_coords([(5, 5), (20, 30)])
    box = paths_bbox([a, b])
    assert box is not None
    assert box.width == 20
    assert box.height == 30
