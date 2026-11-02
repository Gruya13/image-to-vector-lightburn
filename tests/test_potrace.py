"""Testovi potrace adaptera i parsiranja njegovog SVG izlaza."""

from __future__ import annotations

import numpy as np
import pytest

from img2vec.potrace import (PotraceError, PotraceOptions, find_potrace,
                             trace_bitmap)

pytestmark = pytest.mark.skipif(find_potrace() is None,
                                reason="potrace binar nije dostupan")


def _circle_mask(size: int = 300, radius: int = 100) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    return (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2


def test_krug_daje_bezier_krive():
    mask = _circle_mask()
    paths = trace_bitmap(mask, options=PotraceOptions(turdsize=0))
    assert len(paths) >= 1
    path = max(paths, key=lambda item: len(item.segments))
    assert path.closed
    bezier = sum(1 for segment in path.segments if type(segment).__name__ == "CubicBezier")
    assert bezier >= 4


def test_bbox_odgovara_maski():
    mask = _circle_mask()
    paths = trace_bitmap(mask, options=PotraceOptions(turdsize=0))
    points = [point for path in paths for point in path.flatten(0.01)]
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    assert min(xs) == pytest.approx(50, abs=1.0)
    assert max(xs) == pytest.approx(250, abs=1.0)
    assert min(ys) == pytest.approx(50, abs=1.0)
    assert max(ys) == pytest.approx(250, abs=1.0)


def test_turdsize_uklanja_sitne_mrlje():
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True
    mask[10, 10] = True  # usamljeni piksel
    paths = trace_bitmap(mask, options=PotraceOptions(turdsize=2))
    points = [point for path in paths for point in path.flatten(0.5)]
    assert min(point.x for point in points) > 45
    assert min(point.y for point in points) > 45


def test_prazna_maska():
    assert trace_bitmap(np.zeros((50, 50), dtype=bool)) == []


def test_compound_meta_grupiše_podputanje():
    mask = np.zeros((300, 300), dtype=bool)
    mask[20:120, 20:120] = True   # kvadrat
    mask[50:90, 50:90] = False    # rupa
    paths = trace_bitmap(mask, options=PotraceOptions(turdsize=0))
    compounds = {path.meta.get("compound") for path in paths}
    assert compounds == {0}
    assert len(paths) == 2  # spoljna kontura + rupa


def test_nedostupan_binar():
    with pytest.raises(PotraceError):
        trace_bitmap(np.ones((10, 10), dtype=bool),
                     potrace_bin=__import__("pathlib").Path("/nema/potracea"))
