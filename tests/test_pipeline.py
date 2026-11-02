"""End-to-end testovi pipeline-a na sintetickim slikama."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from img2vec.config import MAX_INPUT_BYTES, ProcessParams
from img2vec.image_io import ImageLoadError, load_image_bytes
from img2vec.pipeline import process_image
from tests.helpers import (alpha_circle, black_circle, black_rectangle,
                           horizontal_line, png_bytes, white_canvas)


def _stencil_params(**overrides) -> ProcessParams:
    base = dict(mode="stencil", remove_background=False, target_width_mm=100.0)
    base.update(overrides)
    return ProcessParams(**base)


def test_stencil_kvadrat_jedna_compound_putanja():
    result = process_image(black_rectangle(), _stencil_params())
    assert result.stats.path_count == 1
    assert result.width_mm == pytest.approx(100.0, abs=0.001)
    assert result.height_mm == pytest.approx(100.0, abs=0.001)
    root = ET.fromstring(result.svg)
    assert root.attrib["width"] == "100mm"
    assert root.attrib["viewBox"] == "0 0 100 100"


def test_stencil_cuva_proporciju():
    result = process_image(black_rectangle(400, 200), _stencil_params())
    assert result.width_mm == pytest.approx(100.0, abs=0.001)
    assert result.height_mm == pytest.approx(50.0, abs=0.001)


def test_cut_offset_prosiruje_platno_i_bbox():
    params = _stencil_params(add_cut=True, cut_offset_mm=3.0)
    result = process_image(black_rectangle(), params)
    assert result.width_mm == pytest.approx(106.0, abs=0.001)
    assert result.height_mm == pytest.approx(106.0, abs=0.001)
    cut_paths = [path for path in result.paths if path.style.stroke == "#FF0000"]
    engrave_paths = [path for path in result.paths if path.style.fill == "#000000"]
    assert cut_paths and engrave_paths
    cut_box = cut_paths[0].bbox()
    engrave_box = engrave_paths[0].bbox()
    assert cut_box.width > engrave_box.width + 5.0
    assert cut_box.min_x < engrave_box.min_x


def test_cut_offset_nula_pokriva_gravuru():
    params = _stencil_params(add_cut=True, cut_offset_mm=0.0)
    result = process_image(black_rectangle(), params)
    assert result.width_mm == pytest.approx(100.0, abs=0.001)


def test_centerline_proizvodi_jednu_liniju():
    # debela horizontalna linija ne sme da se pretvori u dve paralelne ivice
    result = process_image(
        horizontal_line(), ProcessParams(mode="centerline", remove_background=False,
                                         target_width_mm=100.0, min_spur_px=5,
                                         line_threshold_method="otsu"))
    assert result.stats.path_count == 1, result.warnings
    line = result.paths[0]
    assert line.style.stroke == "#000000"
    assert line.style.fill is None
    # linija je od x=40 do 360 px na 400 px sirine -> 80 mm
    assert line.length() == pytest.approx(80.0, abs=1.5)


def test_alfa_kanal_se_koristi_kao_maska():
    result = process_image(
        alpha_circle(),
        ProcessParams(mode="stencil", remove_background=False,
                      target_width_mm=60.0, add_cut=True, cut_offset_mm=2.0))
    assert result.mask_png is not None
    assert result.width_mm == pytest.approx(64.0, abs=0.001)
    assert any(path.style.stroke == "#FF0000" for path in result.paths)


def test_bela_slika_daje_upozorenje():
    blank = png_bytes(white_canvas(200, 200))
    result = process_image(blank, _stencil_params())
    assert result.stats.path_count == 0
    assert result.warnings
    assert result.svg  # dimenzije i prazan dokument su i dalje validni


def test_invert_menja_rezultat():
    data = black_rectangle()
    normal = process_image(data, _stencil_params())
    inverted = process_image(data, _stencil_params(invert=True))
    # normal: crni kvadrat na belom; invert: gravura pokriva sve osim kvadrata
    assert normal.stats.path_count == 1
    assert inverted.stats.path_count >= 1
    assert normal.svg != inverted.svg
    # invertovana gravura mora imati vecu ukupnu duzinu kontura
    assert inverted.stats.total_length_m > normal.stats.total_length_m


def test_downscale_velike_slike():
    data = png_bytes(white_canvas(3000, 1500))
    image = load_image_bytes(data)
    assert image.work_size[0] == 2000
    assert image.original_size[0] == 3000


def test_limit_velicine_fajla():
    with pytest.raises(ImageLoadError):
        load_image_bytes(b"0" * (MAX_INPUT_BYTES + 1))


def test_nevalidan_format():
    with pytest.raises(ImageLoadError):
        load_image_bytes(b"nije slika")


def test_nepoznat_rezim():
    params = _stencil_params()
    params.mode = "nepostojeci"
    with pytest.raises(ValueError):
        process_image(black_rectangle(), params)


def test_stats_ukljucuje_cut_boju():
    result = process_image(black_rectangle(), _stencil_params(add_cut=True,
                                                              cut_offset_mm=2.0))
    stats = result.stats.as_dict()
    assert "#FF0000" in stats["stroke_length_m"]
    assert "#000000" in stats["stroke_length_m"]
