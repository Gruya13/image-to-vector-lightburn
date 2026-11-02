"""Acceptance test Faze 1 nad referentnim test datasetom (spor)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from img2vec.config import TEST_IMAGES_DIR, ProcessParams
from img2vec.pipeline import process_image

pytestmark = pytest.mark.slow

RECOMMENDED = {
    "Portret (foto)": "stencil",
    "Silueta": "stencil",
    "Logo (rasterski)": "stencil",
    "Tekst / tipografija": "stencil",
    "Crtez (line art)": "centerline",
    "Kompleksna pozadina": "stencil",
    "Sitni detalji": "stencil",
    "Niska rezolucija": "stencil",
    "Visoka rezolucija": "stencil",
    "Transparentni PNG": "stencil",
}


def _validate(result) -> list[str]:
    problems: list[str] = []
    try:
        root = ET.fromstring(result.svg)
    except ET.ParseError as exc:
        return [f"nevalidan XML: {exc}"]
    if not root.attrib.get("width", "").endswith("mm"):
        problems.append("width nije u mm")
    if not root.attrib.get("height", "").endswith("mm"):
        problems.append("height nije u mm")
    if abs(float(root.attrib["width"][:-2]) - result.width_mm) > 0.01:
        problems.append("width se ne poklapa sa rezultatom")
    if abs(float(root.attrib["height"][:-2]) - result.height_mm) > 0.01:
        problems.append("height se ne poklapa sa rezultatom")
    if result.stats.path_count == 0:
        problems.append("nema putanja")
    if result.stats.node_count > 60000:
        problems.append("previse cvorova")
    if "nan" in result.svg.lower() or "inf" in result.svg.lower():
        problems.append("NaN/inf u SVG-u")
    for path in result.paths:
        box = path.bbox()
        if box.max_x > result.width_mm + 0.6 or box.max_y > result.height_mm + 0.6:
            problems.append("sadrzaj van platna")
            break
    return problems


def test_dataset_acceptance_8_od_10():
    manifest_path = TEST_IMAGES_DIR / "manifest.json"
    assert manifest_path.is_file(), "Pokreni scripts/generate_test_dataset.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    accepted = 0
    report: list[str] = []
    for entry in manifest:
        mode = RECOMMENDED[entry["category"]]
        params = ProcessParams(
            mode=mode,
            target_width_mm=150.0,
            remove_background=False,
            add_cut=(mode == "stencil"),
            cut_offset_mm=3.0,
        )
        result = process_image(TEST_IMAGES_DIR / entry["file"], params,
                               include_dxf=False)
        problems = _validate(result)
        accepted += int(not problems)
        report.append(f"{entry['file']}: {mode} -> "
                      f"{result.stats.path_count} putanja, "
                      f"{result.stats.node_count} cvorova"
                      + (" | PROBLEMI: " + "; ".join(problems) if problems else ""))

    print("\n".join(report))
    assert accepted >= 8, f"Acceptance pao: {accepted}/10\n" + "\n".join(report)
