#!/usr/bin/env python3
"""KORAK 1: Verifikacioni spike Faze 1 nad referentnim test datasetom.

Za svaku od 10 slika iz test_images/ pokrece tri puta:
  1. STENCIL  - auto threshold -> potrace (glatke Bezier krive, Fill sloj)
  2. CENTERLINE - XDoG/threshold -> skeleton -> skan -> vpype/RDP optimizacija
  3. CUT      - rembg/alfa maska -> shapely buffer offset (crveni sloj)

Za svaki rezultat proverava da li je SVG "cist" za LightBurn:
  * validan XML i root <svg> sa width/height u mm i ispravnim viewBox-om
  * tacne fizicke dimenzije (tolerancija 0.01 mm)
  * samo LightBurn paleta boja (#000000 / #0000FF / #FF0000)
  * bez NaN/inf koordinata i bez praznih/degenerisanih putanja
  * sadrzaj unutar platna i razuman broj cvorova

Acceptance kriterijum Faze 1: najmanje 8/10 slika daje cist SVG za svoj
preporuceni rezim.

Pokretanje:
    uv run python spike_phase1.py
    uv run python spike_phase1.py --json          # masinski citljiv izvestaj
    uv run python spike_phase1.py --no-rembg      # brzi spike bez AI maske
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from img2vec.config import ProcessParams  # noqa: E402
from img2vec.pipeline import process_image  # noqa: E402
from img2vec.preview import encode_preview_png  # noqa: E402

TEST_IMAGES = ROOT / "test_images"
OUTPUT = ROOT / "out" / "spike"
SVG_NS = "{http://www.w3.org/2000/svg}"
ALLOWED_COLORS = {"#000000", "#0000FF", "#FF0000"}
MAX_SVG_BYTES = 8 * 1024 * 1024
MAX_NODES = 60000

# Preporuceni rezim po kategoriji iz manifesta
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


@dataclass
class CheckResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    paths: int = 0
    nodes: int = 0
    size_kb: float = 0.0


@dataclass
class RunResult:
    name: str
    mode: str
    ok: bool
    width_mm: float = 0.0
    height_mm: float = 0.0
    paths: int = 0
    nodes: int = 0
    length_m: float = 0.0
    time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    has_cut: bool = False
    output: str = ""


def validate_svg(svg: str, expected_w: float, expected_h: float,
                 result_paths, result_nodes: int) -> CheckResult:
    problems: list[str] = []
    size_kb = len(svg.encode("utf-8")) / 1024
    if size_kb > MAX_SVG_BYTES / 1024:
        problems.append(f"SVG je preveliki ({size_kb:.0f} KB)")

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        return CheckResult(False, [f"nevalidan XML: {exc}"], size_kb=size_kb)

    for attr, expected in (("width", expected_w), ("height", expected_h)):
        value = root.attrib.get(attr, "")
        if not value.endswith("mm"):
            problems.append(f"{attr} nije u mm ({value!r})")
            continue
        try:
            number = float(value[:-2])
        except ValueError:
            problems.append(f"{attr} nije broj ({value!r})")
            continue
        if abs(number - expected) > 0.01:
            problems.append(f"{attr}={number}mm, ocekivano {expected:.3f}mm")

    viewbox = root.attrib.get("viewBox", "")
    parts = viewbox.replace(",", " ").split()
    if len(parts) != 4:
        problems.append(f"neispravan viewBox ({viewbox!r})")
    else:
        vx, vy, vw, vh = (float(p) for p in parts)
        if abs(vx) > 1e-6 or abs(vy) > 1e-6:
            problems.append(f"viewBox ne pocinje od 0 0 ({viewbox!r})")
        if abs(vw - expected_w) > 0.01 or abs(vh - expected_h) > 0.01:
            problems.append(f"viewBox dimenzije {vw}x{vh} != {expected_w:.3f}x{expected_h:.3f}")

    path_count = 0
    for element in root.iter(f"{SVG_NS}path"):
        path_count += 1
        d = element.attrib.get("d", "")
        if not d:
            problems.append("prazan <path> element")
            continue
        if "nan" in d.lower() or "inf" in d.lower():
            problems.append("NaN/inf u path podacima")
        for attr in ("fill", "stroke"):
            color = element.attrib.get(attr)
            if color and color != "none" and color.upper() not in ALLOWED_COLORS:
                problems.append(f"boja van LightBurn palete: {color}")
        width = element.attrib.get("stroke-width")
        if width is not None:
            try:
                if abs(float(width) - 0.1) > 1e-6:
                    problems.append(f"stroke-width={width} (ocekivano 0.1)")
            except ValueError:
                problems.append(f"neispravan stroke-width: {width!r}")

    if path_count == 0:
        problems.append("nijedan <path> nije pronadjen")
    if result_nodes > MAX_NODES:
        problems.append(f"previse cvorova ({result_nodes} > {MAX_NODES})")
    if result_paths:
        tolerance = 0.5
        for path in result_paths:
            box = path.bbox()
            if (box.min_x < -tolerance or box.min_y < -tolerance
                    or box.max_x > expected_w + tolerance
                    or box.max_y > expected_h + tolerance):
                problems.append("sadrzaj izlazi van platna")
                break

    return CheckResult(not problems, problems, path_count, result_nodes, size_kb)


def recommended_mode(category: str) -> str:
    return RECOMMENDED.get(category, "stencil")


def run_case(entry: dict, *, use_rembg: bool) -> tuple[list[RunResult], str]:
    image_path = TEST_IMAGES / entry["file"]
    category = entry["category"]
    recommended = recommended_mode(category)
    target_mm = 150.0
    runs: list[RunResult] = []

    for mode in ("stencil", "centerline"):
        params = ProcessParams(
            mode=mode,
            target_width_mm=target_mm,
            remove_background=use_rembg and mode == "stencil",
            # line-art i cist crtez ne treba da idu kroz rembg
            add_cut=(mode == "stencil" and recommended == "stencil"),
            cut_offset_mm=3.0,
        )
        if mode == "centerline":
            params.remove_background = False
            params.add_cut = False
        started = time.perf_counter()
        try:
            result = process_image(image_path, params, include_dxf=False)
        except Exception as exc:  # noqa: BLE001 - spike prijavljuje sve greske
            runs.append(RunResult(entry["file"], mode, False,
                                  problems=[f"obrada nije uspela: {exc}"]))
            continue
        elapsed = (time.perf_counter() - started) * 1000
        check = validate_svg(result.svg, result.width_mm, result.height_mm,
                             result.paths, result.stats.node_count)
        has_cut = any(path.style.stroke == "#FF0000" for path in result.paths)
        out_path = OUTPUT / f"{image_path.stem}_{mode}.svg"
        out_path.write_text(result.svg, encoding="utf-8")
        preview_path = OUTPUT / f"{image_path.stem}_{mode}_preview.png"
        preview_path.write_bytes(encode_preview_png(
            result.paths, result.width_mm, result.height_mm, max_px=800))
        runs.append(RunResult(
            name=entry["file"], mode=mode, ok=check.ok,
            width_mm=result.width_mm, height_mm=result.height_mm,
            paths=check.paths, nodes=check.nodes,
            length_m=result.stats.total_length_m, time_ms=elapsed,
            warnings=list(result.warnings) + list(check.problems),
            has_cut=has_cut, output=str(out_path.relative_to(ROOT)),
        ))
    return runs, recommended


def main() -> int:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON izvestaj")
    parser.add_argument("--no-rembg", action="store_true",
                        help="preskoci AI uklanjanje pozadine (brzi spike)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    OUTPUT = args.output

    manifest_path = TEST_IMAGES / "manifest.json"
    if not manifest_path.is_file():
        print(f"Greska: nedostaje {manifest_path}. Pokreni "
              f"'python scripts/generate_test_dataset.py'.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("=" * 112)
    print("IMAGE-TO-VECTOR LIGHTBURN :: VERIFIKACIONI SPIKE FAZE 1")
    print(f"dataset: {len(manifest)} slika | izlaz: {args.output} | "
          f"rembg: {'ukljucen' if not args.no_rembg else 'iskljucen'}")
    print("=" * 112)

    cases: list[dict] = []
    accepted_count = 0
    started = time.perf_counter()

    for entry in manifest:
        runs, recommended = run_case(entry, use_rembg=not args.no_rembg)
        recommended_run = next((run for run in runs if run.mode == recommended), None)
        accepted = bool(recommended_run and recommended_run.ok)
        accepted_count += int(accepted)
        cases.append({
            "file": entry["file"],
            "category": entry["category"],
            "recommended": recommended,
            "accepted": accepted,
            "runs": [run.__dict__ for run in runs],
        })
        print(f"\n{entry['file']}  [{entry['category']}]  -> preporuceno: {recommended}")
        for run in runs:
            mark = "OK  " if run.ok else "FAIL"
            cut = " +CUT" if run.has_cut else ""
            print(f"  [{mark}] {run.mode:10s}{cut:5s} {run.width_mm:7.1f}x{run.height_mm:6.1f}mm "
                  f"{run.paths:4d} put {run.nodes:6d} cvor {run.length_m:5.2f}m "
                  f"{run.time_ms:7.0f}ms")
            for problem in run.problems:
                print(f"         - {problem}")
        if recommended_run is not None and not recommended_run.ok:
            print(f"  !! preporuceni rezim nije cist: {recommended_run.problems}")

    elapsed = time.perf_counter() - started
    total = len(manifest)
    print("\n" + "=" * 112)
    print(f"ACCEPTANCE: {accepted_count}/{total} slika daje cist SVG "
          f"(kriterijum Faze 1: >= 8/10) -> "
          f"{'PROSLO' if accepted_count >= 8 else 'NISU PROSLO'}")
    print(f"Ukupno vreme spike-a: {elapsed:.1f}s "
          f"(prosecno {elapsed / max(1, total):.2f}s po slici)")
    print("=" * 112)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_size": total,
        "accepted": accepted_count,
        "passed": accepted_count >= 8,
        "elapsed_s": round(elapsed, 2),
        "rembg": not args.no_rembg,
        "cases": cases,
    }
    report_path = args.output / "spike_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(f"Izvestaj: {report_path.relative_to(ROOT)}")
    return 0 if accepted_count >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
