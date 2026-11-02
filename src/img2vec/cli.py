"""Komandna linija: image-to-vector-lightburn (Faza 1/2/3)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import (MODE_LABELS, SUPPORTED_EXTENSIONS, ProcessParams,
                     PROJECT_ROOT)
from .geometry import format_number
from .pipeline import PipelineError, ProcessingResult, process_image
from .presets import get_preset, list_presets
from .preview import encode_preview_png


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img2vec",
        description="Rasterska slika -> proizvodni vektor (SVG/DXF u mm) za LightBurn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Primeri:",
            "  img2vec slika.png -o out --width 150 --cut-offset 3",
            "  img2vec slika.png --mode centerline --preset led-akril",
            "  img2vec test_images/ -o out/spike --mode stencil --json",
            "  img2vec --list-presets",
        ]),
    )
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="ulazne slike, direktorijumi ili glob)")
    parser.add_argument("-o", "--output", type=Path, default=Path("out"),
                        help="izlazni direktorijum (default: out)")
    parser.add_argument("--mode", choices=sorted(MODE_LABELS),
                        help="rezim vektorizacije")
    parser.add_argument("--preset", help="naziv material preseta")
    parser.add_argument("--width", type=float, help="ciljna sirina u mm")
    parser.add_argument("--height", type=float, help="ciljna visina u mm")
    parser.add_argument("--quality", type=int, choices=(1, 2, 3),
                        help="kvalitet (1 brzo .. 3 maksimalno)")
    parser.add_argument("--cut-offset", type=float,
                        help="ukljuci Cut sloj sa ovim odmakom u mm")
    parser.add_argument("--no-cut", action="store_true",
                        help="iskljuci Cut sloj")
    parser.add_argument("--no-rembg", action="store_true",
                        help="ne uklanjaj pozadinu (brze, za cist line-art)")
    parser.add_argument("--bg-model", help="rembg model (isnet-general-use / birefnet-general)")
    parser.add_argument("--invert", action="store_true", help="invertuj masku")
    parser.add_argument("--formats", default="svg,dxf",
                        help="izlazni formati, npr. svg,dxf (default: svg,dxf)")
    parser.add_argument("--preview", action="store_true",
                        help="sacuvaj i PNG preview")
    parser.add_argument("--json", action="store_true",
                        help="ispisi masinski citljiv JSON po slici")
    parser.add_argument("--list-presets", action="store_true",
                        help="prikazi dostupne material presete")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir()
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif path.is_file():
            files.append(path)
        else:
            matches = sorted(PROJECT_ROOT.glob(str(raw)))
            files.extend(p for p in matches if p.is_file())
    # deduplikacija uz ocuvanje redosleda
    seen: set[Path] = set()
    unique: list[Path] = []
    for file in files:
        resolved = file.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(file)
    return unique


def _params_from_args(args: argparse.Namespace) -> ProcessParams:
    preset = get_preset(args.preset) if args.preset else None
    if preset is not None:
        params = ProcessParams.from_dict(preset.params.to_dict())
    else:
        if args.preset:
            raise SystemExit(f"Nepoznat preset: {args.preset!r} "
                             f"(dostupni: {', '.join(p.key for p in list_presets())})")
        params = ProcessParams()

    if args.mode:
        params.mode = args.mode
    if args.width is not None:
        params.target_width_mm = args.width
        if args.height is None:
            params.target_height_mm = None
    if args.height is not None:
        params.target_height_mm = args.height
        if args.width is None:
            params.target_width_mm = None
    if params.target_width_mm is None and params.target_height_mm is None:
        params.target_width_mm = 150.0
    if args.quality is not None:
        params.quality = args.quality
    if args.cut_offset is not None:
        params.add_cut = True
        params.cut_offset_mm = args.cut_offset
    if args.no_cut:
        params.add_cut = False
    if args.no_rembg:
        params.remove_background = False
    if args.bg_model:
        params.bg_model = args.bg_model
    if args.invert:
        params.invert = True
    params.validate()
    return params


def _print_presets() -> None:
    print(f"{'KLJUC':16s} {'NAZIV':36s} OPIS")
    print("-" * 100)
    for preset in list_presets():
        origin = "ugradjen" if preset.builtin else "timski"
        print(f"{preset.key:16s} {preset.name[:34]:36s} [{origin}] {preset.description}")


def _format_result(result: ProcessingResult) -> str:
    stats = result.stats
    return (
        f"{result.width_mm:8.2f} x {result.height_mm:7.2f} mm | "
        f"{stats.path_count:4d} putanja | {stats.node_count:6d} cvorova | "
        f"{stats.total_length_m:6.2f} m | {result.timings_ms.get('total', 0):7.0f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        _print_presets()
        return 0
    if not args.inputs:
        parser.error("nedostaje ulazna slika (ili koristi --list-presets)")

    try:
        params = _params_from_args(args)
    except SystemExit as exc:  # pragma: no cover
        print(exc, file=sys.stderr)
        return 2

    inputs = _collect_inputs(args.inputs)
    if not inputs:
        print("Greska: nijedna ulazna slika nije pronadjena.", file=sys.stderr)
        return 2

    formats = {part.strip().lower() for part in args.formats.split(",") if part.strip()}
    if not formats <= {"svg", "dxf"}:
        print(f"Greska: nepodrzan format u --formats: {formats - {'svg', 'dxf'}}",
              file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.json:
        print(f"img2vec {__version__} | rezim: {params.mode} | "
              f"izlaz: {args.output}")
        print("-" * 110)

    failures = 0
    started = time.perf_counter()
    for index, input_path in enumerate(inputs, start=1):
        try:
            result = process_image(input_path, params,
                                   include_dxf="dxf" in formats,
                                   include_mask_png=not args.json)
        except (PipelineError, ValueError, OSError) as exc:
            failures += 1
            if args.json:
                print(json.dumps({"file": input_path.name, "ok": False,
                                  "error": str(exc)}, ensure_ascii=False))
            else:
                print(f"[{index:3d}/{len(inputs)}] {input_path.name}: GRESKA - {exc}",
                      file=sys.stderr)
            continue

        stem = input_path.stem
        written: list[str] = []
        if "svg" in formats:
            svg_path = args.output / f"{stem}.svg"
            svg_path.write_text(result.svg, encoding="utf-8")
            written.append(svg_path.name)
        if "dxf" in formats:
            dxf_path = args.output / f"{stem}.dxf"
            dxf_path.write_text(result.dxf, encoding="utf-8")
            written.append(dxf_path.name)
        if args.preview:
            preview_path = args.output / f"{stem}_preview.png"
            preview_path.write_bytes(encode_preview_png(
                result.paths, result.width_mm, result.height_mm))
            written.append(preview_path.name)

        if args.json:
            payload = {"file": input_path.name, "ok": True,
                       "outputs": written, **result.summary()}
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"[{index:3d}/{len(inputs)}] {input_path.name}")
            print(f"        {_format_result(result)}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"        ! {warning}")
            print(f"        -> {', '.join(written)}")

    elapsed = time.perf_counter() - started
    if not args.json:
        ok = len(inputs) - failures
        print("-" * 110)
        print(f"Gotovo: {ok}/{len(inputs)} uspesno za {elapsed:.2f}s")
    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
