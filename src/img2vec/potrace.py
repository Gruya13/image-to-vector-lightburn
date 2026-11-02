"""Adapter za sistemski potrace CLI (bitmapa -> glatke Bezier krive).

Potrace se poziva kao spoljni proces (kako zahteva specifikacija), a njegov
SVG izlaz se parsira i prevodi u milimetarski geometrijski model. Redosled
pronalazenja binarnog fajla:

1. promenljiva okruzenja IMG2VEC_POTRACE (pun put do binar fajla),
2. potrace u sistemskom PATH-u (pacman / brew / apt),
3. lokalno kompajlirani vendor/potrace/bin/potrace.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import PathStyle, VectorPath, parse_path_d

STYLE_FILL = PathStyle(fill="#000000", stroke=None, layer_name="00_ENGRAVE")
STYLE_FILL_DETAIL = PathStyle(fill="#0000FF", stroke=None, layer_name="01_DETAIL")

POTRACE_ENV = "IMG2VEC_POTRACE"
_TRANSFORM_RE = re.compile(
    r"translate\(\s*([-+0-9.eE]+)[,\s]+([-+0-9.eE]+)\s*\)\s*"
    r"scale\(\s*([-+0-9.eE]+)[,\s]+([-+0-9.eE]+)\s*\)"
)
_PATH_D_RE = re.compile(r'<path\b[^>]*\bd="([^"]+)"', re.DOTALL)


class PotraceError(RuntimeError):
    """potrace nije pronadjen ili je vratio gresku."""


def find_potrace() -> Path | None:
    """Pronalazi potrace binar (env -> PATH -> vendor)."""
    override = os.environ.get(POTRACE_ENV)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("potrace")
    if found:
        return Path(found)
    roots = [Path(__file__).resolve().parents[2], Path.cwd()]
    for root in roots:
        candidate = root / "vendor" / "potrace" / "bin" / "potrace"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def potrace_available() -> bool:
    return find_potrace() is not None


def _write_pbm(target: Path, bitmask: np.ndarray) -> None:
    """Upisuje 1-bit PBM (P4) gde True/1 znaci crno (ink)."""
    if bitmask.ndim != 2:
        raise ValueError("PBM maska mora biti 2D")
    height, width = bitmask.shape
    packed = np.packbits(bitmask.astype(np.uint8), axis=1)
    with target.open("wb") as handle:
        handle.write(f"P4\n{width} {height}\n".encode("ascii"))
        handle.write(packed.tobytes())


@dataclass
class PotraceOptions:
    turdsize: int = 2       # potiskuje mrlje do ove velicine (px^2)
    opttolerance: float = 0.2  # tolerancija optimizacije krivih (px)
    unit: int = 10          # kvantizacija na 1/unit piksela
    timeout_s: float = 120.0


def _svg_to_paths(svg_text: str) -> list[VectorPath]:
    """Parsira potrace SVG i vraca putanje u koordinatama piksela (y nadole)."""
    transforms = _TRANSFORM_RE.findall(svg_text)
    if transforms:
        tx, ty, sx, sy = (float(v) for v in transforms[0])
        matrix = (sx, 0.0, 0.0, sy, tx, ty)
    else:  # pragma: no cover - odbrana za nestandardan potrace izlaz
        matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    paths: list[VectorPath] = []
    # Svaki <path> element potracea je JEDNA compound putanja: spoljne konture i
    # rupe su podputanje istog elementa (fill-rule evenodd). Grupisemo ih preko
    # meta["compound"] da bi ih SVG eksporter sacuvao u jednom path elementu.
    for compound_id, d in enumerate(_PATH_D_RE.findall(svg_text)):
        for path in parse_path_d(d):
            transformed = path.affine(matrix, style=STYLE_FILL)
            transformed.meta["compound"] = compound_id
            paths.append(transformed)
    return paths


def trace_bitmap(bitmask: np.ndarray, *, options: PotraceOptions | None = None,
                 potrace_bin: Path | None = None,
                 style: PathStyle = STYLE_FILL) -> list[VectorPath]:
    """Vektorizuje binarnu masku (True = crno) u glatke Bezier putanje.

    Vracene koordinate su u pikselima originalne bitmape (0,0 = gornji levi
    ugao), spremne za skaliranje u milimetre.
    """
    options = options or PotraceOptions()
    binary = find_potrace() if potrace_bin is None else potrace_bin
    if binary is None:
        raise PotraceError(
            "potrace nije pronadjen. Instaliraj ga (npr. 'pacman -S potrace', "
            "'brew install potrace') ili pokreni 'python scripts/setup_potrace.py' "
            "za lokalni build. Put do binar fajla moze se zadati i promenljivom "
            f"{POTRACE_ENV}."
        )
    if bitmask.ndim != 2:
        raise ValueError("Ocekivana 2D binarna maska")

    # Potrace ocekuje crno-na-belom; nasa maska je True = ink.
    binary_mask = np.ascontiguousarray(bitmask.astype(bool))
    if not binary_mask.any():
        return []

    with tempfile.TemporaryDirectory(prefix="img2vec-potrace-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.pbm"
        output_path = tmp_dir / "output.svg"
        _write_pbm(input_path, binary_mask)
        command = [
            str(binary), "--svg", "--flat",
            "--turdsize", str(int(options.turdsize)),
            "--opttolerance", str(float(options.opttolerance)),
            "--unit", str(int(options.unit)),
            "-o", str(output_path), str(input_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=options.timeout_s, check=False)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover
            raise PotraceError(f"potrace je prekoracio vreme od {options.timeout_s}s") from exc
        except (FileNotFoundError, PermissionError) as exc:
            raise PotraceError(f"potrace binar nije izvrsan: {binary}") from exc
        if result.returncode != 0:
            raise PotraceError(
                f"potrace greska (exit {result.returncode}): "
                f"{(result.stderr or result.stdout).strip()[:500]}")
        svg_text = output_path.read_text(encoding="utf-8", errors="replace")

    paths = _svg_to_paths(svg_text)
    if style is not STYLE_FILL:
        paths = [VectorPath(p.segments, p.closed, style, p.meta) for p in paths]
    return paths