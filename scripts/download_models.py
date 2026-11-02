#!/usr/bin/env python3
"""Preuzima ONNX modele pre prvog koriscenja (nikad tokom korisnickog zahteva).

Modeli se smestaju u ./models i kesiraju se (gitignore). Podrzano:
  * rembg: isnet-general-use (brzi), birefnet-general (maksimalni kvalitet)
  * Depth Anything V2 Small (Faza 3) - opciono, veliki fajl

Primeri:
  python scripts/download_models.py               # samo rembg isnet
  python scripts/download_models.py --all         # + birefnet + depth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

REMBG_MODELS = ("isnet-general-use", "birefnet-general")


def download_rembg(model_name: str) -> bool:
    os.environ.setdefault("U2NET_HOME", str(MODELS))
    MODELS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from rembg import new_session
    except ImportError as exc:  # pragma: no cover
        print(f"rembg nije instaliran: {exc}", file=sys.stderr)
        return False
    try:
        print(f"rembg model: {model_name} ...")
        new_session(model_name)
        print(f"  OK ({model_name})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  GRESKA: {exc}", file=sys.stderr)
        return False


def download_depth() -> bool:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from img2vec.phase3 import ensure_depth_model
    except ImportError as exc:  # pragma: no cover
        print(f"Phase 3 modul nije dostupan: {exc}", file=sys.stderr)
        return False
    try:
        path = ensure_depth_model()
        print(f"  OK ({path})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  GRESKA: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="preuzmi i birefnet i Depth Anything V2")
    parser.add_argument("--depth", action="store_true",
                        help="preuzmi samo Depth Anything V2")
    args = parser.parse_args()

    ok = True
    if not args.depth:
        for model in REMBG_MODELS if args.all else REMBG_MODELS[:1]:
            ok = download_rembg(model) and ok
    if args.all or args.depth:
        ok = download_depth() and ok
    print("\nModeli su u:", MODELS)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
