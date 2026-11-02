#!/usr/bin/env python3
"""Cross-platform razvojni pokretac (uvicorn --reload + auto otvaranje UI-ja)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--reload", *sys.argv[1:]]))
