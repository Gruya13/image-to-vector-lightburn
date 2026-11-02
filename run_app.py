#!/usr/bin/env python3
"""Glavni lokalni launcher: podize FastAPI server i otvara Web UI u browseru.

Primeri:
    uv run python run_app.py
    uv run python run_app.py --port 9000 --no-browser
    uv run python run_app.py --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_PORT = 8765  # macOS AirPlay drzi 5000/7000


def find_free_port(start: int, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Nema slobodnog porta u opsegu {start}..{start + attempts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="adresa servera")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port (default {DEFAULT_PORT}, automatski fallback)")
    parser.add_argument("--no-browser", action="store_true",
                        help="ne otvaraj browser automatski")
    parser.add_argument("--reload", action="store_true",
                        help="uvicorn auto-reload (za razvoj)")
    args = parser.parse_args(argv)

    port = find_free_port(args.port) if args.host == "127.0.0.1" else args.port
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{port}"

    if not args.no_browser:
        def _open() -> None:
            webbrowser.open(url)
        threading.Timer(1.5, _open).start()

    print("=" * 72)
    print(" img2vec-lightburn :: lokalni server")
    print(f"   UI:   {url}")
    print(f"   API:  {url}/api/health")
    print("   Za prekid pritisni Ctrl+C")
    print("=" * 72)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("Nedostaje uvicorn. Pokreni: uv sync", file=sys.stderr)
        return 1
    uvicorn.run("backend.main:app", host=args.host, port=port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
