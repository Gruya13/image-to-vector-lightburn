#!/usr/bin/env python3
"""Lokalni build potrace binarnog fajla (bez sistemskog paket menadzera).

Preuzima potrace 1.16 izvor, kompajlira ga i smesta binarni fajl u
vendor/potrace/bin/potrace. Na Linuxu i macOS-u zahteva gcc/cc + make.
Na Windowsu koristi sistemski potrace (npr. preko chocolatey) ili stavi putanju
u promenljivu okruzenja IMG2VEC_POTRACE.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "potrace"
SOURCE_URL = "https://potrace.sourceforge.net/download/1.16/potrace-1.16.tar.gz"


def find_existing() -> Path | None:
    system = shutil.which("potrace")
    if system:
        return Path(system)
    local = VENDOR / "bin" / "potrace"
    return local if local.is_file() and os.access(local, os.X_OK) else None


def build() -> Path:
    if sys.platform.startswith("win"):
        raise SystemExit(
            "Automatski build nije podrzan na Windowsu. Instaliraj potrace "
            "(npr. 'choco install potrace') ili postavi IMG2VEC_POTRACE.")
    if not shutil.which("make"):
        raise SystemExit("Nedostaje 'make' (potreban za build potrace).")
    if not (shutil.which("cc") or shutil.which("gcc")):
        raise SystemExit("Nedostaje C kompajler (cc/gcc) za build potrace.")

    VENDOR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="potrace-build-") as tmp:
        archive = Path(tmp) / "potrace.tar.gz"
        print(f"Preuzimanje: {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, archive)  # noqa: S310
        subprocess.run(["tar", "xzf", str(archive)], cwd=tmp, check=True)
        source_dir = Path(tmp) / "potrace-1.16"
        print("Konfiguracija (bez sistemske instalacije)...")
        subprocess.run(["./configure", "--disable-shared"], cwd=source_dir,
                       check=True, stdout=subprocess.DEVNULL)
        print("Kompajliranje...")
        subprocess.run(["make", "-j", str(os.cpu_count() or 1)], cwd=source_dir,
                       check=True, stdout=subprocess.DEVNULL)
        target = VENDOR / "bin"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "src" / "potrace", target / "potrace")
        os.chmod(target / "potrace", 0o755)
    return target / "potrace"


def main() -> int:
    existing = find_existing()
    if existing is not None:
        print(f"potrace je vec dostupan: {existing}")
        subprocess.run([str(existing), "--version"], check=False)
        return 0
    try:
        binary = build()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Build nije uspeo: {exc}", file=sys.stderr)
        print("Alternativa: instaliraj potrace preko sistemskog paket menadzera "
              "(pacman -S potrace / brew install potrace / apt install potrace).",
              file=sys.stderr)
        return 1
    print(f"Uspesno: {binary}")
    subprocess.run([str(binary), "--version"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
