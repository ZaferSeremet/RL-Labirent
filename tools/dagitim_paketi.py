"""Ekibe gönderilecek zip paketini hazırlar.

Yalnız git'te izlenen dosyalar girer: .venv, results/ içindeki sonuçlar ve
kullanici_ayarlari.json girmez. Commit'lenmemiş değişiklikler de pakete girer
(yeni, henüz git'e eklenmemiş dosyalar girmez; araç bunları listeler).

    .venv\\Scripts\\python tools\\dagitim_paketi.py

Çıktı: dagitim/RL-Labirent_<tarih>_<commit>.zip. Zip'in içinde tek bir
RL-Labirent/ klasörü ve sürümü yazan SURUM.txt bulunur.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dagitim"
PREFIX = "RL-Labirent/"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def main() -> int:
    try:
        head = git("rev-parse", "--short", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"git çalışmadı: {exc}")
        return 1

    # Commit'lenmemiş değişiklikler varsa onları da içeren geçici bir commit nesnesi
    # (çalışma alanına ve dallara dokunmaz); yoksa HEAD
    stash = git("stash", "create")
    tree = stash or "HEAD"
    dirty = bool(stash)

    untracked = [p for p in git("ls-files", "--others", "--exclude-standard").splitlines() if p]
    if untracked:
        print("UYARI: git'e eklenmemiş şu dosyalar pakete GİRMEDİ:")
        for p in untracked:
            print(f"  {p}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    version = f"{head}{'+degisiklik' if dirty else ''}"
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"RL-Labirent_{stamp}_{version}.zip"

    surum = OUT_DIR / "SURUM.txt"
    surum.write_text(
        f"RL Labirent\nSürüm: {version}\nHazırlanma: {datetime.now():%Y-%m-%d %H:%M}\n\n"
        "Kurulum ve kullanım: README.md\n",
        encoding="utf-8",
    )
    try:
        git(
            "archive", "--format=zip", f"--prefix={PREFIX}",
            f"--add-file={surum}", "-o", str(out), tree,
        )
    finally:
        surum.unlink(missing_ok=True)

    size_mb = out.stat().st_size / 1e6
    print(f"Hazır: {out.relative_to(ROOT)}  ({size_mb:.1f} MB, sürüm {version})")
    if dirty:
        print("Not: commit'lenmemiş değişiklikler pakette var; sürüm adında '+degisiklik' yazıyor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
