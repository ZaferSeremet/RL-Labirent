"""Sonuç klasörünün biçimi: okuma/yazma yardımcıları (SÖZLEŞME — imzaları değiştirme).

results/<run_id>/
    ayarlar.json            çalışmanın bütün ayarları (tekrar üretilebilirlik)
    egitim_log.csv          her eğitim bölümü bir satır (TRAIN_FIELDS)
    model_rep<r>.pkl        r. tekrarın eğitilmiş ajanı
    kayitlar/rep0_ep<e>.json izleme için bölüm kayıtları (5 eşit aralıklı)
    kayitlar/son.json       eğitim sırasında en son kaydedilen bölüm (üzerine yazılır)
    test_<kind>.csv         test sonuçları, kind = normal | buyuk (TEST_FIELDS)
    ozet.json               analysis.summarize() çıktısı
    grafikler/*.png         analysis.save_all_plots() çıktısı
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .settings import ROOT

RESULTS_ROOT = ROOT / "results"

TRAIN_FIELDS = (
    "repeat", "episode", "total_reward", "steps", "success",
    "wall_hits", "visited_tiles", "shortest_path", "episode_seed",
)
TEST_FIELDS = (
    "method", "is_baseline", "repeat", "seed", "total_reward", "steps", "success",
    "wall_hits", "visited_tiles", "shortest_path", "efficiency",
)
_INT_FIELDS = {"repeat", "episode", "steps", "wall_hits", "visited_tiles", "shortest_path", "episode_seed", "seed"}
_FLOAT_FIELDS = {"total_reward", "efficiency"}
_BOOL_FIELDS = {"success", "is_baseline"}


# ----------------------------------------------------------------- klasörler

def new_run_dir(agent_name: str, label: str = "", root: Path | None = None) -> Path:
    """results/<YYYY-MM-DD_HH-MM-SS>_<ajan>[_<etiket>] oluşturur (varsa sonuna _2, _3 ...)."""
    root = Path(root or RESULTS_ROOT)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [stamp, _slug(agent_name)] + ([_slug(label)] if label else [])
    base = root / "_".join(parts)
    path, i = base, 2
    while path.exists():
        path = Path(f"{base}_{i}")
        i += 1
    (path / "kayitlar").mkdir(parents=True)
    return path


def list_runs(root: Path | None = None) -> list[Path]:
    """ayarlar.json içeren çalışma klasörleri, yeniden eskiye."""
    root = Path(root or RESULTS_ROOT)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if (p / "ayarlar.json").is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def check_deletable(run_dir: Path, root: Path | None = None) -> Path:
    """Silinecek klasörün gerçekten sonuç kökünün doğrudan altındaki bir çalışma
    olduğunu doğrular (yanlışlıkla başka bir klasör silinmesin). Çözülmüş yolu döndürür."""
    root_r = Path(root or RESULTS_ROOT).resolve()
    run_r = Path(run_dir).resolve()
    if run_r.parent != root_r or not (run_r / "ayarlar.json").is_file():
        raise ValueError(f"Sonuç klasörü değil, silinmedi: {run_dir}")
    return run_r


def delete_run(run_dir: Path, root: Path | None = None) -> None:
    """Çalışma klasörünü kalıcı olarak siler (arayüz önce Geri Dönüşüm Kutusu'nu dener)."""
    shutil.rmtree(check_deletable(run_dir, root))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-") or "x"


# ----------------------------------------------------------------- ayarlar

def write_settings(run_dir: Path, settings: dict[str, Any]) -> None:
    with open(Path(run_dir) / "ayarlar.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2, default=_json_default)


def read_settings(run_dir: Path) -> dict[str, Any]:
    with open(Path(run_dir) / "ayarlar.json", encoding="utf-8") as f:
        return json.load(f)


def update_settings(run_dir: Path, **fields: Any) -> dict[str, Any]:
    s = read_settings(run_dir)
    s.update(fields)
    write_settings(run_dir, s)
    return s


# ----------------------------------------------------------------- CSV

class CsvAppender:
    """Satır satır CSV yazar; dosya yoksa başlık ekler. `with` ile kullan."""

    def __init__(self, path: Path, fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.fields = tuple(fields)
        new = not self.path.exists()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fields)
        if new:
            self._w.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._w.writerow({k: _fmt(row.get(k)) for k in self.fields})

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "CsvAppender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Satırları tiplerine çevirerek okur (int / float / bool; boş -> None)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append({k: _parse(k, v) for k, v in raw.items()})
    return rows


def train_log_path(run_dir: Path) -> Path:
    return Path(run_dir) / "egitim_log.csv"


def test_csv_path(run_dir: Path, kind: str) -> Path:
    return Path(run_dir) / f"test_{kind}.csv"


def model_path(run_dir: Path, repeat: int) -> Path:
    return Path(run_dir) / f"model_rep{repeat}.pkl"


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _parse(k: str, v: str) -> Any:
    if v == "" or v is None:
        return None
    if k in _BOOL_FIELDS:
        return v in ("1", "True", "true")
    if k in _INT_FIELDS:
        return int(float(v))
    if k in _FLOAT_FIELDS:
        return float(v)
    return v


# ----------------------------------------------------------------- bölüm kayıtları

def save_trajectory(path: Path, traj: dict[str, Any]) -> Path:
    """engine.episode.run_episode(record=True) çıktısını JSON'a yazar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(traj, f, default=_json_default)
    # Okuyan taraf yarım dosya görmesin diye önce geçici dosyaya yazılıp yer değiştirilir.
    # Windows'ta hedef o an başka süreçte (arayüz) açıksa replace PermissionError verir:
    # kısa aralıklarla yeniden dene.
    for attempt in range(20):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)
    return path


def load_trajectory(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    t["grid"] = np.asarray(t["grid"], dtype=np.uint8)
    for k in ("start", "exit"):
        t[k] = tuple(t[k])
    t["positions"] = [tuple(p) for p in t["positions"]]
    return t


def snapshot_path(run_dir: Path, repeat: int, episode: int) -> Path:
    return Path(run_dir) / "kayitlar" / f"rep{repeat}_ep{episode:06d}.json"


def live_path(run_dir: Path) -> Path:
    return Path(run_dir) / "kayitlar" / "son.json"


def list_snapshots(run_dir: Path) -> list[Path]:
    """Eşit aralıklı kayıtlar, bölüm sırasıyla (son.json hariç)."""
    return sorted((Path(run_dir) / "kayitlar").glob("rep*_ep*.json"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON'a yazılamıyor: {type(o)}")
