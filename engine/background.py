"""Eğitim/testi ayrı süreçte koşturur; arayüz olayları poll() ile alır.
Ayrıntı: engine/CONTRACT.md "background"."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Literal

_CTX = mp.get_context("spawn")
_TERMINAL = ("done", "error")


def _worker(kind: str, kwargs: dict[str, Any], events, stop_event) -> None:
    """Ayrı sürecin hedefi (Windows spawn için modül düzeyinde). Olayları kuyruğa koyar;
    istisna olursa son olay `error` olur ve süreç normal çıkar."""
    try:
        if kind == "train":
            from engine.trainer import train

            train(kwargs["cfg"], progress=events.put, stop=stop_event)
        elif kind == "test":
            from engine.tester import test_run

            test_run(Path(kwargs["run_dir"]), tuple(kwargs.get("kinds") or ("normal", "buyuk")),
                     progress=events.put, stop=stop_event)
        else:
            raise ValueError(f"Bilinmeyen iş türü: {kind}")
    except BaseException as exc:  # noqa: BLE001 - arayüze iletilmeli
        events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})


class Job:
    """kind="train": kwargs = {"cfg": TrainConfig}
    kind="test" : kwargs = {"run_dir": str|Path, "kinds": [...]}
    Ayrıntı: engine/CONTRACT.md "background"."""

    def __init__(self, kind: Literal["train", "test"], **kwargs: Any) -> None:
        if kind not in ("train", "test"):
            raise ValueError(f"Bilinmeyen iş türü: {kind!r} ('train' ya da 'test' olmalı)")
        need = "cfg" if kind == "train" else "run_dir"
        if need not in kwargs:
            raise ValueError(f"Job('{kind}') için '{need}' gerekli")
        self.kind = kind
        self.kwargs = kwargs
        self._events = _CTX.Queue()
        self._stop = _CTX.Event()
        self._proc: Any = None
        self._finished = False  # done/error olayı poll ile teslim edildi

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Job zaten başlatıldı")
        self._proc = _CTX.Process(
            target=_worker, args=(self.kind, self.kwargs, self._events, self._stop), daemon=True
        )
        self._proc.start()

    def poll(self, max_events: int = 10_000) -> list[dict[str, Any]]:
        """Biriken olayları döndürür, beklemez."""
        out: list[dict[str, Any]] = []
        if self._proc is None:
            return out
        # Önce yaşıyor mu bak, sonra boşalt: süreç ölmüşse bütün veri boruda demektir
        alive = self._proc.is_alive()
        while len(out) < max_events:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        if any(ev.get("type") in _TERMINAL for ev in out):
            self._finished = True
        if not alive and not self._finished and len(out) < max_events:
            code = self._proc.exitcode
            if code not in (0, None):
                out.append({
                    "type": "error",
                    "message": f"Süreç beklenmedik biçimde sonlandı (çıkış kodu {code})",
                    "traceback": "",
                })
                self._finished = True
        return out

    def stop(self) -> None:
        """Nazik durdurma isteği (bölüm biter, model kaydedilir)."""
        self._stop.set()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()
