"""Eğitim sekmesi: arka plan eğitimi, canlı öğrenme eğrisi ve bölüm kaydı izleme."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from engine import analysis, storage
from engine.background import Job
from engine.i18n import tx
from engine.trainer import TrainConfig

from . import maze_view
from .notify import Notifier
from .plot_canvas import LiveLearningPlot

_SNAP_RE = re.compile(r"rep(\d+)_ep(\d+)\.json$")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class TrainTab(QWidget):
    """Eğitim sürecini ayrı süreçte (Job) yönetir ve canlı olarak izler."""

    training_finished = Signal(object)  # Path
    go_to_test_requested = Signal(object)  # Path

    def __init__(
        self,
        config_provider: Callable[[], TrainConfig],
        notifier: Notifier | None = None,
        parent: QWidget | None = None,
        *,
        results_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_provider = config_provider
        self._notifier = notifier
        self._results_root = results_root

        self._job: Job | None = None
        self._run_dir: Path | None = None
        self._start_time: float = 0.0
        self._total_episodes: int = 1
        self._total_repeats: int = 1
        self._completed_episodes: int = 0
        self._snapshot_entries: dict[int, Path] = {}
        self._live_file: Path | None = None
        self._live_episode: int = 0
        self._pending_live: tuple[int, Path] | None = None
        self._last_live_load_time: float = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

        self._build_ui()

    def _get_results_root(self) -> Path:
        return Path(self._results_root) if self._results_root is not None else Path(storage.RESULTS_ROOT)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ------------------------------------------------- Üst kontrol çubuğu
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.start_btn = QPushButton(tx("Eğit", "Train"))
        self.start_btn.setMinimumWidth(100)
        self.start_btn.setProperty("accent", "primary")
        self.start_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton(tx("Durdur", "Stop"))
        self.stop_btn.setMinimumWidth(90)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setProperty("accent", "danger")
        self.stop_btn.clicked.connect(self.stop_training)

        self.go_to_test_btn = QPushButton(tx("Teste geç", "Go to Test"))
        self.go_to_test_btn.setMinimumWidth(100)
        self.go_to_test_btn.setEnabled(False)
        self.go_to_test_btn.clicked.connect(self._on_go_to_test)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat(tx("Hazır", "Ready"))

        self.elapsed_label = QLabel(tx("Süre: 00:00", "Elapsed: 00:00"))
        self.elapsed_label.setMinimumWidth(95)
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.progress_bar, 1)
        ctrl_row.addWidget(self.elapsed_label)
        ctrl_row.addWidget(self.go_to_test_btn)
        root.addLayout(ctrl_row)

        self.status_label = QLabel(
            tx(
                "Eğitim başlatılmadı. Ayarlar sekmesinden parametreleri seçip 'Eğit'e basın.",
                "Training has not started. Select parameters in the Settings tab and press 'Train'.",
            )
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #475569;")
        root.addWidget(self.status_label)

        # ------------------------------------------------- Ana bölünmüş alan
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Sol: Kayıt seçici + ReplayPlayer
        left_box = QGroupBox(tx("Bölüm İzleme", "Episode Replay"))
        left_v = QVBoxLayout(left_box)
        left_v.setSpacing(6)

        snap_row = QHBoxLayout()
        snap_row.setSpacing(8)
        snap_row.addWidget(QLabel(tx("Kayıt:", "Recording:")))
        self.snapshot_combo = QComboBox()
        self.snapshot_combo.setMinimumWidth(160)
        self.snapshot_combo.currentIndexChanged.connect(self._on_snapshot_selected)
        snap_row.addWidget(self.snapshot_combo, 1)

        self.auto_live_check = QCheckBox(tx("En sonu otomatik göster", "Auto-show latest"))
        self.auto_live_check.setChecked(True)
        snap_row.addWidget(self.auto_live_check)
        left_v.addLayout(snap_row)

        self.replay_player = maze_view.ReplayPlayer(left_box)
        left_v.addWidget(self.replay_player, 1)
        splitter.addWidget(left_box)

        # Sağ: Canlı öğrenme eğrisi
        right_box = QGroupBox(tx("Canlı Öğrenme Eğrisi", "Live Learning Curve"))
        right_v = QVBoxLayout(right_box)
        right_v.setContentsMargins(6, 6, 6, 6)
        self.live_plot = LiveLearningPlot(right_box)
        right_v.addWidget(self.live_plot, 1)
        splitter.addWidget(right_box)

        splitter.setSizes([640, 680])

    # ------------------------------------------------------------ durum & yaşam döngüsü

    def is_running(self) -> bool:
        return self._job is not None and (self._timer.isActive() or self._job.is_alive())

    @property
    def last_run_dir(self) -> Path | None:
        return self._run_dir

    def start_training(self) -> bool:
        """Ayarlar sekmesinden TrainConfig alıp arka planda eğitimi başlatır."""
        if self.is_running():
            return False

        try:
            cfg = self._config_provider()
        except ValueError as exc:
            QMessageBox.warning(self, tx("Geçersiz Ayarlar", "Invalid Settings"), str(exc))
            return False

        if cfg.results_root is None:
            cfg = replace(cfg, results_root=self._get_results_root())

        self._run_dir = None
        self._total_episodes = max(1, cfg.episodes)
        self._total_repeats = max(1, cfg.repeats)
        self._completed_episodes = 0
        self._snapshot_entries.clear()
        self._live_file = None
        self._live_episode = 0
        self._pending_live = None
        self._last_live_load_time = time.monotonic()
        self._rebuild_snapshot_combo(select_key=None)

        total_steps = self._total_episodes * self._total_repeats
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {total_steps} (%p%)")
        self.elapsed_label.setText(tx("Süre: 00:00", "Elapsed: 00:00"))
        self.live_plot.reset(total_episodes=self._total_episodes)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.go_to_test_btn.setEnabled(False)
        label_part = f" ({cfg.label})" if cfg.label else ""
        self.status_label.setText(
            tx(
                f"Başlatılıyor: {cfg.agent_name}{label_part} · {cfg.repeats} tekrar × {cfg.episodes} bölüm...",
                f"Starting: {cfg.agent_name}{label_part} · {cfg.repeats} repeats × {cfg.episodes} episodes...",
            )
        )

        self._job = Job("train", cfg=cfg)
        self._job.start()
        self._start_time = time.monotonic()
        self._timer.start()
        return True

    def stop_training(self) -> None:
        """Çalışan eğitime nazik durdurma isteği gönderir."""
        if self._job is not None and self._job.is_alive():
            self._job.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText(
                tx(
                    "Durdurma isteği gönderildi (mevcut bölümün bitmesi bekleniyor)...",
                    "Stop request sent (waiting for the current episode to finish)...",
                )
            )

    def stop_and_wait(self, timeout_s: float = 3.0) -> None:
        """Pencere kapanırken çalışan eğitimi durdurur ve kısa süre bekler."""
        self._timer.stop()
        if hasattr(self.replay_player, "pause"):
            self.replay_player.pause()
        job = self._job
        if job is None:
            return
        if job.is_alive():
            job.stop()
            deadline = time.monotonic() + timeout_s
            while job.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            # Hâlâ hayattaysa zorla sonlandır (daemon süreç pencereyi askıda bırakmasın)
            proc = getattr(job, "_proc", None)
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        self._job = None

    # ------------------------------------------------------------ poll döngüsü

    def _poll(self) -> None:
        if self._job is None:
            self._timer.stop()
            return

        elapsed_str = _format_elapsed(time.monotonic() - self._start_time)
        self.elapsed_label.setText(tx(f"Süre: {elapsed_str}", f"Elapsed: {elapsed_str}"))
        events = self._job.poll()
        if not events:
            if not self._job.is_alive():
                self._finish_idle_unexpected()
            return

        plot_dirty = False
        snapshots_changed = False
        last_ep_info: tuple[int, int] | None = None
        done_ev: dict[str, Any] | None = None
        error_ev: dict[str, Any] | None = None

        for ev in events:
            etype = ev.get("type")
            if etype == "start":
                self._run_dir = Path(ev["run_dir"])
                self._total_episodes = max(1, int(ev["episodes"]))
                self._total_repeats = max(1, int(ev["repeats"]))
                self._last_live_load_time = time.monotonic()
                total = self._total_episodes * self._total_repeats
                self.progress_bar.setRange(0, total)
                self.status_label.setText(
                    tx(
                        f"Eğitiliyor: {ev.get('agent', '')} → {self._run_dir.name}",
                        f"Training: {ev.get('agent', '')} → {self._run_dir.name}",
                    )
                )
            elif etype == "episode":
                rep = int(ev["repeat"])
                ep = int(ev["episode"])
                self.live_plot.add_episode(rep, ep, bool(ev["success"]), float(ev["total_reward"]))
                self._completed_episodes += 1
                last_ep_info = (rep, ep)
                plot_dirty = True
            elif etype == "snapshot":
                ep = int(ev["episode"])
                self._snapshot_entries[ep] = Path(ev["path"])
                snapshots_changed = True
            elif etype == "live":
                ep = int(ev["episode"])
                path = Path(ev["path"])
                self._live_file = path
                self._live_episode = ep
                self._pending_live = (ep, path)
                snapshots_changed = True
            elif etype == "repeat_done":
                rep = int(ev["repeat"])
                self.status_label.setText(
                    tx(
                        f"Tekrar {rep + 1}/{self._total_repeats} tamamlandı, model kaydedildi.",
                        f"Repeat {rep + 1}/{self._total_repeats} completed, model saved.",
                    )
                )
            elif etype == "done":
                done_ev = ev
            elif etype == "error":
                error_ev = ev

        # Poll başına bir kez grafik ve ilerleme güncelle
        if last_ep_info is not None:
            rep, ep = last_ep_info
            total = self._total_episodes * self._total_repeats
            self.progress_bar.setValue(min(self._completed_episodes, total))
            self.progress_bar.setFormat(
                tx(
                    f"Tekrar {rep + 1}/{self._total_repeats} · Bölüm {ep}/{self._total_episodes} (%p%)",
                    f"Repeat {rep + 1}/{self._total_repeats} · Episode {ep}/{self._total_episodes} (%p%)",
                )
            )

        if plot_dirty:
            self.live_plot.refresh()

        if snapshots_changed:
            cur_key = self.snapshot_combo.currentData()
            self._rebuild_snapshot_combo(select_key=cur_key)

        if error_ev is not None:
            self._on_error(error_ev)
        elif done_ev is not None:
            self._on_done(done_ev)
        elif (
            self._pending_live is not None
            and self.auto_live_check.isChecked()
            and not self.replay_player.is_playing()
            and (time.monotonic() - self._last_live_load_time) >= 0.8
        ):
            ep, path = self._pending_live
            self._pending_live = None
            self._last_live_load_time = time.monotonic()
            self._select_live_and_load(path, ep)

    def _finish_idle_unexpected(self) -> None:
        self._timer.stop()
        self._job = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, ev: dict[str, Any]) -> None:
        self._timer.stop()
        self._job = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        msg = str(ev.get("message") or tx("Bilinmeyen eğitim hatası", "Unknown training error"))
        self.status_label.setText(tx(f"Hata: {msg}", f"Error: {msg}"))
        err_title = tx("Eğitim Hatası", "Training Error")
        if self._notifier is not None:
            self._notifier.notify(err_title, msg, error=True)
        QMessageBox.critical(self, err_title, msg)

    def _on_done(self, ev: dict[str, Any]) -> None:
        self._timer.stop()
        self._job = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        run_dir = Path(ev["run_dir"])
        self._run_dir = run_dir
        stopped = bool(ev.get("stopped"))

        # Diskteki bütün kayıtları senkronize et ve en son bölümü göster
        self._pending_live = None
        self._sync_snapshots_from_disk(run_dir)
        if self.auto_live_check.isChecked() and self._live_file is not None and self._live_file.is_file():
            self._select_live_and_load(self._live_file, self._live_episode)

        # Özet ve grafik kaydı
        summary_err = ""
        try:
            analysis.summarize(run_dir)
            analysis.save_all_plots(run_dir)
        except Exception as exc:  # noqa: BLE001
            summary_err = tx(
                f" (özet/grafik uyarısı: {exc})",
                f" (summary/plot warning: {exc})",
            )

        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        if stopped:
            title = tx("Eğitim Durduruldu", "Training Stopped")
            self.status_label.setText(
                tx(
                    f"Eğitim durduruldu ({elapsed}): {run_dir.name}{summary_err}",
                    f"Training stopped ({elapsed}): {run_dir.name}{summary_err}",
                )
            )
        else:
            title = tx("Eğitim Tamamlandı", "Training Completed")
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setFormat(tx("Tamamlandı (%p%)", "Completed (%p%)"))
            self.status_label.setText(
                tx(
                    f"Eğitim tamamlandı ({elapsed}): {run_dir.name}{summary_err}",
                    f"Training completed ({elapsed}): {run_dir.name}{summary_err}",
                )
            )

        self.go_to_test_btn.setEnabled(True)
        if self._notifier is not None:
            self._notifier.notify(title, f"{run_dir.name} ({elapsed})")
        self.training_finished.emit(run_dir)

    # ------------------------------------------------------------ kayıt seçici

    def _sync_snapshots_from_disk(self, run_dir: Path) -> None:
        for p in storage.list_snapshots(run_dir):
            m = _SNAP_RE.search(p.name)
            if m:
                ep = int(m.group(2))
                self._snapshot_entries[ep] = p
        lp = storage.live_path(run_dir)
        if lp.is_file():
            self._live_file = lp
        self._rebuild_snapshot_combo(select_key=self.snapshot_combo.currentData())

    def _rebuild_snapshot_combo(self, select_key: Any) -> None:
        self.snapshot_combo.blockSignals(True)
        self.snapshot_combo.clear()
        if self._live_file is not None and self._live_file.is_file():
            lbl = (
                tx(
                    f"En son bölüm (Bölüm {self._live_episode})",
                    f"Latest episode (Episode {self._live_episode})",
                )
                if self._live_episode
                else tx("En son bölüm", "Latest episode")
            )
            self.snapshot_combo.addItem(lbl, ("live", self._live_file))
        for ep in sorted(self._snapshot_entries):
            self.snapshot_combo.addItem(
                tx(f"Bölüm {ep}", f"Episode {ep}"),
                ("snap", ep, self._snapshot_entries[ep]),
            )

        if select_key is not None:
            idx = self.snapshot_combo.findData(select_key)
            if idx >= 0:
                self.snapshot_combo.setCurrentIndex(idx)
        self.snapshot_combo.blockSignals(False)

    def _select_live_and_load(self, path: Path, episode: int) -> None:
        if not path.is_file():
            return
        self.snapshot_combo.blockSignals(True)
        for i in range(self.snapshot_combo.count()):
            data = self.snapshot_combo.itemData(i)
            if isinstance(data, tuple) and data[0] == "live":
                self.snapshot_combo.setCurrentIndex(i)
                break
        self.snapshot_combo.blockSignals(False)
        self._load_trajectory_file(
            path,
            title=tx(f"En son bölüm (Bölüm {episode})", f"Latest episode (Episode {episode})"),
        )

    def _on_snapshot_selected(self) -> None:
        data = self.snapshot_combo.currentData()
        if not isinstance(data, tuple):
            return
        if data[0] == "live":
            path = Path(data[1])
            title = self.snapshot_combo.currentText()
        else:
            ep = int(data[1])
            path = Path(data[2])
            title = tx(f"Bölüm {ep}", f"Episode {ep}")
        self._load_trajectory_file(path, title=title)

    def _load_trajectory_file(self, path: Path, title: str) -> None:
        if not path.is_file():
            return
        try:
            traj = storage.load_trajectory(path)
            self.replay_player.load(traj, title=title, autoplay=True)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(tx(f"Kayıt yüklenemedi: {exc}", f"Could not load recording: {exc}"))

    def _on_go_to_test(self) -> None:
        if self._run_dir is not None:
            self.go_to_test_requested.emit(self._run_dir)
