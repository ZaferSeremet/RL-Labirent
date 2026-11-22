"""Test sekmesi: kayıtlı çalışmaları normal/büyük labirentlerde test etme ve izleme."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine import analysis, registry, settings as engine_settings, storage, tester
from engine.background import Job

from . import maze_view
from .notify import Notifier

_STATUS_LABELS: dict[str, str] = {
    "tamamlandi": "Tamamlandı",
    "durduruldu": "Durduruldu",
    "egitiliyor": "Eğitiliyor",
    "hata": "Hata",
}

_TABLE_HEADERS = ("Yöntem", "Başarı", "Ort. Adım", "Verim", "Ort. Ödül", "Çarpma")


def format_run_display(run_dir: Path, st: dict[str, Any] | None = None) -> str:
    """Çalışma seçici için açıklayıcı tek satır ad üretir."""
    if st is None:
        try:
            st = storage.read_settings(run_dir)
        except Exception:  # noqa: BLE001
            return run_dir.name
    agent = (st.get("agent") or {}).get("name") or (st.get("train_config") or {}).get("agent_name") or "?"
    label = str((st.get("train_config") or {}).get("label") or "").strip() or "-"
    created = str(st.get("created") or "").replace("T", " ")[:16] or "-"
    raw_status = str(st.get("durum") or "-")
    durum = _STATUS_LABELS.get(raw_status, raw_status)
    return f"{agent} | {label} | {created} | {durum}"


def _fmt_pct(val: float | None, std: float | None = None) -> str:
    if val is None:
        return "-"
    base = f"%{val * 100:.1f}"
    if std is not None and std > 0:
        return f"{base} (±%{std * 100:.1f})"
    return base


def _fmt_num(val: float | None, digits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


class TestTab(QWidget):
    """Eğitilmiş çalışmaları test eder, sonuç tablolarını gösterir ve tek bölüm izletir."""

    testing_finished = Signal(object)  # Path

    def __init__(
        self,
        notifier: Notifier | None = None,
        parent: QWidget | None = None,
        *,
        results_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._notifier = notifier
        self._results_root = results_root

        self._job: Job | None = None
        self._active_run_dir: Path | None = None
        self._current_settings: dict[str, Any] = {}
        self._baseline_names: set[str] = set()
        self._completed: int = 0
        self._total: int = 1

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

        self._build_ui()
        self.refresh_runs()

    def _get_results_root(self) -> Path:
        return Path(self._results_root) if self._results_root is not None else Path(storage.RESULTS_ROOT)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ------------------------------------------------- Üst kontrol çubuğu
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("Çalışma:"))

        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(200)
        self.run_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.setMinimumContentsLength(18)
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        top_row.addWidget(self.run_combo, 1)

        self.refresh_btn = QPushButton("Yenile")
        self.refresh_btn.clicked.connect(lambda: self.refresh_runs())
        top_row.addWidget(self.refresh_btn)

        self.normal_check = QCheckBox("Normal")
        self.normal_check.setChecked(True)
        self.buyuk_check = QCheckBox("Büyük")
        self.buyuk_check.setChecked(True)
        top_row.addWidget(self.normal_check)
        top_row.addWidget(self.buyuk_check)

        self.start_btn = QPushButton("Test et")
        self.start_btn.setMinimumWidth(90)
        self.start_btn.setProperty("accent", "primary")
        self.start_btn.clicked.connect(self.start_test)
        top_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setMinimumWidth(84)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setProperty("accent", "danger")
        self.stop_btn.clicked.connect(self.stop_test)
        top_row.addWidget(self.stop_btn)
        root.addLayout(top_row)

        # İlerleme satırı
        prog_row = QHBoxLayout()
        prog_row.setSpacing(8)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Hazır")
        self.status_label = QLabel("Test etmek için bir çalışma seçin.")
        self.status_label.setStyleSheet("color: #475569;")
        prog_row.addWidget(self.progress_bar, 1)
        prog_row.addWidget(self.status_label, 1)
        root.addLayout(prog_row)

        # ------------------------------------------------- Ana bölünmüş alan
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Sol: İki test tablosu (Normal ve Büyük ayrı)
        tables_container = QWidget()
        tables_v = QVBoxLayout(tables_container)
        tables_v.setContentsMargins(0, 0, 0, 0)
        tables_v.setSpacing(8)

        normal_box = QGroupBox("Normal Labirent Sonuçları")
        normal_v = QVBoxLayout(normal_box)
        self.normal_table = self._create_result_table()
        normal_v.addWidget(self.normal_table)
        tables_v.addWidget(normal_box, 1)

        buyuk_box = QGroupBox("Büyük Labirent Sonuçları")
        buyuk_v = QVBoxLayout(buyuk_box)
        self.buyuk_table = self._create_result_table()
        buyuk_v.addWidget(self.buyuk_table)
        tables_v.addWidget(buyuk_box, 1)

        splitter.addWidget(tables_container)

        # Sağ: Bölüm İzleme
        replay_box = QGroupBox("Test Bölümü İzleme")
        replay_v = QVBoxLayout(replay_box)
        replay_v.setSpacing(6)

        sel_row1 = QHBoxLayout()
        sel_row1.setSpacing(6)
        sel_row1.addWidget(QLabel("Tür:"))
        self.replay_kind_combo = QComboBox()
        self.replay_kind_combo.addItem("Normal", "normal")
        self.replay_kind_combo.addItem("Büyük", "buyuk")
        self.replay_kind_combo.currentIndexChanged.connect(self._populate_seeds)
        sel_row1.addWidget(self.replay_kind_combo)

        sel_row1.addWidget(QLabel("Yöntem:"))
        self.replay_method_combo = QComboBox()
        self.replay_method_combo.setMinimumWidth(110)
        self.replay_method_combo.currentIndexChanged.connect(self._on_replay_method_changed)
        sel_row1.addWidget(self.replay_method_combo, 1)
        replay_v.addLayout(sel_row1)

        sel_row2 = QHBoxLayout()
        sel_row2.setSpacing(6)
        sel_row2.addWidget(QLabel("Tekrar:"))
        self.replay_repeat_combo = QComboBox()
        sel_row2.addWidget(self.replay_repeat_combo)

        sel_row2.addWidget(QLabel("Seed:"))
        self.replay_seed_combo = QComboBox()
        self.replay_seed_combo.setMinimumWidth(90)
        sel_row2.addWidget(self.replay_seed_combo, 1)

        self.replay_btn = QPushButton("İzle")
        self.replay_btn.setMinimumWidth(78)
        self.replay_btn.clicked.connect(self.replay_selected_episode)
        sel_row2.addWidget(self.replay_btn)
        replay_v.addLayout(sel_row2)

        self.replay_player = maze_view.ReplayPlayer(replay_box)
        replay_v.addWidget(self.replay_player, 1)

        splitter.addWidget(replay_box)
        splitter.setSizes([760, 580])

    @staticmethod
    def _create_result_table() -> QTableWidget:
        table = QTableWidget(0, len(_TABLE_HEADERS))
        table.setHorizontalHeaderLabels(list(_TABLE_HEADERS))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setMinimumSectionSize(58)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        return table

    # ------------------------------------------------------------ çalışma listesi

    def selected_run_dir(self) -> Path | None:
        data = self.run_combo.currentData()
        return Path(data) if data else None

    def refresh_runs(self, select_dir: Path | None = None) -> None:
        """Sonuç klasöründeki çalışmaları yeniden tarar."""
        target = select_dir or self.selected_run_dir()
        runs = storage.list_runs(self._get_results_root())

        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for rdir in runs:
            self.run_combo.addItem(format_run_display(rdir), str(rdir))
        if target is not None:
            idx = self.run_combo.findData(str(Path(target)))
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)
        self.run_combo.blockSignals(False)

        self._on_run_selected()

    def select_run(self, run_dir: Path) -> bool:
        """Belirtilen çalışmayı seçer (listede yoksa yenileyip dener)."""
        target = str(Path(run_dir))
        idx = self.run_combo.findData(target)
        if idx < 0:
            self.refresh_runs(select_dir=run_dir)
            idx = self.run_combo.findData(target)
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)
            return True
        return False

    def _on_run_selected(self) -> None:
        run_dir = self.selected_run_dir()
        if run_dir is None or not (run_dir / "ayarlar.json").is_file():
            self._current_settings = {}
            self._fill_table(self.normal_table, {})
            self._fill_table(self.buyuk_table, {})
            self.replay_method_combo.clear()
            self.replay_repeat_combo.clear()
            self.replay_seed_combo.clear()
            self.replay_btn.setEnabled(False)
            return

        self._current_settings = storage.read_settings(run_dir)
        self._load_results_tables(run_dir)
        self._populate_replay_selectors(run_dir)
        self.replay_btn.setEnabled(True)

    def _load_results_tables(self, run_dir: Path) -> None:
        try:
            loaded = analysis.load_run(run_dir)
            summary = analysis.summarize_loaded(loaded)
        except Exception:  # noqa: BLE001
            self._fill_table(self.normal_table, {})
            self._fill_table(self.buyuk_table, {})
            return

        test_sum = summary.get("test") or {}
        self._fill_table(self.normal_table, test_sum.get("normal") or {})
        self._fill_table(self.buyuk_table, test_sum.get("buyuk") or {})

    @staticmethod
    def _fill_table(table: QTableWidget, methods_map: dict[str, dict[str, Any]]) -> None:
        table.setRowCount(0)
        for method, st in methods_map.items():
            row = table.rowCount()
            table.insertRow(row)
            cells = (
                method,
                _fmt_pct(st.get("basari"), st.get("basari_std")),
                _fmt_num(st.get("adim"), 1),
                _fmt_num(st.get("verim"), 2),
                _fmt_num(st.get("odul"), 2),
                _fmt_num(st.get("carpma"), 1),
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, col, item)

    # ------------------------------------------------------------ bölüm izleme seçicileri

    def _populate_replay_selectors(self, run_dir: Path) -> None:
        st = self._current_settings
        agent_name = str((st.get("agent") or {}).get("name") or "")
        reg = registry.discover()
        baselines = [b.name for b in reg.baselines()]
        self._baseline_names = set(baselines)

        self.replay_method_combo.blockSignals(True)
        self.replay_method_combo.clear()
        if agent_name:
            self.replay_method_combo.addItem(agent_name, agent_name)
        for bname in baselines:
            if bname != agent_name:
                self.replay_method_combo.addItem(f"{bname} (baseline)", bname)
        self.replay_method_combo.blockSignals(False)

        # Mevcut model tekrarları
        n_rep = int((st.get("train_config") or {}).get("repeats", 1))
        avail_reps = [r for r in range(n_rep) if storage.model_path(run_dir, r).is_file()]
        if not avail_reps:
            avail_reps = [0]
        self.replay_repeat_combo.clear()
        for r in avail_reps:
            self.replay_repeat_combo.addItem(f"Tekrar {r}", r)

        self._on_replay_method_changed()
        self._populate_seeds()

    def _on_replay_method_changed(self) -> None:
        method = self.replay_method_combo.currentData()
        is_base = method in self._baseline_names
        self.replay_repeat_combo.setEnabled(not is_base)
        if is_base and self.replay_repeat_combo.count() > 0:
            self.replay_repeat_combo.setCurrentIndex(0)

    def _populate_seeds(self) -> None:
        kind = str(self.replay_kind_combo.currentData() or "normal")
        shared = self._current_settings.get("shared") or engine_settings.load_shared_config()
        try:
            seeds = engine_settings.seeds_for(kind, shared)
        except Exception:  # noqa: BLE001
            seeds = []
        self.replay_seed_combo.clear()
        for s in seeds:
            self.replay_seed_combo.addItem(str(s), int(s))

    def replay_selected_episode(self) -> bool:
        """Seçili tür, yöntem, tekrar ve seed için test bölümünü yeniden oynatır."""
        run_dir = self.selected_run_dir()
        if run_dir is None:
            return False
        kind = str(self.replay_kind_combo.currentData() or "normal")
        method = str(self.replay_method_combo.currentData() or "")
        repeat = int(self.replay_repeat_combo.currentData() or 0)
        seed = self.replay_seed_combo.currentData()
        if not method or seed is None:
            return False

        try:
            traj = tester.replay_test_episode(run_dir, kind, method, repeat, int(seed))
            title = f"{kind} · {method} · tekrar {repeat} · seed {seed}"
            self.replay_player.load(traj, title=title, autoplay=True)
            self.status_label.setText(f"İzleniyor: {title}")
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Bölüm İzlenemedi", str(exc))
            return False

    # ------------------------------------------------------------ arka plan testi

    def is_running(self) -> bool:
        return self._job is not None and (self._timer.isActive() or self._job.is_alive())

    def start_test(self) -> bool:
        if self.is_running():
            return False

        run_dir = self.selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, "Çalışma Seçilmedi", "Lütfen test edilecek bir çalışma seçin.")
            return False

        kinds: list[str] = []
        if self.normal_check.isChecked():
            kinds.append("normal")
        if self.buyuk_check.isChecked():
            kinds.append("buyuk")
        if not kinds:
            QMessageBox.warning(self, "Test Türü Seçilmedi", "En az bir test türü (Normal veya Büyük) seçin.")
            return False

        self._active_run_dir = run_dir
        self._completed = 0
        self._total = 1
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Başlatılıyor...")
        self.status_label.setText(f"Test başlatılıyor: {run_dir.name} ({', '.join(kinds)})")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.run_combo.setEnabled(False)

        self._job = Job("test", run_dir=run_dir, kinds=kinds)
        self._job.start()
        self._timer.start()
        return True

    def stop_test(self) -> None:
        if self._job is not None and self._job.is_alive():
            self._job.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Test durduruluyor...")

    def stop_and_wait(self, timeout_s: float = 3.0) -> None:
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
            proc = getattr(job, "_proc", None)
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        self._job = None

    def _poll(self) -> None:
        if self._job is None:
            self._timer.stop()
            return

        events = self._job.poll()
        if not events:
            if not self._job.is_alive():
                self._timer.stop()
                self._job = None
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.run_combo.setEnabled(True)
            return

        last_text = ""
        kind_done = False
        done_ev: dict[str, Any] | None = None
        error_ev: dict[str, Any] | None = None

        for ev in events:
            etype = ev.get("type")
            if etype == "start":
                self._total = max(1, int(ev.get("total") or 1))
                self.progress_bar.setRange(0, self._total)
                self.progress_bar.setValue(0)
            elif etype == "test_episode":
                self._completed += 1
                last_text = (
                    f"{ev.get('kind')} · {ev.get('method')} "
                    f"(tekrar {ev.get('repeat')}, seed {ev.get('seed')})"
                )
            elif etype == "kind_done":
                kind_done = True
            elif etype == "done":
                done_ev = ev
            elif etype == "error":
                error_ev = ev

        if last_text:
            self.progress_bar.setValue(min(self._completed, self._total))
            self.progress_bar.setFormat(f"{self._completed} / {self._total} (%p%)")
            self.status_label.setText(last_text)

        if kind_done and self._active_run_dir is not None:
            self._load_results_tables(self._active_run_dir)

        if error_ev is not None:
            self._on_error(error_ev)
        elif done_ev is not None:
            self._on_done(done_ev)

    def _on_error(self, ev: dict[str, Any]) -> None:
        self._timer.stop()
        self._job = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.run_combo.setEnabled(True)
        msg = str(ev.get("message") or "Bilinmeyen test hatası")
        self.status_label.setText(f"Hata: {msg}")
        if self._notifier is not None:
            self._notifier.notify("Test Hatası", msg, error=True)
        QMessageBox.critical(self, "Test Hatası", msg)

    def _on_done(self, ev: dict[str, Any]) -> None:
        self._timer.stop()
        self._job = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.run_combo.setEnabled(True)

        run_dir = Path(ev["run_dir"])
        stopped = bool(ev.get("stopped"))

        summary_err = ""
        try:
            analysis.summarize(run_dir)
            analysis.save_all_plots(run_dir)
        except Exception as exc:  # noqa: BLE001
            summary_err = f" (özet/grafik uyarısı: {exc})"

        self._load_results_tables(run_dir)
        if stopped:
            title = "Test Durduruldu"
            self.status_label.setText(f"Test durduruldu: {run_dir.name}{summary_err}")
        else:
            title = "Test Tamamlandı"
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setFormat("Tamamlandı (%p%)")
            self.status_label.setText(f"Test tamamlandı: {run_dir.name}{summary_err}")

        if self._notifier is not None:
            self._notifier.notify(title, run_dir.name)
        self.testing_finished.emit(run_dir)
