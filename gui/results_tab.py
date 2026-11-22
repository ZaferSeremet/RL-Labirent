"""Sonuçlar sekmesi: çalışma listesi, karşılaştırmalı grafikler, içe/dışa aktarma ve rapor."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QFile, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine import analysis, report, storage
from engine.i18n import kind_short, tx

from .notify import Notifier
from .plot_canvas import PlotCanvas

_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "tamamlandi": ("Tamamlandı", "Completed"),
    "durduruldu": ("Durduruldu", "Stopped"),
    "egitiliyor": ("Eğitiliyor", "Training"),
    "hata": ("Hata", "Error"),
}


def _status_label(raw: str) -> str:
    pair = _STATUS_LABELS.get(raw)
    return tx(*pair) if pair else raw


def _plot_options() -> tuple[tuple[str, str], ...]:
    return (
        ("learning_success", tx("Öğrenme Eğrisi — Başarı", "Learning Curve — Success")),
        ("learning_reward", tx("Öğrenme Eğrisi — Ödül", "Learning Curve — Reward")),
        ("learning_steps", tx("Öğrenme Eğrisi — Adım", "Learning Curve — Steps")),
        ("baseline_normal", tx("Baseline Karşılaştırma — Normal", "Baseline Comparison — Normal")),
        ("baseline_buyuk", tx("Baseline Karşılaştırma — Büyük", "Baseline Comparison — Large")),
        ("generalization", tx("Genelleme (Normal vs Büyük)", "Generalization (Normal vs Large)")),
        ("compare_normal", tx("Çalışma Karşılaştırma — Normal", "Run Comparison — Normal")),
        ("compare_buyuk", tx("Çalışma Karşılaştırma — Büyük", "Run Comparison — Large")),
    )


def _columns() -> tuple[str, ...]:
    return (
        tx("Seç", "Select"),
        tx("Ajan", "Agent"),
        tx("Etiket", "Label"),
        tx("Durum", "Status"),
        tx("Eğitim Son Başarı", "Final Training Success"),
        tx(f"Test ({kind_short('normal')})", f"Test ({kind_short('normal')})"),
        tx(f"Test ({kind_short('buyuk')})", f"Test ({kind_short('buyuk')})"),
    )


def _runs_word(n: int) -> str:
    return "run" if n == 1 else "runs"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return tx(f"%{val * 100:.1f}", f"{val * 100:.1f}%")


def _agent_test_success(test_kind_dict: dict[str, dict[str, Any]], agent_name: str) -> float | None:
    if not test_kind_dict:
        return None
    if agent_name in test_kind_dict:
        return test_kind_dict[agent_name].get("basari")
    for method, stats in test_kind_dict.items():
        if method not in ("random", "sag_el"):
            return stats.get("basari")
    first = next(iter(test_kind_dict.values()), None)
    return first.get("basari") if first else None


class ResultsTab(QWidget):
    """Kayıtlı çalışmaları listeler, işaretlileri çizer, rapor/zip dışa aktarır ve siler."""

    def __init__(
        self,
        notifier: Notifier | None = None,
        parent: QWidget | None = None,
        *,
        results_root: Path | None = None,
        busy_dirs: Callable[[], list[Path]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._notifier = notifier
        self._results_root = results_root
        # O an eğitim/test sürecinin yazdığı klasörler: bunlar silinemez
        self._busy_dirs = busy_dirs or (lambda: [])
        self._run_dirs: list[Path] = []
        self._updating_table = False

        self._build_ui()
        self.refresh_runs()

    def _get_results_root(self) -> Path:
        return Path(self._results_root) if self._results_root is not None else Path(storage.RESULTS_ROOT)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ------------------------------------------------- Üst araç çubuğu (SİLME BUTONU YOK)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.refresh_btn = QPushButton(tx("Yenile", "Refresh"))
        self.refresh_btn.clicked.connect(self.refresh_runs)
        toolbar.addWidget(self.refresh_btn)

        self.import_btn = QPushButton(tx("İçe aktar", "Import"))
        import_menu = QMenu(self.import_btn)
        import_menu.addAction(tx("Zip dosyası seç...", "Select zip file..."), self.import_zip_dialog)
        import_menu.addAction(tx("Klasör seç...", "Select folder..."), self.import_folder_dialog)
        self.import_btn.setMenu(import_menu)
        toolbar.addWidget(self.import_btn)

        self.export_zip_btn = QPushButton(tx("Zip dışa aktar", "Export zip"))
        self.export_zip_btn.setToolTip(
            tx("Seçili çalışmayı zip dosyası olarak dışa aktar", "Export the selected run as a zip file")
        )
        self.export_zip_btn.clicked.connect(self.export_selected_zip)
        toolbar.addWidget(self.export_zip_btn)

        self.export_report_btn = QPushButton(tx("Rapor dışa aktar", "Export report"))
        self.export_report_btn.setProperty("accent", "primary")
        self.export_report_btn.clicked.connect(self.export_report_dialog)
        toolbar.addWidget(self.export_report_btn)

        self.open_folder_btn = QPushButton(tx("Klasörü aç", "Open folder"))
        self.open_folder_btn.clicked.connect(self.open_selected_folder)
        toolbar.addWidget(self.open_folder_btn)

        self.delete_btn = QPushButton(tx("Sil", "Delete"))
        self.delete_btn.setProperty("accent", "danger")
        self.delete_btn.setToolTip(
            tx(
                "İşaretli çalışmaları (işaret yoksa seçili satırı) Geri Dönüşüm Kutusu'na taşır",
                "Moves checked runs (or the selected row if none checked) to the Trash / Recycle Bin",
            )
        )
        self.delete_btn.clicked.connect(lambda: self.delete_runs())
        toolbar.addWidget(self.delete_btn)

        toolbar.addStretch(1)

        self.select_all_btn = QPushButton(tx("Tümünü seç", "Select all"))
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.clear_sel_btn = QPushButton(tx("Temizle", "Clear"))
        self.clear_sel_btn.setToolTip(tx("Tüm işaretlemeleri kaldır", "Uncheck all runs"))
        self.clear_sel_btn.clicked.connect(lambda: self._set_all_checked(False))
        toolbar.addWidget(self.select_all_btn)
        toolbar.addWidget(self.clear_sel_btn)
        root.addLayout(toolbar)

        # ------------------------------------------------- Dikey bölünmüş alan (Tablo üstte, Grafik altta)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Çalışma listesi
        cols = _columns()
        self.runs_table = QTableWidget(0, len(cols))
        self.runs_table.setHorizontalHeaderLabels(list(cols))
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.verticalHeader().setVisible(False)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for col in range(3, len(cols)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.runs_table.itemChanged.connect(self._on_table_item_changed)
        splitter.addWidget(self.runs_table)

        # Grafik alanı
        plot_panel = QWidget()
        plot_v = QVBoxLayout(plot_panel)
        plot_v.setContentsMargins(0, 4, 0, 0)
        plot_v.setSpacing(6)

        plot_bar = QHBoxLayout()
        plot_bar.setSpacing(8)
        plot_bar.addWidget(QLabel(tx("Grafik:", "Plot:")))
        self.plot_combo = QComboBox()
        for key, title in _plot_options():
            self.plot_combo.addItem(title, key)
        self.plot_combo.setMinimumWidth(260)
        self.plot_combo.currentIndexChanged.connect(self.update_plot)
        plot_bar.addWidget(self.plot_combo)

        self.plot_info_label = QLabel("")
        self.plot_info_label.setStyleSheet("color: #475569;")
        plot_bar.addWidget(self.plot_info_label, 1)
        plot_v.addLayout(plot_bar)

        self.plot_canvas = PlotCanvas(plot_panel, toolbar=True)
        plot_v.addWidget(self.plot_canvas, 1)
        splitter.addWidget(plot_panel)

        splitter.setSizes([250, 490])

    # ------------------------------------------------------------ liste yenileme

    def _read_summary_fast(self, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            settings = storage.read_settings(run_dir)
        except Exception:  # noqa: BLE001
            settings = {}

        ozet_path = run_dir / "ozet.json"
        if ozet_path.is_file():
            try:
                ozet_mtime = ozet_path.stat().st_mtime
                stale = any(
                    p.is_file() and p.stat().st_mtime > ozet_mtime
                    for p in (
                        storage.train_log_path(run_dir),
                        storage.test_csv_path(run_dir, "normal"),
                        storage.test_csv_path(run_dir, "buyuk"),
                    )
                )
                if not stale:
                    with open(ozet_path, encoding="utf-8") as f:
                        return settings, json.load(f)
            except Exception:  # noqa: BLE001
                pass

        try:
            loaded = analysis.load_run(run_dir)
            return settings, analysis.summarize_loaded(loaded)
        except Exception:  # noqa: BLE001
            return settings, {"agent": "", "label": "", "egitim": {}, "test": {}}

    def refresh_runs(self) -> None:
        """Çalışma listesini yeniden yükler; önceki işaretlemeleri korur."""
        prev_checked = {str(p) for p in self.checked_run_dirs()}
        had_rows = len(self._run_dirs) > 0

        self._run_dirs = storage.list_runs(self._get_results_root())
        self._updating_table = True
        self.runs_table.setRowCount(0)

        for idx, rdir in enumerate(self._run_dirs):
            settings, summary = self._read_summary_fast(rdir)
            agent = summary.get("agent") or (settings.get("agent") or {}).get("name") or "-"
            label = summary.get("label") or (settings.get("train_config") or {}).get("label") or "-"
            raw_durum = str(settings.get("durum") or "-")
            durum = _status_label(raw_durum)

            egitim = summary.get("egitim") or {}
            test_map = summary.get("test") or {}
            train_sr = egitim.get("son_basari")
            norm_sr = _agent_test_success(test_map.get("normal") or {}, str(agent))
            buyuk_sr = _agent_test_success(test_map.get("buyuk") or {}, str(agent))

            row = self.runs_table.rowCount()
            self.runs_table.insertRow(row)

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            should_check = (str(rdir) in prev_checked) or (not had_rows and idx == 0)
            chk_item.setCheckState(Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked)
            chk_item.setData(Qt.ItemDataRole.UserRole, str(rdir))
            self.runs_table.setItem(row, 0, chk_item)

            values = (
                str(agent),
                str(label),
                durum,
                _fmt_pct(train_sr),
                _fmt_pct(norm_sr),
                _fmt_pct(buyuk_sr),
            )
            for col_offset, val in enumerate(values, start=1):
                item = QTableWidgetItem(val)
                item.setToolTip(rdir.name)
                if col_offset >= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.runs_table.setItem(row, col_offset, item)

        self._updating_table = False
        self.update_plot()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() != 0:
            return
        self.update_plot()

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating_table = True
        for row in range(self.runs_table.rowCount()):
            it = self.runs_table.item(row, 0)
            if it is not None:
                it.setCheckState(state)
        self._updating_table = False
        self.update_plot()

    def checked_run_dirs(self) -> list[Path]:
        out: list[Path] = []
        for row in range(self.runs_table.rowCount()):
            it = self.runs_table.item(row, 0)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                raw = it.data(Qt.ItemDataRole.UserRole)
                if raw:
                    out.append(Path(raw))
        return out

    def _primary_run_dir(self) -> Path | None:
        row = self.runs_table.currentRow()
        if 0 <= row < self.runs_table.rowCount():
            it = self.runs_table.item(row, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole):
                return Path(it.data(Qt.ItemDataRole.UserRole))
        checked = self.checked_run_dirs()
        return checked[0] if checked else None

    # ------------------------------------------------------------ grafik çizimi

    def update_plot(self) -> None:
        """İşaretli çalışmalara ve seçilen grafik türüne göre matplotlib Figure çizer."""
        run_dirs = self.checked_run_dirs()
        if not run_dirs:
            self.plot_info_label.setText("")
            self.plot_canvas.show_message(
                tx(
                    "Grafik çizmek için yukarıdan en az bir çalışma işaretleyin",
                    "Check at least one run above to display a plot",
                )
            )
            return

        key = str(self.plot_combo.currentData() or "learning_success")
        loaded = [analysis.load_run(rd) for rd in run_dirs]
        n_loaded = len(loaded)

        try:
            if key == "learning_success":
                fig = analysis.learning_curve(loaded, metric="success")
                self.plot_info_label.setText(
                    tx(f"{n_loaded} çalışma gösteriliyor", f"Showing {n_loaded} {_runs_word(n_loaded)}")
                )
            elif key == "learning_reward":
                fig = analysis.learning_curve(loaded, metric="total_reward")
                self.plot_info_label.setText(
                    tx(f"{n_loaded} çalışma gösteriliyor", f"Showing {n_loaded} {_runs_word(n_loaded)}")
                )
            elif key == "learning_steps":
                fig = analysis.learning_curve(loaded, metric="steps")
                self.plot_info_label.setText(
                    tx(f"{n_loaded} çalışma gösteriliyor", f"Showing {n_loaded} {_runs_word(n_loaded)}")
                )
            elif key == "baseline_normal":
                fig = analysis.baseline_bars(loaded[0], kind="normal")
                self.plot_info_label.setText(
                    tx(f"Tek çalışma grafiği: {run_dirs[0].name}", f"Single run plot: {run_dirs[0].name}")
                )
            elif key == "baseline_buyuk":
                fig = analysis.baseline_bars(loaded[0], kind="buyuk")
                self.plot_info_label.setText(
                    tx(f"Tek çalışma grafiği: {run_dirs[0].name}", f"Single run plot: {run_dirs[0].name}")
                )
            elif key == "generalization":
                fig = analysis.generalization_plot(loaded[0])
                self.plot_info_label.setText(
                    tx(f"Tek çalışma grafiği: {run_dirs[0].name}", f"Single run plot: {run_dirs[0].name}")
                )
            elif key == "compare_normal":
                fig = analysis.compare_runs(loaded, kind="normal")
                self.plot_info_label.setText(
                    tx(f"{n_loaded} çalışma karşılaştırılıyor", f"Comparing {n_loaded} {_runs_word(n_loaded)}")
                )
            elif key == "compare_buyuk":
                fig = analysis.compare_runs(loaded, kind="buyuk")
                self.plot_info_label.setText(
                    tx(f"{n_loaded} çalışma karşılaştırılıyor", f"Comparing {n_loaded} {_runs_word(n_loaded)}")
                )
            else:
                self.plot_canvas.show_message(
                    tx(f"Bilinmeyen grafik türü: {key}", f"Unknown plot type: {key}")
                )
                return
        except Exception as exc:  # noqa: BLE001
            self.plot_canvas.show_message(
                tx(f"Grafik çizilemedi: {exc}", f"Failed to draw plot: {exc}")
            )
            return

        self.plot_canvas.set_figure(fig)

    # ------------------------------------------------------------ içe / dışa aktarma

    def import_zip_dialog(self) -> list[Path]:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            tx("Çalışma Zip Dosyası Seç", "Select Run Zip File"),
            "",
            tx("Zip Dosyaları (*.zip);;Tüm Dosyalar (*)", "Zip Files (*.zip);;All Files (*)"),
        )
        if not files:
            return []
        return self._do_import([Path(f) for f in files])

    def import_folder_dialog(self) -> list[Path]:
        folder = QFileDialog.getExistingDirectory(
            self,
            tx("İçe Aktarılacak Çalışma Klasörü Seç", "Select Run Folder to Import"),
        )
        if not folder:
            return []
        return self._do_import([Path(folder)])

    def _do_import(self, sources: list[Path]) -> list[Path]:
        try:
            imported = report.import_runs(sources, root=self._get_results_root())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tx("İçe Aktarma Hatası", "Import Error"), str(exc))
            return []
        self.refresh_runs()
        if self._notifier is not None:
            n = len(imported)
            self._notifier.notify(
                tx("İçe Aktarıldı", "Imported"),
                tx(f"{n} çalışma eklendi.", f"{n} {_runs_word(n)} added."),
            )
        return imported

    def export_selected_zip(self) -> Path | None:
        run_dir = self._primary_run_dir()
        if run_dir is None:
            QMessageBox.warning(
                self,
                tx("Çalışma Seçilmedi", "No Run Selected"),
                tx("Lütfen dışa aktarılacak bir çalışma seçin.", "Please select a run to export."),
            )
            return None

        default_name = f"{run_dir.name}.zip"
        out_file, _ = QFileDialog.getSaveFileName(
            self,
            tx("Zip Olarak Dışa Aktar", "Export as Zip"),
            default_name,
            tx("Zip Dosyaları (*.zip)", "Zip Files (*.zip)"),
        )
        if not out_file:
            return None

        try:
            saved = report.export_run_zip(run_dir, Path(out_file))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tx("Zip Dışa Aktarma Hatası", "Zip Export Error"), str(exc))
            return None

        if self._notifier is not None:
            self._notifier.notify(tx("Zip Oluşturuldu", "Zip Created"), str(saved.name))
        return saved

    def export_report_dialog(self) -> Path | None:
        run_dirs = self.checked_run_dirs()
        if not run_dirs:
            primary = self._primary_run_dir()
            if primary is not None:
                run_dirs = [primary]
        if not run_dirs:
            QMessageBox.warning(
                self,
                tx("Çalışma Seçilmedi", "No Run Selected"),
                tx("Rapor için en az bir çalışma işaretleyin.", "Check at least one run for the report."),
            )
            return None

        out_dir = QFileDialog.getExistingDirectory(
            self,
            tx(
                "Raporun kaydedileceği yeri seçin (içinde yeni bir klasör açılır)",
                "Choose where to save the report (a new folder is created inside)",
            ),
        )
        if not out_dir:
            return None

        try:
            html_path = report.export_report(run_dirs, Path(out_dir))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tx("Rapor Hatası", "Report Error"), str(exc))
            return None

        if self._notifier is not None:
            self._notifier.notify(
                tx("Rapor Oluşturuldu", "Report Created"),
                tx(
                    f"Klasör: {html_path.parent.name}",
                    f"Folder: {html_path.parent.name}",
                ),
            )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path.resolve())))
        return html_path

    def delete_runs(self, run_dirs: list[Path] | None = None, *, confirm: bool = True) -> list[Path]:
        """Çalışmaları Geri Dönüşüm Kutusu'na taşır. Silinenleri döndürür."""
        if run_dirs is None:
            run_dirs = self.checked_run_dirs()
            if not run_dirs:
                primary = self._primary_run_dir()
                run_dirs = [primary] if primary is not None else []
        if not run_dirs:
            QMessageBox.warning(
                self,
                tx("Çalışma Seçilmedi", "No Run Selected"),
                tx(
                    "Silinecek çalışmaları işaretleyin ya da bir satır seçin.",
                    "Check the runs to delete or select a row.",
                ),
            )
            return []

        busy = {Path(p).resolve() for p in self._busy_dirs() if p is not None}
        blocked = [d for d in run_dirs if Path(d).resolve() in busy]
        run_dirs = [d for d in run_dirs if Path(d).resolve() not in busy]
        if blocked:
            QMessageBox.warning(
                self,
                tx("Silinemez", "Cannot Delete"),
                tx(
                    "Şu an eğitilen ya da test edilen çalışma silinemez:\n",
                    "A run currently being trained or tested cannot be deleted:\n",
                )
                + "\n".join(d.name for d in blocked),
            )
        if not run_dirs:
            return []

        if confirm:
            names = "\n".join(f"• {d.name}" for d in run_dirs[:15])
            if len(run_dirs) > 15:
                extra = len(run_dirs) - 15
                names += tx(
                    f"\n… ve {extra} çalışma daha",
                    f"\n… and {extra} more {_runs_word(extra)}",
                )
            n = len(run_dirs)
            answer = QMessageBox.question(
                self,
                tx("Çalışmaları Sil", "Delete Runs"),
                tx(
                    f"{n} çalışma Geri Dönüşüm Kutusu'na taşınacak:\n\n{names}\n\n"
                    "Gerekirse oradan geri yüklenebilir. Devam edilsin mi?",
                    f"{n} {_runs_word(n)} will be moved to the Trash / Recycle Bin:\n\n{names}\n\n"
                    "They can be restored from there if needed. Continue?",
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return []

        root = self._get_results_root()
        deleted: list[Path] = []
        failed: list[str] = []
        for d in run_dirs:
            try:
                target = storage.check_deletable(d, root)
            except ValueError as exc:
                failed.append(str(exc))
                continue
            if QFile.moveToTrash(str(target)):
                deleted.append(d)
            else:
                failed.append(
                    tx(
                        f"{d.name}: Geri Dönüşüm Kutusu'na taşınamadı (dosya açık olabilir)",
                        f"{d.name}: Could not move to Trash / Recycle Bin (a file may be open)",
                    )
                )

        self.refresh_runs()
        if failed:
            QMessageBox.warning(self, tx("Bazıları Silinemedi", "Some Runs Could Not Be Deleted"), "\n".join(failed))
        win = self.window()
        if deleted and hasattr(win, "statusBar"):
            n = len(deleted)
            win.statusBar().showMessage(
                tx(
                    f"{n} çalışma Geri Dönüşüm Kutusu'na taşındı",
                    f"{n} {_runs_word(n)} moved to Trash / Recycle Bin",
                ),
                5000,
            )
        return deleted

    def open_selected_folder(self) -> None:
        target = self._primary_run_dir() or self._get_results_root()
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
