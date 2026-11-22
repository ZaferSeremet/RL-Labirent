"""Ana pencere: Ayarlar, Eğitim, Test, Sonuçlar ve Oyna sekmelerini birleştirir."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QPushButton, QTabWidget

from engine import i18n
from engine.i18n import tx

from . import play_tab
from .notify import Notifier, app_icon
from .results_tab import ResultsTab
from .settings_tab import SettingsTab
from .test_tab import TestTab
from .theme import apply_light_theme
from .train_tab import TrainTab

TAB_TITLES: tuple[str, ...] = ("Ayarlar", "Eğitim", "Test", "Sonuçlar", "Oyna")
TAB_TITLES_EN: tuple[str, ...] = ("Settings", "Training", "Test", "Results", "Play")


def tab_titles() -> tuple[str, ...]:
    return TAB_TITLES_EN if i18n.is_english() else TAB_TITLES


APP_STYLESHEET = """
QMainWindow {
    background: #f8fafc;
}
QTabWidget::pane {
    border: 1px solid #cbd5e1;
    background: #ffffff;
    border-radius: 4px;
}
QTabBar::tab {
    background: #e2e8f0;
    color: #1e293b;
    padding: 7px 18px;
    margin-right: 3px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-bottom: 2px solid #2563eb;
    font-weight: 600;
}
QGroupBox {
    font-weight: 600;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    padding: 5px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    background: #f8fafc;
    color: #0f172a;
}
QPushButton:hover {
    background: #e2e8f0;
}
QPushButton:disabled {
    color: #94a3b8;
    background: #f1f5f9;
}
QPushButton[accent="primary"] {
    background: #2563eb;
    color: #ffffff;
    border-color: #1d4ed8;
    font-weight: 600;
}
QPushButton[accent="primary"]:hover {
    background: #1d4ed8;
}
QPushButton[accent="primary"]:disabled {
    background: #93c5fd;
    border-color: #93c5fd;
    color: #f8fafc;
}
QPushButton[accent="danger"] {
    background: #dc2626;
    color: #ffffff;
    border-color: #b91c1c;
    font-weight: 600;
}
QPushButton[accent="danger"]:hover {
    background: #b91c1c;
}
QPushButton[accent="danger"]:disabled {
    background: #fca5a5;
    border-color: #fca5a5;
    color: #fef2f2;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    text-align: center;
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 500;
    min-height: 22px;
}
QProgressBar::chunk {
    background: #93c5fd;
    border-radius: 4px;
}
QTableWidget {
    gridline-color: #e2e8f0;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QHeaderView::section {
    background: #f1f5f9;
    color: #1e293b;
    padding: 5px 8px;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #cbd5e1;
    font-weight: 600;
}
"""


class MainWindow(QMainWindow):
    """RL Labirent masaüstü uygulamasının ana penceresi."""

    def __init__(
        self,
        parent=None,
        *,
        results_root: Path | None = None,
        persist_language: bool = False,
    ) -> None:
        super().__init__(parent)
        self._results_root = results_root
        # Testlerde dil seçimi diske yazılmasın; app.py True verir
        self._persist_language = persist_language

        self.setWindowIcon(app_icon(self))
        self.resize(1400, 850)
        self.setMinimumSize(920, 620)
        apply_light_theme()  # app.py dışından açılsa da (testler) aynı tema
        self.setStyleSheet(APP_STYLESHEET)

        self.notifier = Notifier(self)
        self.tabs: QTabWidget | None = None
        self._build_tabs()

    # ------------------------------------------------------------ kurulum / dil

    def _build_tabs(self, current_index: int = 0) -> None:
        """Sekmeleri o anki dilde (baştan) kurar."""
        # Sayı biçimi de dile uysun (İngilizcede 0.05, Türkçede sistem ayarı: 0,05)
        locale = (QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
                  if i18n.is_english() else QLocale.system())
        QLocale.setDefault(locale)
        self.setLocale(locale)  # alt widget'lar yerel ayarı pencereden alır
        self.setWindowTitle(tx("RL Labirent", "RL Maze"))
        self.statusBar().showMessage(tx("Hazır", "Ready"))
        results_root = self._results_root

        old = self.tabs
        self.tabs = QTabWidget(self)

        self.lang_btn = QPushButton(tx("English", "Türkçe"), self.tabs)
        self.lang_btn.setToolTip(tx("Arayüzü ve çıktıları İngilizceye çevir",
                                    "Switch the interface and outputs to Turkish"))
        self.lang_btn.clicked.connect(self.toggle_language)
        self.tabs.setCornerWidget(self.lang_btn, Qt.Corner.TopRightCorner)

        self.settings_tab = SettingsTab(self, results_root=results_root)
        self.train_tab = TrainTab(
            self.settings_tab.build_config,
            notifier=self.notifier,
            parent=self,
            results_root=results_root,
        )
        self.test_tab = TestTab(
            notifier=self.notifier,
            parent=self,
            results_root=results_root,
        )
        self.results_tab = ResultsTab(
            notifier=self.notifier,
            parent=self,
            results_root=results_root,
            busy_dirs=self._busy_run_dirs,
        )
        self.play_tab = play_tab.PlayTab(self)

        for widget, title in zip(
            (self.settings_tab, self.train_tab, self.test_tab, self.results_tab, self.play_tab),
            tab_titles(),
        ):
            self.tabs.addTab(widget, title)

        # Sekmeler arası bağlantılar
        self.settings_tab.train_requested.connect(self._go_to_train_tab)
        self.train_tab.training_finished.connect(self._on_training_finished)
        self.train_tab.go_to_test_requested.connect(self._go_to_test_with_run)
        self.test_tab.testing_finished.connect(self._on_testing_finished)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self.tabs)  # eskisini Qt siler
        if old is not None:
            self.tabs.setCurrentIndex(current_index)

    def set_language(self, lang: str) -> bool:
        """Dili değiştirip sekmeleri yeniden kurar. Eğitim/test sürerken değiştirmez."""
        if lang == i18n.get_language():
            return True
        if self.train_tab.is_running() or self.test_tab.is_running():
            QMessageBox.warning(
                self,
                tx("Dil değiştirilemez", "Cannot change language"),
                tx("Eğitim ya da test sürerken dil değiştirilemez. Bitmesini bekleyin ya da durdurun.",
                   "The language cannot be changed while training or testing is running. "
                   "Wait for it to finish or stop it."),
            )
            return False
        index = self.tabs.currentIndex()
        i18n.set_language(lang, persist=self._persist_language)
        self._build_tabs(index)
        return True

    def toggle_language(self) -> bool:
        return self.set_language("tr" if i18n.is_english() else "en")

    def _busy_run_dirs(self) -> list[Path]:
        """O an arka planda yazılan çalışma klasörleri (Sonuçlar'dan silinemez)."""
        busy: list[Path] = []
        if self.train_tab.is_running() and self.train_tab.last_run_dir() is not None:
            busy.append(self.train_tab.last_run_dir())
        if self.test_tab.is_running() and self.test_tab._active_run_dir is not None:
            busy.append(self.test_tab._active_run_dir)
        return busy

    def _go_to_train_tab(self) -> None:
        self.tabs.setCurrentWidget(self.train_tab)

    def _on_training_finished(self, run_dir: Path) -> None:
        if not self.test_tab.is_running():
            self.test_tab.refresh_runs(select_dir=run_dir)
        self.results_tab.refresh_runs()

    def _go_to_test_with_run(self, run_dir: Path) -> None:
        self.tabs.setCurrentWidget(self.test_tab)
        if not self.test_tab.is_running():
            self.test_tab.refresh_runs(select_dir=run_dir)

    def _on_testing_finished(self, _run_dir: Path) -> None:
        self.results_tab.refresh_runs()

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is self.test_tab and not self.test_tab.is_running():
            self.test_tab.refresh_runs()
        elif widget is self.results_tab:
            self.results_tab.refresh_runs()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.train_tab.stop_and_wait(timeout_s=3.0)
        self.test_tab.stop_and_wait(timeout_s=3.0)
        self.notifier.cleanup()
        super().closeEvent(event)
