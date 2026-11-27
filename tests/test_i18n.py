"""Dil değiştirme: modül, ana pencere ve İngilizce modda Türkçe kalan metin denetimi."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from engine import i18n  # noqa: E402
from tests.i18n_check import looks_turkish, turkish_leftovers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_language():
    i18n.set_language("tr")
    yield
    i18n.set_language("tr")


def test_tx_and_labels():
    assert i18n.tx("Eğit", "Train") == "Eğit"
    i18n.set_language("en")
    assert i18n.tx("Eğit", "Train") == "Train"
    assert i18n.kind_label("buyuk") == "Large Maze"
    assert i18n.method_label("sag_el") == "Right-hand"
    assert i18n.method_label("sarsa") == "sarsa"
    with pytest.raises(ValueError):
        i18n.set_language("de")


def test_language_persist(tmp_path, monkeypatch):
    path = tmp_path / "kullanici_ayarlari.json"
    monkeypatch.setattr(i18n, "USER_SETTINGS_PATH", path)
    assert i18n.load_saved_language() == "tr"
    i18n.set_language("en", persist=True)
    assert json.loads(path.read_text(encoding="utf-8"))["dil"] == "en"
    i18n.set_language("tr")
    assert i18n.load_saved_language() == "en"
    path.write_text("bozuk", encoding="utf-8")
    assert i18n.load_saved_language() == "tr"


def test_looks_turkish_helper():
    assert looks_turkish("Eğitim başlat")
    assert looks_turkish("Yenile")
    assert not looks_turkish("Start training")
    assert not looks_turkish("sag_el")
    assert not looks_turkish("Türkçe")


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from engine import storage
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setattr(storage, "RESULTS_ROOT", root)
    monkeypatch.setattr(i18n, "USER_SETTINGS_PATH", tmp_path / "k.json")
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    win = MainWindow(results_root=root)
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def test_window_switches_language_and_rebuilds_tabs(window):
    assert window.tabs.tabText(0) == "Ayarlar"
    window.tabs.setCurrentIndex(3)
    assert window.toggle_language()
    assert i18n.get_language() == "en"
    assert [window.tabs.tabText(i) for i in range(5)] == ["Settings", "Training", "Test", "Results", "Final Test"]
    assert window.tabs.currentIndex() == 3
    assert window.windowTitle() == "RL Maze"
    assert window.lang_btn.text() == "Türkçe"
    window.toggle_language()
    assert window.tabs.tabText(0) == "Ayarlar"


TAB_ATTRS = ("settings_tab", "train_tab", "test_tab", "results_tab", "final_tab")


@pytest.mark.parametrize("attr", TAB_ATTRS)
def test_no_turkish_left_in_english_tab(window, attr):
    window.set_language("en")
    leftovers = turkish_leftovers(getattr(window, attr))
    assert not leftovers, "\n".join(leftovers)
