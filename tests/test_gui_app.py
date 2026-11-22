"""Ana pencere ve sekmeler için offscreen PySide6 testleri (pytest-qt yok)."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from engine import settings as engine_settings
from engine import storage, trainer
from engine.trainer import TrainConfig
from gui.main_window import MainWindow
from gui.settings_tab import parse_hparams


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_until(predicate, timeout_s: float = 30.0, step_ms: int = 50) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(step_ms)
    app.processEvents()
    return bool(predicate())


def _small_shared_config() -> dict:
    cfg = copy.deepcopy(engine_settings.load_shared_config())
    cfg["test"]["normal"]["maze_size"] = 4
    cfg["test"]["normal"]["seeds"] = [10000, 10004]  # 5 seed
    cfg["test"]["buyuk"]["maze_size"] = 5
    cfg["test"]["buyuk"]["seeds"] = [20000, 20002]  # 3 seed
    return cfg


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "results"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "RESULTS_ROOT", root)
    return root


@pytest.fixture
def modal_messages(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """Offscreen testlerde modal QMessageBox penceresi olay döngüsünü kilitlemesin."""
    captured: list[tuple[str, str, str]] = []

    def _cap(kind: str):
        def _fn(_parent, title, text, *a, **kw):
            captured.append((kind, str(title), str(text)))
            return QMessageBox.StandardButton.Ok
        return _fn

    monkeypatch.setattr(QMessageBox, "critical", _cap("critical"))
    monkeypatch.setattr(QMessageBox, "warning", _cap("warning"))
    return captured


@pytest.fixture
def window(isolated_root: Path, modal_messages: list[tuple[str, str, str]]):
    _app()
    win = MainWindow(results_root=isolated_root)
    yield win
    win.close()
    _app().processEvents()


def test_main_window_structure_and_tab_order(window: MainWindow):
    assert window.windowTitle() == "RL Labirent"
    assert window.tabs.count() == 5
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["Ayarlar", "Eğitim", "Test", "Sonuçlar", "Oyna"]

    # Sonuçlar sekmesinde silme butonu var
    result_buttons = [b.text() for b in window.results_tab.findChildren(QPushButton)]
    assert "Sil" in result_buttons


def test_settings_tab_builds_train_config_and_validates(window: MainWindow):
    st = window.settings_tab

    # Baseline'lar ajan listesinde olmamalı, ornek ajan olmalı
    listed_agents = [st.agent_combo.itemData(i) for i in range(st.agent_combo.count())]
    assert "ornek" in listed_agents
    assert "random" not in listed_agents
    assert "sag_el" not in listed_agents

    idx = st.agent_combo.findData("ornek")
    st.agent_combo.setCurrentIndex(idx)
    st.label_edit.setText("deneme-1")
    st.episodes_spin.setValue(40)
    st.repeats_spin.setValue(2)
    st.seed_spin.setValue(7)
    st.snapshots_spin.setValue(4)

    # Başlangıçta hiçbir ödül/ortam değişmedi -> override sözlükleri boş
    cfg_default = st.build_config()
    assert cfg_default.agent_name == "ornek"
    assert cfg_default.label == "deneme-1"
    assert cfg_default.episodes == 40
    assert cfg_default.repeats == 2
    assert cfg_default.seed == 7
    assert cfg_default.snapshots == 4
    assert cfg_default.reward_overrides == {}
    assert cfg_default.env_overrides == {}

    # Yalnız değişen ödül ve ortam alanları override'a girmeli
    st.reward_spins["step"].setValue(-0.05)
    st.reward_spins["new_tile"].setValue(0.1)
    st.maze_size_spin.setValue(6)
    st.hparams_edit.setPlainText("alpha=0.25\ngamma=0.95\nflag=true\nmode=greedy\n# yorum\n")

    cfg = st.build_config()
    assert cfg.reward_overrides == {"step": pytest.approx(-0.05), "new_tile": pytest.approx(0.1)}
    assert cfg.env_overrides == {"maze_size": 6}
    assert cfg.hparams == {"alpha": 0.25, "gamma": 0.95, "flag": True, "mode": "greedy"}

    # JSON hiperparametre desteği
    st.hparams_edit.setPlainText('{"alpha": 0.05, "layers": [32, 32]}')
    cfg_json = st.build_config()
    assert cfg_json.hparams == {"alpha": 0.05, "layers": [32, 32]}

    # Varsayılana dön butonu ödülleri sıfırlar
    st.reset_rewards_btn.click()
    assert st.build_config().reward_overrides == {}

    # Hatalı hiperparametre girişi -> uyarı görünür ve ValueError yükselir
    st.hparams_edit.setPlainText("bozuk_satir_esittir_yok")
    _app().processEvents()
    assert not st.hparams_warning_label.isHidden()
    assert st.hparams_warning_label.text()
    with pytest.raises(ValueError):
        st.build_config()

    # parse_hparams doğrudan kenar durumları
    with pytest.raises(ValueError):
        parse_hparams('{"bozuk": ')
    with pytest.raises(ValueError):
        parse_hparams("123=45")


def test_train_tab_runs_small_training(
    window: MainWindow,
    isolated_root: Path,
    modal_messages: list[tuple[str, str, str]],
):
    st = window.settings_tab
    idx = st.agent_combo.findData("ornek")
    assert idx >= 0
    st.agent_combo.setCurrentIndex(idx)
    st.label_edit.setText("gui-egitim")
    st.episodes_spin.setValue(20)
    st.repeats_spin.setValue(1)
    st.snapshots_spin.setValue(4)
    st.maze_size_spin.setValue(4)

    tt = window.train_tab
    window.tabs.setCurrentWidget(tt)
    tt.start_btn.click()

    assert tt.is_running()
    finished = _wait_until(lambda: not tt.is_running(), timeout_s=60.0)
    if not finished:
        tt.stop_and_wait()
    assert finished, f"Eğitim zaman aşımına uğradı (durum: {tt.status_label.text()!r})"
    assert not modal_messages, f"Beklenmeyen hata mesajı: {modal_messages}"

    assert tt.last_run_dir is not None
    assert tt.last_run_dir.parent == isolated_root
    assert (tt.last_run_dir / "ayarlar.json").is_file()
    assert (tt.last_run_dir / "ozet.json").is_file()
    assert tt.progress_bar.value() == tt.progress_bar.maximum()
    assert tt.live_plot.has_data()
    assert tt.live_plot.point_count() == 20
    assert tt.snapshot_combo.count() >= 4
    assert tt.go_to_test_btn.isEnabled()

    # Teste geç butonu Test sekmesine geçirip çalışmayı seçer
    tt.go_to_test_btn.click()
    assert window.tabs.currentWidget() is window.test_tab
    assert window.test_tab.selected_run_dir() == tt.last_run_dir


def test_test_tab_and_results_tab_end_to_end(
    window: MainWindow,
    isolated_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modal_messages: list[tuple[str, str, str]],
):
    # Job ayrı süreçte yamalı ortak ayarı görmez; tester ayarlar.json'daki shared'ı kullanır.
    # Bu yüzden 5 seed'lik yamalı ortakla hızlı eğitimi doğrudan trainer.train ile oluşturuyoruz.
    small_shared = _small_shared_config()
    monkeypatch.setattr(engine_settings, "load_shared_config", lambda path=None: copy.deepcopy(small_shared))

    run_dir = trainer.train(
        TrainConfig(
            agent_name="ornek",
            episodes=15,
            repeats=1,
            seed=0,
            label="hizli-test",
            env_overrides={"maze_size": 4},
            snapshots=3,
            results_root=isolated_root,
        )
    )

    # Test sekmesi: yalnızca "normal" testini çalıştır
    test_tab = window.test_tab
    window.tabs.setCurrentWidget(test_tab)
    assert test_tab.select_run(run_dir)
    test_tab.normal_check.setChecked(True)
    test_tab.buyuk_check.setChecked(False)

    test_tab.start_btn.click()
    assert test_tab.is_running()
    finished = _wait_until(lambda: not test_tab.is_running(), timeout_s=60.0)
    if not finished:
        test_tab.stop_and_wait()
    assert finished, f"Test zaman aşımına uğradı (durum: {test_tab.status_label.text()!r})"
    assert not modal_messages, f"Beklenmeyen hata mesajı: {modal_messages}"

    # Sonuç tablosu ajan + baseline'larla dolu olmalı
    assert test_tab.normal_table.rowCount() >= 2
    methods_in_table = [test_tab.normal_table.item(r, 0).text() for r in range(test_tab.normal_table.rowCount())]
    assert "ornek" in methods_in_table

    # Bölüm izleme: tek bölüm hatasız oynatıcıya yüklenir
    assert test_tab.replay_selected_episode()

    # Sonuçlar sekmesi: çalışmayı listeler ve tüm grafik seçeneklerini hatasız çizer
    results_tab = window.results_tab
    window.tabs.setCurrentWidget(results_tab)
    results_tab.refresh_runs()
    assert results_tab.runs_table.rowCount() == 1
    results_tab.runs_table.item(0, 0).setCheckState(Qt.CheckState.Checked)

    assert results_tab.plot_combo.count() == 8
    for i in range(results_tab.plot_combo.count()):
        results_tab.plot_combo.setCurrentIndex(i)
        _app().processEvents()
        assert results_tab.plot_canvas.figure is not None

    # Rapor dışa aktarma: dosya diyaloğunu ve tarayıcı açmayı yamala
    report_out = tmp_path / "rapor_cikti"
    report_out.mkdir()
    opened_urls: list[str] = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: str(report_out))
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toLocalFile()) or True)

    html_path = results_tab.export_report_dialog()
    assert html_path is not None
    assert html_path.is_file()
    assert html_path.name == "rapor.html"
    # Her şey seçilen klasörün içindeki tek bir rapor_<tarih> klasöründe
    assert html_path.parent.parent == report_out
    assert html_path.parent.name.startswith("rapor_")
    assert [p.name for p in report_out.iterdir()] == [html_path.parent.name]
    assert (html_path.parent / "rapor.md").is_file()
    assert len(opened_urls) == 1


def test_train_tab_stop_button_stops_running_job(
    window: MainWindow,
    isolated_root: Path,
    modal_messages: list[tuple[str, str, str]],
):
    st = window.settings_tab
    idx = st.agent_combo.findData("ornek")
    st.agent_combo.setCurrentIndex(idx)
    st.label_edit.setText("durdur-testi")
    st.episodes_spin.setValue(4000)
    st.repeats_spin.setValue(1)
    st.maze_size_spin.setValue(8)

    tt = window.train_tab
    tt.start_btn.click()
    assert tt.is_running()

    # Süreç başlayıp en azından birkaç bölüm ilerleyene kadar bekle
    _wait_until(lambda: tt.live_plot.point_count() >= 5 or not tt.is_running(), timeout_s=15.0)
    if tt.is_running():
        tt.stop_btn.click()

    stopped_ok = _wait_until(lambda: not tt.is_running(), timeout_s=30.0)
    if not stopped_ok:
        tt.stop_and_wait()
    assert stopped_ok, f"Durdur butonu eğitimi durduramadı (durum: {tt.status_label.text()!r})"
    assert not modal_messages, f"Beklenmeyen hata mesajı: {modal_messages}"
    assert tt.last_run_dir is not None

    saved_settings = storage.read_settings(tt.last_run_dir)
    assert saved_settings["durum"] == "durduruldu"
    assert tt.live_plot.point_count() < 4000


def test_results_tab_delete_runs(window: MainWindow, isolated_root: Path, monkeypatch, modal_messages):
    import shutil

    from PySide6.QtCore import QFile

    # Testte gerçek Geri Dönüşüm Kutusu'na dokunma
    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(lambda p: (shutil.rmtree(p), True)[1]))
    dirs = []
    for name in ("2026-01-01_00-00-01_a", "2026-01-01_00-00-02_b", "2026-01-01_00-00-03_c"):
        d = isolated_root / name
        (d / "kayitlar").mkdir(parents=True)
        storage.write_settings(d, {"agent": {"name": "ornek"}, "durum": "tamamlandi"})
        dirs.append(d)

    rt = window.results_tab
    rt.refresh_runs()
    assert rt.runs_table.rowCount() == 3

    # Eğitilmekte olan çalışma silinmez
    monkeypatch.setattr(window, "_busy_run_dirs", lambda: [dirs[0]])
    rt._busy_dirs = window._busy_run_dirs
    deleted = rt.delete_runs([dirs[0], dirs[1]], confirm=False)
    assert deleted == [dirs[1]]
    assert dirs[0].exists() and not dirs[1].exists()
    assert any(kind == "warning" for kind, *_ in modal_messages)
    assert rt.runs_table.rowCount() == 2

    # Sonuç kökü dışındaki klasör silinmez
    outside = isolated_root.parent / "baska"
    outside.mkdir()
    (outside / "ayarlar.json").write_text("{}", encoding="utf-8")
    assert rt.delete_runs([outside], confirm=False) == []
    assert outside.exists()
