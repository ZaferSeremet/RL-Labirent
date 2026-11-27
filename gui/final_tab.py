"""Final Test sekmesi: hiç görmediği bir labirentte eğitilmiş ajan ve baseline'ları yarıştırır."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine import race, settings as engine_settings, storage
from engine.i18n import tx

from .race_view import RacePlayer
from .test_tab import format_run_display


def _model_repeats(run_dir: Path, st: dict[str, Any] | None = None) -> list[int]:
    """Çalışma klasöründe model_rep<r>.pkl dosyası bulunan tekrar numaralarını döndürür."""
    if st is None:
        try:
            st = storage.read_settings(run_dir)
        except Exception:  # noqa: BLE001
            st = {}
    n_rep = int((st.get("train_config") or {}).get("repeats", 1))
    reps = [r for r in range(n_rep) if storage.model_path(run_dir, r).is_file()]
    for p in sorted(run_dir.glob("model_rep*.pkl")):
        suffix = p.stem.removeprefix("model_rep")
        if suffix.isdigit():
            r = int(suffix)
            if r not in reps:
                reps.append(r)
    return sorted(reps)


class FinalTab(QWidget):
    """Demo için: seçilen eğitilmiş ajanı istenen boyut ve seed'deki labirentte baseline'larla yarıştırır."""

    def __init__(self, parent: QWidget | None = None, *, results_root: Path | None = None) -> None:
        super().__init__(parent)
        self._results_root = results_root
        self._rng = random.Random()

        try:
            shared = engine_settings.load_shared_config()
            default_size = int(shared["test"]["normal"]["maze_size"])
        except Exception:  # noqa: BLE001
            default_size = 8
        self._default_size = min(race.MAX_MAZE_SIZE, max(race.MIN_MAZE_SIZE, default_size))

        self._build_ui()
        self.refresh_runs()

    def _get_results_root(self) -> Path:
        return Path(self._results_root) if self._results_root is not None else Path(storage.RESULTS_ROOT)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel(tx("Çalışma:", "Run:"), self))
        self.run_combo = QComboBox(self)
        self.run_combo.setMinimumWidth(200)
        self.run_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.setMinimumContentsLength(18)
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        controls.addWidget(self.run_combo, 1)

        self.refresh_btn = QPushButton(tx("Yenile", "Refresh"), self)
        self.refresh_btn.clicked.connect(lambda: self.refresh_runs())
        controls.addWidget(self.refresh_btn)

        controls.addWidget(QLabel(tx("Tekrar:", "Repeat:"), self))
        self.repeat_combo = QComboBox(self)
        controls.addWidget(self.repeat_combo)

        controls.addWidget(QLabel(tx("Boyut:", "Size:"), self))
        self.size_spin = QSpinBox(self)
        self.size_spin.setRange(race.MIN_MAZE_SIZE, race.MAX_MAZE_SIZE)
        self.size_spin.setValue(self._default_size)
        self.size_spin.setToolTip(
            tx(
                f"Labirent boyutu ({race.MIN_MAZE_SIZE}-{race.MAX_MAZE_SIZE}).",
                f"Maze size ({race.MIN_MAZE_SIZE}-{race.MAX_MAZE_SIZE}).",
            )
        )
        controls.addWidget(self.size_spin)

        self.size_8_btn = QPushButton("8x8", self)
        self.size_8_btn.clicked.connect(lambda: self.size_spin.setValue(8))
        controls.addWidget(self.size_8_btn)

        self.size_12_btn = QPushButton("12x12", self)
        self.size_12_btn.clicked.connect(lambda: self.size_spin.setValue(12))
        controls.addWidget(self.size_12_btn)

        controls.addWidget(QLabel(tx("Seed:", "Seed:"), self))
        self.seed_edit = QLineEdit(self)
        self.seed_edit.setPlaceholderText(tx("rastgele", "random"))
        self.seed_edit.setMaxLength(9)
        self.seed_edit.setFixedWidth(110)
        self.seed_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,9}$"), self))
        self.seed_edit.returnPressed.connect(self.start_race)
        controls.addWidget(self.seed_edit)

        self.random_seed_btn = QPushButton(tx("Rastgele", "Random"), self)
        self.random_seed_btn.clicked.connect(self.randomize_seed)
        self.random_btn = self.random_seed_btn
        controls.addWidget(self.random_seed_btn)

        self.start_btn = QPushButton(tx("Başlat", "Start"), self)
        self.start_btn.setMinimumWidth(90)
        self.start_btn.setProperty("accent", "primary")
        self.start_btn.clicked.connect(self.start_race)
        self.start_button = self.start_btn
        controls.addWidget(self.start_btn)

        root.addLayout(controls)

        self.race_player = RacePlayer(self)
        self.player = self.race_player
        root.addWidget(self.race_player, 1)

    # ------------------------------------------------------------ çalışma listesi

    def selected_run_dir(self) -> Path | None:
        data = self.run_combo.currentData()
        return Path(data) if data else None

    def refresh_runs(self, select_dir: Path | None = None) -> None:
        """Sonuç klasöründeki en az bir modeli olan çalışmaları yeniden tarar."""
        target = select_dir or self.selected_run_dir()
        runs = [rdir for rdir in storage.list_runs(self._get_results_root()) if _model_repeats(rdir)]

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
        prev_repeat = self.repeat_combo.currentData()
        self.repeat_combo.clear()
        if run_dir is None or not (run_dir / "ayarlar.json").is_file():
            self.start_btn.setEnabled(False)
            return

        avail_reps = _model_repeats(run_dir)
        if not avail_reps:
            self.start_btn.setEnabled(False)
            return

        for r in avail_reps:
            self.repeat_combo.addItem(tx(f"Tekrar {r}", f"Repeat {r}"), r)
        if prev_repeat in avail_reps:
            idx = self.repeat_combo.findData(prev_repeat)
            if idx >= 0:
                self.repeat_combo.setCurrentIndex(idx)
        self.start_btn.setEnabled(True)

    # ------------------------------------------------------------ yarış

    def randomize_seed(self) -> int:
        """RANDOM_SEED_RANGE içinden rastgele bir seed seçip seed alanına yazar."""
        seed = self._rng.randint(*race.RANDOM_SEED_RANGE)
        self.seed_edit.setText(str(seed))
        return seed

    def start_race(self) -> bool:
        """Seçili çalışma, tekrar, boyut ve seed ile yarışı başlatır."""
        run_dir = self.selected_run_dir()
        if run_dir is None:
            QMessageBox.warning(
                self,
                tx("Çalışma Seçilmedi", "No Run Selected"),
                tx("Lütfen eğitilmiş modeli olan bir çalışma seçin.", "Please select a run with a trained model."),
            )
            return False

        repeat_data = self.repeat_combo.currentData()
        repeat = int(repeat_data) if repeat_data is not None else 0

        text = self.seed_edit.text().strip()
        if not text:
            seed = self.randomize_seed()
        else:
            try:
                seed = int(text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    tx("Geçersiz Seed", "Invalid Seed"),
                    tx("Seed bir tam sayı olmalı.", "Seed must be an integer."),
                )
                return False

        size = int(self.size_spin.value())
        try:
            race_data = race.race_custom(run_dir, repeat, seed, size)
            # Boyut ve seed'i oynatıcının bilgi satırı zaten yazıyor
            title = tx(f"Final test · tekrar {repeat}", f"Final test · repeat {repeat}")
            self.race_player.load(race_data, title=title, autoplay=True)
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tx("Yarış Başlatılamadı", "Could Not Start Race"), str(exc))
            return False

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        self.refresh_runs()
