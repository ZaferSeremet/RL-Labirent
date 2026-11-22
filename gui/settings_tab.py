"""Ayarlar sekmesi: ajan, eğitim, ödül, ortam ve hiperparametre ayarları."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from agents.base import Agent
from engine import registry, settings as engine_settings
from engine.trainer import TrainConfig

_REWARD_META: tuple[tuple[str, str, float, float, float], ...] = (
    ("step", "Adım ödülü (step)", -100.0, 100.0, 0.01),
    ("wall", "Duvar çarpma (wall)", -100.0, 100.0, 0.01),
    ("exit", "Çıkış ödülü (exit)", -100.0, 1000.0, 0.5),
    ("new_tile", "Yeni kare (new_tile)", -100.0, 100.0, 0.01),
    ("timeout", "Süre aşımı (timeout)", -100.0, 100.0, 0.05),
)


def _parse_scalar(raw: str) -> Any:
    """Hiperparametre değerini bool / None / int / float / list / dict / str olarak çözümler."""
    s = raw.strip()
    if not s:
        raise ValueError("Boş değer verilemez")

    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None

    # Tırnaklı metin
    if s[0] in ('"', "'"):
        if len(s) < 2 or s[-1] != s[0]:
            raise ValueError(f"Kapanmamış tırnak: {s}")
        if s[0] == '"':
            try:
                return json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Geçersiz metin değeri ({s}): {exc.msg}") from exc
        return s[1:-1]

    # Liste ya da sözlük JSON
    if s[0] in ("[", "{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Geçersiz JSON değeri ({s}): {exc.msg}") from exc

    # Tamsayı
    try:
        return int(s)
    except ValueError:
        pass

    # Ondalıklı sayı
    try:
        val = float(s)
    except ValueError:
        val = None
    if val is not None:
        if not math.isfinite(val):
            raise ValueError(f"Geçersiz sayısal değer: {s}")
        return val

    # Boşluk içermeyen düz metin değerleri kabul et
    if any(ch.isspace() for ch in s):
        raise ValueError(f"Boşluk içeren metin tırnak içinde yazılmalı: {s}")
    return s


def parse_hparams(text: str) -> dict[str, Any]:
    """`anahtar=değer` satırlarını veya JSON nesnesini sözlüğe çevirir."""
    stripped = text.strip()
    if not stripped:
        return {}

    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Hiperparametre JSON hatası: {exc.msg} (satır {exc.lineno})") from exc
        if not isinstance(data, dict):
            raise ValueError("Hiperparametre JSON bir nesne ({...}) olmalı")
        for k in data:
            if not isinstance(k, str) or not k.isidentifier():
                raise ValueError(f"Geçersiz hiperparametre adı: {k!r}")
        return data

    result: dict[str, Any] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Satır {lineno}: 'anahtar=değer' biçiminde olmalı ({line!r})")
        key_part, val_part = line.split("=", 1)
        key = key_part.strip()
        if not key or not key.isidentifier():
            raise ValueError(f"Satır {lineno}: geçersiz parametre adı ({key!r})")
        val_str = val_part.strip()
        if not val_str:
            raise ValueError(f"Satır {lineno}: '{key}' için değer eksik")
        try:
            result[key] = _parse_scalar(val_str)
        except ValueError as exc:
            raise ValueError(f"Satır {lineno} ('{key}'): {exc}") from exc
    return result


def _agent_hparams_hint(cls: type[Agent]) -> str:
    """Ajanın __init__ imzasındaki varsayılan hiperparametreleri özetler."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return ""
    parts: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "spec", "seed"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            parts.append(f"{name}={param.default!r}")
        else:
            parts.append(name)
    return ", ".join(parts)


class SettingsTab(QWidget):
    """Eğitim ayarlarını düzenler ve TrainConfig üretir."""

    train_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        results_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._results_root = results_root
        self._shared: dict[str, Any] = engine_settings.load_shared_config()
        self._learners: dict[str, type[Agent]] = {}

        self._build_ui()
        self.refresh_agents()
        self.reset_rewards()
        self._reset_env_defaults()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(12)

        # ------------------------------------------------- Registry uyarı kutusu
        self.registry_warning_label = QLabel()
        self.registry_warning_label.setWordWrap(True)
        self.registry_warning_label.setStyleSheet(
            "QLabel { background: #fff4e5; color: #7a4100; border: 1px solid #f0b429; "
            "border-radius: 5px; padding: 8px; }"
        )
        self.registry_warning_label.setVisible(False)
        root.addWidget(self.registry_warning_label)

        # ------------------------------------------------- Üst 2 sütunlu grid
        top_grid = QGridLayout()
        top_grid.setSpacing(12)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)
        root.addLayout(top_grid)

        # 1) Ajan ve Eğitim Ayarları
        agent_box = QGroupBox("Ajan ve Eğitim Ayarları")
        agent_form = QFormLayout(agent_box)
        agent_form.setSpacing(8)
        agent_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        agent_row = QHBoxLayout()
        agent_row.setSpacing(6)
        self.agent_combo = QComboBox()
        self.agent_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.agent_combo.setMinimumContentsLength(12)
        self.agent_combo.currentIndexChanged.connect(self._on_agent_changed)
        self.refresh_agents_btn = QPushButton("Yenile")
        self.refresh_agents_btn.setMinimumWidth(72)
        self.refresh_agents_btn.clicked.connect(self.refresh_agents)
        agent_row.addWidget(self.agent_combo, 1)
        agent_row.addWidget(self.refresh_agents_btn)
        agent_form.addRow("Ajan:", agent_row)

        self.obs_mode_label = QLabel("-")
        self.obs_mode_label.setWordWrap(True)
        self.obs_mode_label.setStyleSheet("color: #334155;")
        agent_form.addRow("Ajan bilgisi:", self.obs_mode_label)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("ör. adim-cezasi-yuksek (isteğe bağlı)")
        agent_form.addRow("Etiket:", self.label_edit)

        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 1_000_000)
        self.episodes_spin.setSingleStep(100)
        self.episodes_spin.setValue(2000)
        agent_form.addRow("Bölüm sayısı:", self.episodes_spin)

        default_repeats = int(self._shared.get("egitim", {}).get("tekrar_sayisi", 3))
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 50)
        self.repeats_spin.setValue(default_repeats)
        agent_form.addRow("Tekrar sayısı:", self.repeats_spin)

        default_seed = int(self._shared.get("egitim", {}).get("seed_baslangic", 0))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(default_seed)
        agent_form.addRow("Seed:", self.seed_spin)

        self.snapshots_spin = QSpinBox()
        self.snapshots_spin.setRange(1, 100)
        self.snapshots_spin.setValue(5)
        self.snapshots_spin.setToolTip("0. tekrarda eşit aralıklı kaydedilecek bölüm sayısı")
        agent_form.addRow("Kayıt sayısı:", self.snapshots_spin)

        top_grid.addWidget(agent_box, 0, 0)

        # 2) Ödüller
        reward_box = QGroupBox("Ödül Ayarları")
        reward_v = QVBoxLayout(reward_box)
        reward_v.setSpacing(8)
        reward_note = QLabel("Yalnız varsayılandan farklı değerler geçersiz kılınır.")
        reward_note.setWordWrap(True)
        reward_note.setStyleSheet("color: #475569; font-size: 12px;")
        reward_v.addWidget(reward_note)
        reward_form = QFormLayout()
        reward_form.setSpacing(8)
        self.reward_spins: dict[str, QDoubleSpinBox] = {}
        for key, label, lo, hi, step in _REWARD_META:
            sp = QDoubleSpinBox()
            sp.setDecimals(4)
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            self.reward_spins[key] = sp
            reward_form.addRow(f"{label}:", sp)
        reward_v.addLayout(reward_form)

        rew_btn_row = QHBoxLayout()
        self.reset_rewards_btn = QPushButton("Varsayılana dön")
        self.reset_rewards_btn.clicked.connect(self.reset_rewards)
        rew_btn_row.addStretch(1)
        rew_btn_row.addWidget(self.reset_rewards_btn)
        reward_v.addLayout(rew_btn_row)

        top_grid.addWidget(reward_box, 0, 1)

        # 3) Ortam Ayarları
        env_box = QGroupBox("Eğitim Ortamı Ayarları")
        env_v = QVBoxLayout(env_box)
        env_v.setSpacing(8)
        env_form = QFormLayout()
        env_form.setSpacing(8)

        self.maze_size_spin = QSpinBox()
        self.maze_size_spin.setRange(2, 50)
        env_form.addRow("Labirent boyutu:", self.maze_size_spin)

        self.view_radius_spin = QSpinBox()
        self.view_radius_spin.setRange(1, 10)
        env_form.addRow("Görüş yarıçapı:", self.view_radius_spin)

        env_v.addLayout(env_form)

        normal_sz = self._shared.get("test", {}).get("normal", {}).get("maze_size", 8)
        buyuk_sz = self._shared.get("test", {}).get("buyuk", {}).get("maze_size", 12)
        env_note = QLabel(
            f"Not: Ortak ayardan farklı değerler eğitim ortamını geçersiz kılar. "
            f"Test boyutları sabittir (Normal: {normal_sz}×{normal_sz}, Büyük: {buyuk_sz}×{buyuk_sz})."
        )
        env_note.setWordWrap(True)
        env_note.setStyleSheet("color: #475569; font-size: 12px;")
        env_v.addWidget(env_note)

        env_btn_row = QHBoxLayout()
        self.reset_env_btn = QPushButton("Ortamı varsayılana dön")
        self.reset_env_btn.clicked.connect(self._reset_env_defaults)
        env_btn_row.addStretch(1)
        env_btn_row.addWidget(self.reset_env_btn)
        env_v.addLayout(env_btn_row)

        top_grid.addWidget(env_box, 1, 0)

        # 4) Hiperparametreler
        hp_box = QGroupBox("Hiperparametreler")
        hp_v = QVBoxLayout(hp_box)
        hp_v.setSpacing(6)

        self.hparams_hint_label = QLabel()
        self.hparams_hint_label.setWordWrap(True)
        self.hparams_hint_label.setStyleSheet("color: #475569; font-size: 12px;")
        hp_v.addWidget(self.hparams_hint_label)

        self.hparams_edit = QPlainTextEdit()
        self.hparams_edit.setPlaceholderText(
            "Örnek:\nalpha=0.1\ngamma=0.99\nepsilon=1.0\n\nveya JSON:\n{\"alpha\": 0.1, \"gamma\": 0.99}"
        )
        self.hparams_edit.setMinimumHeight(110)
        self.hparams_edit.textChanged.connect(self._validate_hparams_live)
        hp_v.addWidget(self.hparams_edit, 1)

        self.hparams_warning_label = QLabel()
        self.hparams_warning_label.setWordWrap(True)
        self.hparams_warning_label.setStyleSheet(
            "QLabel { background: #fef2f2; color: #991b1b; border: 1px solid #f87171; "
            "border-radius: 4px; padding: 6px; }"
        )
        self.hparams_warning_label.setVisible(False)
        hp_v.addWidget(self.hparams_warning_label)

        top_grid.addWidget(hp_box, 1, 1)
        root.addStretch(1)

        # Alt eylem çubuğu
        footer = QHBoxLayout()
        footer.setContentsMargins(4, 2, 4, 2)
        self.summary_hint = QLabel("Ayarları belirledikten sonra Eğitim sekmesinden eğitimi başlatabilirsiniz.")
        self.summary_hint.setWordWrap(True)
        self.summary_hint.setStyleSheet("color: #475569;")
        self.go_train_btn = QPushButton("Eğitim Sekmesine Geç")
        self.go_train_btn.clicked.connect(self.train_requested.emit)
        footer.addWidget(self.summary_hint, 1)
        footer.addWidget(self.go_train_btn)
        outer.addLayout(footer)

    # ------------------------------------------------------------ yükleme / sıfırlama

    def refresh_agents(self) -> None:
        """engine.registry.discover() ile öğrenen ajanları yeniler (baseline hariç)."""
        prev = self.selected_agent_name()
        reg = registry.discover()
        learners = sorted(reg.learners(), key=lambda c: c.name)
        self._learners = {c.name: c for c in learners}

        self.agent_combo.blockSignals(True)
        self.agent_combo.clear()
        for cls in learners:
            self.agent_combo.addItem(f"{cls.name}  ({cls.obs_mode})", cls.name)
        if prev and prev in self._learners:
            idx = self.agent_combo.findData(prev)
            if idx >= 0:
                self.agent_combo.setCurrentIndex(idx)
        self.agent_combo.blockSignals(False)

        if reg.errors:
            lines = [f"• {mod}: {err}" for mod, err in sorted(reg.errors.items())]
            self.registry_warning_label.setText(
                "Bazı ajan modülleri yüklenemedi (diğer ajanlar kullanılabilir):\n" + "\n".join(lines)
            )
            self.registry_warning_label.setVisible(True)
        else:
            self.registry_warning_label.clear()
            self.registry_warning_label.setVisible(False)

        self._on_agent_changed()

    def _on_agent_changed(self) -> None:
        name = self.selected_agent_name()
        cls = self._learners.get(name) if name else None
        if cls is None:
            self.obs_mode_label.setText("Öğrenen ajan bulunamadı.")
            self.hparams_hint_label.setText("Boş bırakılırsa ajanın varsayılan hiperparametreleri kullanılır.")
            return

        obs_desc = "simple (tablo)" if cls.obs_mode == "simple" else "full (vektör)"
        self.obs_mode_label.setText(
            f"Gözlem: {obs_desc}\nSınıf: {cls.__module__}.{cls.__qualname__}"
        )
        hint = _agent_hparams_hint(cls)
        if hint:
            self.hparams_hint_label.setText(f"Varsayılan parametreler: {hint}")
        else:
            self.hparams_hint_label.setText("Boş bırakılırsa ajanın varsayılan hiperparametreleri kullanılır.")

    def reset_rewards(self) -> None:
        """Ödül alanlarını ortak ayardaki varsayılanlara döndürür."""
        defaults = self._shared.get("env", {}).get("rewards", {})
        for key, sp in self.reward_spins.items():
            sp.setValue(float(defaults.get(key, 0.0)))

    def _reset_env_defaults(self) -> None:
        env_defaults = self._shared.get("env", {})
        self.maze_size_spin.setValue(int(env_defaults.get("maze_size", 8)))
        self.view_radius_spin.setValue(int(env_defaults.get("view_radius", 1)))

    # ------------------------------------------------------------ doğrulama & TrainConfig

    def selected_agent_name(self) -> str | None:
        data = self.agent_combo.currentData()
        return str(data) if data else None

    def _validate_hparams_live(self) -> bool:
        try:
            parse_hparams(self.hparams_edit.toPlainText())
        except ValueError as exc:
            self.hparams_warning_label.setText(str(exc))
            self.hparams_warning_label.setVisible(True)
            return False
        self.hparams_warning_label.clear()
        self.hparams_warning_label.setVisible(False)
        return True

    def build_config(self) -> TrainConfig:
        """Mevcut form değerlerinden TrainConfig üretir. Hatalı girişte ValueError yükseltir."""
        agent_name = self.selected_agent_name()
        if not agent_name:
            raise ValueError("Eğitim için seçili bir ajan yok.")

        try:
            hparams = parse_hparams(self.hparams_edit.toPlainText())
        except ValueError as exc:
            self.hparams_warning_label.setText(str(exc))
            self.hparams_warning_label.setVisible(True)
            raise
        else:
            self.hparams_warning_label.clear()
            self.hparams_warning_label.setVisible(False)

        shared_env = self._shared.get("env", {})
        shared_rewards = shared_env.get("rewards", {})

        reward_overrides: dict[str, float] = {}
        for key, sp in self.reward_spins.items():
            val = float(sp.value())
            default_val = float(shared_rewards.get(key, 0.0))
            if not math.isclose(val, default_val, rel_tol=1e-9, abs_tol=1e-9):
                reward_overrides[key] = val

        env_overrides: dict[str, Any] = {}
        maze_size = int(self.maze_size_spin.value())
        if maze_size != int(shared_env.get("maze_size", 8)):
            env_overrides["maze_size"] = maze_size

        view_radius = int(self.view_radius_spin.value())
        if view_radius != int(shared_env.get("view_radius", 1)):
            env_overrides["view_radius"] = view_radius

        return TrainConfig(
            agent_name=agent_name,
            episodes=int(self.episodes_spin.value()),
            repeats=int(self.repeats_spin.value()),
            seed=int(self.seed_spin.value()),
            label=self.label_edit.text().strip(),
            reward_overrides=reward_overrides,
            env_overrides=env_overrides,
            hparams=hparams,
            snapshots=int(self.snapshots_spin.value()),
            results_root=self._results_root,
        )
