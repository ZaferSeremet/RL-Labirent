"""Labirent çizimi ve bölüm tekrar oynatıcı (SÖZLEŞME: imzalar sabit).

Trajectory (bölüm kaydı) sözlüğü: engine.episode.run_episode(record=True),
engine.storage.load_trajectory ve engine.tester.replay_test_episode aynı biçimi döndürür:
    {"grid": (H,W) uint8 0 boş 1 duvar, "start": (y,x), "exit": (y,x), "view_radius": r,
     "positions": [(y,x), ...]  (adım sayısı + 1), "actions": [int ...], "rewards": [float ...]}
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from engine.i18n import tx

# Renkler (sade ve birbirinden ayırt edilebilir)
COLOR_BACKGROUND = QColor("#e9ecf1")
COLOR_WALL = QColor("#2b2f3a")
COLOR_FREE = QColor("#f8f6ef")
COLOR_VISITED = QColor("#bcd9f5")
COLOR_EXIT = QColor("#2fa84f")
COLOR_START = QColor("#6b4fbb")
COLOR_AGENT = QColor("#ee4b1c")
COLOR_AGENT_RING = QColor("#ffffff")
COLOR_WINDOW_FILL = QColor(255, 196, 0, 32)
COLOR_WINDOW_LINE = QColor(224, 150, 0, 230)
COLOR_TEXT = QColor("#5b6270")

# Eylem adları (0 yukarı, 1 sağ, 2 aşağı, 3 sol)
ACTION_LABELS = ("yukarı", "sağ", "aşağı", "sol")
ACTION_LABELS_EN = ("up", "right", "down", "left")


def action_label(action: int) -> str:
    """Eylem numarasını o anki dildeki adına çevirir."""
    if 0 <= action < 4:
        return tx(ACTION_LABELS[action], ACTION_LABELS_EN[action])
    return str(action)

MIN_SPEED = 1.0
MAX_SPEED = 200.0
DEFAULT_SPEED = 20.0
_SPEED_SLIDER_MAX = 100


class MazeView(QWidget):
    """Tek bir kareyi çizer: duvarlar, girilmiş kareler, başlangıç, çıkış, ajan ve
    (istenirse) ajanın görüş penceresi. Boyuta göre ölçeklenir, kare hücreler."""

    MARGIN = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid: np.ndarray | None = None
        self._agent: tuple[int, int] = (0, 0)
        self._exit: tuple[int, int] = (0, 0)
        self._start: tuple[int, int] | None = None
        self._visited: np.ndarray | None = None
        self._view_radius: int | None = None
        self._show_window = True
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------ API

    def sizeHint(self) -> QSize:
        return QSize(500, 500)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, 120)

    def set_frame(
        self,
        grid: np.ndarray,
        agent: tuple[int, int],
        exit_pos: tuple[int, int],
        visited: np.ndarray | None = None,
        view_radius: int | None = None,
        start: tuple[int, int] | None = None,
    ) -> None:
        """Çizilecek durumu ayarlar ve yeniden çizer. view_radius None ise pencere çizilmez."""
        self._grid = np.asarray(grid)
        self._agent = (int(agent[0]), int(agent[1]))
        self._exit = (int(exit_pos[0]), int(exit_pos[1]))
        self._visited = None if visited is None else np.asarray(visited, dtype=bool)
        self._view_radius = None if view_radius is None else int(view_radius)
        self._start = None if start is None else (int(start[0]), int(start[1]))
        self.update()

    def set_show_window(self, show: bool) -> None:
        """Görüş penceresini göster/gizle."""
        self._show_window = bool(show)
        self.update()

    def show_window(self) -> bool:
        return self._show_window

    def clear(self) -> None:
        self._grid = None
        self._visited = None
        self._view_radius = None
        self._start = None
        self.update()

    def has_frame(self) -> bool:
        return self._grid is not None

    def cell_rect(self, y: int, x: int) -> QRectF:
        """(y, x) karesinin widget koordinatlarındaki dikdörtgeni (kare yoksa boş)."""
        geo = self._geometry()
        if geo is None:
            return QRectF()
        ox, oy, cell = geo
        return QRectF(ox + x * cell, oy + y * cell, cell, cell)

    # ------------------------------------------------------------ çizim

    def _geometry(self) -> tuple[float, float, float] | None:
        if self._grid is None or self._grid.ndim != 2 or self._grid.size == 0:
            return None
        h, w = self._grid.shape
        avail_w = self.width() - 2 * self.MARGIN
        avail_h = self.height() - 2 * self.MARGIN
        cell = float(max(1, int(min(avail_w / w, avail_h / h))))
        ox = (self.width() - cell * w) / 2.0
        oy = (self.height() - cell * h) / 2.0
        return float(round(ox)), float(round(oy)), cell

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        p = QPainter(self)
        p.fillRect(self.rect(), COLOR_BACKGROUND)
        geo = self._geometry()
        if geo is None:
            font = QFont(self.font())
            font.setPointSize(max(10, font.pointSize() + 3))
            p.setFont(font)
            p.setPen(COLOR_TEXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tx("Kayıt yok", "No recording"))
            p.end()
            return

        ox, oy, cell = geo
        grid = self._grid
        h, w = grid.shape
        visited = self._visited
        if visited is not None and visited.shape != grid.shape:
            visited = None

        p.setPen(Qt.PenStyle.NoPen)
        for y in range(h):
            row_y = oy + y * cell
            for x in range(w):
                if grid[y, x]:
                    color = COLOR_WALL
                elif visited is not None and visited[y, x]:
                    color = COLOR_VISITED
                else:
                    color = COLOR_FREE
                p.fillRect(QRectF(ox + x * cell, row_y, cell, cell), color)

        # Çıkış: dolu yeşil kare
        ey, ex = self._exit
        p.fillRect(self.cell_rect(ey, ex).adjusted(1, 1, -1, -1), COLOR_EXIT)

        # Başlangıç: küçük işaret
        if self._start is not None:
            r = self.cell_rect(*self._start)
            c = r.center()
            d = r.width() * 0.16
            p.setBrush(COLOR_START)
            p.drawEllipse(c, d, d)

        # Görüş penceresi (ızgara sınırına kırpılmış yarı saydam örtü)
        if self._show_window and self._view_radius is not None:
            ay, ax = self._agent
            rad = self._view_radius
            y0, y1 = max(0, ay - rad), min(h - 1, ay + rad)
            x0, x1 = max(0, ax - rad), min(w - 1, ax + rad)
            top_left = self.cell_rect(y0, x0).topLeft()
            bottom_right = self.cell_rect(y1, x1).bottomRight()
            win = QRectF(top_left, bottom_right)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(COLOR_WINDOW_FILL)
            p.drawRect(win)
            pen = QPen(COLOR_WINDOW_LINE)
            pen.setWidthF(max(1.5, cell * 0.08))
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(win.adjusted(1, 1, -1, -1))

        # Ajan
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.cell_rect(*self._agent)
        rad_px = r.width() * 0.36
        pen = QPen(COLOR_AGENT_RING)
        pen.setWidthF(max(1.0, cell * 0.07))
        p.setPen(pen)
        p.setBrush(COLOR_AGENT)
        p.drawEllipse(r.center(), rad_px, rad_px)
        p.end()


class ReplayPlayer(QWidget):
    """MazeView + kontroller: oynat/duraklat, başa sar, adım kaydırıcısı, hız (adım/sn,
    1-200, logaritmik his), "görüş penceresini göster" kutusu, bilgi satırı
    (adım i/N, toplam ödül, son eylem, başarı). Ziyaret bilgisi positions[:i+1]'den kurulur."""

    finished = Signal()  # oynatma son adıma ulaştı

    TICK_MS = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._traj: dict[str, Any] | None = None
        self._title = ""
        self._grid: np.ndarray | None = None
        self._pos: np.ndarray = np.zeros((0, 2), dtype=int)
        self._cum_reward: np.ndarray = np.zeros(1)
        self._actions: list[int] = []
        self._n_steps = 0
        self._step = 0
        self._visited: np.ndarray | None = None
        self._visited_upto = -1  # visited, positions[0..upto] ile kurulu
        self._speed = DEFAULT_SPEED
        self._acc = 0.0
        self._clock = QElapsedTimer()
        self._updating = False

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        # --- arayüz
        self.view = MazeView(self)
        self.info_label = QLabel(tx("Kayıt yok", "No recording"), self)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.play_button = QPushButton(tx("Oynat", "Play"), self)
        self.play_button.setMinimumWidth(90)
        self.rewind_button = QPushButton(tx("Başa sar", "Rewind"), self)
        self.window_check = QCheckBox(tx("Görüş penceresi", "View window"), self)
        self.window_check.setChecked(True)

        self.step_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.step_slider.setRange(0, 0)
        self.step_slider.setEnabled(False)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setRange(0, _SPEED_SLIDER_MAX)
        self.speed_slider.setFixedWidth(150)
        self.speed_label = QLabel(self)
        self.speed_label.setMinimumWidth(95)

        buttons = QHBoxLayout()
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.rewind_button)
        buttons.addSpacing(12)
        buttons.addWidget(self.window_check)
        buttons.addStretch(1)
        buttons.addWidget(QLabel(tx("Hız:", "Speed:"), self))
        buttons.addWidget(self.speed_slider)
        buttons.addWidget(self.speed_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.info_label)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.step_slider)
        layout.addLayout(buttons)

        self.play_button.clicked.connect(self._on_play_clicked)
        self.rewind_button.clicked.connect(self.rewind)
        self.window_check.toggled.connect(self.view.set_show_window)
        self.step_slider.valueChanged.connect(self._on_step_slider)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)

        self.set_speed(DEFAULT_SPEED)
        self._update_controls()

    # ------------------------------------------------------------------ API

    def load(self, traj: dict[str, Any], title: str = "", autoplay: bool = True) -> None:
        """Yeni kaydı yükler (0. adım), autoplay ise oynatmaya başlar. title bilgi satırında görünür."""
        self.pause()
        self._traj = traj
        self._title = title
        self._grid = np.asarray(traj["grid"])
        self._pos = np.asarray(traj["positions"], dtype=int).reshape(-1, 2)
        self._actions = [int(a) for a in traj.get("actions", [])]
        rewards = np.asarray(traj.get("rewards", []), dtype=float)
        self._cum_reward = np.concatenate([[0.0], np.cumsum(rewards)])
        self._n_steps = max(0, len(self._pos) - 1)
        self._step = 0
        self._visited = np.zeros(self._grid.shape, dtype=bool)
        self._visited_upto = -1
        self._updating = True
        self.step_slider.setRange(0, self._n_steps)
        self._updating = False
        self._show_step(0)
        self._update_controls()
        if autoplay and self._n_steps > 0:
            self.play()

    def play(self) -> None:
        if self._traj is None or self._n_steps == 0:
            return
        if self._step >= self._n_steps:
            self._show_step(0)
        self._acc = 0.0
        self._clock.start()
        self._timer.start()
        self._update_controls()

    def pause(self) -> None:
        self._timer.stop()
        self._update_controls()

    def set_speed(self, steps_per_sec: float) -> None:
        sps = min(MAX_SPEED, max(MIN_SPEED, float(steps_per_sec)))
        self._speed = sps
        self._set_speed_slider(sps)
        self._update_speed_label()

    def is_playing(self) -> bool:
        return self._timer.isActive()

    # -------------------------------------------------- ek yardımcılar

    def speed(self) -> float:
        return self._speed

    def rewind(self) -> None:
        """Başa sarar (oynatma durumu korunur)."""
        self.seek(0)

    def seek(self, step: int) -> None:
        """İstenen adıma gider (0..N)."""
        if self._traj is None:
            return
        self._show_step(min(max(0, int(step)), self._n_steps))

    def current_step(self) -> int:
        return self._step

    def n_steps(self) -> int:
        return self._n_steps

    def visited_map(self) -> np.ndarray | None:
        """O anki ziyaret haritası (yalnız okumak için)."""
        return self._visited

    # ------------------------------------------------------------ iç işler

    def _on_play_clicked(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def _on_step_slider(self, value: int) -> None:
        if self._updating or self._traj is None:
            return
        self._show_step(value)

    def _on_speed_slider(self, value: int) -> None:
        if self._updating:
            return
        self._speed = MIN_SPEED * (MAX_SPEED / MIN_SPEED) ** (value / _SPEED_SLIDER_MAX)
        self._update_speed_label()

    def _set_speed_slider(self, sps: float) -> None:
        v = round(_SPEED_SLIDER_MAX * math.log(sps / MIN_SPEED) / math.log(MAX_SPEED / MIN_SPEED))
        self._updating = True
        self.speed_slider.setValue(int(v))
        self._updating = False

    def _update_speed_label(self) -> None:
        self.speed_label.setText(tx(f"{round(self._speed)} adım/sn", f"{round(self._speed)} steps/s"))

    def _on_tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        self._acc += dt * self._speed
        n = int(self._acc)
        if n <= 0:
            return
        self._acc -= n
        target = min(self._n_steps, self._step + n)
        self._show_step(target)
        if target >= self._n_steps:
            self.pause()
            self.finished.emit()

    def _sync_visited(self, step: int) -> None:
        """Ziyaret haritasını positions[:step+1] ile uyumlu hale getirir (artımlı)."""
        assert self._visited is not None
        if step < self._visited_upto:
            self._visited[:] = False
            self._visited_upto = -1
        lo = self._visited_upto + 1
        if lo <= step:
            seg = self._pos[lo : step + 1]
            self._visited[seg[:, 0], seg[:, 1]] = True
            self._visited_upto = step

    def _show_step(self, step: int) -> None:
        if self._traj is None or self._grid is None or len(self._pos) == 0:
            return
        self._step = step
        self._sync_visited(step)
        traj = self._traj
        agent = (int(self._pos[step, 0]), int(self._pos[step, 1]))
        self.view.set_frame(
            self._grid,
            agent,
            tuple(traj["exit"]),
            self._visited,
            traj.get("view_radius"),
            tuple(traj["start"]) if traj.get("start") is not None else None,
        )
        if self.step_slider.value() != step:
            self._updating = True
            self.step_slider.setValue(step)
            self._updating = False
        self.info_label.setText(self._info_text())

    def _info_text(self) -> str:
        if self._traj is None:
            return tx("Kayıt yok", "No recording")
        i, n = self._step, self._n_steps
        parts = []
        if self._title:
            parts.append(self._title)
        parts.append(tx(f"Adım {i}/{n}", f"Step {i}/{n}"))
        rew = self._cum_reward[min(i, len(self._cum_reward) - 1)]
        parts.append(tx(f"Toplam ödül: {rew:+.2f}", f"Total reward: {rew:+.2f}"))
        if i > 0 and i - 1 < len(self._actions):
            a = self._actions[i - 1]
            act_str = action_label(a)
            parts.append(tx(f"Son eylem: {act_str}", f"Last action: {act_str}"))
        else:
            parts.append(tx("Son eylem: -", "Last action: -"))
        if i == n and n > 0:
            reached = tuple(self._pos[-1]) == tuple(int(v) for v in self._traj["exit"])
            parts.append(tx("Çıkışa ulaştı", "Reached exit") if reached else tx("Süre doldu", "Timed out"))
        return "  |  ".join(parts)

    def _update_controls(self) -> None:
        loaded = self._traj is not None
        self.play_button.setText(tx("Duraklat", "Pause") if self.is_playing() else tx("Oynat", "Play"))
        self.play_button.setEnabled(loaded and self._n_steps > 0)
        self.rewind_button.setEnabled(loaded)
        self.step_slider.setEnabled(loaded and self._n_steps > 0)
