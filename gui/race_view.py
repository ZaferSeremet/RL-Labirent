"""Yarış oynatıcı: birden çok oyuncu aynı labirentte (SÖZLEŞME: genel imzalar sabit).

Girdi: engine/race.py'deki yarış kaydı (race) sözlüğü.
Genel API (imzalar ve özellik adları) değişmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine.i18n import method_label, tx

from .maze_view import (
    COLOR_AGENT_RING,
    COLOR_BACKGROUND,
    COLOR_EXIT,
    COLOR_FREE,
    COLOR_START,
    COLOR_TEXT,
    COLOR_WALL,
    DEFAULT_SPEED,
    MAX_SPEED,
    MIN_SPEED,
)

# Oyuncu renkleri: anahtar -> renk. Bilinmeyen baseline'lar FALLBACK_COLORS'tan sırayla alır.
RACER_COLORS: dict[str, str] = {"agent": "#e4572e", "sag_el": "#2563eb", "random": "#d63384"}
FALLBACK_COLORS: tuple[str, ...] = ("#f59f00", "#0ca678", "#7048e8")

_SPEED_SLIDER_MAX = 100

# Yarış sonu en iyi satır vurgusu (açık yeşil arka plan, koyu yeşil metin)
_BEST_BG = QColor("#dcfce7")
_BEST_FG = QColor("#14532d")


def racer_label(racer: dict[str, Any]) -> str:
    """Oyuncunun o anki dildeki görünen adı."""
    if racer.get("is_baseline"):
        return method_label(str(racer["method"]))
    return tx(f"{racer['method']} (eğitilmiş ajan)", f"{racer['method']} (trained agent)")


def racer_colors(race: dict[str, Any]) -> list[str]:
    """race["racers"] sırasıyla renkler (hex)."""
    out: list[str] = []
    fb = 0
    for r in race.get("racers", []):
        c = RACER_COLORS.get(str(r["key"]))
        if c is None:
            c = FALLBACK_COLORS[fb % len(FALLBACK_COLORS)]
            fb += 1
        out.append(c)
    return out


def _table_headers() -> list[str]:
    """Durum tablosunun o anki dildeki sütun başlıkları."""
    return [
        tx("Oyuncu", "Racer"),
        tx("Durum", "Status"),
        tx("Adım", "Steps"),
        tx("Çarpma", "Wall hits"),
        tx("Gezilen", "Visited"),
        tx("Ödül", "Reward"),
        tx("Sıra", "Rank"),
    ]


@dataclass
class _PreparedRacer:
    """Tek bir oyuncunun önceden hesaplanmış birikimli yarış verisi."""

    label: str
    color: str
    success: bool
    positions: np.ndarray  # (L, 2) int
    own_steps: int
    cum_hits: np.ndarray  # (own_steps + 1,) int
    cum_visited: np.ndarray  # (own_steps + 1,) int
    cum_rewards: np.ndarray  # (own_steps + 1,) float


def _prepare_racer(racer: dict[str, Any], color: str) -> _PreparedRacer:
    """Bir oyuncunun çarpma, ziyaret ve ödül birikimli dizilerini bir kez hazırlar."""
    raw_pos = racer.get("positions") or []
    if len(raw_pos) > 0:
        pos = np.asarray(raw_pos, dtype=int).reshape(-1, 2)
    else:
        pos = np.zeros((0, 2), dtype=int)

    length = len(pos)
    own_steps = max(0, length - 1)

    cum_hits = np.zeros(own_steps + 1, dtype=int)
    if own_steps > 0:
        same = (pos[1:, 0] == pos[:-1, 0]) & (pos[1:, 1] == pos[:-1, 1])
        cum_hits[1:] = np.cumsum(same.astype(int))

    if length == 0:
        cum_visited = np.zeros(1, dtype=int)
    else:
        seen: set[tuple[int, int]] = set()
        cum_visited = np.empty(length, dtype=int)
        for j in range(length):
            seen.add((int(pos[j, 0]), int(pos[j, 1])))
            cum_visited[j] = len(seen)

    raw_rewards = np.asarray(racer.get("rewards") or [], dtype=float).ravel()
    if len(raw_rewards) >= own_steps:
        step_rewards = raw_rewards[:own_steps]
    else:
        step_rewards = np.zeros(own_steps, dtype=float)
        if len(raw_rewards) > 0:
            step_rewards[: len(raw_rewards)] = raw_rewards
    cum_rewards = np.concatenate([[0.0], np.cumsum(step_rewards)])

    return _PreparedRacer(
        label=racer_label(racer),
        color=color,
        success=bool(racer.get("success", False)),
        positions=pos,
        own_steps=own_steps,
        cum_hits=cum_hits,
        cum_visited=cum_visited,
        cum_rewards=cum_rewards,
    )


class _RaceMazeView(QWidget):
    """Aynı labirent üzerinde tüm oyuncuların izlerini ve işaretlerini çizer."""

    MARGIN = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid: np.ndarray | None = None
        self._start: tuple[int, int] | None = None
        self._exit: tuple[int, int] = (0, 0)
        self._paths: list[np.ndarray] = []
        self._colors: list[QColor] = []
        self._step: int = 0
        self._show_trails: bool = True
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(460, 420)

    def minimumSizeHint(self) -> QSize:
        return QSize(140, 140)

    def set_race_data(
        self,
        grid: np.ndarray,
        start: tuple[int, int] | None,
        exit_pos: tuple[int, int],
        paths: list[np.ndarray],
        colors: list[str],
    ) -> None:
        """Sabit labirent ve oyuncu yollarını bir kez yükler."""
        self._grid = np.asarray(grid)
        self._start = None if start is None else (int(start[0]), int(start[1]))
        self._exit = (int(exit_pos[0]), int(exit_pos[1]))
        self._paths = paths
        self._colors = [QColor(c) for c in colors]
        self._step = 0
        self.update()

    def set_step(self, step: int) -> None:
        self._step = int(step)
        self.update()

    def set_show_trails(self, show: bool) -> None:
        self._show_trails = bool(show)
        self.update()

    def show_trails(self) -> bool:
        return self._show_trails

    def clear(self) -> None:
        self._grid = None
        self._start = None
        self._paths = []
        self._colors = []
        self._step = 0
        self.update()

    def has_frame(self) -> bool:
        return self._grid is not None

    def cell_rect(self, y: int, x: int) -> QRectF:
        """(y, x) karesinin widget koordinatlarındaki dikdörtgeni."""
        geo = self._geometry()
        if geo is None:
            return QRectF()
        ox, oy, cell = geo
        return QRectF(ox + x * cell, oy + y * cell, cell, cell)

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
        assert grid is not None
        h, w = grid.shape

        # Duvarlar ve boş kareler (görüş penceresi gösterilmez)
        p.setPen(Qt.PenStyle.NoPen)
        for y in range(h):
            row_y = oy + y * cell
            for x in range(w):
                color = COLOR_WALL if grid[y, x] else COLOR_FREE
                p.fillRect(QRectF(ox + x * cell, row_y, cell, cell), color)

        # Çıkış: dolu yeşil kare
        ey, ex = self._exit
        p.fillRect(self.cell_rect(ey, ex).adjusted(1, 1, -1, -1), COLOR_EXIT)

        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Başlangıç: küçük işaret
        if self._start is not None:
            r = self.cell_rect(*self._start)
            d = r.width() * 0.16
            p.setBrush(COLOR_START)
            p.drawEllipse(r.center(), d, d)

        # Oyuncu izleri (oyuncular arası küçük kaydırma ile üst üste binmez)
        n_racers = len(self._paths)
        if self._show_trails and n_racers > 0:
            for idx, pts in enumerate(self._paths):
                if len(pts) < 2:
                    continue
                upto = min(self._step, len(pts) - 1)
                if upto < 1:
                    continue
                dx, dy = self._trail_offset(idx, n_racers, cell)
                trail_color = QColor(self._colors[idx])
                trail_color.setAlpha(155)
                pen = QPen(trail_color)
                pen.setWidthF(max(1.5, cell * 0.11))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)

                poly = QPolygonF()
                prev_yx: tuple[int, int] | None = None
                for k in range(upto + 1):
                    py = int(pts[k, 0])
                    px = int(pts[k, 1])
                    if (py, px) == prev_yx:
                        continue
                    prev_yx = (py, px)
                    poly.append(QPointF(ox + (px + 0.5) * cell + dx, oy + (py + 0.5) * cell + dy))
                if poly.size() >= 2:
                    p.drawPolyline(poly)

        # Oyuncu işaretleri (aynı karedekiler kareyi paylaşacak şekilde kaydırılır)
        by_cell: dict[tuple[int, int], list[int]] = {}
        for idx, pts in enumerate(self._paths):
            if len(pts) == 0:
                continue
            j = min(self._step, len(pts) - 1)
            pos = (int(pts[j, 0]), int(pts[j, 1]))
            by_cell.setdefault(pos, []).append(idx)

        ring_w = max(1.0, cell * 0.06)
        for (cy, cx), group in by_cell.items():
            center = self.cell_rect(cy, cx).center()
            m = len(group)
            if m == 1:
                rad = max(2.0, cell * 0.34)
                centers = [center]
            else:
                rad = max(2.0, cell * min(0.24, 0.38 / math.sqrt(m)))
                spread = cell * 0.21
                centers = [
                    QPointF(
                        center.x() + math.cos(2.0 * math.pi * k / m - math.pi / 2.0) * spread,
                        center.y() + math.sin(2.0 * math.pi * k / m - math.pi / 2.0) * spread,
                    )
                    for k in range(m)
                ]
            for r_idx, pt in zip(group, centers):
                pen = QPen(COLOR_AGENT_RING)
                pen.setWidthF(ring_w)
                p.setPen(pen)
                p.setBrush(self._colors[r_idx])
                p.drawEllipse(pt, rad, rad)

        p.end()

    @staticmethod
    def _trail_offset(idx: int, total: int, cell: float) -> tuple[float, float]:
        if total <= 1:
            return 0.0, 0.0
        angle = (2.0 * math.pi * idx / total) + (math.pi / 4.0)
        dist = cell * 0.13
        return math.cos(angle) * dist, math.sin(angle) * dist


class RacePlayer(QWidget):
    """Yarış oynatıcı: labirent + renkli oyuncular ve izleri + renk açıklaması (lejant)
    + oynat/duraklat, başa sar, adım kaydırıcısı, hız + durum tablosu.

    Zaman çizelgesi ortaktır: adım i'de her oyuncu positions[min(i, kendi_adım_sayısı)]
    konumundadır (bitiren yerinde kalır). n_steps() = en uzun oyuncunun adım sayısı.
    """

    finished = Signal()  # oynatma son adıma ulaştı

    TICK_MS = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._race: dict[str, Any] | None = None
        self._title = ""
        self._prepared: list[_PreparedRacer] = []
        self._n_steps = 0
        self._step = 0
        self._speed = DEFAULT_SPEED
        self._acc = 0.0
        self._clock = QElapsedTimer()
        self._updating = False

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        # --- arayüz
        self.info_label = QLabel(tx("Kayıt yok", "No recording"), self)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.view = _RaceMazeView(self)

        # Lejant alanı
        self._legend_widget = QWidget(self)
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(12)
        self._legend_layout.addStretch(1)

        # Oynatma kontrolleri
        self.step_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.step_slider.setRange(0, 0)
        self.step_slider.setEnabled(False)

        self.play_button = QPushButton(tx("Oynat", "Play"), self)
        self.play_button.setMinimumWidth(74)
        self.rewind_button = QPushButton(tx("Başa sar", "Rewind"), self)
        self.trails_check = QCheckBox(tx("İzler", "Trails"), self)
        self.trails_check.setChecked(True)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setRange(0, _SPEED_SLIDER_MAX)
        self.speed_slider.setMinimumWidth(65)
        self.speed_slider.setMaximumWidth(140)
        self.speed_label = QLabel(self)
        self.speed_label.setMinimumWidth(72)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.rewind_button)
        buttons.addSpacing(4)
        buttons.addWidget(self.trails_check)
        buttons.addStretch(1)
        buttons.addWidget(QLabel(tx("Hız:", "Speed:"), self))
        buttons.addWidget(self.speed_slider, 1)
        buttons.addWidget(self.speed_label)

        # Durum tablosu
        headers = _table_headers()
        self.table = QTableWidget(0, len(headers), self)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(40)
        # Ad sütunu esner (sığmazsa kısaltılır, tamamı ipucunda); diğerleri içeriği kadar
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setMinimumHeight(90)
        self.table.setMaximumHeight(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.info_label)
        layout.addWidget(self.view, 1)
        layout.addWidget(self._legend_widget)
        layout.addWidget(self.step_slider)
        layout.addLayout(buttons)
        layout.addWidget(self.table)

        self.play_button.clicked.connect(self._on_play_clicked)
        self.rewind_button.clicked.connect(self.rewind)
        self.trails_check.toggled.connect(self.view.set_show_trails)
        self.step_slider.valueChanged.connect(self._on_step_slider)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)

        self.set_speed(DEFAULT_SPEED)
        self._update_controls()

    # ------------------------------------------------------------------ Genel API

    def load(self, race: dict[str, Any], title: str = "", autoplay: bool = True) -> None:
        """Yeni yarışı yükler (0. adım); autoplay ise oynatmaya başlar."""
        self.pause()
        self._race = race
        self._title = title
        colors = racer_colors(race)
        racers = race.get("racers", [])
        self._prepared = [_prepare_racer(r, c) for r, c in zip(racers, colors)]
        self._n_steps = max((p.own_steps for p in self._prepared), default=0)
        self._step = 0

        self.view.set_race_data(
            np.asarray(race["grid"]),
            tuple(race["start"]) if race.get("start") is not None else None,
            tuple(race["exit"]),
            [p.positions for p in self._prepared],
            [p.color for p in self._prepared],
        )
        self._rebuild_legend()
        self._rebuild_table_rows()

        self._updating = True
        self.step_slider.setRange(0, self._n_steps)
        self._updating = False

        self._show_step(0)
        self._update_controls()
        if autoplay and self._n_steps > 0:
            self.play()

    def clear(self) -> None:
        self.pause()
        self._race = None
        self._title = ""
        self._prepared = []
        self._n_steps = 0
        self._step = 0
        self.view.clear()
        self._clear_legend()
        self.table.setRowCount(0)
        self._updating = True
        self.step_slider.setRange(0, 0)
        self.step_slider.setValue(0)
        self._updating = False
        self.info_label.setText(tx("Kayıt yok", "No recording"))
        self._update_controls()

    def play(self) -> None:
        if self._race is None or self._n_steps == 0:
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

    def rewind(self) -> None:
        self.seek(0)

    def seek(self, step: int) -> None:
        clamped = max(0, min(int(step), self.n_steps()))
        if self._race is None:
            self._step = clamped
            return
        self._show_step(clamped)

    def set_speed(self, steps_per_sec: float) -> None:
        sps = min(MAX_SPEED, max(MIN_SPEED, float(steps_per_sec)))
        self._speed = sps
        self._set_speed_slider(sps)
        self._update_speed_label()

    def speed(self) -> float:
        return self._speed

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def current_step(self) -> int:
        return self._step

    def n_steps(self) -> int:
        return self._n_steps

    def race(self) -> dict[str, Any] | None:
        return self._race

    # ------------------------------------------------------------ iç işler

    def _on_play_clicked(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def _on_step_slider(self, value: int) -> None:
        if self._updating or self._race is None:
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

    def _show_step(self, step: int) -> None:
        if self._race is None:
            return
        self._step = step
        self.view.set_step(step)
        if self.step_slider.value() != step:
            self._updating = True
            self.step_slider.setValue(step)
            self._updating = False
        self.info_label.setText(self._info_text())
        self._update_table(step)

    def _info_text(self) -> str:
        if self._race is None:
            return tx("Kayıt yok", "No recording")
        i, n = self._step, self._n_steps
        parts: list[str] = []
        if self._title:
            parts.append(self._title)
        parts.append(tx(f"Adım {i}/{n}", f"Step {i}/{n}"))
        size = self._race.get("maze_size")
        if size is not None:
            parts.append(tx(f"Labirent: {size}x{size}", f"Maze: {size}x{size}"))
        seed = self._race.get("seed")
        if seed is not None:
            parts.append(f"Seed: {seed}")
        return "  |  ".join(parts)

    def _clear_legend(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._legend_layout.addStretch(1)

    def _rebuild_legend(self) -> None:
        self._clear_legend()
        # Sondaki stretch'ten önce öğeleri yerleştir
        insert_pos = 0
        for prep in self._prepared:
            entry = QWidget(self._legend_widget)
            entry_lay = QHBoxLayout(entry)
            entry_lay.setContentsMargins(0, 0, 0, 0)
            entry_lay.setSpacing(5)

            swatch = QLabel(entry)
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: {prep.color}; border: 1px solid #334155; border-radius: 2px;"
            )
            lbl = QLabel(prep.label, entry)
            entry_lay.addWidget(swatch)
            entry_lay.addWidget(lbl)
            self._legend_layout.insertWidget(insert_pos, entry)
            insert_pos += 1

    def _rebuild_table_rows(self) -> None:
        self.table.setHorizontalHeaderLabels(_table_headers())
        self.table.setRowCount(len(self._prepared))
        for row, prep in enumerate(self._prepared):
            pix = QPixmap(12, 12)
            pix.fill(QColor(prep.color))
            name_item = QTableWidgetItem(QIcon(pix), prep.label)
            name_item.setToolTip(prep.label)
            self.table.setItem(row, 0, name_item)

            for col in range(1, 7):
                item = QTableWidgetItem("-")
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)

        # Satır sayısına göre tablo yüksekliğini rahatça sığacak biçimde ayarla
        row_h = self.table.verticalHeader().defaultSectionSize()
        hdr_h = self.table.horizontalHeader().sizeHint().height()
        frame = 2 * self.table.frameWidth()
        target_h = min(260, max(84, hdr_h + len(self._prepared) * row_h + frame + 2))
        self.table.setFixedHeight(target_h)

    def _update_table(self, step: int) -> None:
        n_racers = len(self._prepared)
        if n_racers == 0:
            return

        won_now = [step >= p.own_steps and p.success for p in self._prepared]
        win_steps = [p.own_steps for k, p in enumerate(self._prepared) if won_now[k]]

        # Yarış sonunda en iyi satır vurgusu (en az adım, eşitse en yüksek toplam ödül)
        at_end = step >= self._n_steps
        best_flags = [False] * n_racers
        if at_end and any(won_now):
            best_score = min(
                (p.own_steps, -float(p.cum_rewards[p.own_steps]))
                for k, p in enumerate(self._prepared)
                if won_now[k]
            )
            for k, p in enumerate(self._prepared):
                if won_now[k] and (p.own_steps, -float(p.cum_rewards[p.own_steps])) == best_score:
                    best_flags[k] = True

        status_reached = tx("Çıkışa ulaştı", "Reached exit")
        status_timeout = tx("Süre doldu", "Timed out")
        status_running = tx("Yolda", "In progress")

        for row, prep in enumerate(self._prepared):
            j = min(step, prep.own_steps)
            if step >= prep.own_steps:
                status_text = status_reached if prep.success else status_timeout
            else:
                status_text = status_running

            if won_now[row]:
                rank = 1 + sum(1 for os_val in win_steps if os_val < prep.own_steps)
                rank_text = f"{rank}."
            else:
                rank_text = "-"

            values = (
                status_text,
                str(j),
                str(int(prep.cum_hits[j])),
                str(int(prep.cum_visited[j])),
                f"{float(prep.cum_rewards[j]):+.2f}",
                rank_text,
            )
            for col_offset, val in enumerate(values):
                item = self.table.item(row, col_offset + 1)
                if item is not None:
                    item.setText(val)

            is_best = best_flags[row]
            bg = QBrush(_BEST_BG) if is_best else QBrush()
            fg = QBrush(_BEST_FG) if is_best else QBrush()
            for col in range(7):
                item = self.table.item(row, col)
                if item is None:
                    continue
                item.setBackground(bg)
                item.setForeground(fg)
                font = item.font()
                if font.bold() != is_best:
                    font.setBold(is_best)
                    item.setFont(font)

    def _update_controls(self) -> None:
        loaded = self._race is not None
        self.play_button.setText(tx("Duraklat", "Pause") if self.is_playing() else tx("Oynat", "Play"))
        self.play_button.setEnabled(loaded and self._n_steps > 0)
        self.rewind_button.setEnabled(loaded)
        self.step_slider.setEnabled(loaded and self._n_steps > 0)
