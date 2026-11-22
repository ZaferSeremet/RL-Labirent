""""Oyna" sekmesi: labirenti klavyeyle oyna (SÖZLEŞME: imza sabit)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine.i18n import tx
from engine.settings import env_config_for, load_shared_config
from gui.maze_view import (
    ACTION_LABELS,
    COLOR_AGENT,
    COLOR_AGENT_RING,
    COLOR_BACKGROUND,
    COLOR_EXIT,
    COLOR_FREE,
    COLOR_VISITED,
    COLOR_WALL,
    MazeView,
    action_label,
)
from maze_env import EnvConfig, MazeEnv

MIN_MAZE, MAX_MAZE = 2, 20
MIN_RADIUS, MAX_RADIUS = 1, 3

# Tuş -> eylem (0 yukarı, 1 sağ, 2 aşağı, 3 sol)
KEY_ACTIONS = {
    Qt.Key.Key_Up: 0,
    Qt.Key.Key_W: 0,
    Qt.Key.Key_Right: 1,
    Qt.Key.Key_D: 1,
    Qt.Key.Key_Down: 2,
    Qt.Key.Key_S: 2,
    Qt.Key.Key_Left: 3,
    Qt.Key.Key_A: 3,
}

_STATUS_STYLES = {
    "info": "background:#e3ebf7; color:#1f3556; border:1px solid #b6c8e4;",
    "success": "background:#d6f0dc; color:#14532d; border:1px solid #7cc592;",
    "timeout": "background:#fde7c8; color:#7a3e00; border:1px solid #e8b46c;",
}


class ObservationView(QWidget):
    """Ajanın gördüğü k x k pencereyi büyütülmüş çizer (yalnız observe_dict bilgisi).
    Ajan ortada; duvar koyu, boş açık, girilmiş açık mavi, çıkış yeşil."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: np.ndarray | None = None
        self._visited: np.ndarray | None = None
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(300, 300)

    def set_observation(self, tiles: np.ndarray, visited: np.ndarray) -> None:
        self._tiles = np.asarray(tiles)
        self._visited = np.asarray(visited, dtype=bool)
        self.update()

    def clear(self) -> None:
        self._tiles = None
        self._visited = None
        self.update()

    def cell_rect(self, y: int, x: int) -> QRectF:
        """Pencere karesinin (y, x) widget koordinatlarındaki dikdörtgeni."""
        if self._tiles is None:
            return QRectF()
        k = self._tiles.shape[0]
        margin = 6
        cell = float(max(1, int(min(self.width() - 2 * margin, self.height() - 2 * margin) / k)))
        ox = round((self.width() - cell * k) / 2.0)
        oy = round((self.height() - cell * k) / 2.0)
        return QRectF(ox + x * cell, oy + y * cell, cell, cell)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        p = QPainter(self)
        p.fillRect(self.rect(), COLOR_BACKGROUND)
        if self._tiles is None:
            p.setPen(QColor("#5b6270"))
            font = QFont(self.font())
            font.setPointSize(max(10, font.pointSize() + 2))
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tx("Görüş yok", "No view"))
            p.end()
            return
        k = self._tiles.shape[0]
        p.setPen(Qt.PenStyle.NoPen)
        for y in range(k):
            for x in range(k):
                t = self._tiles[y, x]
                if t == 1:
                    color = COLOR_WALL
                elif t == 2:
                    color = COLOR_EXIT
                elif self._visited is not None and self._visited[y, x]:
                    color = COLOR_VISITED
                else:
                    color = COLOR_FREE
                p.fillRect(self.cell_rect(y, x), color)
        # Hücre çizgileri
        line = QColor(0, 0, 0, 40)
        p.setPen(QPen(line, 1))
        for y in range(k):
            for x in range(k):
                p.drawRect(self.cell_rect(y, x))
        # Ajan ortada
        c = k // 2
        r = self.cell_rect(c, c)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        ring = QPen(COLOR_AGENT_RING)
        ring.setWidthF(max(1.5, r.width() * 0.06))
        p.setPen(ring)
        p.setBrush(COLOR_AGENT)
        p.drawEllipse(r.center(), r.width() * 0.34, r.width() * 0.34)
        p.end()


class PlayTab(QWidget):
    """Kendi MazeEnv'ini kurar (labirent boyutu, görüş yarıçapı, seed alanları + "Yeni labirent").
    Ok tuşları / WASD ile hareket. Solda tam labirent (MazeView), sağda büyütülmüş
    "ajanın gördüğü" pencere (yalnız observe_dict bilgisi) ve son ödül, toplam ödül,
    adım, çarpma, en kısa yol. Bölüm bitince mesaj ve yeni labirent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.env: MazeEnv | None = None
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.success = False
        self._info: dict = {}

        try:
            shared = load_shared_config()
            default_size = int(shared["env"]["maze_size"])
        except Exception:  # ortak ayar okunamazsa varsayılan
            shared = None
            default_size = 8
        self._shared = shared

        # --- kontroller
        self.size_spin = QSpinBox(self)
        self.size_spin.setRange(MIN_MAZE, MAX_MAZE)
        self.size_spin.setValue(min(MAX_MAZE, max(MIN_MAZE, default_size)))
        self.size_spin.setToolTip(
            tx(
                "Labirent boyutu (hücre sayısı, 2-20). Yeni labirentle geçerli olur.",
                "Maze size (number of cells, 2-20). Takes effect on new maze.",
            )
        )
        self.radius_spin = QSpinBox(self)
        self.radius_spin.setRange(MIN_RADIUS, MAX_RADIUS)
        self.radius_spin.setValue(1)
        self.radius_spin.setToolTip(
            tx(
                "Görüş yarıçapı (1-3). Yeni labirentle geçerli olur.",
                "View radius (1-3). Takes effect on new maze.",
            )
        )
        self.seed_edit = QLineEdit(self)
        self.seed_edit.setPlaceholderText(tx("rastgele", "random"))
        self.seed_edit.setMaxLength(9)
        self.seed_edit.setFixedWidth(110)
        self.seed_edit.setToolTip(
            tx(
                "Boş bırakılırsa her seferinde rastgele; aynı seed aynı labirenti verir.",
                "If left empty, random each time; the same seed produces the same maze.",
            )
        )
        self.new_button = QPushButton(tx("Yeni labirent", "New maze"), self)
        for w in (self.size_spin, self.radius_spin, self.seed_edit):
            w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.new_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        controls = QHBoxLayout()
        controls.addWidget(QLabel(tx("Labirent boyutu:", "Maze size:"), self))
        controls.addWidget(self.size_spin)
        controls.addSpacing(10)
        controls.addWidget(QLabel(tx("Görüş yarıçapı:", "View radius:"), self))
        controls.addWidget(self.radius_spin)
        controls.addSpacing(10)
        controls.addWidget(QLabel(tx("Seed:", "Seed:"), self))
        controls.addWidget(self.seed_edit)
        controls.addSpacing(10)
        controls.addWidget(self.new_button)
        controls.addStretch(1)

        # --- durum mesajı
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumHeight(40)
        f = QFont(self.status_label.font())
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.status_label.setFont(f)

        # --- sol: tam labirent
        self.maze_view = MazeView(self)
        self.maze_view.set_show_window(True)
        self.maze_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # --- sağ: ajanın gördüğü + sayaçlar
        self.obs_view = ObservationView(self)
        self.last_action_label = QLabel(self)
        self.last_action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.obs_group = QGroupBox(tx("Ajanın gördüğü", "Agent's view"), self)
        og = QVBoxLayout(self.obs_group)
        og.addWidget(self.obs_view, 1)
        og.addWidget(self.last_action_label)

        self.last_reward_label = QLabel("-", self)
        self.total_reward_label = QLabel("-", self)
        self.steps_label = QLabel("-", self)
        self.wall_label = QLabel("-", self)
        self.shortest_label = QLabel("-", self)
        self.seed_label = QLabel("-", self)
        stats = QFormLayout()
        stats.addRow(tx("Son ödül:", "Last reward:"), self.last_reward_label)
        stats.addRow(tx("Toplam ödül:", "Total reward:"), self.total_reward_label)
        stats.addRow(tx("Adım / limit:", "Steps / limit:"), self.steps_label)
        stats.addRow(tx("Çarpma:", "Wall hits:"), self.wall_label)
        stats.addRow(tx("En kısa yol:", "Shortest path:"), self.shortest_label)
        stats.addRow(tx("Seed:", "Seed:"), self.seed_label)
        self.stats_box = QGroupBox(tx("Durum", "Status"), self)
        self.stats_box.setLayout(stats)

        right = QVBoxLayout()
        right.addWidget(self.obs_group, 1)
        right.addWidget(self.stats_box)

        body = QHBoxLayout()
        body.addWidget(self.maze_view, 3)
        body.addLayout(right, 2)

        hint = QLabel(
            tx(
                "Ok tuşları ya da WASD ile hareket et. Duvara çarpmak ceza verir; yeşil kare çıkıştır.",
                "Move with arrow keys or WASD. Hitting a wall gives a penalty; the green tile is the exit.",
            ),
            self,
        )
        hint.setStyleSheet("color:#5b6270;")
        hint.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addLayout(body, 1)
        layout.addWidget(hint)

        self.new_button.clicked.connect(self.new_maze)
        self.seed_edit.returnPressed.connect(self.new_maze)

        # Klavye odağı: sekmeye gelen odak labirentte olsun
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setFocusProxy(self.maze_view)

        self.new_maze()

    # ------------------------------------------------------------ labirent

    def _seed_value(self) -> int | None:
        text = self.seed_edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _build_config(self) -> EnvConfig:
        n, r = self.size_spin.value(), self.radius_spin.value()
        try:
            return env_config_for("egitim", "full", self._shared, maze_size=n, view_radius=r)
        except Exception:
            return EnvConfig(maze_size=n, view_radius=r, obs_mode="full")

    def new_maze(self) -> None:
        """Ayarlardan yeni bir ortam kurar ve (seed doluysa o seed'le) bölümü başlatır."""
        text = self.seed_edit.text().strip()
        seed = self._seed_value()
        if text and seed is None:
            self._set_status(tx("Seed bir tam sayı olmalı.", "Seed must be an integer."), "timeout")
            return
        self.env = MazeEnv(self._build_config())
        _, self._info = self.env.reset(seed=seed)
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.success = False
        self._set_status(
            tx(
                "Ok tuşları ya da WASD ile hareket et; çıkışı (yeşil kare) bul.",
                "Move with arrow keys or WASD; find the exit (green tile).",
            ),
            "info",
        )
        self._refresh()
        self.maze_view.setFocus()

    # ------------------------------------------------------------ hareket

    def move(self, action: int) -> bool:
        """Bir adım atar (0 yukarı, 1 sağ, 2 aşağı, 3 sol). Bölüm bitmişse False döner."""
        if self.env is None or self.done:
            return False
        _, reward, terminated, truncated, info = self.env.step(int(action))
        self._info = info
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        if terminated or truncated:
            self.done = True
            self.success = bool(terminated)
            if terminated:
                self._set_status(
                    tx(
                        f"Çıkışa ulaştın! {info['steps']} adım (en kısa yol {info['shortest_path']}), "
                        f"{info['wall_hits']} çarpma. Devam için \"Yeni labirent\" ya da Enter.",
                        f"You reached the exit! {info['steps']} steps (shortest path {info['shortest_path']}), "
                        f"{info['wall_hits']} wall hits. Press \"New maze\" or Enter to continue.",
                    ),
                    "success",
                )
            else:
                self._set_status(
                    tx(
                        f"Süre doldu: {info['max_steps']} adımda çıkışa ulaşamadın. "
                        "Devam için \"Yeni labirent\" ya da Enter.",
                        f"Timed out: could not reach the exit in {info['max_steps']} steps. "
                        "Press \"New maze\" or Enter to continue.",
                    ),
                    "timeout",
                )
        self._refresh()
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt)
        fw = QApplication.focusWidget()
        if fw is not None and isinstance(fw, (QLineEdit, QAbstractSpinBox)) and self.isAncestorOf(fw):
            super().keyPressEvent(event)
            return
        mods = event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier
        )
        key = event.key()
        if not mods:
            if key in KEY_ACTIONS:
                self.move(KEY_ACTIONS[key])
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self.done:
                self.new_maze()
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------ odak

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        QTimer.singleShot(0, self.maze_view.setFocus)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt)
        self.maze_view.setFocus()
        super().mousePressEvent(event)

    # ------------------------------------------------------------ görünüm

    def _set_status(self, text: str, kind: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"QLabel {{ {_STATUS_STYLES[kind]} border-radius:6px; padding:6px 10px; }}"
        )

    def _refresh(self) -> None:
        env = self.env
        if env is None:
            return
        rs = env.get_render_state()
        self.maze_view.set_frame(
            rs["grid"], rs["agent"], rs["exit"], rs["visited"], rs["view_radius"], rs["start"]
        )
        obs = env.observe_dict()
        self.obs_view.set_observation(obs["tiles"], obs["visited"])
        la = obs["last_action"]
        act_text = action_label(la) if 0 <= la < 4 else "-"
        self.last_action_label.setText(tx(f"Son eylem: {act_text}", f"Last action: {act_text}"))
        info = self._info
        self.last_reward_label.setText(f"{self.last_reward:+.2f}" if info["steps"] else "-")
        self.total_reward_label.setText(f"{self.total_reward:+.2f}")
        self.steps_label.setText(f"{info['steps']} / {info['max_steps']}")
        self.wall_label.setText(str(info["wall_hits"]))
        self.shortest_label.setText(str(info["shortest_path"]))
        self.seed_label.setText(str(info["episode_seed"]))
