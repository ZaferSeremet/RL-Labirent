"""Matplotlib Figure gömme ve ucuz canlı öğrenme eğrisi bileşenleri."""

from __future__ import annotations

from collections import deque

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from engine.analysis import AGENT_PALETTE
from engine.i18n import tx


def _make_message_figure(message: str, figsize: tuple[float, float] = (8.0, 4.5)) -> Figure:
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


class PlotCanvas(QWidget):
    """engine.analysis.* tarafından üretilen Figure nesnelerini gösteren Qt bileşeni."""

    def __init__(self, parent: QWidget | None = None, *, toolbar: bool = True) -> None:
        super().__init__(parent)
        self._show_toolbar = toolbar
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self.show_message(tx("Grafik göstermek için çalışma seçin", "Select a run to display a plot"))

    @property
    def figure(self) -> Figure | None:
        return self._canvas.figure if self._canvas is not None else None

    def set_figure(self, fig: Figure) -> None:
        """Mevcut grafiği yeni Figure ile değiştirir."""
        if self._toolbar is not None:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.setParent(None)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if self._show_toolbar:
            tb = NavigationToolbar2QT(canvas, self)
            self._layout.addWidget(tb)
            self._toolbar = tb
        self._layout.addWidget(canvas, 1)
        self._canvas = canvas
        canvas.draw_idle()

    def show_message(self, message: str) -> None:
        self.set_figure(_make_message_figure(message))


class _RepeatSeries:
    """Tek bir tekrar için O(1) hareketli ortalama serisi."""

    def __init__(self, window: int) -> None:
        self.window = max(1, window)
        self.episodes: list[int] = []
        self.success_ma: list[float] = []
        self.reward_ma: list[float] = []
        self._succ_q: deque[float] = deque()
        self._rew_q: deque[float] = deque()
        self._succ_sum: float = 0.0
        self._rew_sum: float = 0.0

    def append(self, episode: int, success: bool, reward: float) -> None:
        s_val = 1.0 if success else 0.0
        r_val = float(reward)

        self._succ_q.append(s_val)
        self._succ_sum += s_val
        self._rew_q.append(r_val)
        self._rew_sum += r_val

        if len(self._succ_q) > self.window:
            self._succ_sum -= self._succ_q.popleft()
            self._rew_sum -= self._rew_q.popleft()

        n = len(self._succ_q)
        self.episodes.append(int(episode))
        self.success_ma.append(self._succ_sum / n)
        self.reward_ma.append(self._rew_sum / n)


class LiveLearningPlot(QWidget):
    """Eğitim sırasında ucuz güncellenen canlı öğrenme eğrisi (set_data + draw_idle).

    Üst eksen: Başarı oranı (hareketli ortalama)
    Alt eksen: Ortalama ödül (hareketli ortalama)
    Her tekrar ayrı renkte çizilir.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(6.4, 5.2), dpi=100)
        self._ax_succ = self._fig.add_subplot(2, 1, 1)
        self._ax_rew = self._fig.add_subplot(2, 1, 2, sharex=self._ax_succ)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._window = 20
        self._total_episodes = 100
        self._series: dict[int, _RepeatSeries] = {}
        self._succ_lines: dict[int, Line2D] = {}
        self._rew_lines: dict[int, Line2D] = {}
        self._dirty_repeats: set[int] = set()
        self._total_points = 0

        self.reset(total_episodes=100)

    @property
    def figure(self) -> Figure:
        return self._fig

    def point_count(self) -> int:
        return self._total_points

    def has_data(self) -> bool:
        return self._total_points > 0

    def reset(self, total_episodes: int = 100) -> None:
        """Grafiği yeni bir eğitim için sıfırlar."""
        self._total_episodes = max(1, int(total_episodes))
        if self._total_episodes >= 100:
            self._window = max(1, self._total_episodes // 50)
        else:
            self._window = max(1, min(5, self._total_episodes // 4 or 1))

        self._series.clear()
        self._succ_lines.clear()
        self._rew_lines.clear()
        self._dirty_repeats.clear()
        self._total_points = 0

        self._ax_succ.clear()
        self._ax_rew.clear()

        self._ax_succ.set_title(
            tx(
                f"Canlı Öğrenme Eğrisi (Hareketli Ortalama, pencere={self._window})",
                f"Live Learning Curve (Moving Average, window={self._window})",
            ),
            fontsize=10.5,
            fontweight="bold",
        )
        self._ax_succ.set_ylabel(tx("Başarı Oranı", "Success Rate"))
        self._ax_succ.set_ylim(-0.05, 1.08)
        self._ax_succ.set_xlim(1, max(2, self._total_episodes))
        self._ax_succ.grid(True, linestyle="--", alpha=0.35)
        self._ax_succ.spines["top"].set_visible(False)
        self._ax_succ.spines["right"].set_visible(False)

        self._ax_rew.set_xlabel(tx("Bölüm", "Episode"))
        self._ax_rew.set_ylabel(tx("Ortalama Ödül", "Average Reward"))
        self._ax_rew.set_xlim(1, max(2, self._total_episodes))
        self._ax_rew.grid(True, linestyle="--", alpha=0.35)
        self._ax_rew.spines["top"].set_visible(False)
        self._ax_rew.spines["right"].set_visible(False)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def add_episode(self, repeat: int, episode: int, success: bool, total_reward: float) -> None:
        """Yalnız belleğe ekler; yeniden çizmez (poll başına bir kez refresh() çağır)."""
        rep = int(repeat)
        ser = self._series.get(rep)
        if ser is None:
            ser = _RepeatSeries(self._window)
            self._series[rep] = ser
        ser.append(episode, success, total_reward)
        self._dirty_repeats.add(rep)
        self._total_points += 1

    def refresh(self) -> None:
        """Yalnız değişen serilerin çizgi verisini günceller ve draw_idle() çağırır."""
        if not self._dirty_repeats:
            return

        legend_changed = False
        max_ep = 2
        for rep in sorted(self._dirty_repeats):
            ser = self._series[rep]
            if rep not in self._succ_lines:
                color = AGENT_PALETTE[rep % len(AGENT_PALETTE)]
                label = tx(f"Tekrar {rep + 1}", f"Repeat {rep + 1}")
                (s_line,) = self._ax_succ.plot([], [], color=color, linewidth=1.8, label=label)
                (r_line,) = self._ax_rew.plot([], [], color=color, linewidth=1.8, label=label)
                self._succ_lines[rep] = s_line
                self._rew_lines[rep] = r_line
                legend_changed = True

            self._succ_lines[rep].set_data(ser.episodes, ser.success_ma)
            self._rew_lines[rep].set_data(ser.episodes, ser.reward_ma)

        for ser in self._series.values():
            if ser.episodes and ser.episodes[-1] > max_ep:
                max_ep = ser.episodes[-1]

        upper_x = max(max_ep, self._total_episodes, 2)
        self._ax_succ.set_xlim(1, upper_x)
        self._ax_rew.set_xlim(1, upper_x)
        self._ax_rew.relim()
        self._ax_rew.autoscale_view(scalex=False, scaley=True)

        if legend_changed:
            self._ax_succ.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=3)

        self._dirty_repeats.clear()
        self._canvas.draw_idle()
