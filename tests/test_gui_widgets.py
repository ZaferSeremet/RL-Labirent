"""MazeView ve ReplayPlayer testleri (offscreen Qt, pytest-qt yok)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from agents.base import EnvSpec
from baselines import RightHandAgent
from engine.episode import run_episode
from gui.maze_view import MazeView, ReplayPlayer
from maze_env import MazeEnv

_app = QApplication.instance() or QApplication([])


def make_traj(seed: int = 3, size: int = 5, view_radius: int = 1) -> dict:
    env = MazeEnv(maze_size=size, view_radius=view_radius, obs_mode="full")
    agent = RightHandAgent(EnvSpec.from_env(env))
    res = run_episode(env, agent, seed=seed, train=False, record=True)
    assert res.trajectory is not None
    return res.trajectory


def pixel(widget, rect) -> QColor:
    """grab() görüntüsünde dikdörtgenin merkezindeki renk."""
    img = widget.grab().toImage()
    c = rect.center()
    return QColor(img.pixel(int(c.x()), int(c.y())))


def first_wall(grid, avoid):
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] and (y, x) != avoid:
                return y, x
    raise AssertionError


# ------------------------------------------------------------------ MazeView


def test_maze_view_empty_and_sizes():
    v = MazeView()
    v.resize(300, 300)
    img = v.grab().toImage()
    assert not img.isNull() and img.width() == 300
    assert v.sizeHint().width() >= 300
    assert v.minimumSize().width() <= 200
    assert not v.has_frame()


def test_maze_view_set_frame_colors():
    traj = make_traj()
    grid, start, exit_pos = traj["grid"], traj["start"], traj["exit"]
    v = MazeView()
    v.resize(420, 360)  # kare olmayan boyut: ortalama ve ölçekleme
    visited = np.zeros(grid.shape, dtype=bool)
    visited[start] = True
    v.set_frame(grid, start, exit_pos, visited, view_radius=None, start=start)
    v.set_show_window(False)

    img = v.grab().toImage()
    assert not img.isNull()
    colors = {img.pixel(x, y) for x in range(0, img.width(), 7) for y in range(0, img.height(), 7)}
    assert len(colors) >= 4  # arka plan, duvar, boş, çıkış, ajan...

    wy, wx = first_wall(grid, exit_pos)
    wall = pixel(v, v.cell_rect(wy, wx))
    assert max(wall.red(), wall.green(), wall.blue()) < 90  # koyu

    ex = pixel(v, v.cell_rect(*exit_pos))
    assert ex.green() > ex.red() + 40 and ex.green() > ex.blue() + 40  # yeşil

    ag = pixel(v, v.cell_rect(*start))
    assert ag.red() > 200 and ag.green() < 130 and ag.blue() < 80  # kırmızı/turuncu

    # girilmiş kare ile boş kare farklı (açık mavi): ajan başka yerde
    other = next(
        (y, x)
        for y in range(grid.shape[0])
        for x in range(grid.shape[1])
        if not grid[y, x] and (y, x) not in (start, exit_pos)
    )
    v.set_frame(grid, other, exit_pos, visited, None, start)
    # başlangıç işareti merkezde olabilir; köşeye yakın bir noktadan bak
    r = v.cell_rect(*start)
    corner = QColor(v.grab().toImage().pixel(int(r.left() + 3), int(r.top() + 3)))
    assert corner.blue() > corner.red() + 20  # mavi tonlu

    # hücreler kare ve ortalı
    r00 = v.cell_rect(0, 0)
    assert abs(r00.width() - r00.height()) < 1e-9
    h, w = grid.shape
    left = r00.left()
    right = v.width() - v.cell_rect(0, w - 1).right()
    assert abs(left - right) <= 1.0


def test_maze_view_window_overlay_and_clear():
    traj = make_traj()
    grid, start, exit_pos = traj["grid"], traj["start"], traj["exit"]
    v = MazeView()
    v.resize(400, 400)
    v.set_frame(grid, start, exit_pos, None, view_radius=1, start=start)
    v.set_show_window(False)
    a = v.grab().toImage()
    v.set_show_window(True)
    b = v.grab().toImage()
    assert a != b  # pencere çizimi görüntüyü değiştiriyor
    v.clear()
    assert not v.has_frame()
    assert not v.grab().toImage().isNull()


# -------------------------------------------------------------- ReplayPlayer


def test_replay_seek_last_step_info_and_agent():
    traj = make_traj()
    n = len(traj["actions"])
    assert n > 3
    p = ReplayPlayer()
    p.resize(500, 600)
    p.show()
    QTest.qWait(20)
    p.load(traj, title="Deneme", autoplay=False)
    assert not p.is_playing()
    assert p.step_slider.maximum() == n
    assert "Adım 0/%d" % n in p.info_label.text()

    p.step_slider.setValue(n)
    assert p.current_step() == n
    text = p.info_label.text()
    assert "Deneme" in text and f"Adım {n}/{n}" in text
    assert "Çıkışa ulaştı" in text
    assert f"{sum(traj['rewards']):+.2f}" in text
    assert any(name in text for name in ("yukarı", "sağ", "aşağı", "sol"))
    # ajan son konumda
    assert p.view._agent == tuple(traj["positions"][-1])
    ag = pixel(p.view, p.view.cell_rect(*traj["positions"][-1]))
    assert ag.red() > 200 and ag.green() < 130

    # ara adım: ödül ve eylem o ana göre
    p.step_slider.setValue(2)
    assert "Adım 2/%d" % n in p.info_label.text()
    assert f"{sum(traj['rewards'][:2]):+.2f}" in p.info_label.text()
    assert p.view._agent == tuple(traj["positions"][2])


def test_replay_timeout_text():
    traj = make_traj()
    traj = dict(traj)
    # çıkışa varmadan biten kayıt: son konumu çıkıştan farklı yap
    traj["positions"] = traj["positions"][:3]
    traj["actions"] = traj["actions"][:2]
    traj["rewards"] = traj["rewards"][:2]
    p = ReplayPlayer()
    p.load(traj, autoplay=False)
    p.seek(2)
    assert "Süre doldu" in p.info_label.text()
    assert "Çıkışa ulaştı" not in p.info_label.text()


def test_replay_play_fast_emits_finished():
    traj = make_traj()
    p = ReplayPlayer()
    p.set_speed(200)
    assert p.speed() == pytest.approx(200)
    assert p.speed_slider.value() == p.speed_slider.maximum()
    spy = QSignalSpy(p.finished)
    p.load(traj, title="x", autoplay=True)
    assert p.is_playing()
    assert p.play_button.text() == "Duraklat"
    n = len(traj["actions"])
    for _ in range(100):
        if spy.count() or not p.is_playing():
            break
        QTest.qWait(50)
    assert spy.count() == 1
    assert not p.is_playing()
    assert p.current_step() == n
    assert p.play_button.text() == "Oynat"
    assert "Çıkışa ulaştı" in p.info_label.text()

    # sondayken oynat -> baştan
    p.set_speed(1)
    p.play()
    assert p.current_step() == 0
    p.pause()


def test_replay_speed_slider_log_scale():
    p = ReplayPlayer()
    p.set_speed(1)
    assert p.speed_slider.value() == 0 and "1 adım/sn" in p.speed_label.text()
    p.speed_slider.setValue(50)
    assert 12 < p.speed() < 16  # ~sqrt(200)
    p.set_speed(1000)  # kırpılır
    assert p.speed() == pytest.approx(200)
    p.set_speed(0)
    assert p.speed() == pytest.approx(1)


def test_replay_visited_rebuilt_on_rewind():
    traj = make_traj(seed=5, size=6)
    n = len(traj["actions"])
    pos = [tuple(q) for q in traj["positions"]]
    p = ReplayPlayer()
    p.load(traj, autoplay=False)

    def expected(i):
        m = np.zeros(traj["grid"].shape, dtype=bool)
        for q in pos[: i + 1]:
            m[q] = True
        return m

    assert np.array_equal(p.visited_map(), expected(0))
    p.seek(n)
    assert np.array_equal(p.visited_map(), expected(n))
    p.seek(n // 2)  # geri sarma: yeniden kurulur
    assert np.array_equal(p.visited_map(), expected(n // 2))
    assert p.visited_map().sum() < expected(n).sum() or n // 2 == n
    p.seek(n // 2 + 3)  # artımlı ileri
    assert np.array_equal(p.visited_map(), expected(n // 2 + 3))
    p.rewind()
    assert p.current_step() == 0
    assert np.array_equal(p.visited_map(), expected(0))


def test_replay_pause_play_buttons_and_window_checkbox():
    traj = make_traj()
    p = ReplayPlayer()
    p.set_speed(1)
    p.load(traj, autoplay=False)
    p.play_button.click()
    assert p.is_playing()
    p.step_slider.setValue(3)  # oynarken sarma
    assert p.current_step() == 3 and p.is_playing()
    p.play_button.click()
    assert not p.is_playing()
    p.window_check.setChecked(False)
    assert not p.view.show_window()
    p.window_check.setChecked(True)
    assert p.view.show_window()


def test_replay_large_recording_fast():
    # 900 adımlık sahte kayıt: ileri-geri sarma akıcı olmalı
    env = MazeEnv(maze_size=8, obs_mode="full")
    env.reset(seed=1)
    rs = env.get_render_state()
    free = np.argwhere(rs["grid"] == 0)
    rng = np.random.default_rng(0)
    pos = [tuple(int(v) for v in free[rng.integers(len(free))]) for _ in range(901)]
    traj = {
        "grid": rs["grid"],
        "start": pos[0],
        "exit": rs["exit"],
        "view_radius": 1,
        "positions": pos,
        "actions": [int(a) for a in rng.integers(0, 4, 900)],
        "rewards": [-0.01] * 900,
    }
    p = ReplayPlayer()
    p.resize(500, 600)
    p.load(traj, autoplay=False)
    import time

    t0 = time.perf_counter()
    for i in range(0, 901, 3):
        p.seek(i)
    for i in range(900, -1, -3):
        p.seek(i)
    QCoreApplication.processEvents()
    assert time.perf_counter() - t0 < 5.0
    assert "Adım 0/900" in p.info_label.text()

