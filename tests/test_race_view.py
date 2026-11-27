"""RacePlayer ve yardımcıları için offscreen Qt testleri (pytest-qt yok)."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLabel

from engine import i18n, race
from engine.trainer import TrainConfig, train
from gui.race_view import (
    FALLBACK_COLORS,
    RACER_COLORS,
    RacePlayer,
    racer_colors,
    racer_label,
)
from tests.i18n_check import turkish_leftovers

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _restore_turkish():
    i18n.set_language("tr", persist=False)
    yield
    i18n.set_language("tr", persist=False)


def make_sample_race() -> dict:
    """Farklı bitiş adımları, çarpma, eşit bitiş sırası ve timeout içeren küçük yarış."""
    grid = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return {
        "grid": grid,
        "start": (1, 1),
        "exit": (1, 3),
        "view_radius": 1,
        "maze_size": 3,
        "seed": 42,
        "max_steps": 5,
        "racers": [
            {
                "key": "agent",
                "method": "ornek",
                "is_baseline": False,
                # 3 adım: çarpma + iki hareket -> çıkış
                "positions": [(1, 1), (1, 1), (1, 2), (1, 3)],
                "actions": [0, 1, 1],
                "rewards": [-0.2, -0.05, 1.0],
                "success": True,
                "steps": 3,
                "wall_hits": 1,
                "total_reward": 0.75,
                "visited_tiles": 3,
                "shortest_path": 2,
            },
            {
                "key": "sag_el",
                "method": "sag_el",
                "is_baseline": True,
                # 3 adım: ajanla aynı adım sayısında çıkış, daha düşük ödül
                "positions": [(1, 1), (2, 1), (2, 2), (1, 3)],
                "actions": [2, 1, 0],
                "rewards": [-0.05, -0.05, 0.5],
                "success": True,
                "steps": 3,
                "wall_hits": 0,
                "total_reward": 0.40,
                "visited_tiles": 4,
                "shortest_path": 2,
            },
            {
                "key": "random",
                "method": "random",
                "is_baseline": True,
                # 5 adım: başarısız (süre doldu)
                "positions": [(1, 1), (1, 1), (1, 1), (2, 1), (2, 1), (1, 2)],
                "actions": [0, 3, 2, 3, 1],
                "rewards": [-0.2, -0.2, -0.05, -0.2, -0.05],
                "success": False,
                "steps": 5,
                "wall_hits": 3,
                "total_reward": -0.70,
                "visited_tiles": 3,
                "shortest_path": 2,
            },
        ],
    }


def make_ranked_race() -> dict:
    """1., 2. ve '-' sıralarını ayrı adımlarda doğrulayan yarış."""
    r = make_sample_race()
    r["racers"][1]["positions"] = [(1, 1), (2, 1), (2, 2), (2, 3), (1, 3)]
    r["racers"][1]["actions"] = [2, 1, 1, 0]
    r["racers"][1]["rewards"] = [-0.05, -0.05, -0.05, 0.8]
    r["racers"][1]["steps"] = 4
    return r


def test_racer_label_and_colors():
    assert racer_label({"key": "agent", "method": "ornek", "is_baseline": False}) == "ornek (eğitilmiş ajan)"
    assert racer_label({"key": "sag_el", "method": "sag_el", "is_baseline": True}) == "Sağ el"
    assert racer_label({"key": "random", "method": "random", "is_baseline": True}) == "Random"

    i18n.set_language("en", persist=False)
    assert racer_label({"key": "agent", "method": "ornek", "is_baseline": False}) == "ornek (trained agent)"
    assert racer_label({"key": "sag_el", "method": "sag_el", "is_baseline": True}) == "Right-hand"

    dummy_race = {
        "racers": [
            {"key": "agent"},
            {"key": "sag_el"},
            {"key": "extra_a"},
            {"key": "extra_b"},
        ]
    }
    assert racer_colors(dummy_race) == [
        RACER_COLORS["agent"],
        RACER_COLORS["sag_el"],
        FALLBACK_COLORS[0],
        FALLBACK_COLORS[1],
    ]


def test_race_load_n_steps_and_legend():
    race_data = make_sample_race()
    player = RacePlayer()
    assert player.race() is None
    assert player.n_steps() == 0

    player.load(race_data, title="Test Yarışı", autoplay=False)
    assert player.race() is race_data
    assert player.n_steps() == 5
    assert player.current_step() == 0
    assert not player.is_playing()
    assert player.step_slider.maximum() == 5
    assert "Test Yarışı" in player.info_label.text()
    assert "Adım 0/5" in player.info_label.text()
    assert "3x3" in player.info_label.text()
    assert "Seed: 42" in player.info_label.text()

    legend_texts = [lbl.text() for lbl in player._legend_widget.findChildren(QLabel) if lbl.text()]
    assert "ornek (eğitilmiş ajan)" in legend_texts
    assert "Sağ el" in legend_texts
    assert "Random" in legend_texts


def test_race_seek_updates_table_and_keeps_finished_in_place():
    race_data = make_sample_race()
    player = RacePlayer()
    player.load(race_data, autoplay=False)
    tbl = player.table
    assert tbl.rowCount() == 3

    # Adım 0: herkes yolda
    for row in range(3):
        assert tbl.item(row, 1).text() == "Yolda"
        assert tbl.item(row, 2).text() == "0"
        assert tbl.item(row, 3).text() == "0"
        assert tbl.item(row, 4).text() == "1"
        assert tbl.item(row, 5).text() == "+0.00"
        assert tbl.item(row, 6).text() == "-"

    # Adım 2: çarpma, gezilen kare, ödül kontrolü
    player.seek(2)
    assert tbl.item(0, 1).text() == "Yolda"
    assert tbl.item(0, 2).text() == "2"
    assert tbl.item(0, 3).text() == "1"  # (1,1) -> (1,1)
    assert tbl.item(0, 4).text() == "2"  # (1,1), (1,2)
    assert tbl.item(0, 5).text() == "-0.25"

    assert tbl.item(2, 2).text() == "2"
    assert tbl.item(2, 3).text() == "2"  # iki duvar çarpması
    assert tbl.item(2, 4).text() == "1"  # hep (1,1)'de
    assert tbl.item(2, 5).text() == "-0.40"

    # Adım 3: ajan ve sağ el 3. adımda çıkışa ulaşır, eşit -> ikisi de 1.
    player.seek(3)
    assert tbl.item(0, 1).text() == "Çıkışa ulaştı"
    assert tbl.item(1, 1).text() == "Çıkışa ulaştı"
    assert tbl.item(2, 1).text() == "Yolda"
    assert tbl.item(0, 6).text() == "1."
    assert tbl.item(1, 6).text() == "1."
    assert tbl.item(2, 6).text() == "-"
    # Yarış bitmediği için henüz satır vurgusu yok
    assert not tbl.item(0, 0).font().bold()

    # Adım 4: bitiren yerinde kalır (adım 3, değerler değişmez, konum son karede)
    player.seek(4)
    assert tbl.item(0, 2).text() == "3"
    assert tbl.item(0, 3).text() == "1"
    assert tbl.item(0, 4).text() == "3"
    assert tbl.item(0, 5).text() == "+0.75"
    assert tbl.item(1, 2).text() == "3"
    assert tbl.item(2, 2).text() == "4"
    # Görünümde bitiren oyuncular son konumunda (1, 3) kalmış olmalı
    pos0 = tuple(int(v) for v in player.view._paths[0][min(4, len(player.view._paths[0]) - 1)])
    assert pos0 == (1, 3)

    # Adım 5 (son): random süre doldu; en iyi satır (ajan, yüksek ödül) vurgulanır
    player.seek(5)
    assert tbl.item(2, 1).text() == "Süre doldu"
    assert tbl.item(2, 6).text() == "-"
    assert tbl.item(0, 0).font().bold()
    assert not tbl.item(1, 0).font().bold()
    assert not tbl.item(2, 0).font().bold()

    # Geri sarınca vurgu kalkar
    player.seek(1)
    assert not tbl.item(0, 0).font().bold()


def test_race_rank_column_distinct_steps():
    race_data = make_ranked_race()
    player = RacePlayer()
    player.load(race_data, autoplay=False)
    player.seek(5)
    assert player.table.item(0, 6).text() == "1."
    assert player.table.item(1, 6).text() == "2."
    assert player.table.item(2, 6).text() == "-"


def test_race_play_fast_emits_finished_and_controls():
    race_data = make_sample_race()
    player = RacePlayer()
    player.set_speed(200)
    assert player.speed() == pytest.approx(200)
    assert player.speed_slider.value() == player.speed_slider.maximum()

    spy = QSignalSpy(player.finished)
    player.load(race_data, autoplay=True)
    assert player.is_playing()
    assert player.play_button.text() == "Duraklat"

    for _ in range(100):
        if spy.count() or not player.is_playing():
            break
        QTest.qWait(30)

    assert spy.count() == 1
    assert not player.is_playing()
    assert player.current_step() == player.n_steps()
    assert player.play_button.text() == "Oynat"

    # Sondayken oynat -> 0'dan başlar
    player.set_speed(1)
    player.play()
    assert player.current_step() == 0
    player.pause()
    assert not player.is_playing()

    player.seek(3)
    player.rewind()
    assert player.current_step() == 0

    player.clear()
    assert player.race() is None
    assert player.n_steps() == 0
    assert player.table.rowCount() == 0
    assert "Kayıt yok" in player.info_label.text()


def test_race_paint_grab_and_trails_toggle():
    player = RacePlayer()
    player.resize(520, 640)
    player.show()
    QTest.qWait(20)

    # Boş durumda paintEvent
    empty_img = player.grab().toImage()
    assert not empty_img.isNull() and empty_img.width() == 520

    # Yarış yüklüyken paintEvent
    player.load(make_sample_race(), autoplay=False)
    player.seek(3)
    QCoreApplication.processEvents()
    img_with_trails = player.view.grab().toImage()
    assert not img_with_trails.isNull()

    # İzleri kapatınca görüntü değişmeli
    player.trails_check.setChecked(False)
    QCoreApplication.processEvents()
    img_without_trails = player.view.grab().toImage()
    assert img_without_trails != img_with_trails

    full_img = player.grab().toImage()
    assert not full_img.isNull()
    player.close()


def test_race_with_real_engine_custom_race(tmp_path):
    run = train(TrainConfig("ornek", episodes=2, repeats=1, snapshots=1, results_root=tmp_path))
    r = race.race_custom(run, 0, 123, 5)
    player = RacePlayer()
    player.resize(540, 640)
    player.load(r, title="Gerçek Motor", autoplay=False)
    assert player.n_steps() == max(len(x["positions"]) - 1 for x in r["racers"])
    player.seek(player.n_steps())
    assert not player.grab().toImage().isNull()


def test_race_large_900_steps_fast():
    base = make_sample_race()
    rng = np.random.default_rng(7)
    grid = base["grid"]
    free = np.argwhere(grid == 0)
    racers = []
    for key, method, is_base in [("agent", "ornek", False), ("sag_el", "sag_el", True), ("random", "random", True)]:
        pos = [tuple(int(v) for v in free[rng.integers(len(free))]) for _ in range(901)]
        racers.append(
            {
                "key": key,
                "method": method,
                "is_baseline": is_base,
                "positions": pos,
                "actions": [0] * 900,
                "rewards": [-0.01] * 900,
                "success": key == "agent",
                "steps": 900,
                "wall_hits": 0,
                "total_reward": -9.0,
                "visited_tiles": 9,
                "shortest_path": 2,
            }
        )
    big_race = dict(base, racers=racers, max_steps=900)
    player = RacePlayer()
    player.resize(520, 620)
    player.load(big_race, autoplay=False)
    t0 = time.perf_counter()
    for i in range(0, 901, 5):
        player.seek(i)
    for i in range(900, -1, -5):
        player.seek(i)
    QCoreApplication.processEvents()
    assert time.perf_counter() - t0 < 5.0


def test_race_player_english_no_turkish_leftovers():
    i18n.set_language("en", persist=False)
    player = RacePlayer()
    assert not turkish_leftovers(player)

    player.load(make_sample_race(), title="Demo Race", autoplay=False)
    assert not turkish_leftovers(player)

    player.seek(5)
    assert not turkish_leftovers(player)
    assert player.table.item(0, 1).text() == "Reached exit"
    assert player.table.item(2, 1).text() == "Timed out"
    player.seek(1)
    assert player.table.item(0, 1).text() == "In progress"
