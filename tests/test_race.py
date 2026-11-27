"""engine/race.py: ajan ve baseline'lar aynı labirentte yarışır."""

from __future__ import annotations

import pytest

from engine import race, settings as engine_settings
from engine.trainer import TrainConfig, train


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    return train(TrainConfig("ornek", episodes=2, repeats=2, snapshots=1, results_root=root))


def _check(r: dict, size: int) -> None:
    assert r["maze_size"] == size
    assert r["grid"].shape == (2 * size + 1, 2 * size + 1)
    keys = [x["key"] for x in r["racers"]]
    assert keys[0] == "agent" and r["racers"][0]["method"] == "ornek"
    assert set(keys[1:]) == {"random", "sag_el"}
    for x in r["racers"]:
        assert tuple(x["positions"][0]) == tuple(r["start"])
        assert len(x["positions"]) == len(x["actions"]) + 1 == x["steps"] + 1
        assert x["steps"] <= r["max_steps"]
        assert x["success"] == (tuple(x["positions"][-1]) == tuple(r["exit"]))
    sag = next(x for x in r["racers"] if x["key"] == "sag_el")
    assert sag["success"]  # sağ el döngüsüz labirentte hep çıkar


def test_race_test_episode_same_maze_and_deterministic(run_dir):
    seed = engine_settings.seeds_for("normal")[0]
    a = race.race_test_episode(run_dir, "normal", 1, seed)
    _check(a, engine_settings.load_shared_config()["test"]["normal"]["maze_size"])
    b = race.race_test_episode(run_dir, "normal", 1, seed)
    assert [x["positions"] for x in a["racers"]] == [x["positions"] for x in b["racers"]]


def test_race_custom_size(run_dir):
    r = race.race_custom(run_dir, 0, 123_456, 5)
    _check(r, 5)
    assert r["seed"] == 123_456


def test_race_errors(run_dir):
    with pytest.raises(ValueError):
        race.race_custom(run_dir, 0, 1, race.MAX_MAZE_SIZE + 1)
    with pytest.raises(ValueError):
        race.race_custom(run_dir, 7, 1, 5)  # böyle bir tekrar yok
    with pytest.raises(ValueError):
        race.race_test_episode(run_dir, "kucuk", 0, 1)


def test_old_run_uses_current_test_seeds(run_dir):
    """Eski bir çalışmanın kaydındaki test seed'leri değil, güncel ortak ayardakiler kullanılır."""
    from engine import storage
    from engine.tester import test_shared

    st = storage.read_settings(run_dir)
    st["shared"]["test"]["normal"]["seeds"] = [1, 2]  # eskimiş kayıt
    assert test_shared(st)["test"] == engine_settings.load_shared_config()["test"]
    assert test_shared(st)["env"] == st["shared"]["env"]  # ödüller/ortam çalışmanın kendisinden
