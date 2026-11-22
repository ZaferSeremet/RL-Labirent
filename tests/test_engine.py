"""Faz 3-5: eğitim, test ve arka plan motoru (engine.trainer / tester / background)."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from engine import settings as engine_settings
from engine import storage
from engine.background import Job
from engine.tester import replay_test_episode, test_run
from engine.trainer import TrainConfig, snapshot_episodes, train

N_SEEDS = 5
EPISODES = 12
REPEATS = 2


def small_cfg(root: Path, **kw) -> TrainConfig:
    base = dict(agent_name="ornek", episodes=EPISODES, repeats=REPEATS, label="t",
                env_overrides={"maze_size": 4}, results_root=root)
    base.update(kw)
    return TrainConfig(**base)


def short_shared() -> dict:
    """Ortak ayarın test seed'leri 5'e indirilmiş kopyası (hız için)."""
    shared = copy.deepcopy(engine_settings.load_shared_config())
    for kind in engine_settings.TEST_KINDS:
        first = shared["test"][kind]["seeds"][0]
        shared["test"][kind]["seeds"] = [first, first + N_SEEDS - 1]
    return shared


@pytest.fixture
def short_seeds(monkeypatch):
    shared = short_shared()
    monkeypatch.setattr(engine_settings, "load_shared_config", lambda path=None: copy.deepcopy(shared))
    return shared


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Bir kez eğitilmiş çalışma (kısa test seed'leriyle) ve eğitim olayları."""
    mp = pytest.MonkeyPatch()
    shared = short_shared()
    mp.setattr(engine_settings, "load_shared_config", lambda path=None: copy.deepcopy(shared))
    try:
        root = tmp_path_factory.mktemp("sonuclar")
        events: list[dict] = []
        run_dir = train(small_cfg(root), progress=events.append)
    finally:
        mp.undo()
    return run_dir, events


def wait_job(job: Job, timeout: float = 90.0, until=None) -> list[dict]:
    """Bitiş (done/error) olayına ya da `until(event)` doğru olana dek olayları toplar."""
    events: list[dict] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ev in job.poll():
            events.append(ev)
            if ev["type"] in ("done", "error") or (until and until(ev)):
                return events
        time.sleep(0.05)
    raise AssertionError(f"Job zaman aşımına uğradı; gelen olaylar: {[e['type'] for e in events][-5:]}")


# ------------------------------------------------------------------ trainer

def test_train_folder_structure(trained):
    run_dir, events = trained
    s = storage.read_settings(run_dir)
    assert s["run_id"] == run_dir.name and s["created"]
    assert s["durum"] == "tamamlandi"
    assert s["agent"] == {"name": "ornek", "class": "agents.ornek_ajan.OrnekAjan",
                          "obs_mode": "simple", "is_baseline": False}
    tc = s["train_config"]
    assert tc["episodes"] == EPISODES and tc["repeats"] == REPEATS and tc["label"] == "t"
    assert "results_root" not in tc
    assert s["env"]["maze_size"] == 4 and s["env"]["obs_mode"] == "simple"
    assert s["shared"]["test"]["normal"]["seeds"] == [10000, 10000 + N_SEEDS - 1]
    assert {"alpha", "gamma", "epsilon"} <= set(s["hyperparameters"])
    assert s["tamamlanan_tekrar"] == REPEATS and s["sure_sn"] >= 0

    rows = storage.read_csv(storage.train_log_path(run_dir))
    assert len(rows) == EPISODES * REPEATS
    assert [(r["repeat"], r["episode"]) for r in rows] == [
        (rep, e) for rep in range(REPEATS) for e in range(1, EPISODES + 1)
    ]
    for r in range(REPEATS):
        assert storage.model_path(run_dir, r).is_file()

    expected = snapshot_episodes(EPISODES, 5)
    assert expected[0] == 1 and expected[-1] == EPISODES and len(expected) <= 5
    snaps = storage.list_snapshots(run_dir)
    assert [p.name for p in snaps] == [f"rep0_ep{e:06d}.json" for e in expected]
    assert (run_dir / "kayitlar" / "son.json").is_file()
    traj = storage.load_trajectory(snaps[0])
    assert len(traj["positions"]) == len(traj["actions"]) + 1


def test_snapshot_episode_rule():
    assert snapshot_episodes(2000, 5) == [1, 501, 1000, 1500, 2000]
    assert snapshot_episodes(3, 5) == [1, 2, 3]
    assert snapshot_episodes(10, 1) == [1]


def test_train_events(trained):
    run_dir, events = trained
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert events[0] == {"type": "start", "run_dir": str(run_dir), "episodes": EPISODES,
                         "repeats": REPEATS, "agent": "ornek"}
    assert events[-1] == {"type": "done", "run_dir": str(run_dir), "stopped": False}
    assert types.count("episode") == EPISODES * REPEATS
    assert types.count("repeat_done") == REPEATS
    assert types.count("snapshot") == len(snapshot_episodes(EPISODES, 5))
    assert types.count("live") == EPISODES * REPEATS  # live_every = max(1, 12 // 100) = 1
    json.dumps(events)  # sade tipler
    # her tekrarın repeat_done olayı, o tekrarın son bölümünden sonra gelir
    last_ep = max(i for i, e in enumerate(events) if e["type"] == "episode" and e["repeat"] == 0)
    first_done = next(i for i, e in enumerate(events) if e["type"] == "repeat_done")
    assert last_ep < first_done
    ep = next(e for e in events if e["type"] == "episode")
    assert set(ep) == {"type", "repeat", "episode", "total_reward", "steps", "success"}


def test_train_is_deterministic(tmp_path):
    a = train(small_cfg(tmp_path / "a"))
    b = train(small_cfg(tmp_path / "b"))
    ra = storage.read_csv(storage.train_log_path(a))
    rb = storage.read_csv(storage.train_log_path(b))
    assert ra == rb and len(ra) == EPISODES * REPEATS
    # Farklı seed -> farklı labirent dizisi
    c = train(small_cfg(tmp_path / "c", seed=7))
    rc = storage.read_csv(storage.train_log_path(c))
    assert [r["episode_seed"] for r in rc] != [r["episode_seed"] for r in ra]


@pytest.mark.parametrize("stop_at", [(0, 3), (1, 2)])
def test_train_stop(tmp_path, stop_at):
    stop = threading.Event()
    events: list[dict] = []

    def progress(ev):
        events.append(ev)
        if ev["type"] == "episode" and (ev["repeat"], ev["episode"]) == stop_at:
            stop.set()

    run_dir = train(small_cfg(tmp_path), progress=progress, stop=stop)
    s = storage.read_settings(run_dir)
    assert s["durum"] == "durduruldu"
    rep, ep = stop_at
    rows = storage.read_csv(storage.train_log_path(run_dir))
    assert len(rows) == rep * EPISODES + ep  # mevcut bölüm biter, yenisi başlamaz
    for r in range(rep + 1):
        assert storage.model_path(run_dir, r).is_file()
    assert not storage.model_path(run_dir, rep + 1).exists()
    assert s["tamamlanan_tekrar"] == rep
    assert events[-1] == {"type": "done", "run_dir": str(run_dir), "stopped": True}
    assert [e["repeat"] for e in events if e["type"] == "repeat_done"] == list(range(rep + 1))


def test_train_rejects_baseline_and_unknown(tmp_path):
    with pytest.raises(ValueError):
        train(TrainConfig("sag_el", episodes=3, repeats=1, results_root=tmp_path))
    with pytest.raises(ValueError):
        train(TrainConfig("random", episodes=3, repeats=1, results_root=tmp_path))
    with pytest.raises(KeyError):
        train(TrainConfig("yok_boyle_ajan", episodes=3, repeats=1, results_root=tmp_path))
    assert not any(tmp_path.iterdir())  # boş klasör bırakılmadı


# ------------------------------------------------------------------ tester

def read_tests(run_dir, kind):
    return storage.read_csv(storage.test_csv_path(run_dir, kind))


def test_test_run_csvs(trained):
    run_dir, _ = trained
    events: list[dict] = []
    paths = test_run(run_dir, progress=events.append)
    assert set(paths) == {"normal", "buyuk"}
    for kind in ("normal", "buyuk"):
        assert paths[kind] == storage.test_csv_path(run_dir, kind) and paths[kind].is_file()
        rows = read_tests(run_dir, kind)
        agent_rows = [r for r in rows if r["method"] == "ornek"]
        assert len(agent_rows) == REPEATS * N_SEEDS
        assert all(not r["is_baseline"] for r in agent_rows)
        assert {(r["repeat"], r["seed"]) for r in agent_rows} == {
            (rep, s) for rep in range(REPEATS) for s in engine_settings.seeds_for(kind, short_shared())
        }
        for name in ("random", "sag_el"):
            base = [r for r in rows if r["method"] == name]
            assert len(base) == N_SEEDS
            assert all(r["is_baseline"] and r["repeat"] == 0 for r in base)
        assert len(rows) == (REPEATS + 2) * N_SEEDS
        sag = [r for r in rows if r["method"] == "sag_el"]
        assert sum(r["success"] for r in sag) / len(sag) == 1.0
        for r in rows:  # verim yalnız başarılı bölümde
            assert (r["efficiency"] is not None) == r["success"]
            if r["success"]:
                assert r["efficiency"] == pytest.approx(r["steps"] / r["shortest_path"], rel=1e-3)
    # boyut test türünden gelir (eğitimdeki maze_size=4 testlere taşınmaz)
    assert read_tests(run_dir, "buyuk")[0]["shortest_path"] > 0

    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert events[0] == {"type": "start", "run_dir": str(run_dir), "kinds": ["normal", "buyuk"],
                         "total": 2 * (REPEATS + 2) * N_SEEDS}
    assert events[-1] == {"type": "done", "run_dir": str(run_dir), "stopped": False}
    assert types.count("test_episode") == events[0]["total"]
    assert [e["kind"] for e in events if e["type"] == "kind_done"] == ["normal", "buyuk"]
    json.dumps(events)


def test_test_run_is_deterministic_and_restarts_csv(trained):
    run_dir, _ = trained
    test_run(run_dir)
    first = {k: storage.test_csv_path(run_dir, k).read_bytes() for k in ("normal", "buyuk")}
    test_run(run_dir)
    second = {k: storage.test_csv_path(run_dir, k).read_bytes() for k in ("normal", "buyuk")}
    assert first == second  # baştan yazıldı, eklenmedi; içerik aynı
    only = test_run(run_dir, kinds=("normal",))
    assert set(only) == {"normal"}


def test_test_run_stop_keeps_partial_csv(trained):
    run_dir, _ = trained
    stop = threading.Event()
    events: list[dict] = []

    def progress(ev):
        events.append(ev)
        if ev["type"] == "test_episode" and sum(e["type"] == "test_episode" for e in events) == 3:
            stop.set()

    test_run(run_dir, progress=progress, stop=stop)
    assert len(read_tests(run_dir, "normal")) == 3
    assert events[-1]["type"] == "done" and events[-1]["stopped"] is True
    test_run(run_dir)  # sonraki tam koşu temizce yeniden yazar
    assert len(read_tests(run_dir, "normal")) == (REPEATS + 2) * N_SEEDS


def test_test_run_errors(trained, tmp_path):
    run_dir, _ = trained
    with pytest.raises(ValueError):
        test_run(run_dir, kinds=("orta",))
    # ayarlar.json'daki ajan sınıfı yoksa anlaşılır KeyError
    broken = tmp_path / "bozuk"
    broken.mkdir()
    s = storage.read_settings(run_dir)
    s["agent"]["name"] = "silinmis_ajan"
    storage.write_settings(broken, s)
    with pytest.raises(KeyError, match="silinmis_ajan"):
        test_run(broken)


@pytest.mark.parametrize("kind", ["normal", "buyuk"])
def test_replay_matches_csv(trained, kind):
    run_dir, _ = trained
    test_run(run_dir, kinds=(kind,))
    rows = read_tests(run_dir, kind)
    picks = [next(r for r in rows if r["method"] == "ornek" and r["repeat"] == 1),
             next(r for r in rows if r["method"] == "ornek" and r["repeat"] == 0),
             next(r for r in rows if r["method"] == "sag_el"),
             next(r for r in rows if r["method"] == "random")]
    for row in picks:
        args = (run_dir, kind, row["method"], row["repeat"], row["seed"])
        t1, t2 = replay_test_episode(*args), replay_test_episode(*args)
        assert t1["positions"] == t2["positions"] and t1["actions"] == t2["actions"]
        assert len(t1["actions"]) == row["steps"]
        assert len(t1["positions"]) == row["steps"] + 1
        assert sum(t1["rewards"]) == pytest.approx(row["total_reward"], abs=1e-3)
        if row["success"]:
            assert tuple(t1["positions"][-1]) == tuple(t1["exit"])
    with pytest.raises(KeyError):
        replay_test_episode(run_dir, "normal", "yok_ajan", 0, 10000)


def test_test_run_with_real_shared_size(tmp_path):
    """Gerçek ortak ayarın (100 seed) sag_el için başarı oranı 1.0; ajan modeli olmasa da çalışır."""
    run_dir = train(small_cfg(tmp_path, episodes=3, repeats=1))
    test_run(run_dir, kinds=("normal",))
    rows = read_tests(run_dir, "normal")
    assert len(rows) == 3 * 100  # 1 ajan tekrarı + 2 baseline, 100 seed
    sag = [r for r in rows if r["method"] == "sag_el"]
    assert all(r["success"] for r in sag)


# ------------------------------------------------------------------ background

def test_job_train(tmp_path):
    job = Job("train", cfg=small_cfg(tmp_path, episodes=10, repeats=1))
    assert not job.is_alive() and job.poll() == []
    job.start()
    events = wait_job(job)
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.count("episode") == 10 and events[-1]["stopped"] is False
    run_dir = Path(events[-1]["run_dir"])
    assert storage.read_settings(run_dir)["durum"] == "tamamlandi"
    job._proc.join(15)
    assert not job.is_alive()
    assert job.poll() == []


def test_job_test(trained):
    run_dir, _ = trained  # ayarlar.json 5 seed'lik ortak ayarı taşır
    job = Job("test", run_dir=str(run_dir), kinds=["normal"])
    job.start()
    events = wait_job(job)
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert events[0]["kinds"] == ["normal"]
    assert types.count("test_episode") == (REPEATS + 2) * N_SEEDS
    assert "kind_done" in types
    assert len(read_tests(run_dir, "normal")) == (REPEATS + 2) * N_SEEDS
    job._proc.join(15)


def test_job_error_event(tmp_path):
    job = Job("train", cfg=TrainConfig("olmayan_ajan", episodes=3, repeats=1, results_root=tmp_path))
    job.start()
    events = wait_job(job)
    last = events[-1]
    assert last["type"] == "error"
    assert "olmayan_ajan" in last["message"] and "KeyError" in last["traceback"]
    job._proc.join(15)
    assert job.poll() == []


def test_job_stop(tmp_path):
    job = Job("train", cfg=small_cfg(tmp_path, episodes=1_000_000, repeats=1, snapshots=1, live_every=100000))
    job.start()
    events = wait_job(job, until=lambda e: e["type"] == "episode" and e["episode"] >= 5)
    assert events[-1]["type"] == "episode"
    job.stop()
    events += wait_job(job)
    assert events[-1]["type"] == "done" and events[-1]["stopped"] is True
    run_dir = Path(events[-1]["run_dir"])
    assert storage.read_settings(run_dir)["durum"] == "durduruldu"
    assert storage.model_path(run_dir, 0).is_file()
    n_logged = len(storage.read_csv(storage.train_log_path(run_dir)))
    assert 5 <= n_logged < 1_000_000
    job._proc.join(15)
    assert not job.is_alive()


def test_job_killed_process_gives_error(tmp_path):
    job = Job("train", cfg=small_cfg(tmp_path, episodes=1_000_000, repeats=1, snapshots=1, live_every=100000))
    job.start()
    wait_job(job, until=lambda e: e["type"] == "episode")
    job._proc.terminate()
    job._proc.join(15)
    events = wait_job(job, timeout=10)
    assert events[-1]["type"] == "error" and "çıkış kodu" in events[-1]["message"]


def test_job_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Job("egit")
    with pytest.raises(ValueError):
        Job("train")
    with pytest.raises(ValueError):
        Job("test", kinds=["normal"])


def test_delete_run_only_inside_results_root(tmp_path):
    from engine import storage

    root = tmp_path / "results"
    run = storage.new_run_dir("ornek", root=root)
    storage.write_settings(run, {"agent": "ornek"})
    with pytest.raises(ValueError):
        storage.delete_run(tmp_path, root=root)  # kökün kendisi değil
    not_run = root / "rastgele"
    not_run.mkdir()
    with pytest.raises(ValueError):
        storage.delete_run(not_run, root=root)  # ayarlar.json yok
    storage.delete_run(run, root=root)
    assert not run.exists() and not_run.exists()
