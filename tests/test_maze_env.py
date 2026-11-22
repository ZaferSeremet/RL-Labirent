from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from maze_env import (
    DOWN,
    LEFT,
    NO_ACTION,
    RIGHT,
    SIMPLE_N_STATES,
    TILE_EXIT,
    TILE_FREE,
    TILE_WALL,
    UP,
    EnvConfig,
    MazeEnv,
    RewardConfig,
    decode_simple_state,
)

ACTION_DELTAS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

# (sign(dr), sign(dc)) -> exit_dir code (1..8 = K, KD, D, GD, G, GB, B, KB)
DIR_SIGNS_TO_CODE = {
    (-1, 0): 1,
    (-1, 1): 2,
    (0, 1): 3,
    (1, 1): 4,
    (1, 0): 5,
    (1, -1): 6,
    (0, -1): 7,
    (-1, -1): 8,
}


def bfs_shortest_path_actions(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[int]:
    """Independent BFS returning action sequence from start to goal on free tiles."""
    rows, cols = grid.shape
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
    seen = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), path + [action]))
    raise AssertionError(f"No path from {start} to {goal}")


def bfs_reachable(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    rows, cols = grid.shape
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def expected_full_obs(render_state: dict) -> np.ndarray:
    grid = render_state["grid"]
    agent_r, agent_c = render_state["agent"]
    exit_r, exit_c = render_state["exit"]
    visited = render_state["visited"]
    radius = render_state["view_radius"]
    last_action = render_state["last_action"]

    rows, cols = grid.shape
    k = 2 * radius + 1
    wall_map = np.zeros((k, k), dtype=np.float32)
    exit_map = np.zeros((k, k), dtype=np.float32)
    visited_map = np.zeros((k, k), dtype=np.float32)

    for wr, dr in enumerate(range(-radius, radius + 1)):
        for wc, dc in enumerate(range(-radius, radius + 1)):
            rr, cc = agent_r + dr, agent_c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                wall_map[wr, wc] = 1.0
            else:
                if grid[rr, cc] != 0:
                    wall_map[wr, wc] = 1.0
                if (rr, cc) == (exit_r, exit_c):
                    exit_map[wr, wc] = 1.0
                if visited[rr, cc]:
                    visited_map[wr, wc] = 1.0

    action_onehot = np.zeros(5, dtype=np.float32)
    action_onehot[last_action] = 1.0
    return np.concatenate(
        [wall_map.ravel(), exit_map.ravel(), visited_map.ravel(), action_onehot]
    )


def expected_simple_decoded(render_state: dict) -> dict:
    grid = render_state["grid"]
    agent_r, agent_c = render_state["agent"]
    exit_r, exit_c = render_state["exit"]
    visited = render_state["visited"]
    radius = render_state["view_radius"]
    last_action = render_state["last_action"]
    rows, cols = grid.shape

    def neighbor_code(dr: int, dc: int) -> int:
        rr, cc = agent_r + dr, agent_c + dc
        if not (0 <= rr < rows and 0 <= cc < cols) or grid[rr, cc] != 0:
            return 2  # wall
        if (rr, cc) == (exit_r, exit_c):
            return 3  # exit
        return 1 if visited[rr, cc] else 0

    neighbors = {
        "up": neighbor_code(-1, 0),
        "right": neighbor_code(0, 1),
        "down": neighbor_code(1, 0),
        "left": neighbor_code(0, -1),
    }

    dr = exit_r - agent_r
    dc = exit_c - agent_c
    if max(abs(dr), abs(dc)) > radius or (dr == 0 and dc == 0):
        exit_dir = 0
    else:
        sr = (1 if dr > 0 else -1) if dr != 0 else 0
        sc = (1 if dc > 0 else -1) if dc != 0 else 0
        exit_dir = DIR_SIGNS_TO_CODE[(sr, sc)]

    return {
        "neighbors": neighbors,
        "exit_dir": exit_dir,
        "last_action": last_action,
    }


# 1. Labirent yapisi
@pytest.mark.parametrize("maze_size", [2, 4, 8])
@pytest.mark.parametrize("seed", [0, 42, 999])
def test_maze_structure(maze_size: int, seed: int) -> None:
    env = MazeEnv(maze_size=maze_size)
    env.reset(seed=seed)
    grid = env.get_render_state()["grid"]

    expected_dim = 2 * maze_size + 1
    assert grid.shape == (expected_dim, expected_dim)

    # Dis cerceve tamamen duvar
    assert np.all(grid[0, :] == TILE_WALL)
    assert np.all(grid[-1, :] == TILE_WALL)
    assert np.all(grid[:, 0] == TILE_WALL)
    assert np.all(grid[:, -1] == TILE_WALL)

    # Bos kare sayisi 2n^2 - 1
    free_coords = [tuple(pt) for pt in np.argwhere(grid == 0)]
    expected_free = 2 * maze_size * maze_size - 1
    assert len(free_coords) == expected_free

    # Baglilik: tum bos kareler ulasilabilir
    reachable = bfs_reachable(grid, free_coords[0])
    assert len(reachable) == expected_free

    # Dongusuz (agac): kenar sayisi = dugum sayisi - 1
    directed_edges = 0
    free_set = set(free_coords)
    for r, c in free_coords:
        for dr, dc in ACTION_DELTAS.values():
            if (r + dr, c + dc) in free_set:
                directed_edges += 1
    assert directed_edges % 2 == 0
    undirected_edges = directed_edges // 2
    assert undirected_edges == expected_free - 1


# 2. Baslangic ve cikis
@pytest.mark.parametrize("maze_size", [2, 5, 8])
@pytest.mark.parametrize("seed", [7, 123])
def test_start_exit_and_shortest_path(maze_size: int, seed: int) -> None:
    env = MazeEnv(maze_size=maze_size)
    _, info = env.reset(seed=seed)
    rs = env.get_render_state()
    grid = rs["grid"]
    start = rs["start"]
    exit_pos = rs["exit"]

    assert start != exit_pos
    assert rs["agent"] == start
    assert grid[start] == 0
    assert grid[exit_pos] == 0

    # Hucre karesi (tek indisli)
    assert start[0] % 2 == 1 and start[1] % 2 == 1
    assert exit_pos[0] % 2 == 1 and exit_pos[1] % 2 == 1

    bfs_actions = bfs_shortest_path_actions(grid, start, exit_pos)
    assert info["shortest_path"] == len(bfs_actions)


# 3. Seed tekrarlanabilirligi
def test_seed_reproducibility() -> None:
    env1 = MazeEnv(maze_size=6)
    env2 = MazeEnv(maze_size=6)

    _, info1 = env1.reset(seed=12345)
    _, info2 = env2.reset(seed=12345)
    rs1 = env1.get_render_state()
    rs2 = env2.get_render_state()

    assert info1["episode_seed"] == 12345
    assert info2["episode_seed"] == 12345
    assert np.array_equal(rs1["grid"], rs2["grid"])
    assert rs1["start"] == rs2["start"]
    assert rs1["exit"] == rs2["exit"]

    # seed'siz reset() dizisi ayni labirent dizisini verir
    for _ in range(4):
        _, i1 = env1.reset()
        _, i2 = env2.reset()
        r1 = env1.get_render_state()
        r2 = env2.get_render_state()
        assert i1["episode_seed"] == i2["episode_seed"]
        assert np.array_equal(r1["grid"], r2["grid"])
        assert r1["start"] == r2["start"]
        assert r1["exit"] == r2["exit"]

    # Farkli seed'ler farkli labirentler uretir
    signatures = set()
    for s in range(6):
        env1.reset(seed=s)
        rs = env1.get_render_state()
        signatures.add((rs["grid"].tobytes(), rs["start"], rs["exit"]))
    assert len(signatures) > 1


# 4. Hareket ve yonler
def test_movement_directions_and_wall_hits() -> None:
    env = MazeEnv(maze_size=5)
    env.reset(seed=11)
    rs = env.get_render_state()
    grid = rs["grid"]

    tested_free = set()
    tested_wall = set()
    wall_hits = 0

    # Tum 4 yonde hem bos kareye hareketi hem duvara carpmayi test et
    # Once labirentte her dugumu gezerek veya rastgele/BFS ile ilerleyelim
    visited_nodes = {rs["agent"]}
    # Tum ulasilabilir kareleri DFS ile gezip cikisa girmeden yonleri deneyelim
    exit_pos = rs["exit"]

    def walk_and_test(curr: tuple[int, int]) -> None:
        nonlocal wall_hits
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = curr[0] + dr, curr[1] + dc
            if (nr, nc) == exit_pos:
                continue
            if grid[nr, nc] != 0:
                if action not in tested_wall:
                    _, _, term, trunc, info = env.step(action)
                    assert not term and not trunc
                    wall_hits += 1
                    assert env.get_render_state()["agent"] == curr
                    assert info["wall_hits"] == wall_hits
                    tested_wall.add(action)
            else:
                if (nr, nc) not in visited_nodes or action not in tested_free:
                    was_new = (nr, nc) not in visited_nodes
                    visited_nodes.add((nr, nc))
                    _, _, term, trunc, info = env.step(action)
                    assert not term and not trunc
                    assert env.get_render_state()["agent"] == (nr, nc)
                    assert info["wall_hits"] == wall_hits
                    tested_free.add(action)
                    if was_new:
                        walk_and_test((nr, nc))
                    # Geri don
                    opposite = (action + 2) % 4
                    _, _, term, trunc, info = env.step(opposite)
                    assert not term and not trunc
                    assert env.get_render_state()["agent"] == curr
                    assert info["wall_hits"] == wall_hits
                    tested_free.add(opposite)

    walk_and_test(rs["agent"])
    assert tested_free == {UP, RIGHT, DOWN, LEFT}
    assert tested_wall == {UP, RIGHT, DOWN, LEFT}


# 5. Oduller
def test_rewards_all_components() -> None:
    rewards = RewardConfig(
        step=-0.1,
        wall=-0.5,
        exit=10.0,
        new_tile=0.25,
        timeout=-2.0,
    )
    env = MazeEnv(maze_size=4, rewards=rewards)
    _, info = env.reset(seed=3)
    rs = env.get_render_state()
    grid = rs["grid"]
    start = rs["start"]
    exit_pos = rs["exit"]

    # Default RewardConfig degerlerini de kontrol et
    default_r = RewardConfig()
    assert default_r.step == pytest.approx(-0.01)
    assert default_r.wall == pytest.approx(-0.05)
    assert default_r.exit == pytest.approx(5.0)
    assert default_r.new_tile == pytest.approx(0.0)
    assert default_r.timeout == pytest.approx(0.0)

    path = bfs_shortest_path_actions(grid, start, exit_pos)
    assert len(path) >= 2

    # 1) Ilk bos kareye adim -> step + new_tile
    first_action = path[0]
    _, rew, term, trunc, _ = env.step(first_action)
    assert not term and not trunc
    assert rew == pytest.approx(rewards.step + rewards.new_tile)

    # 2) Baslangica geri adim (zaten girilmis kare) -> yalniz step
    back_action = (first_action + 2) % 4
    _, rew, term, trunc, _ = env.step(back_action)
    assert not term and not trunc
    assert rew == pytest.approx(rewards.step)

    # 3) Duvara carpma -> step + wall
    pos = env.get_render_state()["agent"]
    wall_action = next(
        a for a, (dr, dc) in ACTION_DELTAS.items() if grid[pos[0] + dr, pos[1] + dc] != 0
    )
    _, rew, term, trunc, _ = env.step(wall_action)
    assert not term and not trunc
    assert rew == pytest.approx(rewards.step + rewards.wall)

    # 4) Cikis karesine ilk kez giris -> step + exit + new_tile
    for a in path[:-1]:
        env.step(a)
    _, rew, term, trunc, info = env.step(path[-1])
    assert term and not trunc
    assert info["success"] is True
    assert rew == pytest.approx(rewards.step + rewards.exit + rewards.new_tile)

    # 5) Timeout odulu yalniz truncated adiminda eklenir
    env_to = MazeEnv(maze_size=4, max_steps=3, rewards=rewards)
    env_to.reset(seed=3)
    rs_to = env_to.get_render_state()
    w_act = next(
        a
        for a, (dr, dc) in ACTION_DELTAS.items()
        if rs_to["grid"][rs_to["agent"][0] + dr, rs_to["agent"][1] + dc] != 0
    )
    _, r1, term, trunc, _ = env_to.step(w_act)
    assert not term and not trunc and r1 == pytest.approx(rewards.step + rewards.wall)
    _, r2, term, trunc, _ = env_to.step(w_act)
    assert not term and not trunc and r2 == pytest.approx(rewards.step + rewards.wall)
    _, r3, term, trunc, info_to = env_to.step(w_act)
    assert not term and trunc and info_to["success"] is False
    assert r3 == pytest.approx(rewards.step + rewards.wall + rewards.timeout)


# 6. Bolum sonu, hatalar ve varsayilan max_steps
def test_episode_termination_truncation_and_errors() -> None:
    # Reset oncesi step() RuntimeError
    env = MazeEnv(maze_size=4)
    with pytest.raises(RuntimeError):
        env.step(UP)

    # Varsayilan max_steps = 3 * (2n^2 - 1)
    for n in (2, 4, 8):
        e = MazeEnv(maze_size=n)
        _, info_n = e.reset(seed=0)
        expected_max = 3 * (2 * n * n - 1)
        assert e.max_steps == expected_max
        assert info_n["max_steps"] == expected_max

    # Gecersiz eylemler ValueError
    env.reset(seed=5)
    for bad_action in (-1, 4, 100):
        with pytest.raises(ValueError):
            env.step(bad_action)

    # Cikisa BFS yoluyla gidince terminated=True, truncated=False, success=True
    rs = env.get_render_state()
    path = bfs_shortest_path_actions(rs["grid"], rs["start"], rs["exit"])
    for idx, action in enumerate(path):
        _, _, term, trunc, info = env.step(action)
        assert not (term and trunc)
        if idx < len(path) - 1:
            assert not term and not trunc
        else:
            assert term and not trunc
            assert info["success"] is True
            assert info["steps"] == len(path)

    # Bitmis bolumde step() RuntimeError
    with pytest.raises(RuntimeError):
        env.step(UP)

    # Tam max_steps adiminda cikisa ulasilirsa terminated=True, truncated=False
    env_exact = MazeEnv(maze_size=4, max_steps=len(path))
    env_exact.reset(seed=5)
    for idx, action in enumerate(path):
        _, _, term, trunc, info = env_exact.step(action)
        assert not (term and trunc)
        if idx == len(path) - 1:
            assert term is True
            assert trunc is False
            assert info["success"] is True

    # Kucuk max_steps ile duvara carparak tam max_steps adiminda truncated=True
    env_trunc = MazeEnv(maze_size=4, max_steps=4)
    env_trunc.reset(seed=5)
    rs_t = env_trunc.get_render_state()
    wall_act = next(
        a
        for a, (dr, dc) in ACTION_DELTAS.items()
        if rs_t["grid"][rs_t["agent"][0] + dr, rs_t["agent"][1] + dc] != 0
    )
    for step_idx in range(1, 5):
        _, _, term, trunc, info = env_trunc.step(wall_act)
        assert not (term and trunc)
        if step_idx < 4:
            assert not term and not trunc
        else:
            assert not term and trunc
            assert info["success"] is False
            assert info["steps"] == 4

    with pytest.raises(RuntimeError):
        env_trunc.step(wall_act)


# 7. "full" gozlem
@pytest.mark.parametrize("view_radius", [1, 2])
def test_full_observation(view_radius: int) -> None:
    env = MazeEnv(maze_size=5, view_radius=view_radius, obs_mode="full")
    obs, _ = env.reset(seed=17)
    k = 2 * view_radius + 1
    expected_len = 3 * k * k + 5

    rs = env.get_render_state()
    path = bfs_shortest_path_actions(rs["grid"], rs["start"], rs["exit"])

    def check_obs(current_obs: np.ndarray) -> None:
        assert current_obs.shape == (expected_len,)
        assert current_obs.dtype == np.float32
        assert np.all((current_obs == 0.0) | (current_obs == 1.0))
        assert env.observation_space.contains(current_obs)

        cur_rs = env.get_render_state()
        expected = expected_full_obs(cur_rs)
        assert np.array_equal(current_obs, expected)

        # Ziyaret haritasi merkezi her zaman 1
        visited_slice = current_obs[2 * k * k : 3 * k * k].reshape(k, k)
        assert visited_slice[view_radius, view_radius] == 1.0

    # Baslangicta last_action one-hot indis 4
    assert np.array_equal(obs[-5:], np.array([0, 0, 0, 0, 1], dtype=np.float32))
    check_obs(obs)

    # Koselere yakin giderek sinir disi pencere tasmasini da kapsa:
    # once duvara carp, sonra BFS yoluyla cikisa git
    grid = rs["grid"]
    wall_act = next(
        a
        for a, (dr, dc) in ACTION_DELTAS.items()
        if grid[rs["agent"][0] + dr, rs["agent"][1] + dc] != 0
    )
    obs, _, _, _, _ = env.step(wall_act)
    check_obs(obs)

    for idx, action in enumerate(path):
        obs, _, _, _, _ = env.step(action)
        check_obs(obs)
        # Cikis pencerede ise cikis haritasinda dogru yerde 1, aksi halde 0
        cur_rs = env.get_render_state()
        er, ec = cur_rs["exit"]
        ar, ac = cur_rs["agent"]
        dr, dc = er - ar, ec - ac
        exit_slice = obs[k * k : 2 * k * k].reshape(k, k)
        if abs(dr) <= view_radius and abs(dc) <= view_radius:
            assert exit_slice[view_radius + dr, view_radius + dc] == 1.0
            assert float(np.sum(exit_slice)) == 1.0
        else:
            assert float(np.sum(exit_slice)) == 0.0


def test_full_observation_out_of_bounds_at_border_view_radius_2() -> None:
    """view_radius=2 iken dis duvara bitisik hucrede pencere izgaranin disina tasar."""
    n = 4
    env = MazeEnv(maze_size=n, view_radius=2, obs_mode="full")
    env.reset(seed=2)
    rs = env.get_render_state()
    grid = rs["grid"]
    # Cikis olmayan bir kose hucre sec (her zaman bostur ve iki kenardan disari tasar)
    corners = [(1, 1), (1, 2 * n - 1), (2 * n - 1, 1), (2 * n - 1, 2 * n - 1)]
    target = next(c for c in corners if c != rs["exit"])
    # Cikisa basmadan gitmek icin cikisi gecici engel sayan BFS yerine en kisa yol
    # (target != exit oldugu icin kose uzerinden cikis gecmedigi bir kose secelim)
    best_path = None
    for c in corners:
        if c == rs["exit"]:
            continue
        p = bfs_shortest_path_actions(grid, rs["start"], c)
        # Yol cikistan geciyor mu kontrol et
        r_cur, c_cur = rs["start"]
        hits_exit = False
        for a in p:
            dr, dc = ACTION_DELTAS[a]
            r_cur, c_cur = r_cur + dr, c_cur + dc
            if (r_cur, c_cur) == rs["exit"]:
                hits_exit = True
                break
        if not hits_exit:
            target = c
            best_path = p
            break
    assert best_path is not None
    for act in best_path:
        obs, _, term, trunc, _ = env.step(act)
        assert not term and not trunc
    cur_rs = env.get_render_state()
    assert cur_rs["agent"] == target
    assert np.array_equal(obs, expected_full_obs(cur_rs))
    # k=5 penceresinde izgaranin disina tasan satir ve sutun tamamen duvar olmali
    wall_map = obs[:25].reshape(5, 5)
    oob_row = 0 if target[0] == 1 else 4
    oob_col = 0 if target[1] == 1 else 4
    assert np.all(wall_map[oob_row, :] == 1.0)
    assert np.all(wall_map[:, oob_col] == 1.0)


# 8. "simple" gozlem
@pytest.mark.parametrize("view_radius", [1, 2])
def test_simple_observation_random_episodes(view_radius: int) -> None:
    import random

    env = MazeEnv(maze_size=4, view_radius=view_radius, obs_mode="simple")
    rng = random.Random(2026)

    for seed in (0, 1, 2):
        obs, _ = env.reset(seed=seed)
        for _ in range(40):
            assert 0 <= obs < SIMPLE_N_STATES == 11520
            assert env.observation_space.contains(obs)

            decoded = decode_simple_state(obs)
            rs = env.get_render_state()
            expected = expected_simple_decoded(rs)
            assert decoded == expected

            # Kodlama formulu kontrolu
            nb = decoded["neighbors"]
            encoded = (
                (
                    (((nb["up"] * 4 + nb["right"]) * 4 + nb["down"]) * 4 + nb["left"])
                    * 9
                    + decoded["exit_dir"]
                )
                * 5
                + decoded["last_action"]
            )
            assert encoded == obs

            obs, _, term, trunc, _ = env.step(rng.randrange(4))
            if term or trunc:
                assert 0 <= obs < 11520
                assert decode_simple_state(obs) == expected_simple_decoded(
                    env.get_render_state()
                )
                break


# 9. observe_dict()
@pytest.mark.parametrize("view_radius", [1, 2])
def test_observe_dict_matches_full_obs(view_radius: int) -> None:
    env = MazeEnv(maze_size=4, view_radius=view_radius, obs_mode="full")
    obs, _ = env.reset(seed=9)
    k = 2 * view_radius + 1

    rs = env.get_render_state()
    path = bfs_shortest_path_actions(rs["grid"], rs["start"], rs["exit"])

    def verify(full_obs: np.ndarray) -> None:
        od = env.observe_dict()
        tiles = od["tiles"]
        visited = od["visited"]
        last_action = od["last_action"]

        assert tiles.shape == (k, k)
        assert visited.shape == (k, k)
        assert set(np.unique(tiles).tolist()).issubset({TILE_FREE, TILE_WALL, TILE_EXIT})
        assert 0 <= last_action <= NO_ACTION

        assert np.array_equal((tiles == TILE_WALL).astype(np.float32).ravel(), full_obs[: k * k])
        assert np.array_equal(
            (tiles == TILE_EXIT).astype(np.float32).ravel(), full_obs[k * k : 2 * k * k]
        )
        assert np.array_equal(
            visited.astype(np.float32).ravel(), full_obs[2 * k * k : 3 * k * k]
        )
        expected_oh = np.zeros(5, dtype=np.float32)
        expected_oh[last_action] = 1.0
        assert np.array_equal(full_obs[3 * k * k :], expected_oh)

    verify(obs)
    for act in path:
        obs, _, _, _, _ = env.step(act)
        verify(obs)


# 10. Gymnasium uyumu
@pytest.mark.parametrize("obs_mode", ["full", "simple"])
def test_gymnasium_env_checker(obs_mode: str) -> None:
    env = MazeEnv(maze_size=4, view_radius=1, obs_mode=obs_mode)
    assert isinstance(env, gym.Env)
    check_env(env, skip_render_check=True)


# 11. Config
def test_env_config_roundtrip_validation_and_overrides() -> None:
    cfg = EnvConfig(
        maze_size=6,
        view_radius=2,
        max_steps=50,
        obs_mode="simple",
        rewards=RewardConfig(step=-0.02, wall=-0.1, exit=3.0, new_tile=0.05, timeout=-1.0),
    )
    d = cfg.to_dict()
    cfg2 = EnvConfig.from_dict(d)
    assert cfg2 == cfg

    # Gecersiz degerler ValueError
    with pytest.raises(ValueError):
        EnvConfig(maze_size=1)
    with pytest.raises(ValueError):
        EnvConfig(view_radius=0)
    with pytest.raises(ValueError):
        EnvConfig(obs_mode="invalid")
    with pytest.raises(ValueError):
        EnvConfig(max_steps=0)

    # MazeEnv(config=..., keyword=X) keyword config'i ezer
    base_cfg = EnvConfig(maze_size=4, view_radius=1, max_steps=15, obs_mode="full")
    env = MazeEnv(
        config=base_cfg,
        maze_size=6,
        view_radius=2,
        max_steps=40,
        obs_mode="simple",
    )
    _, info = env.reset(seed=0)
    rs = env.get_render_state()
    assert rs["grid"].shape == (2 * 6 + 1, 2 * 6 + 1)
    assert rs["view_radius"] == 2
    assert env.max_steps == 40
    assert info["max_steps"] == 40
    assert env.observation_space == gym.spaces.Discrete(11520)


# Regresyon: simple modda çıkışa ulaşılan adım çökmemeli (ajan çıkışın üstünde, yön 0)
@pytest.mark.parametrize("view_radius", [1, 2])
@pytest.mark.parametrize("seed", [0, 5, 10000])
def test_simple_observation_until_exit(view_radius: int, seed: int) -> None:
    env = MazeEnv(maze_size=6, view_radius=view_radius, obs_mode="simple")
    obs, _ = env.reset(seed=seed)
    rs = env.get_render_state()
    path = bfs_shortest_path_actions(rs["grid"], rs["start"], rs["exit"])
    for i, a in enumerate(path):
        obs, _, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert decode_simple_state(obs) == expected_simple_decoded(env.get_render_state())
        assert term == (i == len(path) - 1)
    assert info["success"] and decode_simple_state(obs)["exit_dir"] == 0
