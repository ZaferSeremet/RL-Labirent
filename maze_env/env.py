"""Kısmi gözlemli labirent ortamı (Gymnasium arayüzü).

Ajan labirentin tamamını, kendi koordinatını ve çıkışın yerini bilmez.
Her adımda yalnız çevresindeki pencereyi, penceredeki karelere daha önce
girip girmediğini ve bir önceki eylemini görür. Ayrıntılar: maze_env/SPEC.md
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .generator import WALL, generate_maze, pick_start_and_exit
from .rng import StableRandom

# Eylemler
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
NO_ACTION = 4  # bölüm başında "son eylem" değeri
ACTION_NAMES = ("yukari", "sag", "asagi", "sol")
_ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Pencere içindeki kare türleri
TILE_FREE, TILE_WALL, TILE_EXIT = 0, 1, 2

# "simple" gözlem: 4 komşu x 4 kod, 9 çıkış yönü, 5 son eylem
NEIGHBOR_CODES = 4  # 0 boş-girilmedi, 1 boş-girildi, 2 duvar, 3 çıkış
EXIT_DIRS = 9  # 0 görünmüyor, 1..8: K, KD, D, GD, G, GB, B, KB
SIMPLE_N_STATES = NEIGHBOR_CODES**4 * EXIT_DIRS * 5  # 11520

_EXIT_DIR_INDEX = {
    (-1, 0): 1, (-1, 1): 2, (0, 1): 3, (1, 1): 4,
    (1, 0): 5, (1, -1): 6, (0, -1): 7, (-1, -1): 8,
}


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or EnvConfig()
        if overrides:
            cfg = EnvConfig.from_dict({**cfg.to_dict(), **overrides})
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {self.metadata['render_modes']} içinden olmalı")
        self.config = cfg
        self.render_mode = render_mode

        n, r = cfg.maze_size, cfg.view_radius
        self.window_size = 2 * r + 1
        self.free_tile_count = 2 * n * n - 1  # döngüsüz labirent: n² hücre + n²-1 geçit
        self.max_steps = cfg.max_steps if cfg.max_steps is not None else 3 * self.free_tile_count

        self.action_space = spaces.Discrete(4)
        if cfg.obs_mode == "full":
            k2 = self.window_size**2
            self.observation_space = spaces.Box(0.0, 1.0, shape=(3 * k2 + 5,), dtype=np.float32)
        else:
            self.observation_space = spaces.Discrete(SIMPLE_N_STATES)

        self._seed_source = StableRandom(None)
        self._grid: np.ndarray | None = None
        self._done = False

    # ------------------------------------------------------------------ Gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_source = StableRandom(seed)
            episode_seed = int(seed)
        else:
            episode_seed = self._seed_source.seed_int()

        n, r = self.config.maze_size, self.config.view_radius
        rng = StableRandom(episode_seed)
        grid = generate_maze(n, rng)
        start, exit_pos, shortest = pick_start_and_exit(grid, n, rng)

        self._episode_seed = episode_seed
        self._grid = grid
        self._exit = exit_pos
        self._pos = start
        self._start = start
        self._shortest = shortest

        # Pencereyi dilimle almak için r kadar duvarla çevrilmiş kopyalar
        tiles = np.pad(grid.astype(np.int8), r, constant_values=TILE_WALL)
        tiles[exit_pos[0] + r, exit_pos[1] + r] = TILE_EXIT
        self._tiles_padded = tiles
        self._visited_padded = np.zeros(tiles.shape, dtype=np.uint8)
        self._visited_padded[start[0] + r, start[1] + r] = 1

        self._steps = 0
        self._wall_hits = 0
        self._last_action = NO_ACTION
        self._done = False
        return self._observe(), self._info(success=False)

    def step(self, action):
        if self._grid is None:
            raise RuntimeError("Önce reset() çağrılmalı")
        if self._done:
            raise RuntimeError("Bölüm bitti, reset() çağrılmalı")
        action = int(action)
        if not 0 <= action < 4:
            raise ValueError(f"Geçersiz eylem: {action} (0-3 olmalı)")

        rw = self.config.rewards
        r = self.config.view_radius
        dy, dx = _ACTION_DELTAS[action]
        ny, nx = self._pos[0] + dy, self._pos[1] + dx
        reward = rw.step

        h, w = self._grid.shape
        if not (0 <= ny < h and 0 <= nx < w) or self._grid[ny, nx] == WALL:
            self._wall_hits += 1
            reward += rw.wall
        else:
            self._pos = (ny, nx)
            if not self._visited_padded[ny + r, nx + r]:
                self._visited_padded[ny + r, nx + r] = 1
                reward += rw.new_tile

        self._last_action = action
        self._steps += 1

        terminated = self._pos == self._exit
        if terminated:
            reward += rw.exit
        truncated = (not terminated) and self._steps >= self.max_steps
        if truncated:
            reward += rw.timeout
        self._done = terminated or truncated

        return self._observe(), float(reward), terminated, truncated, self._info(success=terminated)

    def render(self):
        if self.render_mode == "ansi":
            return self.render_text()
        return None

    # ------------------------------------------------------------ gözlemler

    def observe_dict(self) -> dict[str, Any]:
        """Ajanın gördüklerinin yapılandırılmış hali (hata ayıklama ve arayüz için).

        tiles   : (k, k) int8, 0 boş, 1 duvar, 2 çıkış; ajan merkezde
        visited : (k, k) uint8, 1 = bu kareye daha önce girildi
        last_action : 0-3, bölüm başında 4
        """
        tiles, visited = self._window()
        return {"tiles": tiles.copy(), "visited": visited.copy(), "last_action": self._last_action}

    def _window(self) -> tuple[np.ndarray, np.ndarray]:
        k = self.window_size
        y, x = self._pos  # dolgulu dizide pencerenin sol üstü tam olarak (y, x)
        return self._tiles_padded[y : y + k, x : x + k], self._visited_padded[y : y + k, x : x + k]

    def _observe(self):
        if self.config.obs_mode == "full":
            return self._observe_full()
        return self._observe_simple()

    def _observe_full(self) -> np.ndarray:
        tiles, visited = self._window()
        last = np.zeros(5, dtype=np.float32)
        last[self._last_action] = 1.0
        return np.concatenate(
            [
                (tiles == TILE_WALL).ravel(),
                (tiles == TILE_EXIT).ravel(),
                visited.ravel(),
                last,
            ]
        ).astype(np.float32)

    def _observe_simple(self) -> int:
        tiles, visited = self._window()
        r = self.config.view_radius
        state = 0
        for dy, dx in _ACTION_DELTAS:  # yukarı, sağ, aşağı, sol
            t = tiles[r + dy, r + dx]
            if t == TILE_WALL:
                code = 2
            elif t == TILE_EXIT:
                code = 3
            else:
                code = int(visited[r + dy, r + dx])
            state = state * NEIGHBOR_CODES + code

        # Ajan çıkışın üstündeyse (yalnız bölümün son gözlemi) yön tanımsız, 0 verilir
        exit_dir = 0
        ey, ex = np.nonzero(tiles == TILE_EXIT)
        if len(ey):
            exit_dir = _EXIT_DIR_INDEX.get((int(np.sign(ey[0] - r)), int(np.sign(ex[0] - r))), 0)

        return (state * EXIT_DIRS + exit_dir) * 5 + self._last_action

    def _info(self, success: bool) -> dict[str, Any]:
        return {
            "success": success,
            "steps": self._steps,
            "wall_hits": self._wall_hits,
            "visited_tiles": int(self._visited_padded.sum()),
            "shortest_path": self._shortest,
            "max_steps": self.max_steps,
            "episode_seed": self._episode_seed,
        }

    # ------------------------------------------------- arayüz ve hata ayıklama

    def get_render_state(self) -> dict[str, Any]:
        """Arayüzün labirenti çizmesi için tam durum. AJANA VERİLMEZ."""
        r = self.config.view_radius
        h, w = self._grid.shape
        return {
            "grid": self._grid.copy(),
            "agent": self._pos,
            "exit": self._exit,
            "start": self._start,
            "visited": self._visited_padded[r : r + h, r : r + w].astype(bool),
            "view_radius": r,
            "steps": self._steps,
            "last_action": self._last_action,
        }

    def render_text(self) -> str:
        """# duvar, . girilmiş kare, boşluk girilmemiş kare, A ajan, E çıkış."""
        s = self.get_render_state()
        lines = []
        for y in range(s["grid"].shape[0]):
            row = []
            for x in range(s["grid"].shape[1]):
                if (y, x) == s["agent"]:
                    row.append("A")
                elif (y, x) == s["exit"]:
                    row.append("E")
                elif s["grid"][y, x] == WALL:
                    row.append("#")
                elif s["visited"][y, x]:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)


def decode_simple_state(state: int) -> dict[str, Any]:
    """"simple" gözlemi okunur hale çevirir (hata ayıklama ve test için)."""
    if not 0 <= state < SIMPLE_N_STATES:
        raise ValueError("Geçersiz simple durum")
    last_action = state % 5
    state //= 5
    exit_dir = state % EXIT_DIRS
    state //= EXIT_DIRS
    codes = []
    for _ in range(4):
        codes.append(state % NEIGHBOR_CODES)
        state //= NEIGHBOR_CODES
    up, right, down, left = reversed(codes)
    return {
        "neighbors": {"up": up, "right": right, "down": down, "left": left},
        "exit_dir": exit_dir,
        "last_action": last_action,
    }
