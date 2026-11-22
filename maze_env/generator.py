"""Seed'li rastgele labirent üretimi (DFS / recursive backtracker).

Izgara gösterimi: `maze_size` x `maze_size` hücrelik labirent,
(2*maze_size+1) x (2*maze_size+1) karelik bir ızgaraya çizilir.
Hücreler tek indisli karelerdir: hücre (cy, cx) -> kare (2*cy+1, 2*cx+1).
İki hücre arasındaki kare duvar ya da geçittir. Dış çerçeve hep duvardır.

DFS döngüsüz ("mükemmel") labirent üretir: her iki hücre arasında tek yol vardır.
"""

from __future__ import annotations

import numpy as np

from .pathfinding import bfs_distance
from .rng import StableRandom

FREE = 0
WALL = 1

# Hücre komşuları: (dy, dx), sabit sıra (yukarı, sağ, aşağı, sol)
_CELL_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def cell_to_tile(cy: int, cx: int) -> tuple[int, int]:
    return 2 * cy + 1, 2 * cx + 1


def generate_maze(maze_size: int, rng: StableRandom) -> np.ndarray:
    """Döngüsüz labirent üretir. Dönüş: uint8 ızgara, 0 = boş, 1 = duvar."""
    n = maze_size
    grid = np.full((2 * n + 1, 2 * n + 1), WALL, dtype=np.uint8)
    visited = np.zeros((n, n), dtype=bool)

    start = (rng.below(n), rng.below(n))
    visited[start] = True
    grid[cell_to_tile(*start)] = FREE
    stack = [start]

    while stack:
        cy, cx = stack[-1]
        options = [
            (cy + dy, cx + dx)
            for dy, dx in _CELL_DIRS
            if 0 <= cy + dy < n and 0 <= cx + dx < n and not visited[cy + dy, cx + dx]
        ]
        if not options:
            stack.pop()
            continue
        ny, nx = rng.choice(options)
        visited[ny, nx] = True
        grid[cell_to_tile(ny, nx)] = FREE
        grid[cy + ny + 1, cx + nx + 1] = FREE  # aradaki duvarı kaldır
        stack.append((ny, nx))

    return grid


def pick_start_and_exit(
    grid: np.ndarray, maze_size: int, rng: StableRandom, tries: int = 50
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Ajanın başlangıcı ve çıkış için iki farklı hücre seçer.

    Çok kısa bölümler olmasın diye aralarındaki en kısa yol en az
    2 * maze_size kare olmaya çalışılır; `tries` denemede tutmazsa
    denemelerin en uzağı alınır.
    Dönüş: (başlangıç karesi, çıkış karesi, en kısa yol uzunluğu)
    """
    n = maze_size
    min_dist = 2 * n
    best: tuple[tuple[int, int], tuple[int, int], int] | None = None
    for _ in range(tries):
        a = cell_to_tile(rng.below(n), rng.below(n))
        b = cell_to_tile(rng.below(n), rng.below(n))
        if a == b:
            continue
        d = bfs_distance(grid, a, b)
        if best is None or d > best[2]:
            best = (a, b, d)
        if d >= min_dist:
            return a, b, d
    if best is None:  # n çok küçükse olabilir
        a = cell_to_tile(0, 0)
        b = cell_to_tile(n - 1, n - 1)
        best = (a, b, bfs_distance(grid, a, b))
    return best
