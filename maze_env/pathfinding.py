"""Izgarada en kısa yol (BFS). Ajan bunu bilmez; verimlilik metriği için ortam kullanır."""

from __future__ import annotations

from collections import deque

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def bfs_distance(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """start'tan goal'e en az adım sayısı. Yol yoksa -1. grid: 0 = boş, 1 = duvar."""
    if start == goal:
        return 0
    h, w = grid.shape
    dist = {start: 0}
    q = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in _DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0 and (ny, nx) not in dist:
                if (ny, nx) == goal:
                    return dist[(y, x)] + 1
                dist[(ny, nx)] = dist[(y, x)] + 1
                q.append((ny, nx))
    return -1


def distance_map(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Her boş kareden goal'e en az adım sayısı (int32). Duvar ve ulaşılamayan kare: -1."""
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[goal] = 0
    q = deque([goal])
    while q:
        y, x = q.popleft()
        for dy, dx in _DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0 and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist
