"""Labirenti terminalde klavyeyle oyna (ortamı elle denemek için).

Kullanım:
    .venv\\Scripts\\python tools\\oyna_terminal.py [--seed 123] [--size 8] [--radius 1]

Komutlar: w yukarı, d sağ, s aşağı, a sol (Enter ile), r yeni labirent, q çıkış.
Problemi elle hissetmek için (arayüzde klavyeyle oynama sekmesi yok).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maze_env import ACTION_NAMES, MazeEnv  # noqa: E402

KEYS = {"w": 0, "d": 1, "s": 2, "a": 3}
WINDOW_CHARS = {0: " ", 1: "#", 2: "E"}


def window_text(env: MazeEnv) -> str:
    d = env.observe_dict()
    r = env.config.view_radius
    rows = []
    for y, row in enumerate(d["tiles"]):
        chars = []
        for x, t in enumerate(row):
            if (y, x) == (r, r):
                chars.append("A")
            elif t == 0 and d["visited"][y, x]:
                chars.append(".")
            else:
                chars.append(WINDOW_CHARS[int(t)])
        rows.append("".join(chars))
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--radius", type=int, default=1)
    args = p.parse_args()

    env = MazeEnv(maze_size=args.size, view_radius=args.radius)
    _, info = env.reset(seed=args.seed)
    total = 0.0
    while True:
        print("\n" + env.render_text())
        print(f"\nAjanın gördüğü pencere (son eylem: "
              f"{ACTION_NAMES[env.observe_dict()['last_action']] if env.observe_dict()['last_action'] < 4 else '-'}):")
        print(window_text(env))
        print(f"adım {info['steps']}/{info['max_steps']}  çarpma {info['wall_hits']}  "
              f"toplam ödül {total:.2f}  en kısa yol {info['shortest_path']}  seed {info['episode_seed']}")
        cmd = input("w/a/s/d, r yeni, q çıkış > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "r":
            _, info = env.reset()
            total = 0.0
            continue
        if cmd not in KEYS:
            continue
        _, reward, term, trunc, info = env.step(KEYS[cmd])
        total += reward
        print(f"ödül {reward:+.2f}")
        if term or trunc:
            sonuc = "ÇIKIŞI BULDUN" if term else "SÜRE DOLDU"
            print(f"\n{sonuc}: {info['steps']} adım, toplam ödül {total:.2f}. Yeni labirent geliyor.")
            _, info = env.reset()
            total = 0.0


if __name__ == "__main__":
    main()
