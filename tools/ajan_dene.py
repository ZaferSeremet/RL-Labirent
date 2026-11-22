"""Bir ajanı komut satırından eğit ve test labirentlerinde baseline'larla karşılaştır.

Arayüz gelene kadar ajan yazanların deneme aracı.

    .venv\\Scripts\\python tools\\ajan_dene.py --agent ornek --episodes 2000
    .venv\\Scripts\\python tools\\ajan_dene.py --list
    .venv\\Scripts\\python tools\\ajan_dene.py --agent sag_el          (baseline: eğitimsiz test)
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base import EnvSpec  # noqa: E402
from engine import discover, env_config_for, load_shared_config, run_episode, seeds_for  # noqa: E402
from maze_env import MazeEnv  # noqa: E402


def make_agent(cls, kind, shared, seed, size=None):
    overrides = {"maze_size": size} if size and kind == "egitim" else {}
    env = MazeEnv(env_config_for(kind, cls.obs_mode, shared, **overrides))
    return env, cls(EnvSpec.from_env(env), seed=seed)


def evaluate(cls, agent, kind, shared, seed):
    """Ajanı (yoksa yeni bir tane kurarak) test labirentlerinde greedy koşturur."""
    env = MazeEnv(env_config_for(kind, cls.obs_mode, shared))
    if agent is None:
        agent = cls(EnvSpec.from_env(env), seed=seed)
    results = [run_episode(env, agent, seed=s, train=False) for s in seeds_for(kind, shared)]
    ok = [r for r in results if r.success]
    return {
        "basari": len(ok) / len(results),
        "adim": statistics.mean(r.steps for r in ok) if ok else float("nan"),
        "verim": statistics.mean(r.efficiency for r in ok) if ok else float("nan"),
        "odul": statistics.mean(r.total_reward for r in results),
        "carpma": statistics.mean(r.wall_hits for r in results),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", help="ajanın name alanı")
    p.add_argument("--episodes", type=int, default=2000, help="eğitim bölümü sayısı")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=None, help="eğitim labirenti boyutu (varsayılan ortak ayar)")
    p.add_argument("--buyuk", action="store_true", help="büyük test labirentlerinde de test et")
    p.add_argument("--list", action="store_true", help="bulunan ajanları listele")
    args = p.parse_args()

    reg = discover()
    for mod, err in reg.errors.items():
        print(f"UYARI: {mod} yüklenemedi -> {err}")
    if args.list or not args.agent:
        print("Ajanlar:")
        for name, cls in sorted(reg.agents.items()):
            print(f"  {name:12s} obs={cls.obs_mode:6s} {'(baseline)' if cls.is_baseline else ''}")
        return

    shared = load_shared_config()
    cls = reg.get(args.agent)
    agent = None

    if not cls.is_baseline:
        env, agent = make_agent(cls, "egitim", shared, args.seed, args.size)
        env.reset(seed=shared["egitim"]["seed_baslangic"] + args.seed)  # eğitim labirent dizisini sabitle
        window = max(1, args.episodes // 20)
        buf = []
        t0 = time.time()
        print(f"Eğitim: {args.agent}, {args.episodes} bölüm, labirent {env.config.maze_size}x{env.config.maze_size}")
        print(f"{'bölüm':>7} {'başarı':>7} {'ort.ödül':>9} {'ort.adım':>9}   (son {window} bölüm)")
        for ep in range(1, args.episodes + 1):
            buf.append(run_episode(env, agent, train=True))
            if ep % window == 0 or ep == args.episodes:
                last = buf[-window:]
                print(
                    f"{ep:7d} {sum(r.success for r in last) / len(last):7.0%} "
                    f"{statistics.mean(r.total_reward for r in last):9.2f} "
                    f"{statistics.mean(r.steps for r in last):9.1f}"
                )
        print(f"Eğitim süresi: {time.time() - t0:.1f} sn")

    kinds = ["normal"] + (["buyuk"] if args.buyuk else [])
    rows = [(args.agent, cls, agent)]
    for b in reg.baselines():
        if b is not cls:
            rows.append((b.name, b, None))
    for kind in kinds:
        n = len(seeds_for(kind, shared))
        size = shared["test"][kind]["maze_size"]
        print(f"\nTest: {kind} ({n} labirent, {size}x{size}), greedy")
        print(f"{'yöntem':12s} {'başarı':>7} {'ort.adım*':>10} {'adım/enkısa*':>13} {'ort.ödül':>9} {'çarpma':>7}")
        for name, c, a in rows:
            m = evaluate(c, a, kind, shared, args.seed)
            print(f"{name:12s} {m['basari']:7.0%} {m['adim']:10.1f} {m['verim']:13.2f} {m['odul']:9.2f} {m['carpma']:7.1f}")
        print("* yalnız başarılı bölümler")


if __name__ == "__main__":
    main()
