"""Elo harness: continuous random-pairing games on random kingdoms ->
results CSV -> anchored Bradley-Terry fit (Random=0, BigMoney=1000).

  python -m train2.elo --seconds 100        # play games, append results
  python -m train2.elo --fit                # fit + print ladder
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from dominion.engine2 import Game
from dominion.bots2 import (RandomBot, BigMoneyBot, BMSmithyBot,
                            GardensRushBot, ChapelMoneyBot)

RESULTS = "runs/elo_results.csv"


def _load_net(path):
    from train2.ppo3 import PolicyNetV3
    net = PolicyNetV3()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def contestants(run_dir=None, sims=(60, 150)):
    """name -> (factory, schedule_weight).

    Anchors are the scripted bots. If run_dir is given, every ckpt_*.pt in it
    is enrolled as raw + MCTS contestants so the LIVE lineage gets rated.
    Falls back to the env var ELO_RUN_DIR.
    """
    pool = {
        "Random": (lambda: RandomBot(seed=random.randrange(9999)), 1.0),
        "BigMoney": (BigMoneyBot, 1.0),
        "BM+Smithy": (BMSmithyBot, 1.0),
        "Rush": (GardensRushBot, 1.0),
        "ChapelMoney": (ChapelMoneyBot, 1.0),
    }
    run_dir = run_dir or os.environ.get("ELO_RUN_DIR")
    if run_dir and os.path.isdir(run_dir):
        import glob
        from train2.ppo3 import NetBot3
        from train2.mcts3 import MCTSBot3
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
        latest = os.path.join(run_dir, "latest.pt")
        if os.path.exists(latest):
            ckpts.append(latest)
        # keep the field manageable: anchor bots + up to ~6 checkpoints
        pick = ckpts[:: max(1, len(ckpts) // 6)] if len(ckpts) > 6 else ckpts
        for p in pick:
            tag = os.path.splitext(os.path.basename(p))[0].replace("ckpt_", "")
            net = _load_net(p)
            pool[f"{tag}-raw"] = (
                lambda n=net: NetBot3(n, greedy=True), 1.0)
            for sm in sims:
                pool[f"{tag}-mcts{sm}"] = (
                    lambda n=net, sm=sm: MCTSBot3(
                        n, n_sims=sm, seed=random.randrange(9999),
                        root_noise=False), 0.3 if sm <= 100 else 0.15)
    return pool


def play_batch(seconds: float, seed0: int):
    pool = contestants()
    names = list(pool)
    weights = [pool[n][1] for n in names]
    os.makedirs("runs", exist_ok=True)
    new = not os.path.exists(RESULTS)
    rng = random.Random(seed0)
    t0 = time.time()
    n = 0
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["a", "b", "result_a", "seed"])
        while time.time() - t0 < seconds:
            a, b = rng.choices(names, weights=weights, k=2)
            if a == b:
                continue
            seed = seed0 + n
            g = Game(seed=seed)
            bots = [pool[a][0](), pool[b][0]()]
            steps = 0
            while not g.done:
                g.step(bots[g.current].act(g))
                steps += 1
                if steps > 6000:
                    break
            res = 0.5 if g.winner is None else (1.0 if g.winner == 0 else 0.0)
            w.writerow([a, b, res, seed]); f.flush()
            n += 1
    print(f"played {n} games in {time.time()-t0:.0f}s "
          f"({sum(1 for _ in open(RESULTS))-1} total on file)")


def fit_elo():
    raw = [r for r in csv.reader(open(RESULTS))
           if r and r[0] != "a"]               # header-agnostic
    rows = [{"a": r[0], "b": r[1], "result_a": r[2]} for r in raw]
    names = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    idx = {n: i for i, n in enumerate(names)}
    R = np.full(len(names), 800.0)
    ANCH = {"Random": 0.0, "BigMoney": 1000.0}
    for n, v in ANCH.items():
        if n in idx:
            R[idx[n]] = v
    A = np.array([idx[r["a"]] for r in rows])
    B = np.array([idx[r["b"]] for r in rows])
    S = np.array([float(r["result_a"]) for r in rows])
    free = np.array([n not in ANCH for n in names])
    for _ in range(4000):                       # gradient ascent on BT loglik
        p = 1.0 / (1.0 + 10 ** ((R[B] - R[A]) / 400.0))
        gerr = S - p
        grad = np.zeros_like(R)
        np.add.at(grad, A, gerr)
        np.add.at(grad, B, -gerr)
        R[free] += 8.0 * grad[free] / max(len(rows) / len(names), 1)
    counts = np.zeros(len(names), int)
    np.add.at(counts, A, 1)
    np.add.at(counts, B, 1)
    print(f"\nELO LADDER  ({len(rows)} games; anchors Random=0, BigMoney=1000)")
    for i in np.argsort(-R):
        se = 400 / max(np.sqrt(counts[i]), 1)   # rough rating error
        print(f"  {names[i]:14s} {R[i]:7.0f}  ±{se:3.0f}  ({counts[i]} games)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.seconds:
        play_batch(a.seconds, seed0=int(time.time()) % 10**6 * 100)
    if a.fit:
        fit_elo()