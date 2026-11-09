"""elo4: within-lineage Elo harness for the v4 stack.

Continuous random-pairing games on random kingdoms -> results CSV ->
anchored Bradley-Terry fit WITH a first-player-advantage term.

This is the purity-clean SELECTION instrument: checkpoints are rated by
play against each other (and two knowledge-free-ish anchors for scale).
Scripted-bot win rates in the training log remain read-only thermometers.

Fixes vs elo.py:
  * loads BOTH PolicyNetV3 and PolicyNetV4 checkpoints (auto-detected
    from the state-dict keys), so mixed-era run dirs work
  * MCTSBot3 root_noise crash fixed (requires the patched mcts3.py)
  * Bradley-Terry fit includes a seat-advantage parameter gamma, so
    Dominion's real first-player edge stops polluting ratings
  * non-terminating games resolve like training: by VP if decisive;
    an exact tie is recorded as 0.5 (pairwise BT cannot express
    "both lose", so this is the closest consistent encoding)

  python -m train2.elo4 --seconds 300 --run-dir runs/v9   # play games
  python -m train2.elo4 --fit                             # print ladder
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import torch

from dominion.engine2 import Game
from dominion.bots2 import (RandomBot, BigMoneyBot, BMSmithyBot,
                            GardensRushBot, ChapelMoneyBot)

RESULTS = "runs/elo4_results.csv"

_NET_CACHE = {}


def _load_net(path):
    """Architecture auto-detection: V4 state dicts contain card_emb.*"""
    if path in _NET_CACHE:
        return _NET_CACHE[path]
    sd = torch.load(path, weights_only=True)
    if any(k.startswith("card_emb") for k in sd):
        from train2.net4 import PolicyNetV4 as Net
    else:
        from train2.ppo3 import PolicyNetV3 as Net
    net = Net()
    net.load_state_dict(sd)
    net.eval()
    _NET_CACHE[path] = net
    return net


class GreedyNetBot:
    """Greedy policy bot; works for any net with (obs, mask)->(logits, v)."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def act(self, g):
        obs = torch.tensor(g.observe_v3()[None])
        mask = torch.tensor(np.asarray(g.legal_actions())[None])
        logits, _ = self.net(obs, mask)
        return int(logits.argmax(-1))


def contestants(run_dir=None, sims=(60, 150), include_latest=True):
    pool = {
        "Random": (lambda: RandomBot(seed=random.randrange(9999)), 1.0),
        "BigMoney": (BigMoneyBot, 1.0),
        "BM+Smithy": (BMSmithyBot, 1.0),
        "Rush": (GardensRushBot, 1.0),
        "ChapelMoney": (ChapelMoneyBot, 1.0),
    }
    run_dir = run_dir or os.environ.get("ELO_RUN_DIR")
    if run_dir and os.path.isdir(run_dir):
        from train2.mcts3 import MCTSBot3   # patched: accepts root_noise
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
        latest = os.path.join(run_dir, "latest.pt")
        if include_latest and os.path.exists(latest):
            ckpts.append(latest)
        pick = ckpts[:: max(1, len(ckpts) // 6)] if len(ckpts) > 6 else ckpts
        if ckpts and ckpts[-1] not in pick:
            pick.append(ckpts[-1])
        for p in pick:
            tag = os.path.splitext(os.path.basename(p))[0] \
                    .replace("ckpt_", "")
            net = _load_net(p)
            pool[f"{tag}-raw"] = (lambda n=net: GreedyNetBot(n), 1.0)
            for sm in sims:
                pool[f"{tag}-mcts{sm}"] = (
                    lambda n=net, sm=sm: MCTSBot3(
                        n, n_sims=sm, seed=random.randrange(9999),
                        root_noise=False),
                    0.3 if sm <= 100 else 0.15)
    return pool


def play_batch(seconds: float, seed0: int, run_dir=None, sims=(),
               focus=None, include_latest=True):
    pool = contestants(run_dir, sims=sims, include_latest=include_latest)
    names = list(pool)
    weights = [pool[n][1] for n in names]
    if focus is not None and focus not in pool:
        raise ValueError(f"focus contestant {focus!r} not in pool: {names}")
    others = [n for n in names if n != focus]
    oweights = [pool[n][1] for n in others]
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
            if focus is not None:
                opp = rng.choices(others, weights=oweights, k=1)[0]
                # randomize the focal player's seat so its rating doesn't
                # ride on the first-player-advantage estimate
                a, b = (focus, opp) if rng.random() < 0.5 else (opp, focus)
            else:
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
            if g.done:
                res = 0.5 if g.winner is None else \
                    (1.0 if g.winner == 0 else 0.0)
            else:
                # non-terminating: resolve like training -- by VP
                s0, s1 = g.score(0), g.score(1)
                res = 1.0 if s0 > s1 else (0.0 if s1 > s0 else 0.5)
            w.writerow([a, b, res, seed])
            f.flush()
            n += 1
    print(f"played {n} games in {time.time()-t0:.0f}s "
          f"({sum(1 for _ in open(RESULTS))-1} total on file)")


def fit_elo():
    raw = [r for r in csv.reader(open(RESULTS)) if r and r[0] != "a"]
    rows = [{"a": r[0], "b": r[1], "s": float(r[2])} for r in raw]
    names = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    idx = {n: i for i, n in enumerate(names)}
    R = np.full(len(names), 800.0)
    ANCH = {"Random": 0.0, "BigMoney": 1000.0}
    for n, v in ANCH.items():
        if n in idx:
            R[idx[n]] = v
    A = np.array([idx[r["a"]] for r in rows])
    B = np.array([idx[r["b"]] for r in rows])
    S = np.array([r["s"] for r in rows])
    free = np.array([n not in ANCH for n in names])
    gamma = 0.0          # first-player (seat-0) advantage, Elo points
    lr = 8.0 / max(len(rows) / len(names), 1)
    for _ in range(4000):
        p = 1.0 / (1.0 + 10 ** ((R[B] - R[A] - gamma) / 400.0))
        gerr = S - p
        grad = np.zeros_like(R)
        np.add.at(grad, A, gerr)
        np.add.at(grad, B, -gerr)
        R[free] += lr * grad[free]
        gamma += 4.0 * gerr.mean()
    counts = np.zeros(len(names), int)
    np.add.at(counts, A, 1)
    np.add.at(counts, B, 1)
    print(f"\nELO LADDER  ({len(rows)} games; anchors Random=0, "
          f"BigMoney=1000; first-player adv = {gamma:+.0f} Elo)")
    for i in np.argsort(-R):
        se = 400 / max(np.sqrt(counts[i]), 1)
        print(f"  {names[i]:18s} {R[i]:7.0f}  ±{se:3.0f}  "
              f"({counts[i]} games)")
    return {n: (R[idx[n]], 400 / max(np.sqrt(counts[idx[n]]), 1),
                int(counts[idx[n]])) for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--mcts", default="",
                    help="comma-separated sim budgets for search-"
                         "strengthened contestants, e.g. '60,150'. "
                         "Default: none (raw-only, ~10-60x more games/min; "
                         "MCTS forwards are unbatched and dominate cost)")
    a = ap.parse_args()
    sims = tuple(int(x) for x in a.mcts.split(",") if x.strip())
    if a.seconds:
        play_batch(a.seconds, seed0=int(time.time()) % 10**6 * 100,
                   run_dir=a.run_dir, sims=sims)
    if a.fit:
        fit_elo()