"""probe_sims.py: does MORE SEARCH (not more training) unlock trashing?

Hypothesis under test: 'the agent needs more sims.' If true, the SAME
trained net with a much larger root search budget should start buying
Chapel / trashing curses on Witch-no-Charlatan boards at inference time.
If deep search still refuses, sims-at-feasible-scale is falsified: the
bottleneck is the value function / data, which search only amplifies.

  python probe_sims.py runs/v10/latest.pt --sims 64,300 --n 30
"""
import sys, argparse
import numpy as np
sys.path.insert(0, '.')
from dominion.engine2 import (Game, NAMES, BUY_BASE, TRASH_BASE, N_CARDS)
from train2.elo4 import _load_net, GreedyNetBot
from train2.mcts3 import MCTSBot3

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="runs/v10/latest.pt")
ap.add_argument("--sims", default="64,300")
ap.add_argument("--n", type=int, default=30)
A = ap.parse_args()
net = _load_net(A.ckpt)
CHAPEL = NAMES.index("Chapel")
WITCH = NAMES.index("Witch")
CHARLATAN = NAMES.index("Charlatan")

def boards(n):
    out, seed = [], 95_000_000
    while len(out) < n:
        seed += 1
        g = Game(seed=seed)
        if g.in_supply[CHAPEL] and g.in_supply[WITCH] \
                and not g.in_supply[CHARLATAN]:
            out.append(seed)
    return out

def run(mk_bot, seeds, tag):
    chap_buys = trashes = curse_trashes = 0
    for sd in seeds:
        g = Game(seed=sd)
        bots = {0: mk_bot(), 1: mk_bot()}
        steps = 0
        while not g.done and steps < 4000:
            a = bots[g.current].act(g)
            if BUY_BASE <= a < BUY_BASE + N_CARDS \
                    and a - BUY_BASE == CHAPEL:
                chap_buys += 1
            if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
                trashes += 1
                if a - TRASH_BASE == NAMES.index("Curse"):
                    curse_trashes += 1
            g.step(a); steps += 1
    n = len(seeds)
    print(f"  {tag:>14s}: Chapel buys {chap_buys/n:.2f}/gm | "
          f"trashes {trashes/n:.2f}/gm (curses {curse_trashes/n:.2f})")

seeds = boards(A.n)
print(f"SIMS PROBE on {A.n} Witch-no-Charlatan boards ({A.ckpt})")
run(lambda: GreedyNetBot(net), seeds, "greedy (0 sims)")
for sm in (int(x) for x in A.sims.split(",")):
    run(lambda s=sm: MCTSBot3(net, n_sims=s, seed=11), seeds, f"search {sm}")
print("\nREAD: rows ~equal at 0/64/300 sims -> search budget is NOT the "
      "binding\nconstraint; value/data is. Chapel+trashing appearing only "
      "at high sims ->\nyou were right: inference compute unlocks it.")
