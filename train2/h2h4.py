"""h2h4: clean head-to-head watcher -- each new checkpoint vs a fixed target.

Training-time win rates are polluted by forced exploration (the learner's
seat gets curriculum/project buys, the league opponent never does). This
plays every new ckpt_*.pt in a run dir against a frozen target (e.g. v10)
with NO forcing: both greedy, fixed seeds shared by every checkpoint,
each board played twice with seats swapped. Appends to <run>/h2h.csv,
including voluntary buys of each trasher (Chapel/Upgrade/Remake) per game
on boards where that card is available.

Read-only instrumentation: never touches training, never selects.

  nohup nice -n 15 python -m train2.h2h4 --run-dir runs/x1 \
      --target runs/v10/ckpt_1781177636.pt > h2h_x1.out 2>&1 &
  python -m train2.h2h4 --run-dir runs/x1 --target ... --cycles 1   # once
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch

import random

from dominion.engine2 import (Game, CHAPEL, UPGRADE, REMAKE, NAMES, BUY_BASE,
                              TRASH_BASE, N_CARDS, COST, KINGDOM_POOL)
from train2.elo4 import _load_net, GreedyNetBot
from train2.selfplay4 import COMMIT_LAST_TURN, _commit_onehot

SEED0 = 44_000_000          # fixed -> identical boards for every checkpoint
TRACKED = (CHAPEL, UPGRADE, REMAKE)   # voluntary buys of these are logged


class CommitBot:
    """Greedy bot playing a commitment exactly as self-play enforces it: told
    the card via the commit input, and forced to buy one copy by turn 4."""

    def __init__(self, net, card):
        self.net, self.card = net, card
        self.flag = _commit_onehot([card])

    @torch.no_grad()
    def act(self, g):
        p, c = g.players[g.current], self.card
        legal = g.legal_actions()
        if (g.pending == 0 and g.phase == 1 and p.turns <= COMMIT_LAST_TURN
                and p.deck_composition()[c] == 0 and legal[BUY_BASE + c]
                and not (p.turns == 0 and g.coins > COST[c] + 1)):
            return BUY_BASE + c
        obs = torch.tensor(g.observe_v3()[None])
        logits, _ = self.net(obs, torch.tensor(np.asarray(legal)[None]),
                             self.flag)
        return int(logits.argmax(-1))


def _board(seed, card):
    """Fixed board per seed; `card` (if any) is forced into the kingdom."""
    if card is None:
        return Game(seed=seed)
    kingdom = random.Random(seed).sample(KINGDOM_POOL, 10)
    if card not in kingdom:
        kingdom[-1] = card
    return Game(seed=seed, kingdom=kingdom)


def head_to_head(net, target, n_boards, seed0=SEED0, commit=None):
    me = GreedyNetBot(net) if commit is None else CommitBot(net, commit)
    opp = GreedyNetBot(target)
    res, trashes = [], 0
    buys = {c: 0 for c in TRACKED}
    boards = {c: 0 for c in TRACKED}
    for i in range(n_boards):
        for seat in (0, 1):
            g = _board(seed0 + i, commit)
            for c in TRACKED:
                boards[c] += g.in_supply[c]
            bots = [me, opp] if seat == 0 else [opp, me]
            steps = 0
            while not g.done and steps < 6000:
                cur = g.current
                a = bots[cur].act(g)
                if cur == seat:
                    trashes += TRASH_BASE <= a < TRASH_BASE + N_CARDS
                    if a - BUY_BASE in buys:
                        buys[a - BUY_BASE] += 1
                g.step(a)
                steps += 1
            if g.done:
                w = g.winner
            else:
                s0, s1 = g.score(0), g.score(1)
                w = None if s0 == s1 else (0 if s0 > s1 else 1)
            res.append(0.5 if w is None else float(w == seat))
    n = len(res)
    r = np.array(res)
    per_board = {c: buys[c] / max(boards[c], 1) for c in TRACKED}
    return r.mean(), r.std() / np.sqrt(n), trashes / n, per_board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--boards", type=int, default=100,
                    help="boards per checkpoint (x2 games, seats swapped)")
    ap.add_argument("--seed0", type=int, default=SEED0,
                    help="first board seed. A fixed set of boards can favour "
                         "one agent by a few points (100 boards did for x2), "
                         "so use enough boards (300+) and keep the set fixed "
                         "within a run")
    ap.add_argument("--commit-card", default="",
                    help="DIAGNOSTIC: checkpoint seat is committed to this "
                         "card (told + forced buy), and it is on every board")
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("--cycles", type=int, default=0, help="0 = forever")
    a = ap.parse_args()
    torch.set_num_threads(1)

    commit = NAMES.index(a.commit_card) if a.commit_card else None
    out = os.path.join(a.run_dir, "h2h.csv" if commit is None else
                       f"h2h_{a.commit_card.lower()}.csv")
    header = (["checkpoint", "games", "winrate", "se", "trashes_per_game"]
              + [f"{NAMES[c].lower()}_buys_per_{NAMES[c].lower()}_game"
                 for c in TRACKED])
    if os.path.exists(out) and next(csv.reader(open(out))) != header:
        os.replace(out, out.replace(".csv", "_old.csv"))   # older format
    if not os.path.exists(out):
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(header)
    seen = {r[0] for r in csv.reader(open(out))} - {"checkpoint"}
    target = _load_net(a.target)
    cycle = 0
    while True:
        cycle += 1
        for p in sorted(glob.glob(os.path.join(a.run_dir, "ckpt_*.pt"))):
            tag = os.path.basename(p)[5:-3]
            if tag in seen:
                continue
            t0 = time.time()
            wr, se, tpg, pb = head_to_head(_load_net(p), target, a.boards,
                                           a.seed0, commit)
            with open(out, "a", newline="") as f:
                csv.writer(f).writerow([tag, 2 * a.boards, f"{wr:.3f}",
                                        f"{se:.3f}", f"{tpg:.2f}"]
                                       + [f"{pb[c]:.2f}" for c in TRACKED])
            seen.add(tag)
            buys = " ".join(f"{NAMES[c]} {pb[c]:.2f}" for c in TRACKED)
            print(f"[{time.strftime('%H:%M:%S')}] ckpt {tag}: vs target "
                  f"{wr:.1%} +/- {se:.1%} | trash/gm {tpg:.2f} | buys per "
                  f"game where available: {buys}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if a.cycles and cycle >= a.cycles:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
