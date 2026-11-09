"""probe_project.py: autopsy of forced-project games for a given card.

Plays greedy mirror games where seat 0 is FORCED to buy the target card
(project-mode rules: turns<=10, <=3 copies, when affordable). Logs:
  * buys forced, plays of the card, and -- for trashers -- exactly WHAT
    got trashed (the sub-decision quality question)
  * seat-0 win rate vs paired same-seed games with NO forcing
If the policy plays the card and trashes junk -> ownership data is good
and the card is just below the detection floor. If it never plays it, or
trashes treasures -> forced games are generating honest negative
evidence and the injection cannot teach this card.

  python probe_project.py runs/v9/latest.pt --card Remake --n 60
"""
import sys, argparse
import numpy as np, torch
sys.path.insert(0, '.')
from dominion.engine2 import (Game, N_CARDS, NAMES, PLAYABLE, PLAY_BASE,
                              BUY_BASE, TRASH_BASE)
from train2.elo4 import _load_net, GreedyNetBot

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="runs/v9/latest.pt")
ap.add_argument("--card", default="Remake")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--copies", type=int, default=3,
                help="max forced copies per game (Chapel wants 1!)")
ap.add_argument("--require", default="",
                help="comma-separated card names that MUST be in supply")
ap.add_argument("--exclude", default="",
                help="comma-separated card names that must NOT be in supply")
A = ap.parse_args()
REQ = [NAMES.index(x) for x in A.require.split(",") if x.strip()]
EXC = [NAMES.index(x) for x in A.exclude.split(",") if x.strip()]
net = _load_net(A.ckpt)
CARD = NAMES.index(A.card)
PLAY_A = PLAY_BASE + PLAYABLE.index(CARD) if CARD in PLAYABLE else None

def run_game(seed, force):
    g = Game(seed=seed)            # random kingdom; skip if card absent
    if not g.in_supply[CARD]:
        return None
    if any(not g.in_supply[c] for c in REQ) \
            or any(g.in_supply[c] for c in EXC):
        return None
    bot = GreedyNetBot(net)
    bought = played = 0
    trashed = []
    steps = 0
    while not g.done and steps < 6000:
        cur = g.current
        a = None
        if (force and cur == 0 and g.phase == 1 and g.pending == 0
                and g.players[0].turns <= 10 and bought < A.copies
                and g.legal_actions()[BUY_BASE + CARD]):
            a = BUY_BASE + CARD
            bought += 1
        if a is None:
            a = bot.act(g)
        if cur == 0:
            if PLAY_A is not None and a == PLAY_A:
                played += 1
            if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
                trashed.append(a - TRASH_BASE)
        g.step(a)
        steps += 1
    win = 1 if g.winner == 0 else 0
    return win, bought, played, trashed

wins_f = wins_c = nf = nc = tot_b = tot_p = 0
all_trash = []
seed = 41_000_000
while min(nf, nc) < A.n:
    seed += 1
    rf = run_game(seed, force=True)
    if rf is None:
        continue
    rc = run_game(seed, force=False)
    w, b, p, t = rf
    wins_f += w; nf += 1; tot_b += b; tot_p += p; all_trash += t
    wins_c += rc[0]; nc += 1

cond = ""
if A.require: cond += f" require[{A.require}]"
if A.exclude: cond += f" exclude[{A.exclude}]"
print(f"PROJECT AUTOPSY: {A.card}{cond}  ({A.ckpt}, {nf} paired games)")
print(f"  forced arm : seat0 wins {100*wins_f/nf:.0f}%  | "
      f"{A.card} bought {tot_b/nf:.2f}/gm, PLAYED {tot_p/nf:.2f}/gm")
print(f"  control arm: seat0 wins {100*wins_c/nc:.0f}%   (same seeds, "
      f"no forcing)")
if tot_b:
    print(f"  play rate per copy owned: {tot_p/max(tot_b,1):.2f}")
if all_trash:
    from collections import Counter
    cnt = Counter(all_trash)
    print("  TRASHED (what its sub-decisions chose):")
    for c, k in cnt.most_common():
        print(f"    {NAMES[c]:12s} x{k}")
else:
    print("  TRASHED: nothing -- the card was never used to trash")
print("\nREAD: plays~0 -> rots in hand (play-prior dead).  Trashes "
      "Copper/Estate/Curse\n-> sub-decisions fine, card just below "
      "detection floor.  Trashes treasures\n-> forced games are "
      "poisoned evidence.  Win% forced vs control = net effect.")