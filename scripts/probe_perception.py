"""probe_perception.py: confound-free test of whether the net READS the
kingdom planes. Takes real observations, surgically swaps ONLY the
kingdom-related features (supply plane, in_supply plane, charlatan_rule
flag) between a curser board and a clean board, holding hand/deck/coins/
scores byte-identical. Compares raw pointer logits for always-legal cards
and the value head. Any delta = the net conditioning on the kingdom.

  python probe_perception.py runs/v9/latest.pt --n 40
"""
import sys, argparse
import numpy as np, torch
sys.path.insert(0, '.')
from dominion.engine2 import (Game, N_CARDS, NAMES, PILE_SIZE, GOLD, SILVER,
                              DUCHY, PROVINCE, BUY_BASE, WORKSHOP, GARDENS,
                              VILLAGE, SMITHY, LABORATORY, MARKET, FESTIVAL,
                              CARAVAN, MONUMENT, COLLECTION, WITCH,
                              CHARLATAN, CHAPEL, UPGRADE)
from train2.elo4 import _load_net, GreedyNetBot

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="runs/v9/latest.pt")
ap.add_argument("--n", type=int, default=40)
A = ap.parse_args()
net = _load_net(A.ckpt)

K = N_CARDS
CLEAN  = sorted([VILLAGE, SMITHY, LABORATORY, MARKET, FESTIVAL, CARAVAN,
                 MONUMENT, COLLECTION, WORKSHOP, GARDENS])
CURSER = sorted([VILLAGE, SMITHY, LABORATORY, MARKET, FESTIVAL, CARAVAN,
                 MONUMENT, COLLECTION, WITCH, CHARLATAN])

def swap_kingdom_planes(obs, kingdom, full_piles=10):
    """Return a copy of obs depicting `kingdom` in planes 0-1 + charlatan
    flag, leaving every other feature untouched."""
    o = obs.copy()
    base = {0,1,2,3,4,5,6}  # copper..province indices stay as-is
    for c in range(K):
        if c in base:
            continue
        if c in kingdom:
            o[c] = full_piles / max(PILE_SIZE.get(c, 1), 1)   # fresh pile
            o[K + c] = 1.0
        else:
            o[c] = 0.0
            o[K + c] = 0.0
    o[12 * K] = 1.0 if CHARLATAN in kingdom else 0.0   # charlatan_rule flag
    return o

@torch.no_grad()
def logits_for(obs):
    mask = np.zeros(106, bool)
    # compare on ALWAYS-legal buys only -> identical mask both sides
    for c in (GOLD, SILVER, DUCHY, PROVINCE):
        mask[BUY_BASE + c] = True
    lo, v = net(torch.tensor(obs[None]), torch.tensor(mask[None]))
    return lo[0].numpy(), float(torch.tanh(v[0]))

bot = GreedyNetBot(net)
deltas = {c: [] for c in (GOLD, SILVER, DUCHY, PROVINCE)}
vdel = []
got = 0
seed = 9_000_000
while got < A.n:
    seed += 1
    g = Game(seed=seed, kingdom=CLEAN)
    steps = 0
    while not g.done and steps < 120:
        if (g.phase == 1 and g.pending == 0 and g.coins >= 6
                and 3 <= g.players[g.current].turns <= 10):
            obs = g.observe_v3()
            lo_clean, v_clean = logits_for(swap_kingdom_planes(obs, CLEAN))
            lo_curse, v_curse = logits_for(swap_kingdom_planes(obs, CURSER))
            for c in deltas:
                deltas[c].append(lo_curse[BUY_BASE + c]
                                 - lo_clean[BUY_BASE + c])
            vdel.append(v_curse - v_clean)
            got += 1
            break
        g.step(bot.act(g))
        steps += 1

print(f"PERCEPTION PROBE  ({A.ckpt}, n={got} matched states, $6+ buy phase)")
print("logit(buy X | CURSER board) - logit(buy X | CLEAN board),")
print("with hand/deck/coins/scores byte-identical:\n")
for c in deltas:
    d = np.array(deltas[c])
    print(f"  {NAMES[c]:10s} mean Δlogit {d.mean():+7.3f}  "
          f"(sd {d.std():.3f}, |Δ|>0.1 in {100*(abs(d)>0.1).mean():.0f}% "
          f"of states)")
v = np.array(vdel)
print(f"\n  value head  mean Δ {v.mean():+7.4f}  (sd {v.std():.4f})")
print("\nREAD: Δ ~ 0 everywhere -> the net IGNORES the kingdom planes")
print("(blindness confirmed). Consistent nonzero Δ -> it reads the board;")
print("the sign/size tells you HOW it re-values Gold etc. under cursers.")
