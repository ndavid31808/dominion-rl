"""
Dominion agent diagnostics. Usage:
    python diagnose.py [checkpoint] [--n N]
Default checkpoint: runs/v9/latest.pt

Reports, in order:
  1. Overall buy table + total trashes (marginal)
  2. Quadrant split: junk/trash board types -> buys, trashes-by-card, pace
  3. Trasher analysis: Chapel/Remake/Upgrade buy / play / trash rates
  4. Opening analysis: per card, % on board, opens/game, %-when-affordable
  5. Curse race: pile-left, curses-eaten, cursers-bought on curser boards
  6. Card-combo affinity: which card PAIRS get bought together above chance
"""
import sys, torch, random, argparse
from collections import defaultdict, Counter
sys.path.insert(0, '.')
from dominion.engine2 import *
from train2.net4 import PolicyNetV4
from train2.eval4 import NetBot4

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="runs/v9/latest.pt")
ap.add_argument("--n", type=int, default=40, help="games per probe cell")
A = ap.parse_args()

net = PolicyNetV4()
net.load_state_dict(torch.load(A.ckpt, weights_only=True))
net.eval()
agent = NetBot4(net, greedy=True)

def gv(name):
    return globals().get(name, -999)
JUNKERS  = {gv(n) for n in ("WITCH", "CHARLATAN", "JESTER")} - {-999}
TRASHERS = {gv(n) for n in ("CHAPEL", "UPGRADE", "REMAKE",
                            "SENTRY", "PRIEST", "REMODEL")} - {-999}
BASE = {COPPER, SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE}


def play_game(seed, kingdom=None):
    """Play one self-play game; return per-seat acquired-card lists,
    trashes Counter (seat 0), turns, final supply."""
    g = Game(seed=seed, kingdom=kingdom)
    acq = {0: [], 1: []}; opn = {0: [], 1: []}; tr = Counter()
    plays = Counter()
    steps = 0
    while not g.done:
        cur = g.current
        a = agent.act(g)
        if BUY_BASE <= a < BUY_BASE + N_CARDS:
            c = a - BUY_BASE; acq[cur].append(c)
            if g.players[cur].turns < 2: opn[cur].append(c)
        elif GAIN_BASE <= a < GAIN_BASE + N_CARDS:
            acq[cur].append(a - GAIN_BASE)
        elif cur == 0 and TRASH_BASE <= a < TRASH_BASE + N_CARDS:
            tr[a - TRASH_BASE] += 1
        elif cur == 0 and PLAY_BASE <= a < PLAY_BASE + len(PLAYABLE):
            plays[PLAYABLE[a]] += 1
        g.step(a); steps += 1
        if steps > 6000: break
    return acq, opn, tr, plays, g


def boards(n, predicate=None, seed0=10**6):
    """Yield n games whose kingdom satisfies predicate(junk,trash)."""
    seed = seed0; got = 0
    while got < n:
        seed += 1
        k = random.Random(seed).sample(KINGDOM_POOL, 10)
        if predicate is not None:
            j = any(c in JUNKERS for c in k)
            t = any(c in TRASHERS for c in k)
            if not predicate(j, t):
                continue
        yield seed, k
        got += 1


print("=" * 64)
print("DIAGNOSTICS:", A.ckpt, f"(n={A.n}/cell)")
print("=" * 64)

# ---- 1. overall marginal buy table ----
present = defaultdict(int); acq = defaultdict(float); trsum = 0
for seed, _ in boards(A.n, seed0=2 * 10**6):
    a, o, tr, pl, g = play_game(seed)
    for c in range(N_CARDS):
        if g.in_supply[c]: present[c] += 1
    for c in a[0]: acq[c] += 1
    trsum += sum(tr.values())
print("\n[1] OVERALL buys/game-where-available:")
for c in sorted(acq, key=lambda c: -acq[c] / max(present[c], 1)):
    if present[c] and acq[c] / present[c] >= 0.1:
        print(f"    {NAMES[c]:18s}{acq[c]/present[c]:5.1f}")
print(f"    overall trashes/game: {trsum/A.n:.2f}")

# ---- 2. quadrant split ----
print("\n[2] BOARD-TYPE QUADRANTS:")
quads = [("junk, no trash", lambda j, t: j and not t),
         ("junk + trash",   lambda j, t: j and t),
         ("trash, no junk", lambda j, t: t and not j),
         ("clean",          lambda j, t: not j and not t)]
for label, pred in quads:
    ac = Counter(); tc = Counter(); turns = []
    for seed, k in boards(A.n, pred, seed0=3 * 10**6):
        a, o, tr, pl, g = play_game(seed, k)
        for c in a[0]: ac[NAMES[c]] += 1
        tc += Counter({NAMES[x]: v for x, v in tr.items()})
        turns.append(g.players[0].turns)
    tg = sum(tc.values()) / A.n
    print(f"  {label:16s} avg {sum(turns)/A.n:4.1f}t | trash/gm {tg:.2f} "
          f"-> {dict(tc.most_common(5)) or 'ZERO'}")
    print(f"  {'':16s} top buys: "
          f"{ {k: round(v/A.n,1) for k,v in ac.most_common(6)} }")

# ---- 3. trasher analysis ----
print("\n[3] TRASHERS (on boards where present):")
for tname in ("CHAPEL", "REMAKE", "UPGRADE", "SENTRY", "PRIEST", "REMODEL"):
    tc = gv(tname)
    if tc == -999 or tc not in KINGDOM_POOL: continue
    bought = played = trashes = 0; games = 0
    for seed, k in boards(A.n, lambda j, t: tc in [gv(tname)] or True,
                          seed0=4 * 10**6):
        if tc not in k: continue
        a, o, tr, pl, g = play_game(seed, k)
        bought += a[0].count(tc); played += pl[tc]
        trashes += sum(tr.values()); games += 1
    if games:
        print(f"    {NAMES[tc]:10s} (n={games:3d}): "
              f"bought {bought/games:.2f} | played {played/games:.2f} "
              f"| trash/gm {trashes/games:.2f}")

# ---- 4. opening analysis ----
print("\n[4] OPENING BUYS (turn 1-2):")
pres = defaultdict(int); opened = defaultdict(int)
aff_opp = defaultdict(int); aff_buy = defaultdict(int)
for seed, _ in boards(A.n * 3, seed0=5 * 10**6):
    g = Game(seed=seed)
    for c in range(N_CARDS):
        if g.in_supply[c]: pres[c] += 1
    steps = 0
    while not g.done:
        cur = g.current; a = agent.act(g)
        if cur == 0 and g.phase == 1 and g.players[0].turns < 2:
            if BUY_BASE <= a < BUY_BASE + N_CARDS:
                opened[a - BUY_BASE] += 1
            for c in range(N_CARDS):
                if (g.in_supply[c] and g.supply[c] > 0
                        and COST[c] <= g.coins and c != CURSE):
                    aff_opp[c] += 1
                    if BUY_BASE <= a < BUY_BASE + N_CARDS and a - BUY_BASE == c:
                        aff_buy[c] += 1
        g.step(a); steps += 1
        if steps > 6000: break
print(f"    {'card':16s} {'board%':>7s} {'opens/gm':>9s} {'when-afford':>12s}")
for c in sorted(range(N_CARDS), key=lambda c: -opened[c] / max(pres[c], 1)):
    if pres[c] == 0 or c in BASE: continue
    opg = opened[c] / max(pres[c], 1)
    waf = aff_buy[c] / aff_opp[c] if aff_opp[c] else 0
    if opg >= 0.02 or c in TRASHERS:
        print(f"    {NAMES[c]:16s} {pres[c]/(A.n*3)*100:6.0f}% "
              f"{opg:9.2f} {waf*100:11.0f}%")

# ---- 5. curse race ----
print("\n[5] CURSE RACE (curser boards):")
for label, req, exc in (("Witch, no trasher", [WITCH], list(TRASHERS) + [CHARLATAN, JESTER]),
                        ("Witch (any)", [WITCH], [CHARLATAN, JESTER]),
                        ("Charlatan", [CHARLATAN], [WITCH, JESTER])):
    left = []; mine = []; cb = []; got = 0; seed = 6 * 10**6
    while got < A.n:
        seed += 1
        k = random.Random(seed).sample(KINGDOM_POOL, 10)
        if any(c not in k for c in req) or any(c in k for c in exc): continue
        a, o, tr, pl, g = play_game(seed, k)
        left.append(g.supply[CURSE])
        mine.append(g.players[0].deck_composition()[CURSE])
        cb.append(sum(1 for c in a[0] if c in JUNKERS))
        got += 1
    print(f"    {label:18s}: pile-left {sum(left)/A.n:.1f}/10 | "
          f"my curses {sum(mine)/A.n:.1f} | cursers bought {sum(cb)/A.n:.1f}")

# ---- 6. card-combo affinity ----
print("\n[6] CARD-PAIR AFFINITY (bought-together lift on shared boards):")
pair_both = Counter(); pair_games = Counter(); solo = Counter(); ngames = 0
for seed, k in boards(A.n * 3, seed0=7 * 10**6):
    a, o, tr, pl, g = play_game(seed, k)
    kingdom = [c for c in k]
    boughtset = set(c for c in a[0] if c in kingdom and a[0].count(c) >= 1)
    ngames += 1
    for c in kingdom:
        if c in boughtset: solo[c] += 1
    for i, c1 in enumerate(kingdom):
        for c2 in kingdom[i+1:]:
            pair_games[(c1, c2)] += 1
            if c1 in boughtset and c2 in boughtset:
                pair_both[(c1, c2)] += 1
# lift = P(both bought | both on board) / (P(buy c1)*P(buy c2) marginal-ish)
rows = []
for (c1, c2), g_ in pair_games.items():
    if g_ < max(8, A.n // 4): continue
    both = pair_both[(c1, c2)] / g_
    # crude baseline: product of each card's solo buy rate on its boards
    b1 = solo[c1] / max(sum(pg for (x, y), pg in pair_games.items()
                            if x == c1 or y == c1), 1)
    b2 = solo[c2] / max(sum(pg for (x, y), pg in pair_games.items()
                            if x == c2 or y == c2), 1)
    base = max(b1 * b2, 1e-6)
    if both >= 0.25:           # only show pairs actually bought together
        rows.append((both / base, both, NAMES[c1], NAMES[c2]))
rows.sort(reverse=True)
print("    (pairs bought together >=25% of shared-board games, by lift)")
for lift, both, n1, n2 in rows[:14]:
    print(f"    {n1:14s}+ {n2:14s}  together {both*100:3.0f}%  lift {lift:4.1f}x")
print("\n" + "=" * 64)