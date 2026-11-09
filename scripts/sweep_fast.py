import sys, os, glob, argparse, random
import torch
sys.path.insert(0, '.')
from dominion.engine2 import (Game, KINGDOM_POOL, N_CARDS, NAMES,
                              BUY_BASE, GAIN_BASE, TRASH_BASE,
                              CHAPEL, WITCH, CHARLATAN, JESTER)
from train2.ppo3 import PolicyNetV3, NetBot3
import dominion.engine2 as E

JUNKERS = {WITCH, CHARLATAN, JESTER}
TRASHER_NAMES = ("CHAPEL", "UPGRADE", "REMAKE", "SENTRY", "PRIEST", "REMODEL")
TRASHERS = {getattr(E, n) for n in TRASHER_NAMES if hasattr(E, n)}

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--all", action="store_true")
A = ap.parse_args()

ckpts = sorted(glob.glob(os.path.join(A.run_dir, "ckpt_*.pt")))
if not A.all and len(ckpts) > 10:
    step = max(1, len(ckpts) // 10)
    keep = ckpts[::step]
    if ckpts[-1] not in keep:
        keep.append(ckpts[-1])
    ckpts = keep
latest = os.path.join(A.run_dir, "latest.pt")
if os.path.exists(latest):
    ckpts.append(latest)

net = PolicyNetV3()

def play(agent, seed, kingdom):
    g = Game(seed=seed, kingdom=kingdom)
    chapel_bought = False
    trashes = 0
    steps = 0
    while not g.done:
        cur = g.current
        a = agent.act(g)
        if cur == 0:
            if a == BUY_BASE + CHAPEL or a == GAIN_BASE + CHAPEL:
                chapel_bought = True
            if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
                trashes += 1
        g.step(a)
        steps += 1
        if steps > 6000:
            break
    zz = (g.score(0) == 0 and g.score(1) == 0)
    return chapel_bought, trashes, g.players[0].turns, steps > 6000, zz

def boards(n, pred, seed0):
    seed, got = seed0, 0
    while got < n:
        seed += 1
        k = random.Random(seed).sample(KINGDOM_POOL, 10)
        j = any(c in JUNKERS for c in k)
        t = any(c in TRASHERS for c in k)
        if pred(j, t, k):
            yield seed, k
            got += 1

print(f"{'checkpoint':16s} {'ChapBuy':>7s} {'trash/gm':>8s} "
      f"{'>40t%':>6s} {'cap%':>5s} {'0-0%':>5s}")
for p in ckpts:
    net.load_state_dict(torch.load(p, weights_only=True))
    net.eval()
    ag = NetBot3(net, greedy=True)
    cb = 0
    for s, k in boards(A.n, lambda j, t, k: CHAPEL in k, 11000000):
        b, *_ = play(ag, s, k)
        cb += b
    tr = long_t = capped = zz_n = 0
    for s, k in boards(A.n, lambda j, t, k: t and not j, 12000000):
        _, trs, turns, cap, zz = play(ag, s, k)
        tr += trs
        long_t += turns > 40
        capped += cap
        zz_n += zz
    tag = os.path.basename(p).replace("ckpt_", "").replace(".pt", "")
    print(f"{tag:16s} {cb/A.n:7.2f} {tr/A.n:8.2f} "
          f"{long_t*100//A.n:5d}% {capped*100//A.n:4d}% {zz_n*100//A.n:4d}%",
          flush=True)
