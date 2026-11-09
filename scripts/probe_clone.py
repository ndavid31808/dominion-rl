"""probe_clone.py: ARCHITECTURE CAPACITY TEST via behavior cloning.

Claim under test: "something in the architecture prevents this model from
even getting to trashing/engines."  If true, a fresh PolicyNetV4 trained
by pure supervised imitation of the scripted ChapelMoneyBot (a competent
thin-deck trasher) should FAIL to reproduce its play. If the clone matches
the script's strength and trashing behavior, the architecture can
represent and execute trasher play, and the failure is discovery/training.

  python probe_clone.py --games 400 --epochs 6 --eval-games 200
"""
import argparse, random, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from dominion.engine2 import Game, TRASH_BASE, N_CARDS, NAMES
from dominion.bots2 import BigMoneyBot, ChapelMoneyBot
from train2.net4 import PolicyNetV4
from train2.elo4 import GreedyNetBot

ap = argparse.ArgumentParser() # ok
ap.add_argument("--games", type=int, default=400)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--eval-games", type=int, default=200)
A = ap.parse_args()
CHAPEL = NAMES.index("Chapel")

def chapel_board(seed):
    g = Game(seed=seed)
    return g if g.in_supply[CHAPEL] else None

# ---- 1. collect imitation data from the scripted trasher --------------
print("collecting demonstrations...", flush=True)
X, M, Y = [], [], []
teacher_wins = teacher_n = 0
trash_per_game = []
seed = 77_000_000
while teacher_n < A.games:
    seed += 1
    g = chapel_board(seed)
    if g is None:
        continue
    bots = {0: ChapelMoneyBot(), 1: BigMoneyBot()}
    tr = 0
    steps = 0
    while not g.done and steps < 6000:
        cur = g.current
        a = bots[cur].act(g)
        if cur == 0:
            X.append(g.observe_v3())
            M.append(np.asarray(g.legal_actions(), dtype=bool))
            Y.append(a)
            if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
                tr += 1
        g.step(a)
        steps += 1
    teacher_wins += 1 if g.winner == 0 else 0
    teacher_n += 1
    trash_per_game.append(tr)
print(f"teacher: {100*teacher_wins/teacher_n:.0f}% vs BigMoney, "
      f"{np.mean(trash_per_game):.1f} trashes/gm, "
      f"{len(X)} decisions collected", flush=True)

# ---- 2. behavior-clone into a FRESH PolicyNetV4 -----------------------
X = torch.tensor(np.asarray(X)); M = torch.tensor(np.asarray(M))
Y = torch.tensor(np.asarray(Y))
net = PolicyNetV4()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
n = len(X); idx = np.arange(n)
for ep in range(A.epochs):
    np.random.shuffle(idx); tot = correct = 0; loss_sum = 0.0
    for i in range(0, n, 256):
        b = idx[i:i+256]
        logits, _ = net(X[b], M[b])
        loss = F.cross_entropy(logits, Y[b])
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss) * len(b); tot += len(b)
        correct += int((logits.argmax(-1) == Y[b]).sum())
    # accuracy on trash decisions specifically
    with torch.no_grad():
        tmask = (Y >= TRASH_BASE) & (Y < TRASH_BASE + N_CARDS)
        if tmask.any():
            tl, _ = net(X[tmask], M[tmask])
            tacc = float((tl.argmax(-1) == Y[tmask]).float().mean())
        else:
            tacc = float("nan")
    print(f"epoch {ep+1}: loss {loss_sum/tot:.3f}  acc {correct/tot:.3f}"
          f"  trash-decision acc {tacc:.3f}", flush=True)

# ---- 3. the verdict: does the CLONE play like a trasher? --------------
print("evaluating clone...", flush=True)
net.eval(); clone = GreedyNetBot(net)
wins = 0; ngames = 0; ctrash = []
seed = 88_000_000
while ngames < A.eval_games:
    seed += 1
    g = chapel_board(seed)
    if g is None:
        continue
    bots = {0: clone, 1: BigMoneyBot()}
    tr = 0; steps = 0
    while not g.done and steps < 6000:
        cur = g.current
        a = bots[cur].act(g)
        if cur == 0 and TRASH_BASE <= a < TRASH_BASE + N_CARDS:
            tr += 1
        g.step(a); steps += 1
    wins += 1 if g.winner == 0 else 0
    ngames += 1; ctrash.append(tr)
print(f"\nVERDICT")
print(f"  teacher (script): {100*teacher_wins/teacher_n:.0f}% vs BigMoney,"
      f" {np.mean(trash_per_game):.1f} trashes/gm")
print(f"  clone (this net): {100*wins/ngames:.0f}% vs BigMoney,"
      f" {np.mean(ctrash):.1f} trashes/gm")
print("\nREAD: clone ~= teacher on BOTH numbers -> architecture can "
      "represent+execute\ntrasher play; failure is discovery/training. "
      "Clone trashes ~0 or wins far less\n-> architecture genuinely "
      "cannot, claim confirmed, failure localized.")