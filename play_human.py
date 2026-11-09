"""play_human.py: play Dominion against the trained agent in the terminal.

  python play_human.py models/v10.pt                # vs the agent
  python play_human.py --opponent bm                     # vs BigMoney
  python play_human.py models/v10.pt --require Chapel,Bazaar
  python play_human.py models/v10.pt --seed 12345   # replayable board

You are seat 0 and go first. Game summaries are appended to
human_games.log (instrumentation only -- never training data).
"""
import sys, argparse, random, datetime
import numpy as np
sys.path.insert(0, '.')
from dominion.engine2 import (Game, NAMES, N_CARDS, PLAYABLE, PLAY_BASE,
                              END_ACTION, BUY_BASE, PASS, GAIN_BASE,
                              TRASH_BASE, STOP_TRASH, JESTER_TO_OPP,
                              JESTER_TO_ME, HG_DUCHY, HG_ESTATES, COPPER,
                              SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE)
from dominion.bots2 import BigMoneyBot

def label(a):
    if PLAY_BASE <= a < PLAY_BASE + len(PLAYABLE):
        return f"play {NAMES[PLAYABLE[a - PLAY_BASE]]}"
    if a == END_ACTION:
        return "end actions (auto-plays treasures)"
    if BUY_BASE <= a < BUY_BASE + N_CARDS:
        return f"buy {NAMES[a - BUY_BASE]}"
    if a == PASS:
        return "pass (end turn)"
    if GAIN_BASE <= a < GAIN_BASE + N_CARDS:
        return f"gain {NAMES[a - GAIN_BASE]}"
    if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
        return f"trash {NAMES[a - TRASH_BASE]}"
    if a == STOP_TRASH:
        return "stop trashing"
    if a == JESTER_TO_OPP:
        return "Jester: opponent gains the copy"
    if a == JESTER_TO_ME:
        return "Jester: I gain the copy"
    if a == HG_DUCHY:
        return "Hunting Grounds: gain a Duchy"
    if a == HG_ESTATES:
        return "Hunting Grounds: gain 3 Estates"
    return f"action #{a}"

def fmt_cards(vec):
    out = [f"{NAMES[c]}x{int(n)}" for c, n in enumerate(vec) if n > 0]
    return ", ".join(out) if out else "(none)"

def show(g, you):
    me = g.players[you]
    opp = g.players[1 - you]
    print("\n" + "=" * 64)
    print(f"TURN {me.turns + 1}   you {g.score(you)} VP | agent "
          f"{g.score(1 - you)} VP   (Provinces left: {g.supply[PROVINCE]})")
    kingdom = [c for c in range(N_CARDS) if g.in_supply[c] and c > 6]
    print("kingdom: " + ", ".join(
        f"{NAMES[c]}[{g.supply[c]}]" for c in kingdom))
    print(f"base: " + ", ".join(
        f"{NAMES[c]}[{g.supply[c]}]"
        for c in (COPPER, SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE)))
    print(f"your hand: {fmt_cards(me.hand)}")
    print(f"your deck: {fmt_cards(me.deck_composition())}")
    print(f"agent deck: {fmt_cards(opp.deck_composition())}")
    print(f"coins {g.coins}  actions {g.actions}  buys {g.buys}  "
          f"phase {'ACTION' if g.phase == 0 else 'BUY'}")

def human_move(g, you):
    legal = [i for i, ok in enumerate(g.legal_actions()) if ok]
    show(g, you)
    for k, a in enumerate(legal):
        print(f"  [{k}] {label(a)}")
    while True:
        try:
            s = input("> ").strip()
            k = int(s)
            if 0 <= k < len(legal):
                return legal[k]
        except (ValueError, EOFError):
            pass
        print("enter a number from the menu")

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="")
ap.add_argument("--opponent", default="net", choices=["net", "bm"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--require", default="")
ap.add_argument("--exclude", default="")
A = ap.parse_args()

if A.opponent == "net":
    from train2.elo4 import _load_net, GreedyNetBot
    agent = GreedyNetBot(_load_net(A.ckpt or "models/v10.pt"))
else:
    agent = BigMoneyBot()

REQ = [NAMES.index(x) for x in A.require.split(",") if x.strip()]
EXC = [NAMES.index(x) for x in A.exclude.split(",") if x.strip()]
seed = A.seed or random.randrange(10**8)
while True:
    g = Game(seed=seed)
    if all(g.in_supply[c] for c in REQ) and \
            not any(g.in_supply[c] for c in EXC):
        break
    seed += 1
print(f"board seed {seed}  (replay with --seed {seed})")

YOU = 0
steps = 0
while not g.done and steps < 6000:
    a = human_move(g, YOU) if g.current == YOU else agent.act(g)
    if g.current != YOU and (BUY_BASE <= a < BUY_BASE + N_CARDS):
        print(f"    agent buys {NAMES[a - BUY_BASE]}")
    g.step(a)
    steps += 1

me_vp, ai_vp = g.score(YOU), g.score(1 - YOU)
you_won = g.winner == YOU
print("\n" + "#" * 64)
print(f"GAME OVER turn {g.players[YOU].turns}: you {me_vp} VP, "
      f"opponent {ai_vp} VP -> {'YOU WIN' if you_won else 'YOU LOSE'}")
with open("human_games.log", "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} seed={seed} "
            f"opp={A.opponent} you={me_vp} ai={ai_vp} "
            f"turns={g.players[YOU].turns} win={you_won}\n")
