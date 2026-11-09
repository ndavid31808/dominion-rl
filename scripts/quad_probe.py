import sys, torch, random; sys.path.insert(0,'.')
from dominion.engine2 import *
from train2.ppo3 import PolicyNetV3, NetBot3
from collections import Counter

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'runs/fixed10/latest.pt'
net=PolicyNetV3(); net.load_state_dict(torch.load(CKPT, weights_only=True)); net.eval()
agent=NetBot3(net, greedy=True)

def has(name):           # engine-agnostic constant lookup
    return globals().get(name, -999)
JUNKERS  = {has(n) for n in ("WITCH","CHARLATAN","JESTER")} - {-999}
TRASHERS = {has(n) for n in ("CHAPEL","UPGRADE","REMAKE",
                             "SENTRY","PRIEST","REMODEL")} - {-999}

def run(predicate, label, target=25):
    acq=Counter(); trd=Counter(); turns=[]; seed=90_000_000; done=0; tries=0
    while done<target and tries<40000:
        tries+=1; seed+=1
        k=random.Random(seed).sample(KINGDOM_POOL,10)
        j = any(c in JUNKERS for c in k)
        t = any(c in TRASHERS for c in k)
        if not predicate(j,t): continue
        g=Game(seed=seed, kingdom=k); bots=[agent,agent]; steps=0
        while not g.done:
            cur=g.current; a=bots[cur].act(g)
            if cur==0:
                if BUY_BASE<=a<BUY_BASE+N_CARDS or GAIN_BASE<=a<GAIN_BASE+N_CARDS:
                    acq[NAMES[a-BUY_BASE if a<GAIN_BASE else a-GAIN_BASE]]+=1
                elif TRASH_BASE<=a<TRASH_BASE+N_CARDS: trd[NAMES[a-TRASH_BASE]]+=1
            g.step(a); steps+=1
            if steps>6000: break
        turns.append(g.players[0].turns); done+=1
    tg = sum(trd.values())/max(done,1)
    print(f"\n{label}  (n={done}, avg {sum(turns)/max(done,1):.1f} turns)")
    print("  buys:", {k:round(v/max(done,1),1) for k,v in acq.most_common(7)})
    print(f"  trashes/game: {tg:.2f}  ->", dict(trd) or "ZERO")

print("="*60, "\nCKPT:", CKPT)
run(lambda j,t: j and not t,     "JUNKERS, no trasher")
run(lambda j,t: j and t,         "JUNK + TRASH available")
run(lambda j,t: t and not j,     "TRASHER, no junkers")
run(lambda j,t: not j and not t, "CLEAN (no junk, no trash)")
