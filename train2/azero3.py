"""azero3: the learn-from-your-own-thinking loop for engine2 (20-card pool).

Every move in self-play is chosen by MCTS; the policy head trains by
cross-entropy toward the root VISIT DISTRIBUTION (search's per-decision
verdict) and the value head regresses on the game outcome. Pure self-play:
current mirror + a league of FROZEN PAST SELVES (AlphaStar-style population
play -- still 100% self-play). Scripted bots remain eval-only. Resumable;
checkpoints feed the Elo harness.

  python -m train2.azero3 --iters 200 --sims 240 --out runs/v8 \
      --init runs/fixed10/latest.pt --curr 0.25 --buffer 25000 --league-p 0.3
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from dominion.engine2 import Game, N_ACTIONS
from train2.ppo3 import PolicyNetV3, evaluate
from train2.mcts3 import MCTS

TEMP_TURNS = 8
DIR_ALPHA, DIR_EPS = 0.5, 0.25


def dirich_alpha(rng, n, alpha):
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    t = sum(xs) or 1.0
    return [x / t for x in xs]


def dirich(rng, n):
    x = np.array([rng.gammavariate(DIR_ALPHA, 1.0) for _ in range(n)])
    s = x.sum()
    return x / s if s > 0 else np.ones(n) / n


# (removed dead selfplay_game: the live path is batched3.parallel_selfplay)

def train_step(net, opt, buf, batch=512, steps=150):
    if len(buf) < batch:
        return 0.0, 0.0
    pl = vl = 0.0
    for _ in range(steps):
        b = random.sample(buf, batch)
        obs = torch.tensor(np.stack([x[0] for x in b]))
        mask = torch.tensor(np.stack([x[1] for x in b]))
        pi = torch.tensor(np.stack([x[2] for x in b]))
        z = torch.tensor([x[3] for x in b], dtype=torch.float32)
        logits, v = net(obs, mask)
        ploss = -(pi * F.log_softmax(logits, -1)).sum(-1).mean()
        vloss = F.mse_loss(torch.tanh(v), z)
        loss = ploss + vloss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        pl += ploss.item()
        vl += vloss.item()
    return pl / steps, vl / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games-per-iter", type=int, default=16)
    ap.add_argument("--sims", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=25_000)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="runs/az3")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--batch-games", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--curr", type=float, default=0.0,
                    help="curriculum: P(player gets 2 random opening gains)")
    ap.add_argument("--piles", default="10",
                    help="kingdom piles per training game (1-10)")
    ap.add_argument("--league-p", type=float, default=0.3,
                    help="fraction of self-play games vs a frozen past self "
                         "(0 = pure current-mirror)")
    ap.add_argument("--league-size", type=int, default=6,
                    help="how many frozen past selves to keep in the league")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)
    net = PolicyNetV3()
    latest = os.path.join(a.out, "latest.pt")
    state_path = os.path.join(a.out, "trainer_state.pt")
    if os.path.exists(latest):
        net.load_state_dict(torch.load(latest, weights_only=True))
        print("resumed latest.pt", flush=True)
    elif a.init and os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init, weights_only=True))
        print(f"warm-started from {a.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    buf: deque = deque(maxlen=a.buffer)
    start_iter = 1
    if os.path.exists(state_path):                 # resume continuously
        st = torch.load(state_path, weights_only=False)
        try:
            opt.load_state_dict(st["opt"])
            for rec in st["buf"]:
                buf.append(rec)
            start_iter = st.get("iter", 0) + 1
            print(f"resumed optimizer + {len(buf)} buffer positions "
                  f"(iter {start_iter})", flush=True)
        except Exception as e:
            print(f"warn: could not restore trainer state ({e})", flush=True)
    log = os.path.join(a.out, "log.csv")
    if not os.path.exists(log):
        open(log, "w").write(
            "iter,positions,ploss,vloss,Random,BigMoney,BMSmithy,Rush,"
            "ChapelMoney,secs\n")

    def current_league():
        """Frozen past selves spread across the whole run's history (older
        checkpoints are more strategically diverse than near-identical recent
        ones). Falls back to the warm-start checkpoint before any ckpt exists,
        so the league is non-empty from iteration 1. Returns None only when
        there is genuinely no past self to spar against."""
        cks = sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")))
        if not cks:
            return [a.init] if (a.init and os.path.exists(a.init)) else None
        return cks[:: max(1, len(cks) // a.league_size)][-a.league_size:]

    t0 = time.time()
    seed = int(time.time()) % 10**6 * 100
    for it in range(start_iter, start_iter + a.iters):
        net.eval()
        from train2.batched3 import parallel_selfplay, CAPPED_LOSS
        n_pos = 0
        league = current_league()
        results = parallel_selfplay(net, a.games_per_iter, a.sims,
                                    seed0=seed, workers=a.workers,
                                    batch_games=a.batch_games, curr_p=a.curr,
                                    piles=tuple(int(x)
                                                for x in str(a.piles).split(',')),
                                    league=league,
                                    league_p=(a.league_p if league else 0.0))
        seed += a.games_per_iter
        for recs, w in results:
            for obs, mask, pi, mover in recs:
                # Pure-outcome value target. CAPPED_LOSS is a true unresolved
                # stalemate -> both seats lose (kills the free-draw incentive
                # that made over-trashing look break-even). A decisive capped
                # game already arrives as a normal 0/1 winner. w is None only
                # for genuine engine-level exact ties (rare, legitimately 0).
                if w == CAPPED_LOSS:
                    z = -1.0
                elif w is None:
                    z = 0.0
                else:
                    z = 1.0 if w == mover else -1.0
                buf.append((obs, mask, pi, z))
            n_pos += len(recs)
        net.train()
        pl, vl = train_step(net, opt, buf)
        torch.save(net.state_dict(), latest)
        torch.save({"opt": opt.state_dict(),
                    "buf": list(buf), "iter": it},
                   state_path)
        line = f"iter {it:3d}  +{n_pos} pos (buf {len(buf)})  " \
               f"ploss {pl:.3f}  vloss {vl:.3f}"
        if it % a.eval_every == 0 or it == a.iters:
            net.eval()
            ev = evaluate(net, n_games=200, seed=12345)  # held-out fixed kingdoms, larger n
            line += "  eval " + " ".join(f"{k}:{v*100:.0f}%"
                                         for k, v in ev.items())
            torch.save(net.state_dict(),
                       os.path.join(a.out, f"ckpt_{int(time.time())}.pt"))
            with open(log, "a") as f:
                f.write(f"{it},{n_pos},{pl:.3f},{vl:.3f}," +
                        ",".join(f"{v:.3f}" for v in ev.values()) +
                        f",{time.time()-t0:.0f}\n")
        print(line, flush=True)
    print(f"done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()