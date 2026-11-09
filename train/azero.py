"""AlphaZero-style training loop: MCTS self-play -> (state, visit-dist,
outcome) targets -> supervised updates. The search-improvement ratchet:
the net is trained to imitate search's choices, which makes search stronger,
which improves the targets.

Differences from train/ppo.py:
  * No PPO, no advantages, no reward shaping. Policy target = root visit
    distribution; value target = final game outcome (mover's perspective).
  * Self-play only (both seats = current net + MCTS). Scripted bots are
    EVAL-ONLY -- the external yardstick, never trained against.
  * Temperature: moves sampled ~ visits for the first TEMP_TURNS player-turns
    (exploration), argmax afterwards. Dirichlet noise at each root.

Usage:
  python -m train.azero --init runs/v1/best.pt --out runs/az --iters 50
  (resumable; rerun the same command to continue)
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from dominion.engine import Game, N_ACTIONS
from train.ppo import PolicyNet, DEVICE, evaluate, make_net, obs_for
from train.mcts import MCTS
from train.batched import parallel_selfplay

TEMP_TURNS = 8          # player-turns of proportional sampling
DIRICHLET_ALPHA = 0.6   # ~10/avg_legal_moves heuristic
DIRICHLET_EPS = 0.25


def selfplay_game(net, sims, seed, rng):
    """Play one self-play game. Returns list of (obs, mask, pi, player) and
    the winner (0/1/None)."""
    g = Game(seed=seed)
    mcts = MCTS(net, n_sims=sims, seed=rng.randrange(1 << 30))
    records = []
    while not g.done:
        mover = g.current
        root_legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        if len(root_legal) == 1:
            g.step(root_legal[0])      # forced move: no search, no record
            continue
        _, visits = mcts.search(g)
        # Dirichlet noise is folded in by perturbing the visit sampling
        # (simpler than noising priors; same exploratory effect at the root)
        acts = list(visits.keys())
        v = np.array([visits[a] for a in acts], dtype=np.float64)
        if v.sum() == 0:
            v += 1.0
        pi = v / v.sum()
        noise = rng_dirichlet(rng, len(acts))
        pi_explore = (1 - DIRICHLET_EPS) * pi + DIRICHLET_EPS * noise

        turn = g.players[mover].turns
        if turn < TEMP_TURNS:
            a = acts[int(np.searchsorted(np.cumsum(pi_explore),
                                         rng.random()))]
        else:
            a = acts[int(np.argmax(pi))]

        full_pi = np.zeros(N_ACTIONS, dtype=np.float32)
        for act, p in zip(acts, pi):
            full_pi[act] = p
        records.append((np.asarray(obs_for(net, g), dtype=np.float32),
                        np.array(g.legal_actions(), dtype=bool),
                        full_pi, mover))
        g.step(a)
    return records, g.winner


def rng_dirichlet(rng, n):
    x = np.array([rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in range(n)])
    s = x.sum()
    return x / s if s > 0 else np.ones(n) / n


def train_step(net, opt, buffer, batch_size=512, steps=200):
    if len(buffer) < batch_size:
        return 0.0, 0.0
    ploss_t = vloss_t = 0.0
    for _ in range(steps):
        batch = random.sample(buffer, batch_size)
        obs = torch.tensor(np.stack([b[0] for b in batch]), device=DEVICE)
        mask = torch.tensor(np.stack([b[1] for b in batch]), device=DEVICE)
        pi = torch.tensor(np.stack([b[2] for b in batch]), device=DEVICE)
        z = torch.tensor([b[3] for b in batch], dtype=torch.float32,
                         device=DEVICE)
        logits, v = net(obs, mask)
        logp = F.log_softmax(logits, dim=-1)
        ploss = -(pi * logp).sum(-1).mean()       # CE to visit distribution
        vloss = F.mse_loss(torch.tanh(v), z)      # outcome regression
        loss = ploss + vloss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        ploss_t += ploss.item(); vloss_t += vloss.item()
    return ploss_t / steps, vloss_t / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=60_000)
    ap.add_argument("--init", default="", help="warm-start weights (.pt)")
    ap.add_argument("--out", default="runs/az")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--workers", type=int, default=0,
                    help="self-play worker processes (0 = in-process batched)")
    ap.add_argument("--batch-games", type=int, default=32,
                    help="games run in lockstep per worker")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    net = make_net(args.v2)
    latest = os.path.join(args.out, "latest.pt")
    if os.path.exists(latest):
        net.load_state_dict(torch.load(latest, weights_only=True))
        print("resumed from latest.pt", flush=True)
    elif args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))
        print(f"warm-started from {args.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    buffer: deque = deque(maxlen=args.buffer)
    log = os.path.join(args.out, "log.csv")
    if not os.path.exists(log):
        with open(log, "w") as f:
            f.write("iter,games,positions,ploss,vloss,"
                    "vsRandom,vsBM,vsSmithy,vsRush,secs\n")

    t0 = time.time()
    seed_ctr = int(time.time()) % 1_000_000
    for it in range(1, args.iters + 1):
        # ---- self-play data generation (batched / parallel)
        net.eval()
        n_pos = 0
        results = parallel_selfplay(net, args.games_per_iter, args.sims,
                                    seed0=seed_ctr, workers=args.workers,
                                    batch_games=args.batch_games)
        seed_ctr += args.games_per_iter
        for recs, winner in results:
            for obs, mask, pi, mover in recs:
                z = 0.0 if winner is None else \
                    (1.0 if winner == mover else -1.0)
                buffer.append((obs, mask, pi, z))
            n_pos += len(recs)

        # ---- supervised update on search targets
        net.train()
        ploss, vloss = train_step(net, opt, buffer)
        torch.save(net.state_dict(), latest)

        line = (f"iter {it:3d}  +{n_pos} pos (buf {len(buffer)})  "
                f"ploss {ploss:.3f}  vloss {vloss:.3f}")
        evals = {}
        if it % args.eval_every == 0 or it == args.iters:
            net.eval()
            evals = evaluate(net, n_games=60, seed=it * 31337)
            line += "   eval(raw net) " + "  ".join(
                f"{k}:{v*100:.0f}%" for k, v in evals.items())
            torch.save(net.state_dict(),
                       os.path.join(args.out, f"ckpt_{int(time.time())}.pt"))
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(f"{it},{args.games_per_iter},{n_pos},{ploss:.4f},"
                    f"{vloss:.4f},{evals.get('Random','')},"
                    f"{evals.get('BigMoney','')},{evals.get('BM+Smithy','')},"
                    f"{evals.get('Rush','')},{time.time()-t0:.0f}\n")

    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
