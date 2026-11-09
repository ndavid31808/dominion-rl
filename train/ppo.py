"""PPO trainer for the Dominion vanilla board.

Learner plays one seat (alternating per env); the opponent is sampled per
episode from a pool: scripted bots + frozen past checkpoints of the learner
("league"), which prevents overfitting to any single strategy and prevents
self-play cycling.

Reward: +1 win / -1 loss / 0 draw at terminal. gamma=1 (short episodic game).

Usage:
  python -m train.ppo --iters 300 --out runs/v1          # full run
  python -m train.ppo --iters 25 --quick                 # smoke / proof of life
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from dominion.engine import Game, N_ACTIONS, OBS_SIZE
from dominion.bots import (RandomBot, BigMoneyBot, BMSmithyBot,
                           GardensRushBot, Bot)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- network --
class PolicyNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


class PolicyNetV2(nn.Module):
    """3x512 residual MLP on obs-v2 (190 floats), ~650k params."""

    obs_size = 190

    def __init__(self, hidden=512):
        super().__init__()
        from dominion.engine import OBS_SIZE_V2
        self.inp = nn.Linear(OBS_SIZE_V2, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = torch.relu(self.inp(obs))
        r = torch.relu(self.fc1(h))
        h = torch.relu(h + self.fc2(r))      # residual connection
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


def obs_for(net, g):
    """Dispatch: v2 nets get observe_v2, v1 nets get observe."""
    return g.observe_v2() if getattr(net, "obs_size", 108) == 190 \
        else g.observe()


def make_net(v2: bool):
    return (PolicyNetV2() if v2 else PolicyNet()).to(DEVICE)


class NetBot(Bot):
    """Wraps a (frozen) network as an opponent / eval agent."""

    def __init__(self, net: PolicyNet, greedy: bool = False):
        self.net = net
        self.greedy = greedy

    @torch.no_grad()
    def act(self, g: Game) -> int:
        obs = torch.tensor(np.asarray(obs_for(self.net, g),
                                      dtype=np.float32)[None], device=DEVICE)
        mask = torch.tensor([g.legal_actions()], dtype=torch.bool, device=DEVICE)
        logits, _ = self.net(obs, mask)
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())


# ----------------------------------------------------------------- rollout --
@dataclass
class EnvSlot:
    game: Game
    learner_seat: int
    opponent: Bot
    # learner trajectory
    obs: list
    masks: list
    acts: list
    logps: list
    vals: list


def make_opponent(pool, rng) -> Bot:
    kind = rng.choices([p[0] for p in pool], weights=[p[1] for p in pool])[0]
    return kind() if callable(kind) else kind


def new_slot(opp_pool, rng, seed) -> EnvSlot:
    return EnvSlot(Game(seed=seed), rng.randint(0, 1),
                   make_opponent(opp_pool, rng), [], [], [], [], [])


def collect(net, opp_pool, n_envs, steps_target, rng, seed0):
    """Vectorized-ish rollout: batch NN forwards across envs whose current
    player is the learner. Scripted/frozen opponents act inline. Returns
    flat tensors for PPO + episode stats."""
    slots = [new_slot(opp_pool, rng, seed0 + i) for i in range(n_envs)]
    OBS, MASK, ACT, LOGP, VAL, ADV = [], [], [], [], [], []
    wins = games = draws = 0

    def finish(slot, reward):
        nonlocal wins, games, draws
        T = len(slot.obs)
        if T:
            OBS.extend(slot.obs); MASK.extend(slot.masks)
            ACT.extend(slot.acts); LOGP.extend(slot.logps)
            VAL.extend(slot.vals)
            # gamma=1, terminal-only reward: return for every step = reward
            ADV.extend([reward - v for v in slot.vals])
        games += 1
        wins += reward > 0.5
        draws += reward == 0

    seed_ctr = seed0 + n_envs
    while len(OBS) < steps_target:
        # advance every env until it's the learner's turn (or done)
        for s in slots:
            while not s.game.done and s.game.current != s.learner_seat:
                s.game.step(s.opponent.act(s.game))
        # batch the learner's decision
        live = [s for s in slots if not s.game.done]
        if live:
            obs = torch.tensor(np.array([obs_for(net, s.game)
                                         for s in live], dtype=np.float32),
                               device=DEVICE)
            mask = torch.tensor([s.game.legal_actions() for s in live],
                                dtype=torch.bool, device=DEVICE)
            with torch.no_grad():
                logits, vals = net(obs, mask)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            for i, s in enumerate(live):
                a = int(acts[i])
                s.obs.append(obs[i].cpu().numpy())
                s.masks.append(mask[i].cpu().numpy())
                s.acts.append(a); s.logps.append(float(logps[i]))
                s.vals.append(float(vals[i]))
                s.game.step(a)
        # settle finished games, respawn
        for idx, s in enumerate(slots):
            if not s.game.done:
                # opponent may need to finish the game after learner's move
                while not s.game.done and s.game.current != s.learner_seat:
                    s.game.step(s.opponent.act(s.game))
            if s.game.done:
                w = s.game.winner
                r = 0.0 if w is None else (1.0 if w == s.learner_seat else -1.0)
                # Terminal score-diff shaping (OK on this board: no trashing,
                # so score diff is a monotone proxy). Capped so winning still
                # dominates the signal.
                diff = (s.game.score(s.learner_seat)
                        - s.game.score(1 - s.learner_seat))
                r += max(-0.5, min(0.5, 0.015 * diff))
                finish(s, r)
                slots[idx] = new_slot(opp_pool, rng, seed_ctr)
                seed_ctr += 1

    to = lambda x, dt: torch.tensor(np.array(x), dtype=dt, device=DEVICE)
    adv = to(ADV, torch.float32)
    ret = adv + to(VAL, torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (to(OBS, torch.float32), to(MASK, torch.bool),
            to(ACT, torch.int64), to(LOGP, torch.float32), ret, adv,
            wins / max(games, 1), draws / max(games, 1), games)


# -------------------------------------------------------------------- PPO ---
def ppo_update(net, opt, batch, epochs=4, minibatch=2048,
               clip=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, mask, act, logp_old, ret, adv = batch
    n = len(obs)
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        for s in range(0, n, minibatch):
            b = idx[s:s + minibatch]
            logits, v = net(obs[b], mask[b])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[b])
            ratio = (logp - logp_old[b]).exp()
            pg = -torch.min(ratio * adv[b],
                            ratio.clamp(1 - clip, 1 + clip) * adv[b]).mean()
            vloss = ((v - ret[b]) ** 2).mean()
            loss = pg + vf_coef * vloss - ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


# -------------------------------------------------------------------- eval --
def evaluate(net, n_games=100, seed=0):
    """Greedy policy vs each scripted bot, seats balanced."""
    me = NetBot(net, greedy=True)
    out = {}
    for name, mk in [("Random", lambda: RandomBot(seed=7)),
                     ("BigMoney", BigMoneyBot),
                     ("BM+Smithy", BMSmithyBot),
                     ("Rush", GardensRushBot)]:
        wins = 0
        for i in range(n_games):
            g = Game(seed=seed + i)
            seat = i % 2
            bots = [me, mk()] if seat == 0 else [mk(), me]
            steps = 0
            while not g.done:
                g.step(bots[g.current].act(g))
                steps += 1
                if steps > 4000:
                    raise RuntimeError("non-terminating eval game")
            if g.winner == seat:
                wins += 1
        out[name] = wins / n_games
    return out


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16384,
                    help="learner transitions per iteration")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="small run for smoke testing")
    ap.add_argument("--v2", action="store_true",
                    help="obs-v2 (190 floats) + residual PolicyNetV2")
    args = ap.parse_args()
    if args.quick:
        args.steps = 8192
        args.eval_every = 5

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    net = make_net(args.v2)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # opponent pool: (constructor-or-instance, weight)
    frozen: list[PolicyNet] = []

    # resume: load latest checkpoint + rebuild league from saved ckpts
    ckpts = sorted(f for f in os.listdir(args.out) if f.startswith("ckpt_"))
    if ckpts:
        net.load_state_dict(torch.load(os.path.join(args.out, ckpts[-1]),
                                       weights_only=True))
        for c in ckpts[-5:]:
            f = make_net(args.v2)
            f.load_state_dict(torch.load(os.path.join(args.out, c),
                                         weights_only=True))
            f.eval()
            for p in f.parameters():
                p.requires_grad_(False)
            frozen.append(f)
        print(f"resumed from {ckpts[-1]} with {len(frozen)} league nets",
              flush=True)
    base_pool = [(lambda: RandomBot(seed=rng.randint(0, 9999)), 0.5),
                 (BigMoneyBot, 4.0), (BMSmithyBot, 3.0),
                 (GardensRushBot, 1.5)]

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "a" if ckpts else "w") as f:
        f.write("iter,games,win_rate,draw_rate,vsRandom,vsBM,vsSmithy,vsRush,secs\n")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        pool = list(base_pool)
        if frozen:  # league: 30% of games vs a frozen past self
            w = sum(p[1] for p in base_pool) * (0.3 / 0.7)
            for fnet in frozen[-5:]:
                pool.append((NetBot(fnet), w / min(len(frozen), 5)))

        obs, mask, act, logp, ret, adv, wr, dr, games = collect(
            net, pool, args.envs, args.steps, rng, seed0=it * 100_000)
        ppo_update(net, opt, (obs, mask, act, logp, ret, adv))

        line = (f"iter {it:3d}  games {games:4d}  "
                f"train-win {wr*100:5.1f}%  draw {dr*100:4.1f}%")
        evals = {}
        if it % args.eval_every == 0 or it == args.iters:
            evals = evaluate(net, n_games=60 if args.quick else 100,
                             seed=it * 7777)
            line += "   eval " + "  ".join(f"{k}:{v*100:.0f}%"
                                           for k, v in evals.items())
            # freeze a league checkpoint
            f = copy.deepcopy(net).eval()
            for p in f.parameters():
                p.requires_grad_(False)
            frozen.append(f)
            stamp = int(time.time())
            torch.save(net.state_dict(),
                       os.path.join(args.out, f"ckpt_{stamp}.pt"))
            avg = sum(evals.values()) / len(evals)
            best_file = os.path.join(args.out, "best_score.txt")
            prev = float(open(best_file).read()) if os.path.exists(best_file) else -1
            if avg > prev:
                torch.save(net.state_dict(), os.path.join(args.out, "best.pt"))
                open(best_file, "w").write(str(avg))
                print(f"   ^ new best (avg eval {avg*100:.0f}%) -> best.pt", flush=True)
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(f"{it},{games},{wr:.4f},{dr:.4f},"
                    f"{evals.get('Random','')},{evals.get('BigMoney','')},"
                    f"{evals.get('BM+Smithy','')},{evals.get('Rush','')},"
                    f"{time.time()-t0:.0f}\n")

    torch.save(net.state_dict(), os.path.join(args.out, "final.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min; weights in {args.out}/")


if __name__ == "__main__":
    main()
