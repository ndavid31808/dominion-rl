"""PPO baseline for engine2 (variable 10-of-15 kingdoms). Port of train/ppo.py
with v2's proven config: terminal score-diff shaping, BM-weighted pool plus
ChapelMoney, 30% league vs frozen checkpoints. Eval = greedy vs all bots2
baselines on RANDOM kingdoms.

  python -m train2.ppo3 --iters 15 --out runs/v3ppo     (resumable chunks)
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

from dominion.engine2 import Game, N_ACTIONS, OBS_SIZE_V3
from dominion.bots2 import (Bot, RandomBot, BigMoneyBot, BMSmithyBot,
                            GardensRushBot, ChapelMoneyBot)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetV3(nn.Module):
    obs_size = OBS_SIZE_V3

    def __init__(self, hidden=512):
        super().__init__()
        self.inp = nn.Linear(OBS_SIZE_V3, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = torch.relu(self.inp(obs))
        r = torch.relu(self.fc1(h))
        h = torch.relu(h + self.fc2(r))
        return self.pi(h).masked_fill(~mask, -1e9), self.v(h).squeeze(-1)


class NetBot3(Bot):
    def __init__(self, net, greedy=False):
        self.net = net
        self.greedy = greedy

    @torch.no_grad()
    def act(self, g: Game) -> int:
        obs = torch.tensor(g.observe_v3()[None], device=DEVICE)
        mask = torch.tensor(np.asarray(g.legal_actions())[None], device=DEVICE)
        logits, _ = self.net(obs, mask)
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())


@dataclass
class EnvSlot:
    game: Game
    learner_seat: int
    opponent: Bot
    obs: list
    masks: list
    acts: list
    logps: list
    vals: list


def make_opponent(pool, rng):
    kind = rng.choices([p[0] for p in pool], weights=[p[1] for p in pool])[0]
    return kind() if callable(kind) else kind


def new_slot(pool, rng, seed):
    return EnvSlot(Game(seed=seed), rng.randint(0, 1),
                   make_opponent(pool, rng), [], [], [], [], [])


def collect(net, pool, n_envs, steps_target, rng, seed0):
    slots = [new_slot(pool, rng, seed0 + i) for i in range(n_envs)]
    OBS, MASK, ACT, LOGP, VAL, ADV = [], [], [], [], [], []
    wins = games = draws = 0
    seed_ctr = seed0 + n_envs

    def finish(s, r):
        nonlocal wins, games, draws
        if s.obs:
            OBS.extend(s.obs); MASK.extend(s.masks); ACT.extend(s.acts)
            LOGP.extend(s.logps); VAL.extend(s.vals)
            ADV.extend([r - v for v in s.vals])
        games += 1
        wins += r > 0.5
        draws += abs(r) <= 0.5

    while len(OBS) < steps_target:
        for s in slots:
            while not s.game.done and s.game.current != s.learner_seat:
                s.game.step(s.opponent.act(s.game))
        live = [s for s in slots if not s.game.done]
        if live:
            obs = torch.tensor(np.stack([s.game.observe_v3() for s in live]),
                               device=DEVICE)
            mask = torch.tensor(np.array([s.game.legal_actions()
                                          for s in live]), device=DEVICE)
            with torch.no_grad():
                logits, vals = net(obs, mask)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            for i, s in enumerate(live):
                a = int(acts[i])
                s.obs.append(obs[i].cpu().numpy())
                s.masks.append(mask[i].cpu().numpy())
                s.acts.append(a)
                s.logps.append(float(logps[i]))
                s.vals.append(float(vals[i]))
                s.game.step(a)
        for idx, s in enumerate(slots):
            if not s.game.done:
                while not s.game.done and s.game.current != s.learner_seat:
                    s.game.step(s.opponent.act(s.game))
            if s.game.done:
                w = s.game.winner
                r = 0.0 if w is None else (1.0 if w == s.learner_seat else -1.0)
                diff = (s.game.score(s.learner_seat)
                        - s.game.score(1 - s.learner_seat))
                r += max(-0.5, min(0.5, 0.015 * diff))
                finish(s, r)
                slots[idx] = new_slot(pool, rng, seed_ctr)
                seed_ctr += 1

    to = lambda x, dt: torch.tensor(np.array(x), dtype=dt, device=DEVICE)
    adv = to(ADV, torch.float32)
    ret = adv + to(VAL, torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (to(OBS, torch.float32), to(MASK, torch.bool),
            to(ACT, torch.int64), to(LOGP, torch.float32), ret, adv,
            wins / max(games, 1), draws / max(games, 1), games)


def ppo_update(net, opt, batch, epochs=4, minibatch=2048, clip=0.2,
               vf=0.5, ent=0.01):
    obs, mask, act, logp_old, ret, adv = batch
    n = len(obs)
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        for s in range(0, n, minibatch):
            b = idx[s:s + minibatch]
            logits, v = net(obs[b], mask[b])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(act[b]) - logp_old[b]).exp()
            pg = -torch.min(ratio * adv[b],
                            ratio.clamp(1 - clip, 1 + clip) * adv[b]).mean()
            loss = pg + vf * ((v - ret[b]) ** 2).mean() \
                - ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


EVAL_BOTS = [("Random", lambda: RandomBot(seed=7)),
             ("BigMoney", BigMoneyBot), ("BM+Smithy", BMSmithyBot),
             ("Rush", GardensRushBot), ("ChapelMoney", ChapelMoneyBot)]


def evaluate(net, n_games=60, seed=0):
    me = NetBot3(net, greedy=True)
    out = {}
    for name, mk in EVAL_BOTS:
        w = 0
        for i in range(n_games):
            g = Game(seed=seed + i)         # random kingdom per game
            seat = i % 2
            bots = [me, mk()] if seat == 0 else [mk(), me]
            steps = 0
            while not g.done:
                g.step(bots[g.current].act(g))
                steps += 1
                assert steps < 6000
            w += g.winner == seat
        out[name] = w / n_games
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/v3ppo")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--piles", default="10",
                    help="kingdom piles per game, e.g. '1,2' samples each")
    args = ap.parse_args()
    global PILES
    PILES = [int(x) for x in str(args.piles).split(',')]

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)
    net = PolicyNetV3().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    frozen = []
    latest = os.path.join(args.out, "latest.pt")
    ckpts = sorted(f for f in os.listdir(args.out) if f.startswith("ckpt_"))
    if os.path.exists(latest):
        net.load_state_dict(torch.load(latest, weights_only=True))
        for c in ckpts[-5:]:
            f = PolicyNetV3().to(DEVICE)
            f.load_state_dict(torch.load(os.path.join(args.out, c),
                                         weights_only=True))
            f.eval()
            [p.requires_grad_(False) for p in f.parameters()]
            frozen.append(f)
        print("resumed latest.pt", flush=True)

    # PURE SELF-PLAY: no scripted bots in training. Current self (mirror)
    # + league of frozen past checkpoints. Bots are eval-only thermometers.
    base_pool = []
    log = os.path.join(args.out, "log.csv")
    if not os.path.exists(log):
        with open(log, "w") as f:
            f.write("iter,games,win,draw," +
                    ",".join(n for n, _ in EVAL_BOTS) + ",secs\n")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        pool = [(NetBot3(net), 0.7)]            # mirror vs current self
        if frozen:                              # 30% vs league of past selves
            for fz in frozen[-5:]:
                pool.append((NetBot3(fz), 0.3 / min(len(frozen), 5)))
        batch = collect(net, pool, args.envs, args.steps, rng,
                        seed0=it * 100_000 + int(time.time()) % 1000)
        *tensors, wr, dr, games = batch
        ppo_update(net, opt, tuple(tensors))
        torch.save(net.state_dict(), os.path.join(args.out, "latest.pt"))
        line = f"iter {it:3d}  games {games:4d}  train-win {wr*100:5.1f}%"
        if it % args.eval_every == 0 or it == args.iters:
            ev = evaluate(net, seed=it * 7777)
            line += "   eval " + "  ".join(f"{k}:{v*100:.0f}%"
                                           for k, v in ev.items())
            fz = copy.deepcopy(net).eval()
            [p.requires_grad_(False) for p in fz.parameters()]
            frozen.append(fz)
            torch.save(net.state_dict(),
                       os.path.join(args.out, f"ckpt_{int(time.time())}.pt"))
            with open(log, "a") as f:
                f.write(f"{it},{games},{wr:.3f},{dr:.3f}," +
                        ",".join(f"{ev[n]:.3f}" for n, _ in EVAL_BOTS) +
                        f",{time.time()-t0:.0f}\n")
        print(line, flush=True)
    torch.save(net.state_dict(), os.path.join(args.out, "final.pt"))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()