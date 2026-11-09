"""Batched lockstep MCTS self-play for engine2 — port of train/batched.py.
Runs G games simultaneously; each loop advances every game by one simulation
and pools all leaf evaluations into ONE network forward pass. Plus
multiprocess sharding via --workers. Produces azero-style records:
(obs_v3, legal_mask, visit_distribution, mover)."""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from dominion.engine2 import (Game, N_ACTIONS, N_CARDS, BUY_BASE,
                              KINGDOM_POOL)
from train2.mcts3 import _determinize, _outcome
from train2.azero3 import (TEMP_TURNS, DIR_EPS, DIR_ALPHA,
                           dirich, dirich_alpha)

C_PUCT = 1.5


def fast_clone(g: Game) -> Game:
    c = Game.__new__(Game)
    c.rng = None
    c.kingdom = g.kingdom
    c.in_supply = g.in_supply              # immutable post-reset: share
    c.supply = list(g.supply)
    c.total_cards = g.total_cards
    c.trash = list(g.trash)
    c.players = [p.clone() for p in g.players]
    for a in ("current", "phase", "actions", "buys", "coins", "pending",
              "pending_n", "pending_cost", "pending_card", "remake_left",
              "charlatan_rule", "done", "winner"):
        setattr(c, a, getattr(g, a))
    c.hg_resume = g.hg_resume
    # v7 fields (default safely if running an older engine)
    c.priest_bonus = getattr(g, "priest_bonus", 0)
    c.sentry_idx = getattr(g, "sentry_idx", 0)
    c.sentry_cards = list(getattr(g, "sentry_cards", []))
    c.sentry_kept = list(getattr(g, "sentry_kept", []))
    # v8 multiplier fields
    c.queued_plays = list(getattr(g, "queued_plays", []))
    c.mult_times = getattr(g, "mult_times", 0)
    c.mm_pending = getattr(g, "mm_pending", 0)
    c._replaying = False
    return c


class Node:
    __slots__ = ("expanded", "P", "N", "W", "children")

    def __init__(self):
        self.expanded = False
        self.P, self.N, self.W = {}, {}, {}
        self.children = {}


def _select(node, legal):
    N, W, P = node.N, node.W, node.P
    tot = 0
    for a in legal:
        tot += N.get(a, 0)
    k = C_PUCT * math.sqrt(tot + 1)
    inv = 1.0 / len(legal)
    best_a, best = legal[0], -1e30
    for a in legal:
        n = N.get(a, 0)
        s = (W[a] / n if n else 0.0) + k * P.get(a, inv) / (1 + n)
        if s > best:
            best, best_a = s, a
    return best_a


def _walk(root, g):
    path = []
    node = root
    while True:
        if g.done:
            return path, None, _outcome(g, g.current)
        if not node.expanded:
            return path, node, 0.0
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        a = _select(node, legal)
        me = g.current
        g.step(a)
        child = node.children.get(a)
        if child is None:
            child = node.children[a] = Node()
        path.append((node, a, me, g.current))
        node = child


def _backup(path, v):
    for node, a, me, nxt in reversed(path):
        v_me = v if nxt == me else -v
        node.N[a] = node.N.get(a, 0) + 1
        node.W[a] = node.W.get(a, 0.0) + v_me
        v = v_me


_CURR_EXCLUDE = (3, 4, 5, 6, 7)   # CURSE, ESTATE, DUCHY, PROVINCE, GARDENS


MAX_DECISIONS = 3000     # hard cap on a single game's decisions
# Sentinel winner for a game that hit the cap WITHOUT a decisive VP result,
# i.e. a true unresolved stalemate. Both seats are scored as a loss (z=-1)
# upstream in azero3, so over-trashing into a non-terminating deck is no
# longer a free draw. A capped game that IS decisive on VP is returned as a
# normal 0/1 winner instead (the agent that built a working deck still wins).
CAPPED_LOSS = -2

_LEAGUE_CACHE = {}


def _load_league(paths):
    from train2.ppo3 import PolicyNetV3
    nets = []
    for p in paths:
        if p not in _LEAGUE_CACHE:
            n = PolicyNetV3()
            n.load_state_dict(torch.load(p, weights_only=True))
            n.eval()
            _LEAGUE_CACHE[p] = n
        nets.append(_LEAGUE_CACHE[p])
    return nets


class _Ctx:
    __slots__ = ("g", "root", "sims", "records", "rng", "curr_seats",
                 "decisions", "opp_seat", "opp_idx")

    def __init__(self, seed, rng_seed, curr_p=0.0, piles=(10,),
                 league=None, league_p=0.0):
        if isinstance(piles, int):
            piles = (piles,)
        rng0 = random.Random(seed ^ 0xA5A5)
        n = rng0.choice(piles)
        kingdom = sorted(rng0.sample(KINGDOM_POOL, n)) if n < 10 else None
        self.g = Game(seed=seed, kingdom=kingdom)
        self.root = Node()
        self.sims = 0
        self.records = []
        self.rng = random.Random(rng_seed)
        # curriculum: per-seat independently, this player's first TWO buys
        # are uniform-random among cards it can actually AFFORD that turn
        self.curr_seats = {i for i in (0, 1)
                           if curr_p > 0.0 and self.rng.random() < curr_p}
        self.decisions = 0
        self.opp_seat = -1
        self.opp_idx = -1
        if league and self.rng.random() < league_p:
            self.opp_seat = self.rng.randrange(2)
            self.opp_idx = self.rng.randrange(len(league))

    def opponent_move(self, league_nets):
        if self.g.current != self.opp_seat:
            return None
        net = league_nets[self.opp_idx]
        obs = torch.tensor(self.g.observe_v3()[None])
        mask = torch.tensor(np.array(self.g.legal_actions())[None])
        with torch.no_grad():
            logits, _ = net(obs, mask)
        return int(logits[0].argmax())

    def forced_curriculum_buy(self):
        """If the current decision is a curriculum opening buy, return the
        action; else None. Forced buys are NOT recorded as policy targets."""
        g = self.g
        cur = g.current
        if cur not in self.curr_seats or g.pending != 0 or g.phase != 1:
            return None
        if g.players[cur].turns >= 2:          # only turns 1 and 2
            return None
        legal = g.legal_actions()
        elig = [a for a in range(BUY_BASE, BUY_BASE + N_CARDS)
                if legal[a] and (a - BUY_BASE) not in _CURR_EXCLUDE]
        return self.rng.choice(elig) if elig else None


def selfplay_batch(net, n_games, sims, seed0, batch_games=24, curr_p=0.0,
                   piles=10, league=None, league_p=0.0):
    league_nets = _load_league(league) if league else []
    out: List[Tuple[list, Optional[int]]] = []
    master = random.Random(seed0)
    next_seed = seed0
    active: List[_Ctx] = []

    def spawn():
        nonlocal next_seed
        c = _Ctx(next_seed, master.randrange(1 << 30), curr_p=curr_p,
                 piles=piles, league=league, league_p=league_p)
        next_seed += 1
        return c

    while len(out) < n_games:
        while len(active) < batch_games and len(active) + len(out) < n_games:
            active.append(spawn())
        # forced moves + finished games
        still = []
        for c in active:
            if c.decisions > MAX_DECISIONS:      # runaway game: resolve it
                s0, s1 = c.g.score(0), c.g.score(1)
                if s0 != s1:                     # decisive on VP at the cap
                    out.append((c.records, 0 if s0 > s1 else 1))
                else:                            # true stalemate -> both lose
                    out.append((c.records, CAPPED_LOSS))
                continue
            while not c.g.done:
                legal = [a for a, ok in enumerate(c.g.legal_actions()) if ok]
                if len(legal) == 1:
                    c.g.step(legal[0])
                    c.decisions += 1
                    continue
                om = c.opponent_move(league_nets)
                if om is not None:
                    c.g.step(om)               # frozen sparring move
                    c.root = Node(); c.sims = 0; c.decisions += 1
                    continue
                fa = c.forced_curriculum_buy()
                if fa is not None:
                    c.g.step(fa)               # random affordable opening buy
                    c.root = Node()            # (unrecorded; fresh tree)
                    c.sims = 0
                    c.decisions += 1
                    continue
                break
            if c.g.done:
                out.append((c.records, c.g.winner))
            else:
                still.append(c)
        active = still
        if not active:
            continue
        # one simulation per game; pool evaluations
        pending = []
        for c in active:
            sim = fast_clone(c.g)
            sim.rng = random.Random(c.rng.randrange(1 << 30))
            _determinize(sim, c.g.current, sim.rng)
            path, leaf, tv = _walk(c.root, sim)
            if leaf is None:
                _backup(path, tv)
                c.sims += 1
            else:
                pending.append((c, path, leaf, sim.observe_v3(),
                                sim.legal_actions()))
        if pending:
            obs = torch.tensor(np.stack([p[3] for p in pending]))
            mask = torch.tensor(np.array([p[4] for p in pending]))
            with torch.no_grad():
                logits, vals = net(obs, mask)
                probs = torch.softmax(logits, -1).numpy()
            for (c, path, leaf, _, m), pr, v in zip(pending, probs,
                                                    vals.numpy()):
                legal = [a for a, ok in enumerate(m) if ok]
                s = float(pr[legal].sum()) or 1.0
                leaf.P = {a: float(pr[a]) / s for a in legal}
                if leaf is c.root:                     # root exploration noise
                    eps, al = DIR_EPS, DIR_ALPHA
                    noise = dirich_alpha(c.rng, len(legal), al)
                    leaf.P = {a: (1 - eps) * leaf.P[a] + eps * noise[i]
                              for i, a in enumerate(legal)}
                leaf.N = {a: 0 for a in leaf.P}
                leaf.W = {a: 0.0 for a in leaf.P}
                leaf.expanded = True
                _backup(path, math.tanh(float(v)))   # match training scale
                c.sims += 1
        # budget spent -> pick a move
        for c in active:
            if c.sims < sims:
                continue
            acts = list(c.root.P)
            vv = np.array([c.root.N.get(a, 0) for a in acts], float)
            if vv.sum() == 0:
                vv += 1.0
            pi = vv / vv.sum()
            if c.g.players[c.g.current].turns < TEMP_TURNS:
                pe = (1 - DIR_EPS) * pi + DIR_EPS * dirich(c.rng, len(acts))
                a = acts[int(np.searchsorted(np.cumsum(pe), c.rng.random()))]
            else:
                a = acts[int(np.argmax(pi))]
            full = np.zeros(N_ACTIONS, np.float32)
            for act, p in zip(acts, pi):
                full[act] = p
            c.records.append((np.asarray(c.g.observe_v3(), np.float32),
                              np.array(c.g.legal_actions(), bool),
                              full, c.g.current))
            c.g.step(a)
            c.root = Node()
            c.sims = 0
            c.decisions += 1
    return out


# ---- persistent worker pool (#6): spawn once, reuse across iterations ----
_WORKER = {}


def _init_worker():
    import torch as _t
    _t.set_num_threads(1)
    from train2.ppo3 import PolicyNetV3
    _WORKER["net"] = PolicyNetV3()        # built ONCE per worker lifetime


def _worker(args):
    (blob, n_games, sims, seed0, batch_games, curr_p, piles,
     league, league_p) = args
    net = _WORKER["net"]
    net.load_state_dict(blob)             # cheap: just refresh weights
    net.eval()
    return selfplay_batch(net, n_games, sims, seed0, batch_games, curr_p,
                          piles, league, league_p)


_POOL = {"pool": None, "size": 0}


def _get_pool(n):
    import multiprocessing as mp
    if _POOL["pool"] is not None and _POOL["size"] == n:
        return _POOL["pool"]
    if _POOL["pool"] is not None:
        try:
            _POOL["pool"].terminate(); _POOL["pool"].join()
        except Exception:
            pass
    ctx = mp.get_context("spawn")
    _POOL["pool"] = ctx.Pool(n, initializer=_init_worker)
    _POOL["size"] = n
    return _POOL["pool"]


def _drop_pool():
    if _POOL["pool"] is not None:
        try:
            _POOL["pool"].terminate(); _POOL["pool"].join()
        except Exception:
            pass
    _POOL["pool"] = None
    _POOL["size"] = 0


def parallel_selfplay(net, n_games, sims, seed0, workers=0, batch_games=24,
                      curr_p=0.0, piles=10, league=None, league_p=0.0):
    if workers <= 1:
        return selfplay_batch(net, n_games, sims, seed0, batch_games, curr_p,
                              piles, league, league_p)
    import multiprocessing as mp
    blob = {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_games // workers + (1 if i < n_games % workers else 0)
           for i in range(workers)]
    tasks = [(blob, per[i], sims, seed0 + i * 10**6, batch_games, curr_p,
              piles, league, league_p)
             for i in range(workers) if per[i] > 0]
    games_per_task = max(per) if per else n_games
    # sims-aware timeout so honest slow workers are not executed
    timeout = 300 + games_per_task * 60 * max(1.0, sims / 100.0)
    pool = _get_pool(len(tasks))
    try:
        async_res = pool.map_async(_worker, tasks)
        res = async_res.get(timeout=timeout)
        return [g for r in res for g in r]
    except mp.TimeoutError:
        print(f"[batched3] pool timed out after {timeout:.0f}s; rebuilding "
              "pool and falling back to in-process for this iter", flush=True)
        _drop_pool()                      # next iter gets a fresh pool
        return selfplay_batch(net, n_games, sims, seed0 + 31337,
                              batch_games, curr_p, piles, league, league_p)