"""Phase 1 throughput: batched self-play.

Runs G games in lockstep. Each loop iteration advances every game by ONE
MCTS simulation; all simulations that need a network evaluation pool their
leaf states into a single batched forward pass. This amortizes the per-call
torch overhead (the v1 profile showed ~45% of self-play time in 1-observation
forwards) across the whole batch.

Also: `fast_clone` skips copying RNG state (the search re-seeds the clone
anyway), which the profile showed as pure waste.

Multiprocess scaling: `parallel_selfplay` shards games across worker
processes via multiprocessing.Pool. Each worker runs its own batched loop
with torch limited to 1 thread. On an N-core laptop expect ~(N-1)x on top
of batching; on a 1-core box it adds nothing (use workers=0).
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from dominion.engine import Game, N_ACTIONS
from train.mcts import _determinize, _outcome


class ANode:
    __slots__ = ("expanded", "to_move", "P", "N", "W", "children")

    def __init__(self):
        self.expanded = False
        self.to_move = None
        self.P = {}
        self.N = {}
        self.W = {}
        self.children = {}
from train.ppo import PolicyNet, PolicyNetV2, DEVICE, obs_for

TEMP_TURNS = 8
DIRICHLET_ALPHA = 0.6
DIRICHLET_EPS = 0.25
C_PUCT = 1.5


def fast_clone(g: Game) -> Game:
    """Game.clone() minus the RNG-state copy (search re-seeds immediately)."""
    c = Game.__new__(Game)
    c.rng = None  # set by caller
    c.supply = list(g.supply)
    c.players = [p.clone() for p in g.players]
    for attr in ("current", "phase", "actions", "buys", "coins",
                 "pending_gain", "done", "winner"):
        setattr(c, attr, getattr(g, attr))
    return c


def _select(node, legal) -> int:
    import math
    N, W, P = node.N, node.W, node.P
    total = 0
    for a in legal:
        total += N.get(a, 0)
    k = C_PUCT * math.sqrt(total + 1)
    best_a, best = legal[0], -1e30
    inv = 1.0 / len(legal)
    for a in legal:
        n = N.get(a, 0)
        s = (W[a] / n if n else 0.0) + k * P.get(a, inv) / (1 + n)
        if s > best:
            best, best_a = s, a
    return best_a


def _walk(root: Node, g: Game):
    """Iterative descent. Returns (path, leaf_node_or_None, terminal_value).
    path entries: (node, action, me, mover_next). If leaf_node is None the
    walk hit a terminal state and terminal_value is from the perspective of
    the player to move at that terminal (matching the recursive version)."""
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
            child = node.children[a] = ANode()
        path.append((node, a, me, g.current if not g.done else g.current))
        node = child


def _backup(path, v_leaf_perspective_next):
    """v is from the perspective of the mover at the deepest reached state."""
    v = v_leaf_perspective_next
    for node, a, me, mover_next in reversed(path):
        v_me = v if mover_next == me else -v
        node.N[a] = node.N.get(a, 0) + 1
        node.W[a] = node.W.get(a, 0.0) + v_me
        v = v_me


def _expand(node, priors, value, mover):
    node.to_move = mover
    node.P = priors
    node.N = {a: 0 for a in priors}
    node.W = {a: 0.0 for a in priors}
    node.expanded = True
    return value


def _dirichlet(rng, n):
    x = np.array([rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in range(n)])
    s = x.sum()
    return x / s if s > 0 else np.ones(n) / n


class _Ctx:
    __slots__ = ("g", "root", "sims", "records", "rng", "seed")

    def __init__(self, seed, rng_seed):
        self.g = Game(seed=seed)
        self.root = ANode()
        self.sims = 0
        self.records = []
        self.rng = random.Random(rng_seed)
        self.seed = seed


def selfplay_batch(net: PolicyNet, n_games: int, sims: int,
                   seed0: int, batch_games: int = 32,
                   temp_turns: int = TEMP_TURNS):
    """Returns list of (records, winner); records = [(obs, mask, pi, mover)]."""
    out: List[Tuple[list, Optional[int]]] = []
    master = random.Random(seed0)
    next_seed = seed0
    active: List[_Ctx] = []

    def spawn():
        nonlocal next_seed
        c = _Ctx(next_seed, master.randrange(1 << 30))
        next_seed += 1
        return c

    while len(out) < n_games:
        while len(active) < min(batch_games, n_games - len(out) - 0) \
                and len(active) + len(out) < n_games:
            active.append(spawn())

        # 1) fast-forward forced moves; settle finished games
        still = []
        for c in active:
            while not c.g.done:
                legal = [a for a, ok in enumerate(c.g.legal_actions()) if ok]
                if len(legal) == 1:
                    c.g.step(legal[0])
                else:
                    break
            if c.g.done:
                out.append((c.records, c.g.winner))
            else:
                still.append(c)
        active = still
        if not active:
            continue

        # 2) one simulation per game; pool the evaluations
        pending = []   # (ctx, path, leaf_node, obs, mask, mover)
        for c in active:
            sim = fast_clone(c.g)
            sim.rng = random.Random(c.rng.randrange(1 << 30))
            _determinize(sim, c.g.current, sim.rng)
            path, leaf, term_v = _walk(c.root, sim)
            if leaf is None:
                _backup(path, term_v)
                c.sims += 1
            else:
                pending.append((c, path, leaf,
                                obs_for(net, sim), sim.legal_actions(),
                                sim.current))

        if pending:
            obs = torch.tensor(np.array([p[3] for p in pending],
                                        dtype=np.float32), device=DEVICE)
            mask = torch.tensor(np.array([p[4] for p in pending],
                                         dtype=bool), device=DEVICE)
            with torch.no_grad():
                logits, vals = net(obs, mask)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for (c, path, leaf, _, m, mover), pr, v in \
                    zip(pending, probs, vals.cpu().numpy()):
                legal = [a for a, ok in enumerate(m) if ok]
                s = float(pr[legal].sum()) or 1.0
                priors = {a: float(pr[a]) / s for a in legal}
                _expand(leaf, priors, float(v), mover)
                _backup(path, float(v))
                c.sims += 1

        # 3) games whose search budget is spent: pick a move
        for c in active:
            if c.sims < sims:
                continue
            acts = list(c.root.P)
            vv = np.array([c.root.N.get(a, 0) for a in acts],
                          dtype=np.float64)
            if vv.sum() == 0:
                vv += 1.0
            pi = vv / vv.sum()
            turn = c.g.players[c.g.current].turns
            if turn < temp_turns:
                pe = (1 - DIRICHLET_EPS) * pi \
                     + DIRICHLET_EPS * _dirichlet(c.rng, len(acts))
                a = acts[int(np.searchsorted(np.cumsum(pe), c.rng.random()))]
            else:
                a = acts[int(np.argmax(pi))]
            full = np.zeros(N_ACTIONS, dtype=np.float32)
            for act, p in zip(acts, pi):
                full[act] = p
            c.records.append((np.asarray(obs_for(net, c.g),
                                         dtype=np.float32),
                              np.array(c.g.legal_actions(), dtype=bool),
                              full, c.g.current))
            c.g.step(a)
            c.root = ANode()
            c.sims = 0
    return out


# ---------------------------------------------------------------- workers --
def _worker(args):
    blob, n_games, sims, seed0, batch_games, v2 = args
    torch.set_num_threads(1)
    net = (PolicyNetV2() if v2 else PolicyNet()).to("cpu")
    net.load_state_dict(blob)
    net.eval()
    return selfplay_batch(net, n_games, sims, seed0, batch_games)


def parallel_selfplay(net: PolicyNet, n_games: int, sims: int, seed0: int,
                      workers: int, batch_games: int = 32):
    """Shard self-play across processes. workers=0 -> in-process batched."""
    if workers <= 1:
        return selfplay_batch(net, n_games, sims, seed0, batch_games)
    import multiprocessing as mp
    blob = {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_games // workers + (1 if i < n_games % workers else 0)
           for i in range(workers)]
    v2 = getattr(net, "obs_size", 108) == 190
    tasks = [(blob, per[i], sims, seed0 + i * 1_000_000, batch_games, v2)
             for i in range(workers) if per[i] > 0]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        results = pool.map(_worker, tasks)
    return [g for r in results for g in r]
