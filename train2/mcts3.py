"""Determinized PUCT MCTS for engine2 (port of train/mcts.py).
Hidden info (own draw order; opponent hand+draw split) re-randomized per
simulation; net supplies priors + leaf values."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from dominion.engine2 import Game, N_CARDS
from dominion.bots2 import Bot


class Node:
    __slots__ = ("to_move", "expanded", "P", "N", "W", "children")

    def __init__(self):
        self.to_move = None
        self.expanded = False
        self.P, self.N, self.W = {}, {}, {}
        self.children = {}


def _determinize(g: Game, me: int, rng: random.Random) -> None:
    rng.shuffle(g.players[me].draw)
    opp = g.players[1 - me]
    pool = list(opp.draw)
    for c in range(N_CARDS):
        pool.extend([c] * opp.hand[c])
    rng.shuffle(pool)
    hs = sum(opp.hand)
    opp.hand = [0] * N_CARDS
    for c in pool[:hs]:
        opp.hand[c] += 1
    opp.draw = pool[hs:]


def _outcome(g, me):
    return 0.0 if g.winner is None else (1.0 if g.winner == me else -1.0)


class MCTS:
    def __init__(self, net, n_sims=80, c_puct=1.5, seed=0,
                 root_noise=False):
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.root_noise = root_noise
        self.rng = random.Random(seed)

    @torch.no_grad()
    def _evaluate(self, g):
        obs = torch.tensor(g.observe_v3()[None])
        mask = torch.tensor(np.asarray(g.legal_actions())[None])
        logits, v = self.net(obs, mask)
        probs = torch.softmax(logits, -1)[0]
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        pri = {a: float(probs[a]) for a in legal}
        s = sum(pri.values()) or 1.0
        return {a: p / s for a, p in pri.items()}, math.tanh(float(v[0]))

    def _simulate(self, node, g):
        me = g.current
        if g.done:
            return _outcome(g, me)
        if not node.expanded:
            node.to_move = me
            node.P, v = self._evaluate(g)
            for a in node.P:
                node.N[a] = 0
                node.W[a] = 0.0
            node.expanded = True
            return v
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        tot = sum(node.N.get(a, 0) for a in legal)
        k = self.c_puct * math.sqrt(tot + 1)
        inv = 1.0 / len(legal)
        best_a, best = legal[0], -1e30
        for a in legal:
            n = node.N.get(a, 0)
            s = (node.W[a] / n if n else 0.0) + k * node.P.get(a, inv) / (1 + n)
            if s > best:
                best, best_a = s, a
        g.step(best_a)
        nxt = g.current
        child = node.children.get(best_a)
        if child is None:
            child = node.children[best_a] = Node()
        v_next = self._simulate(child, g)
        v_me = v_next if nxt == me else -v_next
        node.N[best_a] = node.N.get(best_a, 0) + 1
        node.W[best_a] = node.W.get(best_a, 0.0) + v_me
        return v_me

    def search(self, root_game):
        me = root_game.current
        root = Node()
        for i in range(self.n_sims):
            g = root_game.clone()
            g.rng = random.Random(self.rng.randrange(1 << 30))
            _determinize(g, me, g.rng)
            self._simulate(root, g)
            if i == 0 and root.expanded and self.root_noise:
                legal = list(root.P)
                xs = [self.rng.gammavariate(0.5, 1.0) for _ in legal]
                t = sum(xs) or 1.0
                eps = 0.25
                for a, x in zip(legal, xs):
                    root.P[a] = (1 - eps) * root.P[a] + eps * (x / t)
        vis = {a: root.N.get(a, 0) for a in root.P}
        if not vis:
            return next(a for a, ok in
                        enumerate(root_game.legal_actions()) if ok), {}
        return max(vis, key=vis.get), vis


class MCTSBot3(Bot):
    def __init__(self, net, n_sims=80, seed=0, root_noise=False):
        self.mcts = MCTS(net, n_sims=n_sims, seed=seed,
                         root_noise=root_noise)

    def act(self, g):
        return self.mcts.search(g)[0]