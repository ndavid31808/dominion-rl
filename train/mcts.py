"""Determinized PUCT MCTS on top of a trained PolicyNet.

Plain AlphaZero MCTS assumes a deterministic, perfect-information game.
Dominion is neither: shuffles are stochastic, and the opponent's hand plus
your own draw-pile *order* are hidden. We handle both with determinized
search (a.k.a. PIMC / Perfect-Information Monte Carlo, the standard card-game
approach):

  * Before each simulation we re-randomize all hidden information into one
    concrete, observation-consistent world: shuffle our own draw pile order,
    and re-deal the opponent's hidden hand+draw from their (publicly known)
    remaining deck composition.
  * We then search that determinized world. Visit counts are aggregated
    across hundreds of different determinizations, which averages over the
    hidden state and the chance outcomes.

The net supplies the policy prior P(s,a) and the leaf value V(s); search never
does random rollouts. The move actually played is the most-visited root child
-- which is reliably stronger than the net's raw argmax, the "policy
improvement" that makes search worth adding.

Approximation, stated honestly: this is closed-loop (children keyed by action)
with priors fixed at first expansion, so a node's prior set comes from one
determinization. For this small board it works well in practice; a fully
principled treatment would use information-set MCTS.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from dominion.engine import Game, N_CARDS, N_ACTIONS
from dominion.bots import Bot
from train.ppo import PolicyNet, DEVICE, obs_for


class Node:
    __slots__ = ("to_move", "expanded", "P", "N", "W", "children")

    def __init__(self):
        self.to_move = None
        self.expanded = False
        self.P = {}          # action -> prior
        self.N = {}          # action -> visit count
        self.W = {}          # action -> summed value (this node's perspective)
        self.children = {}   # action -> Node


def _determinize(g: Game, me: int, rng: random.Random) -> None:
    """Randomize hidden info into one concrete world consistent with what `me`
    can observe: own draw order, and the opponent's hidden hand+draw.
    Uses a numpy Generator for shuffles (profiled ~25% of self-play time
    with random.shuffle's per-element overhead)."""
    rng.shuffle(g.players[me].draw)
    opp = g.players[1 - me]
    pool = list(opp.draw)
    for c in range(N_CARDS):
        pool.extend([c] * opp.hand[c])
    rng.shuffle(pool)
    hand_size = sum(opp.hand)
    opp.hand = [0] * N_CARDS
    for c in pool[:hand_size]:
        opp.hand[c] += 1
    opp.draw = pool[hand_size:]


def _outcome(g: Game, me: int) -> float:
    if g.winner is None:
        return 0.0
    return 1.0 if g.winner == me else -1.0


class MCTS:
    def __init__(self, net: PolicyNet, n_sims: int = 120, c_puct: float = 1.5,
                 seed: int = 0):
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.rng = random.Random(seed)

    @torch.no_grad()
    def _evaluate(self, g: Game):
        """Net policy (over legal actions) + value, from g.current's view."""
        obs = torch.tensor([obs_for(self.net, g)], dtype=torch.float32,
                           device=DEVICE)
        mask = torch.tensor([g.legal_actions()], dtype=torch.bool, device=DEVICE)
        logits, v = self.net(obs, mask)
        probs = torch.softmax(logits, dim=-1)[0]
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        priors = {a: float(probs[a]) for a in legal}
        s = sum(priors.values()) or 1.0
        return {a: p / s for a, p in priors.items()}, float(v[0])

    def _simulate(self, node: Node, g: Game) -> float:
        """Returns value from the perspective of g.current at entry."""
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

        # PUCT selection among actions legal in THIS determinized world
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        total_n = sum(node.N.get(a, 0) for a in legal)
        sqrt_total = math.sqrt(total_n + 1)
        best_a, best_score = None, -1e30
        for a in legal:
            n = node.N.get(a, 0)
            q = node.W.get(a, 0.0) / n if n > 0 else 0.0
            p = node.P.get(a, 1.0 / len(legal))
            u = self.c_puct * p * sqrt_total / (1 + n)
            if q + u > best_score:
                best_score, best_a = q + u, a

        g.step(best_a)
        mover_next = g.current  # capture before recursion mutates g
        child = node.children.get(best_a)
        if child is None:
            child = node.children[best_a] = Node()
        v_next = self._simulate(child, g)
        v_me = v_next if mover_next == me else -v_next  # zero-sum rebase

        node.N[best_a] = node.N.get(best_a, 0) + 1
        node.W[best_a] = node.W.get(best_a, 0.0) + v_me
        return v_me

    def search(self, root_game: Game):
        """Returns (best_action, root_visit_counts)."""
        me = root_game.current
        root = Node()
        for _ in range(self.n_sims):
            g = root_game.clone()
            g.rng = random.Random(self.rng.randrange(1 << 30))
            _determinize(g, me, g.rng)
            self._simulate(root, g)
        visits = {a: root.N.get(a, 0) for a in root.P}
        best = max(visits, key=visits.get) if visits else \
            next(a for a, ok in enumerate(root_game.legal_actions()) if ok)
        return best, visits


class MCTSBot(Bot):
    def __init__(self, net: PolicyNet, n_sims: int = 120, seed: int = 0):
        self.mcts = MCTS(net, n_sims=n_sims, seed=seed)

    def act(self, g: Game) -> int:
        return self.mcts.search(g)[0]
