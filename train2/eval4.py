"""eval4: instrumentation-only evaluation for the v4 stack.

Scripted bots are THERMOMETERS. Per the tabula-rasa protocol, nothing
here should drive training or checkpoint selection -- selection belongs
to within-lineage play (run train2.elo with ELO_RUN_DIR=<run dir>).
"""
from __future__ import annotations

import numpy as np
import torch

from dominion.engine2 import Game
from dominion.bots2 import (Bot, RandomBot, BigMoneyBot, BMSmithyBot,
                            GardensRushBot, ChapelMoneyBot)


class NetBot4(Bot):
    def __init__(self, net, greedy=True):
        self.net = net
        self.greedy = greedy

    @torch.no_grad()
    def act(self, g: Game) -> int:
        obs = torch.tensor(g.observe_v3()[None])
        mask = torch.tensor(np.asarray(g.legal_actions())[None])
        logits, _ = self.net(obs, mask)
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())


EVAL_BOTS = [("Random", lambda: RandomBot(seed=7)),
             ("BigMoney", BigMoneyBot), ("BM+Smithy", BMSmithyBot),
             ("Rush", GardensRushBot), ("ChapelMoney", ChapelMoneyBot)]


def evaluate(net, n_games=100, seed=0):
    me = NetBot4(net, greedy=True)
    out = {}
    for name, mk in EVAL_BOTS:
        w = 0
        for i in range(n_games):
            g = Game(seed=seed + i)
            seat = i % 2
            bots = [me, mk()] if seat == 0 else [mk(), me]
            steps = 0
            while not g.done:
                g.step(bots[g.current].act(g))
                steps += 1
                if steps > 6000:
                    break
            w += g.winner == seat
        out[name] = w / n_games
    return out
