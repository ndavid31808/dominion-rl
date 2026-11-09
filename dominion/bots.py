"""Scripted baseline bots. These double as engine validation: their relative
strengths and game lengths are known quantities from Dominion theory."""
from __future__ import annotations

import random
from .engine import (
    Game, COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, GARDENS,
    WORKSHOP, SMITHY,
    PLAY_BASE, END_ACTION, BUY_BASE, PASS, GAIN_BASE, MOAT,
)


class Bot:
    def act(self, g: Game) -> int:
        raise NotImplementedError


class RandomBot(Bot):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, g: Game) -> int:
        legal = [i for i, ok in enumerate(g.legal_actions()) if ok]
        return self.rng.choice(legal)


def _buy_priority(g: Game, priorities) -> int:
    """Return the buy action for the first affordable, in-stock card in
    `priorities`, else PASS."""
    mask = g.legal_actions()
    for card in priorities:
        a = BUY_BASE + card
        if mask[a]:
            return a
    return PASS


class BigMoneyBot(Bot):
    """Province>=8, Gold>=6, Duchy>=5 once Provinces<=4, Silver>=3."""

    def act(self, g: Game) -> int:
        if g.pending_gain:  # never buys Workshop, but be safe
            mask = g.legal_actions()
            for card in (SILVER, ESTATE, COPPER):
                if mask[GAIN_BASE + card]:
                    return GAIN_BASE + card
            return [i for i, ok in enumerate(mask) if ok][0]
        if g.phase == 0:
            return END_ACTION
        prios = [PROVINCE, GOLD]
        if g.supply[PROVINCE] <= 4:
            prios.append(DUCHY)
        prios.append(SILVER)
        return _buy_priority(g, prios)


class BMSmithyBot(BigMoneyBot):
    """Big Money + up to 2 Smithies, played on sight."""

    def act(self, g: Game) -> int:
        me = g.players[g.current]
        if not g.pending_gain and g.phase == 0:
            if g.actions > 0 and me.hand[SMITHY] > 0:
                return PLAY_BASE + (SMITHY - MOAT)
            return END_ACTION
        if not g.pending_gain and g.phase == 1:
            owned = me.deck_composition()[SMITHY]
            prios = [PROVINCE, GOLD]
            if g.supply[PROVINCE] <= 4:
                prios.append(DUCHY)
            if owned < 2:
                prios.append(SMITHY)
            prios.append(SILVER)
            return _buy_priority(g, prios)
        return super().act(g)


class GardensRushBot(Bot):
    """Workshop/Gardens rush: spam Workshops, gain Gardens + junk, bloat the
    deck, and race to empty three piles (Gardens, Estates, Workshops)."""

    def act(self, g: Game) -> int:
        me = g.players[g.current]
        comp = me.deck_composition()
        mask = g.legal_actions()

        # Workshops first while ramping (each one compounds acquisition
        # speed AND drains a pile), then Gardens, then Estates, then Copper.
        if comp[WORKSHOP] < 5:
            prios = [WORKSHOP, GARDENS, ESTATE, COPPER]
        else:
            prios = [GARDENS, WORKSHOP, ESTATE, COPPER]

        if g.pending_gain:
            for card in prios:
                if mask[GAIN_BASE + card]:
                    return GAIN_BASE + card
            return [i for i, ok in enumerate(mask) if ok][0]

        if g.phase == 0:
            if g.actions > 0 and me.hand[WORKSHOP] > 0:
                return PLAY_BASE + (WORKSHOP - MOAT)
            return END_ACTION

        # A couple of early Silvers so $4 hands (Gardens) stay reachable.
        if comp[SILVER] < 2 and g.coins == 3 and mask[BUY_BASE + SILVER]:
            return BUY_BASE + SILVER
        return _buy_priority(g, prios)


def play_game(bot0: Bot, bot1: Bot, seed=None, max_steps=4000):
    """Returns (winner, turns_p0, turns_p1). winner in {0, 1, None}."""
    g = Game(seed=seed)
    bots = [bot0, bot1]
    steps = 0
    while not g.done:
        g.step(bots[g.current].act(g))
        steps += 1
        if steps > max_steps:
            raise RuntimeError("game failed to terminate")
    return g.winner, g.players[0].turns, g.players[1].turns


def matchup(bot_a, bot_b, n_games=400, base_seed=0):
    """Plays n_games with seats swapped halfway. Returns
    (win_rate_a, draw_rate, avg_turns)."""
    wins_a = draws = 0
    total_turns = 0
    half = n_games // 2
    for i in range(n_games):
        if i < half:
            w, t0, t1 = play_game(bot_a, bot_b, seed=base_seed + i)
            if w == 0:
                wins_a += 1
        else:
            w, t0, t1 = play_game(bot_b, bot_a, seed=base_seed + i)
            if w == 1:
                wins_a += 1
        if w is None:
            draws += 1
        total_turns += (t0 + t1) / 2
    return wins_a / n_games, draws / n_games, total_turns / n_games
