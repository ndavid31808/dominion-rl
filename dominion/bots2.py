"""Scripted baselines for engine2 (variable kingdoms). Bots check card
availability and degrade gracefully when their key cards weren't dealt."""
from __future__ import annotations

import random
from dominion.engine2 import (
    Game, N_CARDS, COST,
    COPPER, SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE, GARDENS,
    VILLAGE, WORKSHOP, SMITHY, CHAPEL,
    PLAY_BASE, END_ACTION, BUY_BASE, PASS, GAIN_BASE, TRASH_BASE, STOP_TRASH,
    P_NONE, P_GAIN_WS, P_TRASH_CHAPEL, P_TRASH_UPGRADE, P_GAIN_UPGRADE,
)


class Bot:
    def act(self, g: Game) -> int:
        raise NotImplementedError


class RandomBot(Bot):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, g):
        return self.rng.choice(
            [i for i, ok in enumerate(g.legal_actions()) if ok])


def _buy(g, prios):
    mask = g.legal_actions()
    for c in prios:
        if mask[BUY_BASE + c]:
            return BUY_BASE + c
    return PASS


def _handle_pending(g, gain_prios=(SILVER, ESTATE, COPPER)):
    """Generic fallback for sub-decisions a bot doesn't strategize about."""
    mask = g.legal_actions()
    if g.pending in (P_GAIN_WS, P_GAIN_UPGRADE):
        for c in gain_prios:
            if mask[GAIN_BASE + c]:
                return GAIN_BASE + c
        for a in range(N_CARDS):
            if mask[GAIN_BASE + a]:
                return GAIN_BASE + a
    if g.pending in (P_TRASH_CHAPEL,):
        return STOP_TRASH if mask[STOP_TRASH] else \
            next(i for i, ok in enumerate(mask) if ok)
    if g.pending == P_TRASH_UPGRADE:        # forced: trash cheapest junk
        for c in (COPPER, ESTATE):
            if mask[TRASH_BASE + c]:
                return TRASH_BASE + c
        return next(i for i, ok in enumerate(mask) if ok)
    return next(i for i, ok in enumerate(mask) if ok)


class BigMoneyBot(Bot):
    def act(self, g):
        if g.pending != P_NONE:
            return _handle_pending(g)
        if g.phase == 0:
            return END_ACTION
        prios = [PROVINCE, GOLD]
        if g.supply[PROVINCE] <= 4:
            prios.append(DUCHY)
        prios.append(SILVER)
        return _buy(g, prios)


class BMSmithyBot(BigMoneyBot):
    def act(self, g):
        me = g.players[g.current]
        if g.pending == P_NONE and g.phase == 0:
            if g.actions > 0 and me.hand[SMITHY] > 0:
                return PLAY_BASE + (SMITHY - VILLAGE)
            return END_ACTION
        if g.pending == P_NONE and g.phase == 1:
            owned = me.deck_composition()[SMITHY]
            prios = [PROVINCE, GOLD]
            if g.supply[PROVINCE] <= 4:
                prios.append(DUCHY)
            if g.in_supply[SMITHY] and owned < 2:
                prios.append(SMITHY)
            prios.append(SILVER)
            return _buy(g, prios)
        return super().act(g)


class GardensRushBot(Bot):
    """Workshop/Gardens rush; degrades to BigMoney if either pile is absent."""
    def __init__(self):
        self.bm = BigMoneyBot()

    def act(self, g):
        if not (g.in_supply[WORKSHOP] and g.in_supply[GARDENS]):
            return self.bm.act(g)
        me = g.players[g.current]
        comp = me.deck_composition()
        mask = g.legal_actions()
        prios = ([WORKSHOP, GARDENS, ESTATE, COPPER] if comp[WORKSHOP] < 5
                 else [GARDENS, WORKSHOP, ESTATE, COPPER])
        if g.pending in (P_GAIN_WS, P_GAIN_UPGRADE):
            for c in prios:
                if mask[GAIN_BASE + c]:
                    return GAIN_BASE + c
            return _handle_pending(g)
        if g.pending != P_NONE:
            return _handle_pending(g)
        if g.phase == 0:
            if g.actions > 0 and me.hand[WORKSHOP] > 0:
                return PLAY_BASE + (WORKSHOP - VILLAGE)
            return END_ACTION
        if comp[SILVER] < 2 and g.coins == 3 and mask[BUY_BASE + SILVER]:
            return BUY_BASE + SILVER
        return _buy(g, prios)


class ChapelMoneyBot(Bot):
    """The classic: open Chapel, trash down to thin money, then race
    Provinces. Degrades to BigMoney without Chapel."""
    def __init__(self):
        self.bm = BigMoneyBot()

    def act(self, g):
        if not g.in_supply[CHAPEL]:
            return self.bm.act(g)
        me = g.players[g.current]
        comp = me.deck_composition()
        mask = g.legal_actions()
        treasure = comp[COPPER] + 2 * comp[SILVER] + 3 * comp[GOLD]
        if g.pending == P_TRASH_CHAPEL:
            # trash Estates always; Coppers while total coin stock > 3
            if me.hand[ESTATE] > 0 and g.supply[PROVINCE] > 2:
                return TRASH_BASE + ESTATE
            if me.hand[COPPER] > 0 and treasure > 4:
                return TRASH_BASE + COPPER
            return STOP_TRASH
        if g.pending != P_NONE:
            return _handle_pending(g)
        if g.phase == 0:
            if g.actions > 0 and me.hand[CHAPEL] > 0 and \
                    (me.hand[ESTATE] > 0 or (me.hand[COPPER] > 0
                                             and treasure > 4)):
                return PLAY_BASE + (CHAPEL - VILLAGE)
            return END_ACTION
        prios = [PROVINCE, GOLD]
        if g.supply[PROVINCE] <= 4:
            prios.append(DUCHY)
        prios.append(SILVER)
        if comp[CHAPEL] == 0 and g.coins in (2, 3):
            return _buy(g, [CHAPEL] + prios)
        return _buy(g, prios)


def play_game(b0, b1, seed=None, kingdom=None, max_steps=6000):
    g = Game(seed=seed, kingdom=kingdom)
    bots = [b0, b1]
    steps = 0
    while not g.done:
        g.step(bots[g.current].act(g))
        steps += 1
        if steps > max_steps:
            raise RuntimeError("non-terminating")
    return g.winner, g


def matchup(a, b, n=300, base_seed=0, kingdom=None):
    wins = draws = 0
    for i in range(n):
        if i % 2 == 0:
            w, _ = play_game(a, b, seed=base_seed + i, kingdom=kingdom)
            wins += (w == 0)
        else:
            w, _ = play_game(b, a, seed=base_seed + i, kingdom=kingdom)
            wins += (w == 1)
        draws += (w is None)
    return wins / n, draws / n
