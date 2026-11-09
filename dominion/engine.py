"""Dominion engine — fixed vanilla board, 2 players.

Board: Moat, Village, Workshop, Smithy, Bazaar, Council Room, Festival,
Laboratory, Market, Gardens (+ Copper/Silver/Gold, Estate/Duchy/Province).

Design decisions (see v1 scope doc):
  * Flat masked action space, 43 discrete actions.
  * Treasures are auto-played when the Action phase ends (safe on this board:
    no card ever rewards holding treasures back).
  * Workshop is the only sub-decision, handled via a `pending_gain` flag that
    restricts legality to gain actions.
  * No trashing exists, so total card count is invariant (used by tests).
  * Game end (Province empty or any 3 piles empty) is checked at end of turn,
    per the official rules.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------
COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, GARDENS = range(7)
(MOAT, VILLAGE, WORKSHOP, SMITHY, BAZAAR,
 COUNCIL_ROOM, FESTIVAL, LABORATORY, MARKET) = range(7, 16)

N_CARDS = 16

NAMES = [
    "Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Gardens",
    "Moat", "Village", "Workshop", "Smithy", "Bazaar", "Council Room",
    "Festival", "Laboratory", "Market",
]

COST = [0, 3, 6, 2, 5, 8, 4,  # treasures + victory + gardens
        2, 3, 3, 4, 5, 5, 5, 5, 5]  # moat..market

TREASURE_VALUE = {COPPER: 1, SILVER: 2, GOLD: 3}
ACTION_CARDS = list(range(MOAT, MARKET + 1))  # 9 playable actions

INITIAL_SUPPLY = [46, 40, 30, 8, 8, 8, 8] + [10] * 9  # indexed by card

TOTAL_CARDS = sum(INITIAL_SUPPLY) + 2 * 10  # supply + two starting decks

# --------------------------------------------------------------------------
# Flat action space (43 actions)
#   0..8   play action card (Moat..Market  ->  card index 7 + a)
#   9      end Action phase (auto-plays all treasures)
#   10..25 buy card (a - 10)
#   26     end Buy phase / pass
#   27..42 Workshop gain card (a - 27), cost <= 4
# --------------------------------------------------------------------------
PLAY_BASE = 0
END_ACTION = 9
BUY_BASE = 10
PASS = 26
GAIN_BASE = 27
N_ACTIONS = 43

ACTION_PHASE, BUY_PHASE = 0, 1


@dataclass
class Player:
    draw: List[int] = field(default_factory=list)   # ordered; top of deck = end
    hand: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    discard: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    in_play: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    turns: int = 0
    top_discard: int = -1            # -1 = unknown/empty (post-cleanup mixing)
    recent: List[int] = field(default_factory=list)  # last 3 acquisitions

    def deck_composition(self) -> List[int]:
        comp = [0] * N_CARDS
        for c in self.draw:
            comp[c] += 1
        for i in range(N_CARDS):
            comp[i] += self.hand[i] + self.discard[i] + self.in_play[i]
        return comp

    def deck_size(self) -> int:
        return (len(self.draw) + sum(self.hand) + sum(self.discard)
                + sum(self.in_play))

    def clone(self) -> "Player":
        p = Player()
        p.draw = list(self.draw)
        p.hand = list(self.hand)
        p.discard = list(self.discard)
        p.in_play = list(self.in_play)
        p.turns = self.turns
        p.top_discard = self.top_discard
        p.recent = list(self.recent)
        return p


class Game:
    """Two-player game. Public API: reset / legal_actions / step / clone /
    score / observe. `step` always acts for `self.current`."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()

    # -- setup -------------------------------------------------------------
    def reset(self) -> None:
        self.supply = list(INITIAL_SUPPLY)
        self.players = [Player(), Player()]
        for p in self.players:
            cards = [COPPER] * 7 + [ESTATE] * 3
            self.rng.shuffle(cards)
            p.draw = cards
            self._draw(p, 5)
        self.current = 0
        self.phase = ACTION_PHASE
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.pending_gain = False
        self.done = False
        self.winner: Optional[int] = None  # 0 / 1 / None (draw or ongoing)

    # -- internals ----------------------------------------------------------
    def _draw(self, p: Player, n: int) -> None:
        for _ in range(n):
            if not p.draw:
                pile = []
                for c in range(N_CARDS):
                    pile.extend([c] * p.discard[c])
                    p.discard[c] = 0
                p.top_discard = -1
                if not pile:
                    return  # nothing left to draw
                self.rng.shuffle(pile)
                p.draw = pile
            p.hand[p.draw.pop()] += 1

    def _play_effect(self, card: int) -> None:
        me = self.players[self.current]
        opp = self.players[1 - self.current]
        if card == MOAT:
            self._draw(me, 2)
        elif card == VILLAGE:
            self._draw(me, 1); self.actions += 2
        elif card == WORKSHOP:
            self.pending_gain = True
        elif card == SMITHY:
            self._draw(me, 3)
        elif card == BAZAAR:
            self._draw(me, 1); self.actions += 2; self.coins += 1
        elif card == COUNCIL_ROOM:
            self._draw(me, 4); self.buys += 1; self._draw(opp, 1)
        elif card == FESTIVAL:
            self.actions += 2; self.buys += 1; self.coins += 2
        elif card == LABORATORY:
            self._draw(me, 2); self.actions += 1
        elif card == MARKET:
            self._draw(me, 1); self.actions += 1; self.buys += 1
            self.coins += 1
        else:
            raise ValueError(f"not a playable action card: {card}")

    def _gain(self, p: Player, c: int) -> None:
        """Move card c from supply to p's discard (publicly visible event)."""
        self.supply[c] -= 1
        p.discard[c] += 1
        p.top_discard = c
        p.recent.append(c)
        if len(p.recent) > 3:
            del p.recent[0]

    def _autoplay_treasures(self) -> None:
        me = self.players[self.current]
        for t, v in TREASURE_VALUE.items():
            n = me.hand[t]
            if n:
                me.hand[t] = 0
                me.in_play[t] += n
                self.coins += v * n

    def _cleanup_and_pass_turn(self) -> None:
        me = self.players[self.current]
        dumped = sum(me.hand) + sum(me.in_play)
        for c in range(N_CARDS):
            me.discard[c] += me.hand[c] + me.in_play[c]
            me.hand[c] = 0
            me.in_play[c] = 0
        if dumped:
            me.top_discard = -1
        self._draw(me, 5)
        me.turns += 1

        if self._game_over():
            self.done = True
            self._decide_winner()
            return

        self.current = 1 - self.current
        self.phase = ACTION_PHASE
        self.actions, self.buys, self.coins = 1, 1, 0
        self.pending_gain = False

    def _game_over(self) -> bool:
        if self.supply[PROVINCE] == 0:
            return True
        return sum(1 for n in self.supply if n == 0) >= 3

    def _decide_winner(self) -> None:
        s0, s1 = self.score(0), self.score(1)
        if s0 != s1:
            self.winner = 0 if s0 > s1 else 1
            return
        t0, t1 = self.players[0].turns, self.players[1].turns
        if t0 != t1:
            self.winner = 0 if t0 < t1 else 1  # fewer turns wins ties
        else:
            self.winner = None  # true draw

    # -- public ---------------------------------------------------------------
    def score(self, idx: int) -> int:
        p = self.players[idx]
        comp = p.deck_composition()
        return (comp[ESTATE] + 3 * comp[DUCHY] + 6 * comp[PROVINCE]
                + comp[GARDENS] * (p.deck_size() // 10))

    def legal_actions(self) -> List[bool]:
        mask = [False] * N_ACTIONS
        if self.done:
            return mask
        me = self.players[self.current]

        if self.pending_gain:  # Workshop: must gain a card costing <= 4
            any_gain = False
            for c in range(N_CARDS):
                if COST[c] <= 4 and self.supply[c] > 0:
                    mask[GAIN_BASE + c] = True
                    any_gain = True
            if not any_gain:           # degenerate: nothing gainable
                mask[PASS if self.phase == BUY_PHASE else END_ACTION] = True
            return mask

        if self.phase == ACTION_PHASE:
            if self.actions > 0:
                for card in ACTION_CARDS:
                    if me.hand[card] > 0:
                        mask[PLAY_BASE + (card - MOAT)] = True
            mask[END_ACTION] = True
        else:  # BUY_PHASE
            if self.buys > 0:
                for c in range(N_CARDS):
                    if self.supply[c] > 0 and COST[c] <= self.coins:
                        mask[BUY_BASE + c] = True
            mask[PASS] = True
        return mask

    def step(self, action: int) -> None:
        assert not self.done, "game is over"
        assert self.legal_actions()[action], \
            f"illegal action {action} ({describe_action(action)})"
        me = self.players[self.current]

        if GAIN_BASE <= action < GAIN_BASE + N_CARDS:      # Workshop gain
            c = action - GAIN_BASE
            self._gain(me, c)
            self.pending_gain = False
        elif PLAY_BASE <= action < PLAY_BASE + 9:           # play action card
            card = MOAT + action
            me.hand[card] -= 1
            me.in_play[card] += 1
            self.actions -= 1
            self._play_effect(card)
        elif action == END_ACTION:
            self._autoplay_treasures()
            self.phase = BUY_PHASE
        elif BUY_BASE <= action < BUY_BASE + N_CARDS:       # buy
            c = action - BUY_BASE
            self.coins -= COST[c]
            self.buys -= 1
            self._gain(me, c)
            if self.buys == 0:
                self._cleanup_and_pass_turn()
        elif action == PASS:
            self._cleanup_and_pass_turn()
        else:
            raise ValueError(f"bad action id {action}")

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        g.supply = list(self.supply)
        g.players = [p.clone() for p in self.players]
        for attr in ("current", "phase", "actions", "buys", "coins",
                     "pending_gain", "done", "winner"):
            setattr(g, attr, getattr(self, attr))
        return g

    # -- observation (for RL, current player's perspective) -----------------
    def observe(self) -> List[float]:
        me = self.players[self.current]
        opp = self.players[1 - self.current]
        o = np.empty(108, dtype=np.float32)
        o[0:16] = self.supply
        o[0:16] /= _SUPPLY_NORM
        o[16:32] = me.hand
        draw_comp = np.bincount(me.draw, minlength=N_CARDS)[:N_CARDS] \
            if me.draw else _ZERO16
        o[32:48] = draw_comp
        o[48:64] = me.discard
        o[64:80] = me.in_play
        opp_draw = np.bincount(opp.draw, minlength=N_CARDS)[:N_CARDS] \
            if opp.draw else _ZERO16
        o[80:96] = opp_draw
        o[80:96] += opp.hand
        o[80:96] += opp.discard
        o[80:96] += opp.in_play
        o[16:96] /= 10.0
        o[96] = sum(opp.hand) / 5.0
        o[97] = me.deck_size() / 30.0
        o[98] = opp.deck_size() / 30.0
        o[99] = self.score(self.current) / 50.0
        o[100] = self.score(1 - self.current) / 50.0
        o[101] = self.coins / 10.0
        o[102] = self.actions / 5.0
        o[103] = self.buys / 5.0
        o[104] = float(self.phase)
        o[105] = float(self.pending_gain)
        o[106] = me.turns / 30.0
        o[107] = self.supply[PROVINCE] / 8.0
        return o


_SUPPLY_NORM = np.maximum(np.array(INITIAL_SUPPLY, dtype=np.float32), 1.0)
_ZERO16 = np.zeros(16, dtype=np.float32)


def _observe_v2(self) -> "np.ndarray":
    """190 floats. Exposes exactly what a perfect-memory player could know:
    opponent discard composition (public events) and the COMBINED hand+draw
    pool -- never the draw/hand split, which is hidden in real play."""
    me = self.players[self.current]
    opp = self.players[1 - self.current]
    o = np.zeros(190, dtype=np.float32)
    o[0:16] = self.supply
    o[0:16] /= _SUPPLY_NORM
    o[16:32] = me.hand
    if me.draw:
        o[32:48] = np.bincount(me.draw, minlength=N_CARDS)[:N_CARDS]
    o[48:64] = me.discard
    o[64:80] = me.in_play
    o[80:96] = opp.discard
    o[96:112] = opp.in_play
    # hidden pool = total deck - discard - in_play  (= hand + draw, unsplit)
    comp = opp.deck_composition()
    for c in range(N_CARDS):
        o[112 + c] = comp[c] - opp.discard[c] - opp.in_play[c]
    o[16:128] /= 10.0
    if opp.top_discard >= 0:
        o[128 + opp.top_discard] = 1.0
    for c in me.recent:
        o[144 + c] += 1.0 / 3.0
    for c in opp.recent:
        o[160 + c] += 1.0 / 3.0
    s = 176
    o[s + 0] = sum(opp.hand) / 5.0          # public: hand size
    o[s + 1] = len(me.draw) / 15.0          # cards until own reshuffle
    o[s + 2] = len(opp.draw) / 15.0         # public: their pile size
    o[s + 3] = me.deck_size() / 30.0
    o[s + 4] = opp.deck_size() / 30.0
    o[s + 5] = self.score(self.current) / 50.0
    o[s + 6] = self.score(1 - self.current) / 50.0
    o[s + 7] = self.coins / 10.0
    o[s + 8] = self.actions / 5.0
    o[s + 9] = self.buys / 5.0
    o[s + 10] = float(self.phase)
    o[s + 11] = float(self.pending_gain)
    o[s + 12] = me.turns / 30.0
    o[s + 13] = self.supply[PROVINCE] / 8.0
    return o


Game.observe_v2 = _observe_v2
OBS_SIZE_V2 = 190


OBS_SIZE = 108


def describe_action(a: int) -> str:
    if PLAY_BASE <= a < PLAY_BASE + 9:
        return f"play {NAMES[MOAT + a]}"
    if a == END_ACTION:
        return "end action phase"
    if BUY_BASE <= a < BUY_BASE + N_CARDS:
        return f"buy {NAMES[a - BUY_BASE]}"
    if a == PASS:
        return "pass"
    if GAIN_BASE <= a < GAIN_BASE + N_CARDS:
        return f"gain {NAMES[a - GAIN_BASE]} (Workshop)"
    return f"?{a}"
