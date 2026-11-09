"""engine2: variable kingdoms from a 15-card pool.

Per game, 10 of 15 kingdom cards are dealt (KINGDOM_POOL below); base
treasures + Estate/Duchy/Province are always present. New mechanics over v1:

  Chapel   ($2)  trash up to 4 cards from hand (optional, sequential)
  Upgrade  ($5)  +1 Card +1 Action; trash a card, gain one costing exactly +1
  Worker's Village ($4) +1 Card +2 Actions +1 Buy (vanilla)
  Monument ($4)  +$2, +1 VP token (score outside the deck)
  Caravan  ($4)  +1 Card +1 Action; duration: +1 Card at start of next turn

State machine for sub-decisions: pending in {NONE, GAIN_WS, TRASH_CHAPEL,
TRASH_UPGRADE, GAIN_UPGRADE}. Trashed cards go to an explicit trash pile so
the conservation invariant becomes supply + all zones + trash == game total.

Action space (80):
  0..13   play action card (card index 7 + a)
  14      end Action phase (auto-plays treasures)
  15..35  buy card c
  36      pass / end Buy phase
  37..57  gain card c (Workshop: cost<=4; Upgrade: cost == trashed+1)
  58..78  trash card c from hand (Chapel: optional; Upgrade: mandatory)
  79      stop trashing (Chapel only)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------- cards ----
(COPPER, SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE, GARDENS,
 VILLAGE, WORKSHOP, SMITHY, BAZAAR, COUNCIL_ROOM, FESTIVAL,
 LABORATORY, MARKET, CHAPEL, UPGRADE, WORKERS_VILLAGE, MONUMENT,
 CARAVAN, CHARLATAN, WITCH, JESTER, COLLECTION, REMAKE,
 HUNTING_GROUNDS) = range(27)
N_CARDS = 27

NAMES = ["Copper", "Silver", "Gold", "Curse", "Estate", "Duchy", "Province",
         "Gardens", "Village", "Workshop", "Smithy", "Bazaar", "Council Room",
         "Festival", "Laboratory", "Market", "Chapel", "Upgrade",
         "Worker's Village", "Monument", "Caravan", "Charlatan", "Witch",
         "Jester", "Collection", "Remake", "Hunting Grounds"]

COST = [0, 3, 6, 0, 2, 5, 8, 4,
        3, 3, 4, 5, 5, 5, 5, 5, 2, 5, 4, 4, 4, 5, 5, 5, 5, 4, 6]

TREASURE_VALUE = {COPPER: 1, SILVER: 2, GOLD: 3}
VICTORY_CARDS = (ESTATE, DUCHY, PROVINCE, GARDENS)
PLAYABLE = list(range(VILLAGE, JESTER + 1)) + [REMAKE, HUNTING_GROUNDS]
ACTION_CARDS = PLAYABLE                                # 18 playable
KINGDOM_POOL = ([GARDENS] + list(range(VILLAGE, JESTER + 1))
                + [COLLECTION, REMAKE, HUNTING_GROUNDS])  # 20 dealable
BASE_PILES = [COPPER, SILVER, GOLD, CURSE, ESTATE, DUCHY, PROVINCE]

PILE_SIZE = {COPPER: 46, SILVER: 40, GOLD: 30, CURSE: 10,
             ESTATE: 8, DUCHY: 8, PROVINCE: 8, GARDENS: 8}
for _c in PLAYABLE + [COLLECTION]:
    PILE_SIZE[_c] = 10

# ------------------------------------------------------------- action ids --
PLAY_BASE = 0          # 0..17 -> PLAYABLE[a]
END_ACTION = 18
BUY_BASE = 19          # 19..45
PASS = 46
GAIN_BASE = 47         # 47..73
TRASH_BASE = 74        # 74..100
STOP_TRASH = 101
JESTER_TO_OPP = 102
JESTER_TO_ME = 103
HG_DUCHY = 104         # Hunting Grounds on-trash: gain a Duchy
HG_ESTATES = 105       # ... or 3 Estates
N_ACTIONS = 106

ACTION_PHASE, BUY_PHASE = 0, 1
(P_NONE, P_GAIN_WS, P_TRASH_CHAPEL, P_TRASH_UPGRADE, P_GAIN_UPGRADE,
 P_JESTER, P_TRASH_REMAKE, P_HG) = range(8)


@dataclass
class Player:
    draw: List[int] = field(default_factory=list)
    hand: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    discard: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    in_play: List[int] = field(default_factory=lambda: [0] * N_CARDS)
    dur_active: int = 0          # Caravans set aside until next turn start
    dur_new: int = 0             # Caravans PLAYED this turn (will set aside)
    vp_tokens: int = 0
    turns: int = 0
    top_discard: int = -1
    recent: List[int] = field(default_factory=list)

    def deck_composition(self) -> List[int]:
        comp = [0] * N_CARDS
        for c in self.draw:
            comp[c] += 1
        for i in range(N_CARDS):
            comp[i] += self.hand[i] + self.discard[i] + self.in_play[i]
        comp[CARAVAN] += self.dur_active
        return comp

    def deck_size(self) -> int:
        return (len(self.draw) + sum(self.hand) + sum(self.discard)
                + sum(self.in_play) + self.dur_active)

    def clone(self) -> "Player":
        p = Player()
        p.draw = list(self.draw)
        p.hand = list(self.hand)
        p.discard = list(self.discard)
        p.in_play = list(self.in_play)
        p.dur_active = self.dur_active
        p.dur_new = self.dur_new
        p.vp_tokens = self.vp_tokens
        p.turns = self.turns
        p.top_discard = self.top_discard
        p.recent = list(self.recent)
        return p


class Game:
    def __init__(self, seed: Optional[int] = None,
                 kingdom: Optional[List[int]] = None):
        self.rng = random.Random(seed)
        self.reset(kingdom)

    # ------------------------------------------------------------- setup --
    def reset(self, kingdom: Optional[List[int]] = None) -> None:
        if kingdom is None:
            kingdom = self.rng.sample(KINGDOM_POOL, 10)
        self.kingdom = sorted(kingdom)
        assert 1 <= len(self.kingdom) <= 10 and \
            all(c in KINGDOM_POOL for c in self.kingdom)
        self.in_supply = [False] * N_CARDS
        self.supply = [0] * N_CARDS
        for c in BASE_PILES + self.kingdom:
            self.in_supply[c] = True
            self.supply[c] = PILE_SIZE[c]
        self.total_cards = sum(self.supply) + 20
        self.trash = [0] * N_CARDS

        self.players = [Player(), Player()]
        for p in self.players:
            cards = [COPPER] * 7 + [ESTATE] * 3
            self.rng.shuffle(cards)
            p.draw = cards
            self._draw(p, 5)
        self.current = 0
        self.phase = ACTION_PHASE
        self.actions, self.buys, self.coins = 1, 1, 0
        self.pending = P_NONE
        self.pending_n = 0       # Chapel trashes remaining
        self.pending_cost = -1   # Upgrade exact-gain cost
        self.pending_card = -1   # Jester revealed card
        self.remake_left = 0     # Remake trashes remaining
        self.hg_resume = None    # pending state to restore after HG choice
        self.charlatan_rule = CHARLATAN in self.kingdom  # Curse = $1 treasure
        self.done = False
        self.winner: Optional[int] = None

    # --------------------------------------------------------- internals --
    def _draw(self, p: Player, n: int) -> None:
        for _ in range(n):
            if not p.draw:
                pile = []
                for c in range(N_CARDS):
                    pile.extend([c] * p.discard[c])
                    p.discard[c] = 0
                p.top_discard = -1
                if not pile:
                    return
                self.rng.shuffle(pile)
                p.draw = pile
            p.hand[p.draw.pop()] += 1

    def _gain(self, p: Player, c: int) -> None:
        self.supply[c] -= 1
        p.discard[c] += 1
        if c in _PLAYSET and p.in_play[COLLECTION] > 0:
            p.vp_tokens += p.in_play[COLLECTION]
        p.top_discard = c
        p.recent.append(c)
        if len(p.recent) > 3:
            del p.recent[0]

    def _play_effect(self, card: int) -> None:
        me = self.players[self.current]
        opp = self.players[1 - self.current]
        if card == VILLAGE:
            self._draw(me, 1); self.actions += 2
        elif card == WORKSHOP:
            self.pending = P_GAIN_WS
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
        elif card == CHAPEL:
            if sum(me.hand) > 0:
                self.pending = P_TRASH_CHAPEL
                self.pending_n = 4
        elif card == UPGRADE:
            self._draw(me, 1); self.actions += 1
            if sum(me.hand) > 0:
                self.pending = P_TRASH_UPGRADE
        elif card == WORKERS_VILLAGE:
            self._draw(me, 1); self.actions += 2; self.buys += 1
        elif card == MONUMENT:
            self.coins += 2; me.vp_tokens += 1
        elif card == CARAVAN:
            self._draw(me, 1); self.actions += 1
            me.dur_new += 1      # this copy stays out at cleanup
        elif card == WITCH:
            self._draw(me, 2)
            if self.supply[CURSE] > 0:
                self._gain(opp, CURSE)
        elif card == CHARLATAN:
            self.coins += 3
            if self.supply[CURSE] > 0:
                self._gain(opp, CURSE)
        elif card == REMAKE:
            if sum(me.hand) > 0:
                self.pending = P_TRASH_REMAKE
                self.remake_left = 2
        elif card == HUNTING_GROUNDS:
            self._draw(me, 4)
        elif card == JESTER:
            self.coins += 2
            t = self._reveal_and_discard_top(opp)
            if t is None:
                pass                          # nothing to reveal
            elif t in VICTORY_CARDS:
                if self.supply[CURSE] > 0:
                    self._gain(opp, CURSE)
            elif self.supply[t] > 0:          # copy available: chooser
                self.pending = P_JESTER
                self.pending_card = t
        else:
            raise ValueError(card)

    def _autoplay_treasures(self) -> None:
        me = self.players[self.current]
        vals = dict(TREASURE_VALUE)
        vals[COLLECTION] = 2         # Collection: $2 Treasure, +1 Buy
        if self.charlatan_rule:
            vals[CURSE] = 1          # Charlatan: Curses are $1 Treasures
        for t, v in vals.items():
            if t == COLLECTION:
                self.buys += me.hand[t]
            n = me.hand[t]
            if n:
                me.hand[t] = 0
                me.in_play[t] += n
                self.coins += v * n

    def _reveal_and_discard_top(self, p: Player) -> Optional[int]:
        """Jester: discard p's top deck card (reshuffling if needed)."""
        if not p.draw:
            pile = []
            for c in range(N_CARDS):
                pile.extend([c] * p.discard[c])
                p.discard[c] = 0
            p.top_discard = -1
            if not pile:
                return None
            self.rng.shuffle(pile)
            p.draw = pile
        t = p.draw.pop()
        p.discard[t] += 1
        p.top_discard = t
        return t

    def _cleanup_and_pass_turn(self) -> None:
        me = self.players[self.current]
        # Caravans PLAYED THIS TURN stay out; ones returned from duration
        # (already paid their +1 Card this turn) discard normally.
        staying = me.dur_new
        me.dur_new = 0
        me.in_play[CARAVAN] -= staying
        dumped = sum(me.hand) + sum(me.in_play)
        for c in range(N_CARDS):
            me.discard[c] += me.hand[c] + me.in_play[c]
            me.hand[c] = 0
            me.in_play[c] = 0
        me.dur_active += staying
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
        self.pending = P_NONE
        self.pending_n = 0
        self.pending_cost = -1
        self.pending_card = -1
        # duration payoff: start of the new current player's turn
        nxt = self.players[self.current]
        if nxt.dur_active:
            self._draw(nxt, nxt.dur_active)          # +1 Card per Caravan
            nxt.in_play[CARAVAN] += nxt.dur_active   # discard at THIS cleanup
            nxt.dur_active = 0

    def _game_over(self) -> bool:
        if self.supply[PROVINCE] == 0:
            return True
        empty = sum(1 for c in range(N_CARDS)
                    if self.in_supply[c] and self.supply[c] == 0)
        return empty >= 3

    def _decide_winner(self) -> None:
        s0, s1 = self.score(0), self.score(1)
        if s0 != s1:
            self.winner = 0 if s0 > s1 else 1
            return
        t0, t1 = self.players[0].turns, self.players[1].turns
        self.winner = (0 if t0 < t1 else 1) if t0 != t1 else None

    # ------------------------------------------------------------ public --
    def score(self, idx: int) -> int:
        p = self.players[idx]
        comp = p.deck_composition()
        return (comp[ESTATE] + 3 * comp[DUCHY] + 6 * comp[PROVINCE]
                - comp[CURSE]
                + comp[GARDENS] * (p.deck_size() // 10) + p.vp_tokens)

    def legal_actions(self) -> List[bool]:
        mask = [False] * N_ACTIONS
        if self.done:
            return mask
        me = self.players[self.current]

        if self.pending == P_GAIN_WS:
            any_g = False
            for c in range(N_CARDS):
                if self.in_supply[c] and self.supply[c] > 0 and COST[c] <= 4:
                    mask[GAIN_BASE + c] = True
                    any_g = True
            if not any_g:
                mask[STOP_TRASH] = True   # degenerate escape hatch
            return mask
        if self.pending == P_TRASH_CHAPEL:
            for c in range(N_CARDS):
                if me.hand[c] > 0:
                    mask[TRASH_BASE + c] = True
            mask[STOP_TRASH] = True
            return mask
        if self.pending in (P_TRASH_UPGRADE, P_TRASH_REMAKE):
            for c in range(N_CARDS):
                if me.hand[c] > 0:
                    mask[TRASH_BASE + c] = True
            return mask                    # mandatory: no stop
        if self.pending == P_HG:
            mask[HG_DUCHY] = True
            mask[HG_ESTATES] = True
            return mask
        if self.pending == P_GAIN_UPGRADE:
            for c in range(N_CARDS):
                if self.in_supply[c] and self.supply[c] > 0 \
                        and COST[c] == self.pending_cost:
                    mask[GAIN_BASE + c] = True
            return mask                    # caller guarantees non-empty
        if self.pending == P_JESTER:
            mask[JESTER_TO_OPP] = True
            mask[JESTER_TO_ME] = True
            return mask

        if self.phase == ACTION_PHASE:
            if self.actions > 0:
                for i, card in enumerate(PLAYABLE):
                    if me.hand[card] > 0:
                        mask[PLAY_BASE + i] = True
            mask[END_ACTION] = True
        else:
            if self.buys > 0:
                for c in range(N_CARDS):
                    if self.in_supply[c] and self.supply[c] > 0 \
                            and COST[c] <= self.coins:
                        mask[BUY_BASE + c] = True
            mask[PASS] = True
        return mask

    def step(self, action: int) -> None:
        assert not self.done, "game over"
        assert self.legal_actions()[action], \
            f"illegal {action} ({describe_action(action)})"
        me = self.players[self.current]

        if TRASH_BASE <= action < TRASH_BASE + N_CARDS:
            c = action - TRASH_BASE
            me.hand[c] -= 1
            self.trash[c] += 1
            if self.pending == P_TRASH_CHAPEL:
                self.pending_n -= 1
                if self.pending_n == 0 or sum(me.hand) == 0:
                    self.pending = P_NONE
            elif self.pending == P_TRASH_REMAKE:
                self.remake_left -= 1
                tgt = COST[c] + 1
                if any(self.in_supply[x] and self.supply[x] > 0
                       and COST[x] == tgt for x in range(N_CARDS)):
                    self.pending = P_GAIN_UPGRADE
                    self.pending_cost = tgt
                elif self.remake_left > 0 and sum(me.hand) > 0:
                    self.pending = P_TRASH_REMAKE
                else:
                    self.pending = P_NONE
                    self.remake_left = 0
            else:                                   # Upgrade
                tgt = COST[c] + 1
                if any(self.in_supply[x] and self.supply[x] > 0
                       and COST[x] == tgt for x in range(N_CARDS)):
                    self.pending = P_GAIN_UPGRADE
                    self.pending_cost = tgt
                else:
                    self.pending = P_NONE           # gain fizzles
            if c == HUNTING_GROUNDS:                # on-trash trigger
                self.hg_resume = (self.pending, self.pending_n,
                                  self.pending_cost, self.remake_left)
                self.pending = P_HG
        elif action == STOP_TRASH:
            self.pending = P_NONE
            self.pending_n = 0
        elif action in (JESTER_TO_OPP, JESTER_TO_ME):
            target = self.players[1 - self.current] \
                if action == JESTER_TO_OPP else me
            self._gain(target, self.pending_card)
            self.pending = P_NONE
            self.pending_card = -1
        elif action in (HG_DUCHY, HG_ESTATES):
            if action == HG_DUCHY:
                if self.supply[DUCHY] > 0:
                    self._gain(me, DUCHY)
            else:
                for _ in range(min(3, self.supply[ESTATE])):
                    self._gain(me, ESTATE)
            (self.pending, self.pending_n,
             self.pending_cost, self.remake_left) = self.hg_resume
            self.hg_resume = None
        elif GAIN_BASE <= action < GAIN_BASE + N_CARDS:
            self._gain(me, action - GAIN_BASE)
            self.pending_cost = -1
            if self.remake_left > 0 and sum(me.hand) > 0:
                self.pending = P_TRASH_REMAKE       # Remake's 2nd round
            else:
                self.pending = P_NONE
                self.remake_left = 0
        elif PLAY_BASE <= action < PLAY_BASE + len(PLAYABLE):
            card = PLAYABLE[action]
            me.hand[card] -= 1
            me.in_play[card] += 1
            self.actions -= 1
            self._play_effect(card)
        elif action == END_ACTION:
            self._autoplay_treasures()
            self.phase = BUY_PHASE
        elif BUY_BASE <= action < BUY_BASE + N_CARDS:
            c = action - BUY_BASE
            self.coins -= COST[c]
            self.buys -= 1
            self._gain(me, c)
            if self.buys == 0:
                self._cleanup_and_pass_turn()
        elif action == PASS:
            self._cleanup_and_pass_turn()
        else:
            raise ValueError(action)

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        g.kingdom = list(self.kingdom)
        g.in_supply = list(self.in_supply)
        g.supply = list(self.supply)
        g.total_cards = self.total_cards
        g.trash = list(self.trash)
        g.players = [p.clone() for p in self.players]
        for a in ("current", "phase", "actions", "buys", "coins", "pending",
                  "pending_n", "pending_cost", "pending_card", "remake_left",
                  "charlatan_rule", "done", "winner"):
            setattr(g, a, getattr(self, a))
        g.hg_resume = self.hg_resume
        for a in ():
            setattr(g, a, getattr(self, a))
        return g

    # ------------------------------------------------- observation (v3) --
    def observe_v3(self) -> np.ndarray:
        me = self.players[self.current]
        opp = self.players[1 - self.current]
        K = N_CARDS
        o = np.zeros(OBS_SIZE_V3, dtype=np.float32)
        o[0:K] = [self.supply[c] / max(PILE_SIZE.get(c, 1), 1)
                  for c in range(K)]
        o[K:2*K] = self.in_supply                       # kingdom indicator
        i = 2 * K
        o[i:i+K] = me.hand
        if me.draw:
            o[i+K:i+2*K] = np.bincount(me.draw, minlength=K)[:K]
        o[i+2*K:i+3*K] = me.discard
        o[i+3*K:i+4*K] = me.in_play
        o[i+4*K:i+5*K] = opp.discard
        o[i+5*K:i+6*K] = opp.in_play
        comp = opp.deck_composition()
        for c in range(K):
            o[i+6*K+c] = comp[c] - opp.discard[c] - opp.in_play[c]
        o[i:i+7*K] /= 10.0
        i += 7 * K
        if opp.top_discard >= 0:
            o[i + opp.top_discard] = 1.0
        i += K
        for c in me.recent:
            o[i + c] += 1 / 3
        i += K
        for c in opp.recent:
            o[i + c] += 1 / 3
        i += K
        o[i:i+21] = [
            float(self.charlatan_rule),
            sum(opp.hand) / 5.0, len(me.draw) / 15.0, len(opp.draw) / 15.0,
            me.deck_size() / 30.0, opp.deck_size() / 30.0,
            self.score(self.current) / 50.0,
            self.score(1 - self.current) / 50.0,
            me.vp_tokens / 10.0, opp.vp_tokens / 10.0,
            me.dur_active / 3.0, opp.dur_active / 3.0,
            self.coins / 10.0, self.actions / 5.0, self.buys / 5.0,
            float(self.phase), self.pending / 7.0,
            self.remake_left / 2.0,
            me.turns / 30.0, self.supply[PROVINCE] / 8.0,
            sum(self.trash) / 20.0,
        ]
        return o


OBS_SIZE_V3 = 27 * 12 + 21   # 345


_PLAYSET = set(PLAYABLE)


def describe_action(a: int) -> str:
    if a == HG_DUCHY:
        return "hunting grounds: gain Duchy"
    if a == HG_ESTATES:
        return "hunting grounds: gain 3 Estates"
    if a == JESTER_TO_OPP:
        return "jester: opponent gains copy"
    if a == JESTER_TO_ME:
        return "jester: I gain copy"
    if PLAY_BASE <= a < PLAY_BASE + len(PLAYABLE):
        return f"play {NAMES[PLAYABLE[a]]}"
    if a == END_ACTION:
        return "end action phase"
    if BUY_BASE <= a < BUY_BASE + N_CARDS:
        return f"buy {NAMES[a - BUY_BASE]}"
    if a == PASS:
        return "pass"
    if GAIN_BASE <= a < GAIN_BASE + N_CARDS:
        return f"gain {NAMES[a - GAIN_BASE]}"
    if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
        return f"trash {NAMES[a - TRASH_BASE]}"
    if a == STOP_TRASH:
        return "stop trashing"
    return f"?{a}"
