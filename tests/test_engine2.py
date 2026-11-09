"""engine2 tests: every new mechanic hand-verified + random-kingdom fuzz
with the trash-inclusive conservation invariant."""
import random

import numpy as np
import pytest

from dominion.engine2 import (
    Game, N_CARDS, N_ACTIONS, OBS_SIZE_V3, KINGDOM_POOL, BASE_PILES, COST,
    COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, GARDENS,
    VILLAGE, WORKSHOP, SMITHY, MARKET, WITCH, CHARLATAN, JESTER, CURSE,
    CHAPEL, UPGRADE, WORKERS_VILLAGE, MONUMENT, CARAVAN,
    PLAY_BASE, END_ACTION, BUY_BASE, PASS, GAIN_BASE, TRASH_BASE, STOP_TRASH,
    P_NONE, P_TRASH_CHAPEL, P_TRASH_UPGRADE, P_GAIN_UPGRADE,
)

FULL_K = [GARDENS, VILLAGE, WORKSHOP, SMITHY, CHAPEL, UPGRADE,
          WORKERS_VILLAGE, MONUMENT, CARAVAN, WITCH]   # fixed kingdom for unit tests


def total(g):
    n = sum(g.supply) + sum(g.trash)
    for p in g.players:
        n += p.deck_size()
    return n


def hand(g, cards):
    """Replace current hand (test fixture). Recomputes the game's card total
    since this conjures/destroys cards by design."""
    me = g.players[g.current]
    me.hand = [0] * N_CARDS
    for c in cards:
        me.hand[c] += 1
    g.total_cards = total(g)


def play(g, card):
    g.step(PLAY_BASE + (card - VILLAGE))


def test_kingdom_sampling():
    seen = set()
    for i in range(30):
        g = Game(seed=i)
        assert len(g.kingdom) == 10
        assert all(c in KINGDOM_POOL for c in g.kingdom)
        for c in BASE_PILES:
            assert g.in_supply[c] and g.supply[c] > 0
        for c in KINGDOM_POOL:
            if c not in g.kingdom:
                assert not g.in_supply[c] and g.supply[c] == 0
        seen.add(tuple(g.kingdom))
    assert len(seen) > 15  # actually randomizing


def test_chapel_trash_up_to_four_and_stop():
    g = Game(seed=1, kingdom=FULL_K)
    hand(g, [CHAPEL, ESTATE, ESTATE, COPPER, COPPER, COPPER])
    play(g, CHAPEL)
    assert g.pending == P_TRASH_CHAPEL and g.pending_n == 4
    g.step(TRASH_BASE + ESTATE)
    g.step(TRASH_BASE + ESTATE)
    assert g.trash[ESTATE] == 2 and g.pending_n == 2
    g.step(STOP_TRASH)                       # optional early stop
    assert g.pending == P_NONE
    assert total(g) == g.total_cards         # conservation incl. trash


def test_chapel_caps_at_four():
    g = Game(seed=2, kingdom=FULL_K)
    hand(g, [CHAPEL] + [COPPER] * 5)
    play(g, CHAPEL)
    for _ in range(4):
        g.step(TRASH_BASE + COPPER)
    assert g.pending == P_NONE               # 4 max, auto-closes
    assert g.trash[COPPER] == 4
    assert g.players[g.current].hand[COPPER] == 1


def test_upgrade_trash_then_exact_gain():
    g = Game(seed=3, kingdom=FULL_K)
    hand(g, [UPGRADE, ESTATE, COPPER, COPPER])
    play(g, UPGRADE)                         # +1 card +1 action first
    me = g.players[g.current]
    assert g.actions == 1
    assert g.pending == P_TRASH_UPGRADE
    assert not g.legal_actions()[STOP_TRASH]  # mandatory
    g.step(TRASH_BASE + ESTATE)              # cost 2 -> gain exactly 3
    assert g.pending == P_GAIN_UPGRADE and g.pending_cost == 3
    mask = g.legal_actions()
    legal = {a for a, ok in enumerate(mask) if ok}
    assert GAIN_BASE + SILVER in legal       # cost 3
    assert GAIN_BASE + VILLAGE in legal      # cost 3, in kingdom
    assert GAIN_BASE + WORKSHOP in legal
    assert GAIN_BASE + COPPER not in legal
    g.step(GAIN_BASE + SILVER)
    assert me.discard[SILVER] == 1 and g.pending == P_NONE


def test_upgrade_gain_fizzles_when_no_pile_at_exact_cost():
    g = Game(seed=4, kingdom=FULL_K)
    hand(g, [UPGRADE, COPPER, COPPER])
    play(g, UPGRADE)
    g.step(TRASH_BASE + COPPER)              # cost 0 -> gain exactly 1: none
    assert g.pending == P_NONE
    assert g.trash[COPPER] == 1


def test_workers_village_vanilla():
    g = Game(seed=5, kingdom=FULL_K)
    hand(g, [WORKERS_VILLAGE, COPPER, COPPER])
    play(g, WORKERS_VILLAGE)
    assert g.actions == 2 and g.buys == 2
    assert sum(g.players[g.current].hand) == 2 + 1


def test_monument_vp_tokens_score():
    g = Game(seed=6, kingdom=FULL_K)
    me = g.players[g.current]
    hand(g, [MONUMENT])
    base = g.score(g.current)
    play(g, MONUMENT)
    assert g.coins == 2
    assert me.vp_tokens == 1
    assert g.score(g.current) == base + 1    # tokens count, deck unchanged


def test_caravan_duration_cycle():
    g = Game(seed=7, kingdom=FULL_K)
    me = g.players[0]
    hand(g, [CARAVAN, COPPER])
    play(g, CARAVAN)
    assert g.actions == 1
    assert me.in_play[CARAVAN] == 1
    size_before = me.deck_size()
    g.step(END_ACTION)
    g.step(PASS)                             # my cleanup: caravan STAYS out
    assert me.dur_active == 1
    assert me.in_play[CARAVAN] == 0
    assert me.deck_size() == size_before     # still owns it (set aside)
    # opponent's turn passes
    g.step(END_ACTION)
    g.step(PASS)
    # my next turn started: duration paid +1 card (hand=6), caravan in play
    assert g.current == 0
    assert me.dur_active == 0
    assert me.in_play[CARAVAN] == 1
    assert sum(me.hand) == 6
    g.step(END_ACTION)
    g.step(PASS)                             # now it discards normally
    assert me.deck_composition()[CARAVAN] == 1
    assert me.dur_active == 0


def test_three_pile_counts_only_active_piles():
    g = Game(seed=8, kingdom=FULL_K)
    # absent piles are 0 but must NOT count toward game end
    assert not g._game_over()
    g.supply[SMITHY] = 0
    g.supply[CHAPEL] = 0
    assert not g._game_over()                # 2 empty
    g.supply[VILLAGE] = 0
    assert g._game_over()                    # 3 empty active piles


def test_observe_v3_size_and_kingdom_indicator():
    g = Game(seed=9, kingdom=FULL_K)
    o = g.observe_v3()
    assert o.shape == (OBS_SIZE_V3,)
    assert np.isfinite(o).all()
    for c in range(N_CARDS):
        assert o[N_CARDS + c] == (1.0 if g.in_supply[c] else 0.0)


def test_hidden_split_leak_still_absent():
    g = Game(seed=10, kingdom=FULL_K)
    opp = g.players[1]
    opp.hand = [0] * N_CARDS
    opp.hand[GOLD] = 1
    opp.hand[COPPER] = 4
    opp.draw = [SILVER, COPPER, COPPER, ESTATE, ESTATE]
    o1 = g.observe_v3().copy()
    opp.hand[GOLD] = 0
    opp.hand[SILVER] = 1
    opp.draw = [GOLD, COPPER, COPPER, ESTATE, ESTATE]
    assert np.array_equal(o1, g.observe_v3())


def test_fuzz_random_kingdoms_conservation():
    rng = random.Random(0)
    for i in range(150):
        g = Game(seed=4000 + i)              # random kingdom each game
        steps = 0
        while not g.done:
            legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
            assert legal, f"deadlock (pending={g.pending})"
            g.step(rng.choice(legal))
            assert total(g) == g.total_cards
            steps += 1
            assert steps < 6000
        assert g.observe_v3().shape == (OBS_SIZE_V3,)
        for p in g.players:
            assert p.dur_active >= 0 and p.vp_tokens >= 0


# ----------------------------------------------------- junker expansion ----
def test_witch_curses_opponent():
    g = Game(seed=20, kingdom=FULL_K)        # FULL_K includes WITCH
    hand(g, [WITCH])
    opp = g.players[1]
    play(g, WITCH)
    assert sum(g.players[0].hand) == 2       # +2 cards
    assert opp.discard[CURSE] == 1
    assert g.supply[CURSE] == 9
    assert g.score(1) == 3 - 1               # 3 estates, -1 curse


def test_charlatan_curse_treasure_rule():
    K = sorted([GARDENS, VILLAGE, WORKSHOP, SMITHY, CHAPEL, UPGRADE,
                WORKERS_VILLAGE, MONUMENT, CARAVAN, CHARLATAN])
    g = Game(seed=21, kingdom=K)
    assert g.charlatan_rule
    hand(g, [CHARLATAN, CURSE, COPPER])
    play(g, CHARLATAN)
    assert g.coins == 3
    assert g.players[1].discard[CURSE] == 1
    g.step(END_ACTION)                       # autoplay: Curse worth $1 here
    assert g.coins == 3 + 1 + 1
    # rule OFF without Charlatan in kingdom
    g2 = Game(seed=22, kingdom=FULL_K)
    assert not g2.charlatan_rule
    hand(g2, [CURSE, COPPER])
    g2.step(END_ACTION)
    assert g2.coins == 1                     # curse worthless


def test_jester_victory_branch_gives_curse():
    g = Game(seed=23, kingdom=sorted([GARDENS, VILLAGE, WORKSHOP, SMITHY,
             CHAPEL, UPGRADE, WORKERS_VILLAGE, MONUMENT, CARAVAN, JESTER]))
    opp = g.players[1]
    opp.draw = [ESTATE]                      # top card = victory
    hand(g, [JESTER])
    play(g, JESTER)
    assert g.coins == 2
    assert opp.discard[ESTATE] == 1          # revealed+discarded
    assert opp.discard[CURSE] == 1           # cursed
    assert g.pending == P_NONE


def test_jester_copy_choice_both_ways():
    K = sorted([GARDENS, VILLAGE, WORKSHOP, SMITHY, CHAPEL, UPGRADE,
                WORKERS_VILLAGE, MONUMENT, CARAVAN, JESTER])
    from dominion.engine2 import JESTER_TO_ME, JESTER_TO_OPP, P_JESTER
    for choice, who in ((JESTER_TO_ME, 0), (JESTER_TO_OPP, 1)):
        g = Game(seed=24, kingdom=K)
        opp = g.players[1]
        opp.draw = [SILVER]                  # non-victory, pile available
        hand(g, [JESTER])
        play(g, JESTER)
        assert g.pending == P_JESTER and g.pending_card == SILVER
        legal = g.legal_actions()
        assert legal[JESTER_TO_ME] and legal[JESTER_TO_OPP]
        g.step(choice)
        assert g.players[who].discard[SILVER] == (1 if who == 0 else 2)  # opp: revealed + copy
        assert g.pending == P_NONE


def test_jester_empty_deck_reshuffles_or_noops():
    K = sorted([GARDENS, VILLAGE, WORKSHOP, SMITHY, CHAPEL, UPGRADE,
                WORKERS_VILLAGE, MONUMENT, CARAVAN, JESTER])
    g = Game(seed=25, kingdom=K)
    opp = g.players[1]
    opp.draw = []
    opp.discard = [0] * N_CARDS
    opp.discard[GOLD] = 1                    # reshuffles, reveals the Gold
    hand(g, [JESTER])
    play(g, JESTER)
    assert opp.discard[GOLD] == 1            # back in discard after reveal
    # truly nothing anywhere: no crash
    g2 = Game(seed=26, kingdom=K)
    o2 = g2.players[1]
    o2.draw = []; o2.discard = [0] * N_CARDS; o2.hand = [0] * N_CARDS
    hand(g2, [JESTER])
    play(g2, JESTER)
    assert g2.pending == P_NONE


def test_curse_pile_counts_for_three_pile_end():
    g = Game(seed=27, kingdom=FULL_K)
    g.supply[CURSE] = 0
    g.supply[CHAPEL] = 0
    g.supply[VILLAGE] = 0
    assert g._game_over()


# ------------------------------------------------- collection/remake/HG ----
from dominion.engine2 import (COLLECTION, REMAKE, HUNTING_GROUNDS, PLAYABLE,
                              P_TRASH_REMAKE, HG_DUCHY, HG_ESTATES)

K3 = sorted([GARDENS, VILLAGE, SMITHY, CHAPEL, UPGRADE, MONUMENT,
             COLLECTION, REMAKE, HUNTING_GROUNDS, WITCH])

def play2(g, card):
    g.step(PLAY_BASE + PLAYABLE.index(card))

def test_hunting_grounds_draw_and_on_trash():
    g = Game(seed=30, kingdom=K3)
    hand(g, [HUNTING_GROUNDS, COPPER])
    play2(g, HUNTING_GROUNDS)
    assert sum(g.players[0].hand) == 1 + 4           # +4 cards
    # trash it via Chapel -> on-trash choice fires
    g2 = Game(seed=31, kingdom=K3)
    hand(g2, [CHAPEL, HUNTING_GROUNDS])
    play2(g2, CHAPEL)
    g2.step(TRASH_BASE + HUNTING_GROUNDS)
    legal = g2.legal_actions()
    assert legal[HG_DUCHY] and legal[HG_ESTATES]
    g2.step(HG_ESTATES)
    assert g2.players[0].discard[ESTATE] == 3
    # resumes chapel afterwards (nothing left in hand -> closed)
    assert g2.pending == P_NONE

def test_hg_duchy_branch_and_chapel_resume():
    g = Game(seed=32, kingdom=K3)
    hand(g, [CHAPEL, HUNTING_GROUNDS, CURSE, CURSE])
    play2(g, CHAPEL)
    g.step(TRASH_BASE + HUNTING_GROUNDS)
    g.step(HG_DUCHY)
    assert g.players[0].discard[DUCHY] == 1
    # chapel still open (2 trashes used? no: 1 used, curses remain)
    assert g.pending == P_TRASH_CHAPEL
    g.step(TRASH_BASE + CURSE)
    g.step(STOP_TRASH)
    assert g.trash[CURSE] == 1

def test_remake_double_trash_exact_gain():
    g = Game(seed=33, kingdom=K3)
    hand(g, [REMAKE, ESTATE, ESTATE, COPPER])
    play2(g, REMAKE)
    assert g.pending == P_TRASH_REMAKE and g.remake_left == 2
    g.step(TRASH_BASE + ESTATE)              # cost 2 -> gain exactly 3
    g.step(GAIN_BASE + SILVER)
    assert g.pending == P_TRASH_REMAKE       # second round mandatory
    g.step(TRASH_BASE + ESTATE)
    g.step(GAIN_BASE + VILLAGE)
    assert g.pending == P_NONE and g.remake_left == 0
    me = g.players[0]
    assert me.discard[SILVER] == 1 and me.discard[VILLAGE] == 1
    assert g.trash[ESTATE] == 2

def test_remake_fizzle_continues_to_second_trash():
    g = Game(seed=34, kingdom=K3)
    hand(g, [REMAKE, COPPER, COPPER])
    play2(g, REMAKE)
    g.step(TRASH_BASE + COPPER)              # cost 0 -> $1: nothing, fizzle
    assert g.pending == P_TRASH_REMAKE       # but round 2 still due
    g.step(TRASH_BASE + COPPER)
    assert g.pending == P_NONE

def test_collection_treasure_and_vp_trigger():
    g = Game(seed=35, kingdom=K3)
    hand(g, [COLLECTION, GOLD])
    g.step(END_ACTION)                       # autoplay: $2 + $3, +1 buy
    assert g.coins == 5 and g.buys == 2
    me = g.players[0]
    tok = me.vp_tokens
    g.step(BUY_BASE + SMITHY)                # action gain w/ Collection out
    assert me.vp_tokens == tok + 1
    g.step(BUY_BASE + SILVER) if g.legal_actions()[BUY_BASE+SILVER] else g.step(PASS)
    # non-action gain must NOT trigger: check on fresh game
    g2 = Game(seed=36, kingdom=K3)
    hand(g2, [COLLECTION, GOLD, GOLD])
    g2.step(END_ACTION)
    t0 = g2.players[0].vp_tokens
    g2.step(BUY_BASE + SILVER)
    assert g2.players[0].vp_tokens == t0

def test_fuzz_20card_pool():
    import random as _r
    rng = _r.Random(1)
    for i in range(80):
        g = Game(seed=7000 + i)
        steps = 0
        while not g.done:
            legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
            assert legal, f"deadlock pending={g.pending}"
            g.step(rng.choice(legal))
            assert total(g) == g.total_cards
            steps += 1
            assert steps < 6000
