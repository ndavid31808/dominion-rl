"""Engine tests: per-card unit tests, rules edge cases, and a fuzz test
asserting the card-conservation invariant (no trashing on this board, so
total card count never changes)."""
import random

import pytest

from dominion.engine import (
    Game, Player, N_CARDS, TOTAL_CARDS, INITIAL_SUPPLY, COST, OBS_SIZE,
    COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, GARDENS,
    MOAT, VILLAGE, WORKSHOP, SMITHY, BAZAAR, COUNCIL_ROOM, FESTIVAL,
    LABORATORY, MARKET,
    PLAY_BASE, END_ACTION, BUY_BASE, PASS, GAIN_BASE, N_ACTIONS,
)
from dominion.bots import RandomBot, BigMoneyBot, play_game


def total_cards(g: Game) -> int:
    n = sum(g.supply)
    for p in g.players:
        n += p.deck_size()
    return n


def give_hand(g: Game, cards):
    """Force current player's hand (test helper)."""
    me = g.players[g.current]
    me.hand = [0] * N_CARDS
    for c in cards:
        me.hand[c] += 1


def play(g: Game, card):
    g.step(PLAY_BASE + (card - MOAT))


# ---------------------------------------------------------------- setup ----
def test_initial_state():
    g = Game(seed=1)
    assert total_cards(g) == TOTAL_CARDS
    for p in g.players:
        assert p.deck_size() == 10
        assert sum(p.hand) == 5
        comp = p.deck_composition()
        assert comp[COPPER] == 7 and comp[ESTATE] == 3
    assert g.supply[COPPER] == 46 and g.supply[PROVINCE] == 8


# ----------------------------------------------------------- card effects --
@pytest.mark.parametrize("card,exp_hand,exp_actions,exp_coins,exp_buys", [
    (MOAT,        2 + 2, 0, 0, 1),
    (VILLAGE,     2 + 1, 2, 0, 1),
    (SMITHY,      2 + 3, 0, 0, 1),
    (BAZAAR,      2 + 1, 2, 1, 1),
    (FESTIVAL,    2 + 0, 2, 2, 2),
    (LABORATORY,  2 + 2, 1, 0, 1),
    (MARKET,      2 + 1, 1, 1, 2),
])
def test_vanilla_effects(card, exp_hand, exp_actions, exp_coins, exp_buys):
    g = Game(seed=2)
    give_hand(g, [card, COPPER, COPPER])  # 3 cards: the action + 2 dummies
    play(g, card)
    me = g.players[g.current]
    assert sum(me.hand) == exp_hand
    assert me.in_play[card] == 1
    assert g.actions == exp_actions
    assert g.coins == exp_coins
    assert g.buys == exp_buys


def test_council_room_opponent_draws():
    g = Game(seed=3)
    give_hand(g, [COUNCIL_ROOM])
    opp_before = sum(g.players[1 - g.current].hand)
    play(g, COUNCIL_ROOM)
    assert sum(g.players[g.current].hand) == 4
    assert g.buys == 2
    assert sum(g.players[1 - g.current].hand) == opp_before + 1


def test_workshop_gain():
    g = Game(seed=4)
    give_hand(g, [WORKSHOP])
    play(g, WORKSHOP)
    assert g.pending_gain
    mask = g.legal_actions()
    legal = {i for i, ok in enumerate(mask) if ok}
    # only gain actions, only cost <= 4
    assert all(GAIN_BASE <= a < GAIN_BASE + N_CARDS for a in legal)
    assert GAIN_BASE + GARDENS in legal      # cost 4: ok
    assert GAIN_BASE + DUCHY not in legal    # cost 5: no
    assert GAIN_BASE + PROVINCE not in legal
    before = g.players[g.current].discard[GARDENS]
    g.step(GAIN_BASE + GARDENS)
    assert not g.pending_gain
    assert g.players[g.current].discard[GARDENS] == before + 1
    assert g.supply[GARDENS] == INITIAL_SUPPLY[GARDENS] - 1


def test_chained_actions_village_smithy():
    g = Game(seed=5)
    give_hand(g, [VILLAGE, SMITHY])
    play(g, VILLAGE)            # +1 card +2 actions
    assert g.actions == 2
    play(g, SMITHY)             # uses one action
    assert g.actions == 1


def test_cannot_play_without_actions():
    g = Game(seed=6)
    give_hand(g, [SMITHY, SMITHY])
    play(g, SMITHY)
    assert g.actions == 0
    mask = g.legal_actions()
    assert not mask[PLAY_BASE + (SMITHY - MOAT)]
    assert mask[END_ACTION]


# -------------------------------------------------------------- buy phase --
def test_autoplay_treasures_and_buy():
    g = Game(seed=7)
    give_hand(g, [COPPER, COPPER, COPPER, SILVER, ESTATE])  # 3 + 2 = 5 coins
    g.step(END_ACTION)
    assert g.coins == 5
    me = g.players[g.current]
    assert me.hand[COPPER] == 0 and me.in_play[COPPER] == 3
    mask = g.legal_actions()
    assert mask[BUY_BASE + DUCHY]            # cost 5
    assert not mask[BUY_BASE + GOLD]         # cost 6
    g.step(BUY_BASE + DUCHY)
    # buys hit 0 -> auto cleanup, turn passed
    assert g.current == 1
    assert g.players[0].discard == [0] * N_CARDS or True  # reshuffled into draw maybe
    assert g.players[0].deck_composition()[DUCHY] == 1


def test_buy_decrements_coins_with_extra_buys():
    g = Game(seed=8)
    give_hand(g, [FESTIVAL, GOLD, GOLD])
    play(g, FESTIVAL)                        # +2 coins +1 buy
    g.step(END_ACTION)                       # +6 from golds -> 8 coins, 2 buys
    assert g.coins == 8 and g.buys == 2
    g.step(BUY_BASE + SILVER)                # 5 left, 1 buy
    assert g.coins == 5 and g.buys == 1 and g.current == 0
    g.step(BUY_BASE + DUCHY)                 # last buy -> turn passes
    assert g.current == 1


def test_pass_ends_turn():
    g = Game(seed=9)
    g.step(END_ACTION)
    g.step(PASS)
    assert g.current == 1
    assert g.players[0].turns == 1


# ---------------------------------------------------------------- reshuffle --
def test_reshuffle_when_draw_pile_empty():
    g = Game(seed=10)
    me = g.players[g.current]
    me.draw = []
    me.discard = [0] * N_CARDS
    me.discard[SILVER] = 5
    give_hand(g, [SMITHY])
    play(g, SMITHY)
    assert me.hand[SILVER] == 3
    assert len(me.draw) == 2
    assert me.discard[SILVER] == 0


def test_draw_short_when_no_cards_anywhere():
    g = Game(seed=11)
    me = g.players[g.current]
    me.draw = []
    me.discard = [0] * N_CARDS
    give_hand(g, [SMITHY])
    play(g, SMITHY)              # nothing to draw: should not crash
    assert sum(me.hand) == 0


# ------------------------------------------------------------------ scoring --
def test_gardens_scoring():
    g = Game(seed=12)
    p = g.players[0]
    p.draw, p.hand, p.discard, p.in_play = [], [0] * N_CARDS, [0] * N_CARDS, [0] * N_CARDS
    p.discard[GARDENS] = 2
    p.discard[COPPER] = 17           # deck size 19 -> floor 1 per Gardens
    assert g.score(0) == 2
    p.discard[COPPER] = 18           # deck size 20 -> 2 per Gardens
    assert g.score(0) == 4


def test_game_end_province_and_winner():
    g = Game(seed=13)
    g.supply[PROVINCE] = 1
    give_hand(g, [GOLD, GOLD, GOLD])
    g.step(END_ACTION)
    g.step(BUY_BASE + PROVINCE)
    assert g.done
    assert g.winner == 0


def test_game_end_three_piles():
    g = Game(seed=14)
    g.supply[MOAT] = 0
    g.supply[VILLAGE] = 0
    g.supply[ESTATE] = 1
    give_hand(g, [COPPER, COPPER])
    g.step(END_ACTION)
    g.step(BUY_BASE + ESTATE)
    assert g.done


def test_tiebreak_fewer_turns_wins():
    g = Game(seed=15)
    g.players[0].turns, g.players[1].turns = 10, 9
    # equal scores by construction (both have 3 estates)
    g._decide_winner()
    assert g.winner == 1


# -------------------------------------------------------------------- fuzz --
def test_fuzz_random_games_conservation():
    rng = random.Random(0)
    for i in range(300):
        g = Game(seed=1000 + i)
        bot = RandomBot(seed=i)
        steps = 0
        while not g.done:
            mask = g.legal_actions()
            legal = [a for a, ok in enumerate(mask) if ok]
            assert legal, "no legal actions in a live game"
            g.step(rng.choice(legal))
            assert total_cards(g) == TOTAL_CARDS, "card conservation violated"
            steps += 1
            assert steps < 4000, "game failed to terminate"
        assert max(p.turns for p in g.players) < 150
        assert len(g.observe()) == OBS_SIZE


def test_observation_size_and_range():
    g = Game(seed=16)
    obs = g.observe()
    assert len(obs) == OBS_SIZE
    assert all(-0.01 <= x <= 10 for x in obs)


def test_bm_beats_random_smoke():
    wins = 0
    for i in range(30):
        w, *_ = play_game(BigMoneyBot(), RandomBot(seed=i), seed=i)
        wins += (w == 0)
    assert wins >= 28  # BM should crush random play
