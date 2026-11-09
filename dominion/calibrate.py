"""Phase 2 calibration: run the matchups whose outcomes are known from
Dominion theory. If these numbers are off, the ENGINE is buggy — fix it
before starting RL.

Expected:
  BM vs Random        ~100% BM
  BM+Smithy vs BM     ~60-70% BM+Smithy
  avg game length     ~15-22 turns/player for money mirrors
"""
from dominion.bots import (RandomBot, BigMoneyBot, BMSmithyBot,
                           GardensRushBot, matchup)


def report(name_a, bot_a, name_b, bot_b, n=400):
    wr, dr, turns = matchup(bot_a, bot_b, n_games=n)
    print(f"{name_a:>12} vs {name_b:<12}  "
          f"win% {name_a}: {wr*100:5.1f}   draws: {dr*100:4.1f}%   "
          f"avg turns/player: {turns:5.1f}")


if __name__ == "__main__":
    print("Calibration (seat-balanced):\n")
    report("BM", BigMoneyBot(), "Random", RandomBot(seed=42))
    report("BM+Smithy", BMSmithyBot(), "BM", BigMoneyBot())
    report("Rush", GardensRushBot(), "BM", BigMoneyBot())
    report("BM+Smithy", BMSmithyBot(), "Rush", GardensRushBot())
    report("BM", BigMoneyBot(), "BM", BigMoneyBot())
