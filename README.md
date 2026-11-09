# Dominion RL

A from-scratch reinforcement-learning agent for the deck-building card game
**Dominion** (2-player), built on a custom rules engine and an
AlphaZero-style self-play loop. No human games or strategy rules are used in
training: the agent learns what every card is worth purely from self-play
outcomes.

The strongest agent (`models/v10.pt`, raw policy with no search) beats
scripted baselines on random kingdoms:

| Opponent | Win rate |
|---|---|
| Random | 100% |
| Big Money | 95% |
| Big Money + Smithy | 94% |
| Workshop/Gardens rush | 88% |
| Chapel + money | 97% |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q                 # 57 tests
python play_human.py models/v10.pt        # play against the agent in the terminal
```

## What's here

| Path | Contents |
|---|---|
| `dominion/engine2.py` | Rules engine: 10-of-20 random kingdoms, trashing, Curses/attacks, VP tokens, durations, on-trash triggers, 3-pile endings. Hidden-information observation (345 floats), 106-action masked action space. |
| `dominion/bots2.py` | Scripted baselines: Random, Big Money, BM+Smithy, Gardens rush, Chapel+money. |
| `train2/azero4.py`, `selfplay4.py`, `net4.py` | Main trainer: AlphaZero-style self-play with Gumbel root search, playout-cap randomization, whole-turn search, a league of past checkpoints, and a card-embedding + self-attention network with pointer action heads. |
| `train2/elo4.py`, `h2h4.py`, `eval4.py`, `monitor4.py` | Evaluation: Elo ladder, head-to-head watcher vs a target checkpoint, baseline evals, training monitor. |
| `train/`, `train2/ppo3.py`, `azero3.py`, `mcts3.py` | Earlier generations (PPO, bolt-on MCTS, first AlphaZero loops), kept for the record. |
| `dominion/engine.py`, `bots.py` | Original fixed-board engine. |
| `tests/` | Per-card unit tests, edge cases, card-conservation fuzzing, hidden-information leak test. |
| `scripts/` | One-off diagnostic probes. Run from the repo root. |
| `docs/DEVLOG.md` | Detailed development log of the earlier phases. |

### Train

```bash
python -m train2.azero4 --out runs/myrun --iters 600 --games-per-iter 64 \
    --batch-games 32 --sims 256 --workers 5 --turn-search --deep-sims 1024
python -m train2.h2h4 --run-dir runs/myrun --target models/v10.pt --boards 300
```

Runs are resumable from `<out>/latest.pt`. `--init` warm-starts a new run
from a checkpoint. See `python -m train2.azero4 --help` for exploration
options (curriculum openings, commitments, league, late-green weighting).

## Research findings

**Search is a policy-improvement operator, not an oracle.** Bolt-on MCTS with
identical weights beats the raw network 70–77%, but cannot fix matchups the
value network misjudges. Putting search inside training (AlphaZero) is what
improved play.

**The agent discovers immediate-payoff cards on its own** (Monument,
Charlatan, Witch, Gardens) within tens of iterations.

**It never discovers trashing.** Across a dozen experimental runs (x1–x11)
the agent essentially never buys Chapel, Upgrade or Remake, even on Witch
boards where expert players open Chapel every game. Things tried:

- Forced commitments to a kingdom card (bandit-chosen, told or hidden from the net)
- Chapel mirror games, where both seats must open the same card
- Deep search on opening buys (up to 4× budget)
- Training from scratch with deeper lookahead (x10)
- Over-sampling games where the winner bought Victory cards later than the loser (x11)

Diagnosis from probes:

- When the agent *does* play Chapel holding a Curse, it trashes the Curse
  100% of the time. The mechanics are learned.
- Forcing a trasher opening on Witch boards *lowers* its win rate by 9–21
  points. The network has never practised thin decks, so it misplays them
  and judges those positions as losing. The search uses that same network,
  so deeper search doesn't escape it.
- Self-play settles into a stable but poor equilibrium. 96% of x10's
  self-play games end on three piles (Duchy/Estate/Curse) with Provinces
  barely touched. Since no opponent in its training ever punishes that
  style, it is never forced out of it.

This is the classic exploration problem for delayed-payoff strategies, on a
laptop-scale budget (~10⁴ self-play games vs AlphaZero's ~10⁷).

## Requirements

Python 3.12, PyTorch 2.12, NumPy 2.4. Trained on a single laptop CPU.
