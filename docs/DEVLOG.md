# Dominion RL — Engine (Phase 1 + 2)

2-player Dominion engine on the fixed vanilla board: **Moat, Village,
Workshop, Smithy, Bazaar, Council Room, Festival, Laboratory, Market,
Gardens** (+ Copper/Silver/Gold, Estate/Duchy/Province).

## Layout
```
dominion/engine.py     core engine (Game class, ~330 lines)
dominion/bots.py       scripted baselines: Random, BigMoney, BM+Smithy, GardensRush
dominion/calibrate.py  known-result matchups (engine validation)
tests/test_engine.py   24 tests: per-card units, edge cases, conservation fuzz
```

## Run
```bash
pip install pytest
python -m pytest tests/ -q          # 24 passed
python -m dominion.calibrate        # seat-balanced matchup table
```

## Design decisions
- **Flat masked action space, 43 actions:** 0–8 play action card, 9 end
  Action phase, 10–25 buy, 26 pass, 27–42 Workshop gain. One change from the
  scope doc's 44: Gardens isn't playable, so play actions are 9 not 10.
- **Treasures auto-play** when the Action phase ends (no card on this board
  rewards holding them back).
- **Workshop** = `pending_gain` flag restricting legality to gain actions.
- **Conservation invariant:** no trashing exists, so total card count is
  constant; the fuzz test asserts it after every step of 300 random games.
- `Game.observe()` returns a 108-float vector (current-player perspective)
  ready for Phase 3. `Game.clone()` exists for a future MCTS layer.

## Calibration results (400 games each, seats swapped halfway)
| Matchup | Win % | Avg turns/player |
|---|---|---|
| BM vs Random | 100.0 | 25.8 |
| BM+Smithy vs BM | 68.2 | 16.3 |
| Rush vs BM | 98.0 | 19.1 |
| Rush vs BM+Smithy | 79.5 | 19.1 |
| BM mirror | 42.8 / 38.7 / 18.5 draw | 17.3 |

First three rows match Dominion theory (BM+Smithy over BM at 60–70% is the
classic figure). The rush numbers are the headline: on this board a scripted
Workshop/Gardens rush, after one tuning pass (take Workshops before Gardens
while ramping — acquisition compounds), three-piles around turn 17–19 and
dominates money strategies. **Implication for Phase 3:** beating the rush
bot, not BM, is the real success bar; watch whether the RL agent learns to
contest piles, out-rush, or find a faster money line.

## Known simplifications (documented, deliberate)
- Treasures auto-played; Moat's reaction text inert (no attacks on board).
- Engine is unoptimized pure Python (~roughly 300+ games/sec random play);
  profile before Phase 3 if training throughput becomes the bottleneck.

## MCTS (search on top of the trained net)

`train/mcts.py` adds determinized PUCT search using the trained net for
priors + leaf values (no random rollouts). Hidden info and shuffles are
handled by re-randomizing the unknown card order each simulation and
averaging over determinizations (PIMC).

Results with `best.pt`, 80 sims/move (same weights throughout):

| Opponent | Raw net (greedy) | MCTS(80) |
|---|---|---|
| MCTS vs raw net (head-to-head) | — | **70% win** |
| BigMoney | 91% | 92% |
| Rush | 17% | 21% |

**Interpretation — this is the point of the whole exercise.** MCTS beats the
raw net 70% using *identical weights*: search is a policy-improvement
operator, refining the net's instincts with lookahead. But it's improvement
*relative to the net*, not an oracle. Where the net's value function is sound
(money play) search holds at ceiling; where the net fundamentally misvalues
the matchup (Rush) search can't rescue it — it just pursues the net's flawed
plan more thoroughly. More simulations wouldn't help; they'd amplify the same
bias.

The real fix is full AlphaZero: put MCTS *inside* the training loop so
search-improved moves become the training targets, teaching the value
function the matchups it currently gets wrong. That closes the loop and is
the natural next build.

## AlphaZero loop (`train/azero.py`)

Full self-play training: MCTS plays both seats, the policy head is trained
by cross-entropy toward root **visit distributions** and the value head
regressed on game **outcomes**. No PPO, no reward shaping (search replaces
the shaping crutch), no scripted opponents in training -- the four bots are
eval-only yardsticks. Temperature sampling for the first 8 turns + Dirichlet
noise at the root keep self-play diverse; resumable via latest.pt.

Run (warm-started from the PPO net):

    python -m train.azero --init runs/v1/best.pt --out runs/az --iters 200

Throughput is ~2-5s per self-play game at 64 sims on one CPU core, so a few
hundred iterations is an overnight job. Watch log.csv: the question that
matters is whether eval-vs-Rush climbs above the PPO ceiling (~20%) -- that's
the signature of search-improved targets teaching the value function the
matchup it previously couldn't learn.

## Phase 1 (v2): self-play throughput

`train/batched.py` runs G games in lockstep; every loop advances each game
by one MCTS simulation and pools all leaf evaluations into one batched
forward (the v1 profile showed ~45% of self-play in single-observation
torch calls). `train/azero.py` now collects via this path
(`--batch-games`, default 32) and shards across processes with `--workers N`
(multiprocessing.Pool, torch pinned to 1 thread per worker).

Measured (this 1-core sandbox, batch=32):

| sims | sequential v1 | batched | speedup |
|---|---|---|---|
| 8  | ~5.0 (est.) | 10.5 g/s | ~2x |
| 16 | 2.58        | 4.9–5.2 g/s | ~2x |
| 32 | ~1.3 (est.) | 2.3 g/s | ~2x |

Negative results, kept honestly: numpy-ifying the PUCT select (array node
stats) and per-call `np.random.default_rng` determinization both made things
SLOWER (5.2 -> 2.7 g/s combined) — small-N numpy per-call overhead beats the
python it replaces, and Generator construction is expensive. Reverted after
piecewise measurement. Lesson: measure every "optimization" in isolation.

**Laptop projection (the actual Phase 1 target):** with W workers expect
~W-fold on top of batching. 8 cores -> 7 workers x ~5 g/s/core (faster
per-core than this sandbox) ~= 30-50 games/sec at 16 sims, i.e. ~100k+
games per overnight run. VERIFY on real hardware:

    python -m train.azero --init runs/v1/best.pt --out runs/az \
        --iters 200 --games-per-iter 512 --sims 16 --workers 7

Remaining Phase-1 definition-of-done item: the multiprocess scaling number
must be measured on a multi-core machine (untestable in this 1-core sandbox).

## Phase 2 (v2): observation + model upgrade

**obs-v2 (190 floats, `Game.observe_v2`)** adds: opponent discard
composition + in-play (public), opponent hidden pool (hand+draw COMBINED --
the split is hidden in real play and a unit test proves the observation is
bit-identical under hand/draw swaps), opponent top-of-discard one-hot,
last-3 acquisitions for both players, cards-until-reshuffle for both.
Engine now tracks gains centrally (`_gain`), top-of-discard (-1 = mixed
after cleanup / empty after reshuffle), and a rolling acquisition window.
9 new tests; 33 total pass.

**PolicyNetV2:** 3x512 residual MLP, 646k params.

**Baseline sanity (PPO from scratch, v1's final config, 30 iters):**

| Opponent | v1 best-ever (~200 iters) | v2 @ 30 iters (n=300) |
|---|---|---|
| Random | 100% | 100% |
| BigMoney | 91% | 98% |
| BM+Smithy | 69% | 88% |
| Rush | 36% | **94.7%** |

The Rush column is the finding: v1 never beat Rush under any configuration;
v2 dominates it almost immediately. Attribution is partly config head-start
(v1 spent ~145 iters before shaping existed), but the Rush ceiling break is
representation/capacity: the recent-acquisitions + opponent-discard features
make the rush legible early enough to counter. PPO baseline is now frozen
(runs/v2ppo/final.pt); v2 gauntlet thresholds already nearly met by the
baseline alone (BM+Smithy 88% vs the 97% bar is the remaining gap).

## Board expansion: engine2 (variable kingdoms, 15-card pool)

`dominion/engine2.py` + `dominion/bots2.py` + `tests/test_engine2.py`.
Each game deals 10 of 15 kingdom piles (original 10 + Chapel, Upgrade,
Worker's Village, Monument, Caravan). New mechanics: trashing (explicit
trash pile; conservation invariant is now supply+zones+trash == per-game
total), two-step trash->exact-cost gain (Upgrade), VP tokens (Monument),
durations (Caravan, with dur_new/dur_active separation -- a returned
Caravan must NOT be set aside again; this was a real bug caught by tests).
80-action masked space with a pending-mode state machine. observe_v3
(271 floats) adds kingdom-present indicators, VP tokens, durations, trash.
45 tests pass, incl. 150-game random-kingdom fuzz and the hidden-info leak
test ported to v3.

Bot calibration (300 games, seat-balanced, random kingdoms):
BM>Random 100%; BM+Smithy>BM 57%; Rush>BM 59%; ChapelMoney>BM 51%
(55.7% on Chapel boards -- below the ~65%+ theory says good Chapel play
achieves; the bot is a machinery validator, not a strong yardstick);
Rush>ChapelMoney 66% (real Dominion: rushes punish slow trash-down setups).

**Not yet done:** trainer plumbing. ppo/mcts/batched/azero import engine v1
(108/190-float obs, 43 actions). Porting them to engine2 (obs_v3, 80
actions, PolicyNetV3) is the next work item before any training on the
expanded board.

## v3 baseline: PPO on variable kingdoms (train2/ppo3.py)

PolicyNetV3 (271->512 residual->80, ~705k params), v2's proven config,
ChapelMoney added to pool+eval, latest.pt saved EVERY iteration (a timeout
mid-chunk previously lost work silently). ~70 effective iters in-sandbox:

eval on RANDOM kingdoms: Random 100% | BigMoney 72-77% | BM+Smithy 67-68%
| Rush 70% | ChapelMoney 68-77%

Behavioral probe (buys conditioned on dealt kingdom, vs BM):
- Discovered Monument unprompted (~3.4/game where present).
- Plays a robust generalist money+VP line; does NOT revert to v2's
  hyper-rush even on Workshop+Gardens boards (universal rush no longer
  dominates when it's only dealt ~43% of the time).
- Kingdom conditioning is WEAK at this stage (similar buy profiles across
  board types) and Chapel is COMPLETELY undiscovered: zero trashes in 25
  Chapel-board games. The delayed-payoff exploration cliff is back, exactly
  as predicted for a card whose value requires multi-turn commitment.

Continue training:  python -m train2.ppo3 --iters 100 --out runs/v3ppo
(resumable; ~1.5-2.5x v2's iterations expected for equivalent strength).

## Junker expansion: Moat out; Curse, Witch, Charlatan, Jester in

Pool is now 10-of-17. New base pile: Curse (-1 VP, always in supply, counts
for 3-pile endings, feeds Gardens). Witch (+2 Cards, curse opponent),
Charlatan (+$3, curse opponent, AND kingdom-wide rule: Curses are $1
Treasures while Charlatan is dealt -- charlatan_rule flag, exposed in obs),
Jester (+$2; opponent discards top card: victory -> they get a Curse,
otherwise a NEW two-option pending decision: you choose who gains a copy).
Deleting Moat removed the only reaction card, so no reaction system needed.

Card indices renumbered (N_CARDS=24, N_ACTIONS=93, OBS_SIZE_V3=308):
**the runs/v3ppo checkpoint is invalidated; v3 training restarts from
scratch.** 51 tests pass (6 new junker tests). Random-play sanity on a
Witch board: curses flow (6/3 gained) and get trashed (11 cards trashed).
Calibration: ChapelMoney edges BM 53% on Witch+Chapel boards -- modest only
because NEITHER scripted bot plays Witch; the curse pressure that makes
trashing essential will come from learned agents cursing each other in
self-play, which is the point of the expansion.

## v4: RL + MCTS on the junker world

train2/ppo3 retrained from scratch (runs/v4ppo, ~36 iters): Random 100% |
BM 63% | BM+Smithy 52% | Rush 48% | ChapelMoney 65%. train2/mcts3.py ports
determinized PUCT search to engine2.

**MCTS(60) beats the raw net 77% head-to-head** on random junker kingdoms
(same weights) -- the policy-improvement operator confirmed on the new world.

Witch+Chapel board probe (vs BM): the raw net ALREADY discovered Charlatan
(0.9 buys, 23 plays/10 games) -- immediate-payoff cursing found within 36
iterations, exactly like Monument. Search shifts behavior further: the MCTS
agent buys Witch 8x more than the raw net (0.8 vs 0.1/game) and plays it 19
times -- lookahead unlocking a card the priors underrate, with no retraining.
**Trashing remains at zero for both** raw and search agents: Chapel is still
undiscovered even with curses flowing, consistent with the prediction that
bolt-on search can't repair what the value function hasn't learned. Next:
azero-loop port (search-in-training) + long runs; watch for first Chapel
trashes on curse boards.

## Elo harness (train2/elo.py)

Random-pairing games on random kingdoms -> append-only CSV (flushed per
game) -> anchored Bradley-Terry fit (Random=0, BigMoney=1000). Search
variants are separate contestants, so sims-per-move has a measurable rating.
First ladder (616 games):
v4-MCTS150 1307 | v4-MCTS60 1264 | v4-raw 1247 | BM+Smithy 1134 | Rush 1130
| ChapelMoney 1034 | BigMoney 1000 | Random 0.
Usage: `python -m train2.elo --seconds 600` (repeat / background),
`--fit` to print the ladder. Add checkpoints to contestants() as training
runs produce them; filter the CSV by kingdom for per-board-type ladders.

## 20-card pool + azero3 (search-in-the-loop)

Added Collection ($5 Treasure: $2 +1 Buy; +1 VP token per Collection in
play when you gain an Action), Remake ($4: trash-then-gain-exactly-+1,
TWICE, mandatory), Hunting Grounds ($6: +4 Cards; ON-TRASH choice of a
Duchy or 3 Estates -- trigger interrupts and resumes mid-Chapel sequences
via hg_resume). Pool is now 10-of-20; N_CARDS=27, N_ACTIONS=106,
OBS_SIZE_V3=345. 57 tests incl. 80-game fuzz. v4 checkpoints invalidated.

train2/azero3.py: the AlphaZero loop for this engine. Pure self-play, every
move via MCTS (--sims, default 120); policy trained by cross-entropy to root
visit distributions, value on outcomes; Dirichlet root noise + temperature
sampling for 8 turns; resumable; checkpoints -> Elo harness contestants.
Smoke-tested end to end (fresh net, losses decreasing).

THE RUN (laptop, overnight, 2-3 seeds):
  python -m train2.azero3 --iters 300 --games-per-iter 32 --sims 150 --out runs/az3_s0
Watch: first nonzero Chapel/Remake trashes in self-play (expect on Witch
boards first), Elo-vs-checkpoint curve, buy-table evolution per board type.

## Throughput fix: batched search self-play for azero3

train2/batched3.py ports the Phase-1 lockstep design to engine2: G games
advance one simulation per loop, all leaf evals pooled into one forward
pass; fast_clone skips RNG-state copies; --workers shards across processes.
azero3 now collects through it (--batch-games, --workers). Validated
end-to-end (azero3 iter through the batched path, 4 games @16 sims, ~30s
incl. training). Sandbox caveat: embryonic-net games run ~500 decisions,
so 110-second windows can't produce clean speedup benchmarks here — the
architecture measured 2x single-core at batch 24-32 on v1, and the real
batch sizes + --workers on a laptop are where it pays. Recommended run:
  python -m train2.azero3 --iters 300 --games-per-iter 64 --sims 100 \
      --batch-games 32 --workers 7 --out runs/az3_s0
