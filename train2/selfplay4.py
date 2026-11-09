"""selfplay4: batched lockstep self-play for engine2, rebuilt from scratch.

Tabula rasa training data generator. Differences from batched3:

  * GUMBEL ROOT SELECTION (Danihelka et al. 2022): at the root, sample
    m candidate actions without replacement via Gumbel-Top-k over
    logits, evaluate them with sequential halving, and pick the action
    maximizing g + logit + sigma(q). Gives a valid policy-improvement
    target at SMALL simulation budgets (16-64), which is the whole
    economics of CPU training. Non-root selection stays PUCT.

  * PLAYOUT-CAP RANDOMIZATION (Wu 2020, KataGo): only a random fraction
    of moves get the full simulation budget and become policy-training
    targets; the rest use a cheap budget and contribute value data only.
    Multiplies games-per-hour at equal policy-target quality.

  * SEARCH-VALUE TARGETS: each record stores the root mean value from
    search, so the trainer can mix it with the game outcome z (lower-
    variance value learning; the only grounding signal is still win/loss).

  * PURE-OUTCOME CAP SCORING: a game hitting MAX_DECISIONS resolves by
    final VP if decisive, else scores as a loss for BOTH seats
    (CAPPED_LOSS) -- a non-terminating stalemate is a degenerate state,
    not a draw. Rules-derived; no strategy content.

  * UNRESTRICTED OPENING CURRICULUM: with prob curr_p per seat, that
    seat's first two buys are uniform-random over ALL affordable cards
    (no exclusion list -- zero human strategy priors). Forced moves are
    never recorded as policy targets.

  * Frozen-self league: with prob league_p a game is played against a
    frozen past checkpoint (greedy). 100% self-lineage.

Records: (obs, mask, pi, mover, root_v, is_full, commit)
  commit is the committed card the net was TOLD about, -1 if none (or if
  the commitment was hidden via commit_hide).
  pi is the improved policy (Gumbel weights at full-budget roots, visit
  distribution otherwise); is_full marks policy-trainable records.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from dominion.engine2 import (Game, N_ACTIONS, N_CARDS, BUY_BASE,
                              KINGDOM_POOL, COST, COPPER, CURSE, TRASH_BASE,
                              ESTATE, DUCHY, PROVINCE, GARDENS, VICTORY_CARDS,
                              PLAY_BASE, PLAYABLE)

C_PUCT = 1.5
FPU_REDUCTION = 0.25   # first-play urgency: unvisited children assume the
                       # parent's running value MINUS this, instead of 0.0
                       # ("draw"). At small budgets the init dominates, so
                       # this materially changes which lines get explored.
MAX_DECISIONS = 3000
CAPPED_LOSS = -2

# Gumbel root parameters (Danihelka et al. defaults, scaled small)
GUMBEL_M = 8           # root candidates sampled via Gumbel-Top-k
SIGMA_SCALE = 1.0      # q -> sigma(q) value-logit scale (c_visit folded in)

# playout-cap randomization
FULL_PROB = 0.25       # fraction of decisions searched at full budget
OPEN_MAX_TURNS = 4      # curriculum seats: random buys for the first
                        # 1..this turns (depth drawn uniformly per seat)
MID_CURR_LAST_TURN = 8
PROJ_LAST_TURN = 10     # project forcing only during the build window
PROJ_FORCE_P = 0.6      # P(force a project buy | one is affordable)
PROJ_MAX_COPIES = 3     # cap is randomized 1..this per card per game:
                        # uniform over dosage, so singleton-type cards
                        # (Chapel!) get fair-dose experiments instead of
                        # always-overdosed ones  # mid-game forced buys occur turns 2..this. Front-
                        # loaded on purpose: engine pieces only generate
                        # INFORMATIVE outcome data if acquired with runway
                        # left; late forced buys of slow cards manufacture
                        # anti-engine evidence.
CHEAP_SIMS_FRAC = 0.25  # cheap budget as a fraction of full sims

# Exploration-cleanup options (all off by default = original behaviour).
# Forced experiments should vary ONE thing: stacked handicaps make the
# value net blame the experimented card for losses it didn't cause.
#   proj_max    cap on forced copies (cap drawn uniformly 1..proj_max)
#   proj_cheap  only force a project buy when it displaces little:
#               coins <= card cost + 1
#   curr_clean  random-opening buys never pick Curse or Copper
#   commit_p    COMMITMENTS: P(a learner seat commits to one kingdom
#               card). The seat must own one copy by its 4th turn (bought
#               at its first affordable buy, waiting on turn 1 if that
#               hand is >$1 over the cost); every other decision is its
#               own. The net is TOLD (commit input) so the experiment's
#               losses are attributed to the commitment, not the card.
#   commit_w    sampling weight per card (trainer sets it from LEARNING
#               PROGRESS of committed-seat results, not their level, so a
#               commitment that is improving gets more practice even while
#               it still loses). Cards with no weight yet get the max
#               weight (optimistic: unexplored).
#   commit_hide P(a committed seat is NOT told): it still must buy the card,
#               but the net sees no commit input, so those games teach the
#               ordinary (uncommitted) policy what owning the card is worth.
COMMIT_LAST_TURN = 3
#   commit_hide_cards  if not None: only these cards may be hidden (the
#               trainer passes cards whose committed play has reached the
#               uncommitted baseline -- hiding a still-losing commitment
#               re-teaches the ordinary policy that the card loses)
EXPLORE_DEFAULTS = {"proj_max": PROJ_MAX_COPIES, "proj_cheap": False,
                    "curr_clean": False, "commit_p": 0.0, "commit_w": {},
                    "commit_hide": 0.0, "commit_hide_cards": None,
                    "start_p": 0.0, "require_card": None, "commit_card": None,
                    "commit_mirror": False, "deep_sims": 0, "deep_turns": 2}
#   deep_sims   DEEP OPENING SEARCH: buy decisions in a learner seat's
#               first deep_turns turns are searched with this many sims (and
#               always used as policy targets), with EVERY legal buy a root
#               candidate, so a low-prior opening like Chapel is always
#               evaluated several turns deep. 0 = off. Opening buys shape
#               the whole game; the rest keeps the normal budget.
#   commit_mirror  with prob commit_p a game is a MIRROR: both learner
#               seats commit to the same card, so each side learns to play
#               that card against an opponent playing it too (results sit
#               near 50%, keeping the value target informative).
#   require_card / commit_card  DIAGNOSTIC (not pure exploration): put this
#               card on every board / make every committed seat commit to it.
#               Used to test whether a plan around one card is learnable at
#               all (e.g. "can a Chapel-committed agent learn to beat v10?").
#   start_p     RANDOM STARTING DECKS: P(each seat, independently, starts the
#               game with 0-10 Coppers, 0-5 Estates and 0-2 random non-Victory
#               kingdom cards from this game's supply) instead of 7 Copper +
#               3 Estate. Lets the agent practise thin and unusual decks
#               without first having to build them. Rules-only randomness:
#               no strategy content. Never applied to h2h/eval games.
#               (Kingdom cards are 0-2, not always 2: two free cards win ~72%
#               vs a normal start, which would drown out plain thin decks.)
START_COPPERS = (0, 10)
START_ESTATES = (0, 5)
START_KINGDOM_CARDS = (0, 2)


def _random_start(g, p, rng):
    """Replace player p's starting deck (before any move) with a random one."""
    kcards = [c for c in KINGDOM_POOL
              if g.in_supply[c] and c not in VICTORY_CARDS]
    deck = ([COPPER] * rng.randint(*START_COPPERS)
            + [ESTATE] * rng.randint(*START_ESTATES))
    for _ in range(rng.randint(*START_KINGDOM_CARDS)):
        c = rng.choice(kcards)
        if g.supply[c] > 0:
            g.supply[c] -= 1       # kingdom cards come out of the supply
            deck.append(c)
    # starting Copper/Estate never come from the supply (as in reset()), so
    # only their count change enters the conservation total; count before
    # drawing, since _draw pops from this same list
    g.total_cards += (sum(1 for x in deck if x in (COPPER, ESTATE))
                      - p.deck_size())
    rng.shuffle(deck)
    p.draw = deck
    p.hand = [0] * N_CARDS
    p.discard = [0] * N_CARDS
    g._draw(p, 5)


def _outcome(g, me):
    return 0.0 if g.winner is None else (1.0 if g.winner == me else -1.0)


def fast_clone(g: Game) -> Game:
    c = Game.__new__(Game)
    c.rng = None
    c.kingdom = g.kingdom
    c.in_supply = g.in_supply
    c.supply = list(g.supply)
    c.total_cards = g.total_cards
    c.trash = list(g.trash)
    c.players = [p.clone() for p in g.players]
    for a in ("current", "phase", "actions", "buys", "coins", "pending",
              "pending_n", "pending_cost", "pending_card", "remake_left",
              "charlatan_rule", "done", "winner"):
        setattr(c, a, getattr(g, a))
    c.hg_resume = g.hg_resume
    c.priest_bonus = getattr(g, "priest_bonus", 0)
    c.sentry_idx = getattr(g, "sentry_idx", 0)
    c.sentry_cards = list(getattr(g, "sentry_cards", []))
    c.sentry_kept = list(getattr(g, "sentry_kept", []))
    c.queued_plays = list(getattr(g, "queued_plays", []))
    c.mult_times = getattr(g, "mult_times", 0)
    c.mm_pending = getattr(g, "mm_pending", 0)
    c._replaying = False
    return c


def _determinize(g: Game, me: int, rng: random.Random) -> None:
    rng.shuffle(g.players[me].draw)
    opp = g.players[1 - me]
    pool = list(opp.draw)
    for c in range(N_CARDS):
        pool.extend([c] * opp.hand[c])
    rng.shuffle(pool)
    hs = sum(opp.hand)
    opp.hand = [0] * N_CARDS
    for c in pool[:hs]:
        opp.hand[c] += 1
    opp.draw = pool[hs:]


class Node:
    __slots__ = ("expanded", "P", "N", "W", "children")

    def __init__(self):
        self.expanded = False
        self.P, self.N, self.W = {}, {}, {}
        self.children = {}


def _select_puct(node, legal):
    N, W, P = node.N, node.W, node.P
    tot = 0
    for a in legal:
        tot += N.get(a, 0)
    # FPU: unvisited actions score as the node's running value minus a
    # reduction, not as 0.0. Before any visits, fpu = -FPU_REDUCTION,
    # which makes the prior term dominate (follow the policy first).
    v_node = (sum(W.get(a, 0.0) for a in legal) / tot) if tot else 0.0
    fpu = v_node - FPU_REDUCTION
    k = C_PUCT * math.sqrt(tot + 1)
    inv = 1.0 / len(legal)
    best_a, best = legal[0], -1e30
    for a in legal:
        n = N.get(a, 0)
        q = (W[a] / n) if n else fpu
        s = q + k * P.get(a, inv) / (1 + n)
        if s > best:
            best, best_a = s, a
    return best_a


def _walk(root, g, forced_root_action=None):
    """Descend the tree. If forced_root_action is given, take it at the
    root (Gumbel candidate evaluation), PUCT below."""
    path = []
    node = root
    first = True
    while True:
        if g.done:
            return path, None, _outcome(g, g.current)
        if not node.expanded:
            return path, node, 0.0
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        if first and forced_root_action is not None:
            a = forced_root_action
            first = False
        else:
            a = _select_puct(node, legal)
        me = g.current
        g.step(a)
        child = node.children.get(a)
        if child is None:
            child = node.children[a] = Node()
        path.append((node, a, me, g.current))
        node = child


def _backup(path, v):
    for node, a, me, nxt in reversed(path):
        v_me = v if nxt == me else -v
        node.N[a] = node.N.get(a, 0) + 1
        node.W[a] = node.W.get(a, 0.0) + v_me
        v = v_me


# ---------------------------------------------------------------------------
# TURN-LEVEL SEARCH (explore["turn_search"]).
# Per-decision search spends its depth on micro-decisions (each action
# played, each Chapel trash, each buy), so even 1000 sims rarely see past
# the current turn -- but trashing pays off turns later. HYBRID tree:
#   * the CURRENT turn (the root player's) branches on every decision, so
#     play order -- what makes engines work -- is searched exactly as before
#   * LATER turns branch only where the deck can change (buy phase and
#     pending gain/trash/choice sub-decisions); their action-phase play is
#     filled in by the current policy's top move (one network call each,
#     batched across games like every other evaluation)
# Later turns were almost never reached by the old search anyway, so this
# trades nothing for several turns of extra lookahead. Every real decision
# is still the root of its own full search and is what the policy trains on.
# ---------------------------------------------------------------------------
TS_MAX_STEPS = 2000            # safety cap on engine steps per simulation


def _is_branch(g, s):
    if g.pending != 0 or g.phase == 1:
        return True                       # deck-changing decision
    return (g.current == s.root_player    # action phase of the root turn
            and g.players[s.root_player].turns == s.root_turn)


class _TSim:
    """One in-flight turn-level simulation."""
    __slots__ = ("g", "node", "path", "first", "forced", "steps", "t0",
                 "root_player", "root_turn")

    def __init__(self, g, root, forced):
        self.g, self.node, self.path = g, root, []
        self.first, self.forced, self.steps = True, forced, 0
        self.t0 = _turns(g)
        self.root_player = g.current
        self.root_turn = g.players[g.current].turns


def _ts_finish(c, s, v):
    """End simulation s with value v from the CURRENT player's view."""
    if s.path and s.path[-1][3] is None:
        s.path[-1][3] = s.g.current
    _backup(s.path, v)
    c.sims += 1
    c.depth_sum += _turns(s.g) - s.t0
    c.depth_n += 1
    if c.deep:
        c.deep_depth[0] += _turns(s.g) - s.t0
        c.deep_depth[1] += 1
    c.tsim = None


def _ts_advance(c):
    """Drive c's simulation until it needs the network: returns 'leaf'
    (expand c.tsim.node) or 'auto' (pick a policy move), or None if the
    simulation ended at a terminal state (already backed up)."""
    s = c.tsim
    g = s.g
    while True:
        if g.done or s.steps > TS_MAX_STEPS:
            _ts_finish(c, s, _outcome(g, g.current) if g.done else 0.0)
            return None
        if s.node is None:                      # between branch points
            legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
            if len(legal) == 1:                 # forced: no node needed
                g.step(legal[0])
                s.steps += 1
                continue
            if _is_branch(g, s):                # arrive at the next node
                entry = s.path[-1]
                entry[3] = g.current
                child = entry[0].children.get(entry[1])
                if child is None:
                    child = entry[0].children[entry[1]] = Node()
                s.node = child
                continue
            return "auto"
        node = s.node
        if not node.expanded:
            return "leaf"
        legal = [a for a, ok in enumerate(g.legal_actions()) if ok]
        if s.first and s.forced is not None:
            a = s.forced
        else:
            a = _select_puct(node, legal)
        s.first = False
        me = g.current
        g.step(a)
        s.steps += 1
        s.path.append([node, a, me, None])      # nxt filled on arrival
        s.node = None


def _turn_search_round(net, active):
    """One network round of turn-level search for every active game."""
    reqs = []
    for c in active:
        if c.sims >= c.budget and c.root.expanded:
            continue                            # ready to commit a move
        for _ in range(8):                      # skip sims ending terminal
            if c.tsim is None:
                sim = fast_clone(c.g)
                sim.rng = random.Random(c.rng.randrange(1 << 30))
                _determinize(sim, c.g.current, sim.rng)
                forced = None
                if c.gum is not None and c.root.expanded:
                    forced = c.gum.next_action(c.root)
                c.tsim = _TSim(sim, c.root, forced)
            kind = _ts_advance(c)
            if kind is not None:
                reqs.append((c, kind, c.tsim))
                break
            if c.sims >= c.budget and c.root.expanded:
                break
    if not reqs:
        return
    obs = torch.tensor(np.stack([s.g.observe_v3() for _, _, s in reqs]))
    masks = [np.array(s.g.legal_actions(), bool) for _, _, s in reqs]
    commit = _commit_onehot([c.commit_flag[s.g.current] for c, _, s in reqs])
    with torch.no_grad():
        logits_t, vals = net(obs, torch.tensor(np.stack(masks)), commit)
        logp = torch.log_softmax(logits_t, -1).numpy()
        probs = torch.softmax(logits_t, -1).numpy()
    for (c, kind, s), m, lp, pr, v in zip(reqs, masks, logp, probs,
                                          vals.numpy()):
        legal = [a for a, ok in enumerate(m) if ok]
        if kind == "auto":                      # policy fills in the move
            s.g.step(max(legal, key=lambda a: pr[a]))
            s.steps += 1
            continue
        node = s.node
        tot = float(pr[legal].sum()) or 1.0
        node.P = {a: float(pr[a]) / tot for a in legal}
        node.N = {a: 0 for a in node.P}
        node.W = {a: 0.0 for a in node.P}
        node.expanded = True
        if node is c.root and c.gum is None:
            c.gum = _Gumbel(legal, {a: float(lp[a]) for a in legal},
                            c.rng, c.budget,
                            len(legal) if c.deep else GUMBEL_M)
        _ts_finish(c, s, math.tanh(float(v)))


def _turns(g):
    return g.players[0].turns + g.players[1].turns


def _classic_round(net, active):
    """One simulation per game of the original per-decision search."""
    pending = []
    for c in active:
        sim = fast_clone(c.g)
        sim.rng = random.Random(c.rng.randrange(1 << 30))
        _determinize(sim, c.g.current, sim.rng)
        forced = None
        if c.gum is not None and c.root.expanded:
            forced = c.gum.next_action(c.root)
        path, leaf, tv = _walk(c.root, sim, forced_root_action=forced)
        c.depth_sum += _turns(sim) - _turns(c.g)
        c.depth_n += 1
        if c.deep:
            c.deep_depth[0] += _turns(sim) - _turns(c.g)
            c.deep_depth[1] += 1
        if leaf is None:
            _backup(path, tv)
            c.sims += 1
        else:
            pending.append((c, path, leaf, sim.observe_v3(),
                            sim.legal_actions(),
                            c.commit_flag[sim.current]))
    if not pending:
        return
    obs = torch.tensor(np.stack([p[3] for p in pending]))
    mask = torch.tensor(np.array([p[4] for p in pending]))
    commit = _commit_onehot([p[5] for p in pending])
    with torch.no_grad():
        logits_t, vals = net(obs, mask, commit)
        logp = torch.log_softmax(logits_t, -1).numpy()
        probs = torch.softmax(logits_t, -1).numpy()
    for (c, path, leaf, _, m, _), lp, pr, v in zip(
            pending, logp, probs, vals.numpy()):
        legal = [a for a, ok in enumerate(m) if ok]
        s = float(pr[legal].sum()) or 1.0
        leaf.P = {a: float(pr[a]) / s for a in legal}
        leaf.N = {a: 0 for a in leaf.P}
        leaf.W = {a: 0.0 for a in leaf.P}
        leaf.expanded = True
        if leaf is c.root and c.gum is None:
            # root just expanded: set up Gumbel candidates from
            # the raw policy log-probs
            c.gum = _Gumbel(legal, {a: float(lp[a]) for a in legal},
                            c.rng, c.budget,
                            len(legal) if c.deep else GUMBEL_M)
        _backup(path, math.tanh(float(v)))
        c.sims += 1


def _root_q(node, a):
    n = node.N.get(a, 0)
    return node.W.get(a, 0.0) / n if n else 0.0


class _Gumbel:
    """Per-decision Gumbel-Top-k + sequential halving.

    Halving uses an explicit cumulative visit TARGET per candidate: each
    round grants every live candidate budget/(rounds*|live|) extra sims;
    when all live candidates have reached their target, the worse half
    (by g + logit + sigma(q)) is eliminated and the next round's targets
    are issued. This concentrates simulations onto survivors, which is
    the point of the method (Danihelka et al. 2022)."""

    __slots__ = ("cands", "g_noise", "logits", "live", "budget", "target")

    def __init__(self, legal, logits, rng, budget, m=GUMBEL_M):
        m = min(m, len(legal))
        gn = {a: -math.log(-math.log(max(rng.random(), 1e-12)))
              for a in legal}
        scored = sorted(legal, key=lambda a: gn[a] + logits.get(a, 0.0),
                        reverse=True)
        self.cands = scored[:m]
        self.g_noise = gn
        self.logits = logits
        self.live = list(self.cands)
        self.budget = budget
        self.target = {a: 0 for a in self.cands}
        self._issue_round()

    def _issue_round(self):
        rounds = max(1, (len(self.cands) - 1).bit_length())  # ceil(log2 m)
        per = max(1, self.budget // (rounds * len(self.live)))
        for a in self.live:
            self.target[a] += per

    def next_action(self, node):
        """Next root candidate to simulate; halve when targets are met."""
        if len(self.live) == 1:
            return self.live[0]
        for a in self.live:
            if node.N.get(a, 0) < self.target[a]:
                return a
        # all targets met -> eliminate worse half, issue next round
        self.live.sort(key=lambda a: self._score(node, a), reverse=True)
        self.live = self.live[:max(1, len(self.live) // 2)]
        if len(self.live) > 1:
            self._issue_round()
        return self.live[0]

    def _score(self, node, a):
        return (self.g_noise[a] + self.logits.get(a, 0.0)
                + SIGMA_SCALE * _root_q(node, a)
                * (1 + sum(node.N.get(x, 0) for x in self.cands)) ** 0.5)

    def best(self, node):
        return max(self.live, key=lambda a: self._score(node, a))

    def improved_pi(self, node):
        """Policy target: softmax(logit + sigma(q)) over ALL legal actions
        with visited q where available (Gumbel AZ's completed-Q target)."""
        acts = list(self.logits)
        mx = -1e30
        zs = {}
        vroot = self.root_value(node)
        for a in acts:
            q = _root_q(node, a) if node.N.get(a, 0) else vroot
            z = self.logits[a] + SIGMA_SCALE * q * \
                (1 + sum(node.N.get(x, 0) for x in self.cands)) ** 0.5
            zs[a] = z
            mx = max(mx, z)
        tot = sum(math.exp(z - mx) for z in zs.values())
        return {a: math.exp(z - mx) / tot for a, z in zs.items()}

    def root_value(self, node):
        ns = sum(node.N.get(a, 0) for a in self.cands)
        if ns == 0:
            return 0.0
        return sum(node.W.get(a, 0.0) for a in self.cands) / ns


class _Ctx:
    __slots__ = ("vp_n", "green", "g", "root", "sims", "records", "rng", "curr_seats",
                 "decisions", "opp_seat", "opp_idx", "gum", "budget",
                 "is_full", "mid_p", "proj", "proj_cap", "open_turns",
                 "explore", "trashes", "commit", "commit_flag", "plays",
                 "trashed", "tsim", "depth_sum", "depth_n",
                 "deep", "deep_checked", "deep_depth")

    def __init__(self, seed, rng_seed, curr_p, piles, league, league_p,
                 sims_full, mid_p=0.0, proj_p=0.0, explore=None):
        self.explore = dict(EXPLORE_DEFAULTS, **(explore or {}))
        self.trashes = 0    # trash actions by learner (non-league) seats
        self.tsim = None              # in-flight turn-level simulation
        self.depth_sum = 0            # turns looked ahead, summed over sims
        self.depth_n = 0
        self.deep = False
        self.deep_checked = False
        self.deep_depth = [0, 0]      # turns looked ahead / sims, deep roots
        self.plays = [0, 0]           # per seat: plays of its committed card
        self.trashed = [{}, {}]       # per seat: card -> times trashed
        if isinstance(piles, int):
            piles = (piles,)
        rng0 = random.Random(seed ^ 0xA5A5)
        n = rng0.choice(piles)
        kingdom = sorted(rng0.sample(KINGDOM_POOL, n)) if n < 10 else None
        req = self.explore["require_card"]
        if req is not None:                 # diagnostic runs: card on board
            kingdom = kingdom or rng0.sample(KINGDOM_POOL, 10)
            if req not in kingdom:
                kingdom[-1] = req
            kingdom = sorted(kingdom)
        self.g = Game(seed=seed, kingdom=kingdom)
        self.root = Node()
        self.sims = 0
        self.records = []
        self.rng = random.Random(rng_seed)
        sp_ = self.explore["start_p"]
        if sp_ > 0.0:
            for p in self.g.players:
                if self.rng.random() < sp_:
                    _random_start(self.g, p, self.rng)
        # first turn each player gains a Victory card (Estate/Duchy/Province/Gardens; not Curses), for --late-green-weight
        self.vp_n = [_vp_count(p) for p in self.g.players]
        self.green = [None, None]
        self.curr_seats = {i for i in (0, 1)
                           if curr_p > 0.0 and self.rng.random() < curr_p}
        self.open_turns = {i: self.rng.randint(1, OPEN_MAX_TURNS)
                           for i in self.curr_seats}
        self.decisions = 0
        self.opp_seat = -1
        self.opp_idx = -1
        if league and self.rng.random() < league_p:
            self.opp_seat = self.rng.randrange(2)
            self.opp_idx = self.rng.randrange(len(league))
        self.gum = None
        self.is_full = False
        self.budget = sims_full
        self.mid_p = mid_p
        # COMMITMENTS (learner seats only, disjoint from curriculum seats)
        self.commit = [-1, -1]
        cp = self.explore["commit_p"]
        if cp > 0.0:
            kcards = [c for c in KINGDOM_POOL if self.g.in_supply[c]]
            w = self.explore["commit_w"]
            top = max(w.values(), default=1.0)
            weights = [w.get(c, top) for c in kcards]
            fixed = self.explore["commit_card"]
            pick = lambda: (fixed if fixed in kcards else
                            self.rng.choices(kcards, weights)[0])
            elig = [s_ for s_ in (0, 1)
                    if s_ != self.opp_seat and s_ not in self.curr_seats]
            if self.explore["commit_mirror"]:
                # both seats, same card; games vs league/curriculum seats
                # are not mirrors, so they get no commitment at all
                if len(elig) == 2 and self.rng.random() < cp:
                    card = pick()
                    for s_ in elig:
                        self.commit[s_] = card
            else:
                for s_ in elig:
                    if self.rng.random() < cp:
                        self.commit[s_] = pick()
        # what the NET is told (commit is always enforced; flag may be hidden)
        hide = self.explore["commit_hide"]
        hideable = self.explore["commit_hide_cards"]
        self.commit_flag = [
            -1 if (c >= 0 and (hideable is None or c in hideable)
                   and self.rng.random() < hide) else c
            for c in self.commit]
        # PROJECT SEATS: uniform-random 1-3 kingdom-card combo, force-
        # bought during the build window. Manufactures multi-card
        # archetype data (e.g. Village+Smithy decks) that single random
        # buys never produce. Disjoint from curriculum seats. The combo
        # is invisible to the net: scaffolding, not a rule.
        self.proj = [dict(), dict()]
        self.proj_cap = [dict(), dict()]
        if proj_p > 0.0:
            kingdom = [c for c in range(N_CARDS) if self.g.in_supply[c]
                       and c not in (0, 1, 2, 3, 4, 5, 6)]
            for s_ in (0, 1):
                if (s_ not in self.curr_seats and self.commit[s_] < 0
                        and self.rng.random() < proj_p):
                    k = self.rng.randint(1, 3)
                    cards = self.rng.sample(kingdom, min(k, len(kingdom)))
                    self.proj[s_] = {c: 0 for c in cards}
                    self.proj_cap[s_] = {c: self.rng.randint(
                        1, self.explore["proj_max"]) for c in cards}

    def step(self, a):
        """Advance the real game, counting learner-seat trashes and plays
        of the committed card (instrumentation only)."""
        s = self.g.current
        if s != self.opp_seat:
            if TRASH_BASE <= a < TRASH_BASE + N_CARDS:
                self.trashes += 1
                c = a - TRASH_BASE
                self.trashed[s][c] = self.trashed[s].get(c, 0) + 1
            elif (PLAY_BASE <= a < PLAY_BASE + len(PLAYABLE)
                    and PLAYABLE[a - PLAY_BASE] == self.commit[s]):
                self.plays[s] += 1
        self.g.step(a)
        for i, p in enumerate(self.g.players):
            if self.green[i] is None and _vp_count(p) > self.vp_n[i]:
                self.green[i] = p.turns + (1 if i == s and self.g.current == s else 0)

    def new_decision(self, sims_full):
        """Playout-cap randomization: decide this decision's budget."""
        self.root = Node()
        self.gum = None
        self.sims = 0
        self.tsim = None              # a new root invalidates in-flight sims
        self.deep = False
        self.deep_checked = False
        self.is_full = self.rng.random() < FULL_PROB
        self.budget = sims_full if self.is_full else \
            max(8, int(sims_full * CHEAP_SIMS_FRAC))

    def maybe_deepen(self):
        """Called once a decision is known to be searched: switch opening
        buy decisions of learner seats to the deep budget."""
        if self.deep_checked or self.sims:
            return
        self.deep_checked = True
        d, g = self.explore["deep_sims"], self.g
        if (d and g.phase == 1 and g.pending == 0
                and g.current != self.opp_seat
                and g.players[g.current].turns < self.explore["deep_turns"]):
            self.deep = True
            self.is_full = True
            self.budget = d

    def forced_curriculum_buy(self):
        """Uniform-random opening buy over ALL affordable cards (turns 1-2).
        No exclusions: zero human priors. Never recorded as a target."""
        g = self.g
        cur = g.current
        if g.pending != 0 or g.phase != 1:
            return None
        c = self.commit[cur]
        if c >= 0:
            p = g.players[cur]
            if (p.turns <= COMMIT_LAST_TURN and p.deck_composition()[c] == 0
                    and g.legal_actions()[BUY_BASE + c]
                    and not (p.turns == 0 and g.coins > COST[c] + 1)):
                return BUY_BASE + c
            return None
        pj = self.proj[cur]
        if pj and g.players[cur].turns <= PROJ_LAST_TURN:
            legal = g.legal_actions()
            cap = self.proj_cap[cur]
            elig = [c for c, n in pj.items()
                    if n < cap[c] and legal[BUY_BASE + c]
                    and (not self.explore["proj_cheap"]
                         or g.coins <= COST[c] + 1)]
            if elig and self.rng.random() < PROJ_FORCE_P:
                c = self.rng.choice(elig)
                pj[c] += 1
                return BUY_BASE + c
        if cur not in self.curr_seats:
            return None
        t = g.players[cur].turns
        if t >= self.open_turns[cur] and not (
                t <= MID_CURR_LAST_TURN
                and self.rng.random() < self.mid_p):
            return None   # random-depth openings; mid only if mid_p>0
        legal = g.legal_actions()
        elig = [a for a in range(BUY_BASE, BUY_BASE + N_CARDS) if legal[a]]
        if self.explore["curr_clean"]:
            elig = [a for a in elig
                    if a - BUY_BASE not in (CURSE, COPPER)] or elig
        return self.rng.choice(elig) if elig else None

    def opponent_move(self, league_nets):
        if self.g.current != self.opp_seat:
            return None
        net = league_nets[self.opp_idx]
        obs = torch.tensor(self.g.observe_v3()[None])
        mask = torch.tensor(np.array(self.g.legal_actions())[None])
        with torch.no_grad():
            logits, _ = net(obs, mask)
        return int(logits[0].argmax())


def _commit_onehot(cards):
    """(B,27) one-hot of committed cards; -1 rows stay all-zero."""
    t = torch.zeros(len(cards), N_CARDS)
    for i, c in enumerate(cards):
        if c >= 0:
            t[i, c] = 1.0
    return t


_VP_CARDS = (ESTATE, DUCHY, PROVINCE, GARDENS)


def _vp_count(p):
    comp = p.deck_composition()
    return sum(comp[c] for c in _VP_CARDS)


def _stats(c):
    """Per-game instrumentation; 'commit' also feeds the commit bandit."""
    return {"opp_seat": c.opp_seat, "trashes": c.trashes,
            "commit": list(c.commit), "plays": list(c.plays),
            "trashed": c.trashed,
            "depth": c.depth_sum / c.depth_n if c.depth_n else 0.0,
            "deep_depth": list(c.deep_depth), "green": list(c.green)}


_LEAGUE_CACHE = {}


def _load_league(paths, net_factory):
    nets = []
    for p in paths:
        if p not in _LEAGUE_CACHE:
            n = net_factory()
            n.load_state_dict(torch.load(p, weights_only=True))
            n.eval()
            _LEAGUE_CACHE[p] = n
        nets.append(_LEAGUE_CACHE[p])
    return nets


def selfplay_batch(net, net_factory, n_games, sims, seed0, batch_games=32,
                   curr_p=0.0, piles=10, league=None, league_p=0.0,
                   mid_p=0.0, proj_p=0.0, explore=None):
    league_nets = _load_league(league, net_factory) if league else []
    out: List[Tuple[list, Optional[int]]] = []
    master = random.Random(seed0)
    next_seed = seed0
    active: List[_Ctx] = []

    def spawn():
        nonlocal next_seed
        c = _Ctx(next_seed, master.randrange(1 << 30), curr_p, piles,
                 league, league_p, sims, mid_p=mid_p, proj_p=proj_p,
                 explore=explore)
        c.new_decision(sims)
        next_seed += 1
        return c

    while len(out) < n_games:
        while len(active) < batch_games and len(active) + len(out) < n_games:
            active.append(spawn())

        still = []
        for c in active:
            if c.decisions > MAX_DECISIONS:
                s0, s1 = c.g.score(0), c.g.score(1)
                w = (0 if s0 > s1 else 1) if s0 != s1 else CAPPED_LOSS
                out.append((c.records, w,
                            (s0 - s1, s1 - s0), (s0, s1), _stats(c)))
                continue
            while not c.g.done:
                legal = [a for a, ok in enumerate(c.g.legal_actions()) if ok]
                if len(legal) == 1:
                    c.step(legal[0])
                    c.decisions += 1
                    continue
                om = c.opponent_move(league_nets)
                if om is not None:
                    c.step(om)
                    c.new_decision(sims)
                    c.decisions += 1
                    continue
                fa = c.forced_curriculum_buy()
                if fa is not None:
                    c.step(fa)
                    c.new_decision(sims)
                    c.decisions += 1
                    continue
                c.maybe_deepen()
                break
            if c.g.done:
                s0, s1 = c.g.score(0), c.g.score(1)
                out.append((c.records, c.g.winner,
                            (s0 - s1, s1 - s0), (s0, s1), _stats(c)))
            else:
                still.append(c)
        active = still
        if not active:
            continue

        # one simulation (step) per game, pooled network evaluation
        if (explore or {}).get("turn_search"):
            _turn_search_round(net, active)
        else:
            _classic_round(net, active)

        # budget exhausted -> commit a move
        for c in active:
            if c.sims < c.budget or not c.root.expanded:
                continue
            if c.gum is not None:
                a = c.gum.best(c.root)
                pi_map = c.gum.improved_pi(c.root)
                root_v = c.gum.root_value(c.root)
            else:  # degenerate (shouldn't happen): fall back to visits
                acts = list(c.root.P)
                vv = np.array([c.root.N.get(x, 0) for x in acts], float)
                vv = vv / vv.sum() if vv.sum() else np.ones(len(acts)) / len(acts)
                pi_map = dict(zip(acts, vv))
                a = max(pi_map, key=pi_map.get)
                root_v = 0.0
            full = np.zeros(N_ACTIONS, np.float32)
            for act, p in pi_map.items():
                full[act] = p
            c.records.append((np.asarray(c.g.observe_v3(), np.float32),
                              np.array(c.g.legal_actions(), bool),
                              full, c.g.current, float(root_v),
                              c.is_full, c.commit_flag[c.g.current]))
            c.step(a)
            c.new_decision(sims)
            c.decisions += 1
    return out


# ---- persistent worker pool: spawn once, reuse across iterations ----
_WORKER = {}


def _init_worker():
    import torch as _t
    _t.set_num_threads(1)
    from train2.net4 import PolicyNetV4
    _WORKER["net"] = PolicyNetV4()


def _worker(args):
    (blob, n_games, sims, seed0, batch_games, curr_p, piles,
     league, league_p, mid_p, proj_p, explore) = args
    from train2.net4 import PolicyNetV4
    net = _WORKER["net"]
    net.load_state_dict(blob)
    net.eval()
    return selfplay_batch(net, PolicyNetV4, n_games, sims, seed0,
                          batch_games, curr_p, piles, league, league_p,
                          mid_p, proj_p, explore)


_POOL = {"pool": None, "size": 0}


def _get_pool(n):
    import multiprocessing as mp
    if _POOL["pool"] is not None and _POOL["size"] == n:
        return _POOL["pool"]
    if _POOL["pool"] is not None:
        try:
            _POOL["pool"].terminate(); _POOL["pool"].join()
        except Exception:
            pass
    ctx = mp.get_context("spawn")
    _POOL["pool"] = ctx.Pool(n, initializer=_init_worker)
    _POOL["size"] = n
    return _POOL["pool"]


def _drop_pool():
    if _POOL["pool"] is not None:
        try:
            _POOL["pool"].terminate(); _POOL["pool"].join()
        except Exception:
            pass
    _POOL["pool"] = None
    _POOL["size"] = 0


def parallel_selfplay(net, n_games, sims, seed0, workers=0, batch_games=32,
                      curr_p=0.0, piles=10, league=None, league_p=0.0,
                      mid_p=0.0, proj_p=0.0, explore=None):
    from train2.net4 import PolicyNetV4
    if workers <= 1:
        return selfplay_batch(net, PolicyNetV4, n_games, sims, seed0,
                              batch_games, curr_p, piles, league, league_p,
                              mid_p, proj_p, explore)
    import multiprocessing as mp
    blob = {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_games // workers + (1 if i < n_games % workers else 0)
           for i in range(workers)]
    tasks = [(blob, per[i], sims, seed0 + i * 10**6, batch_games, curr_p,
              piles, league, league_p, mid_p, proj_p, explore)
             for i in range(workers) if per[i] > 0]
    games_per_task = max(per) if per else n_games
    timeout = 300 + games_per_task * 60 * max(1.0, sims / 100.0)
    pool = _get_pool(len(tasks))
    try:
        async_res = pool.map_async(_worker, tasks)
        res = async_res.get(timeout=timeout)
        return [g for r in res for g in r]
    except mp.TimeoutError:
        print(f"[selfplay4] pool timed out after {timeout:.0f}s; rebuilding "
              "and falling back in-process for this iter", flush=True)
        _drop_pool()
        return selfplay_batch(net, PolicyNetV4, n_games, sims,
                              seed0 + 31337, batch_games, curr_p, piles,
                              league, league_p, mid_p, proj_p, explore)