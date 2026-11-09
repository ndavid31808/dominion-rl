"""azero4: tabula-rasa training loop for engine2 with the v4 stack.

  python -m train2.azero4 --iters 600 --out runs/v9 \
      --games-per-iter 64 --batch-games 32 --sims 64 --workers 5

Design (all decisions documented; zero human strategy content):
  * Network: PolicyNetV4 (card embeddings + attention + pointer heads).
  * Search: Gumbel root (works at small sims) + playout-cap randomization
    -> set --sims SMALL (32-96), the savings go into MORE GAMES.
  * Value target: (1-SV_MIX)*z + SV_MIX*root_search_value. The only
    grounding signal is win/loss; search value is the same signal,
    variance-reduced. Set --sv-mix 0 for pure Monte Carlo z.
  * Aux heads: predict final score margin + own VP (rules-derived
    quantities). Prediction-only; NEVER enters the reward. --aux 0 disables.
  * Policy loss only on full-budget (is_full) records.
  * Training volume SCALES WITH DATA: steps = new_positions * EPOCHS_EQ
    / batch, so replay ratio stays constant as throughput varies.
  * Capped stalemates score as losses for both seats (z = -1).
  * Eval vs scripted bots is INSTRUMENTATION ONLY. Selection guidance
    should come from within-lineage play (elo harness on this run dir).
  * stats.csv (instrumentation): learner trashes/game and win rate vs
    league opponents, every iteration.

Exploiter league (AlphaStar-style), still pure RL:
  # exploiter: rewarded only for beating a frozen main agent
  python -m train2.azero4 --out runs/x1 --init runs/v10/latest.pt \
      --exploit runs/v10/ckpt_1781177636.pt --proj-max-copies 1 \
      --proj-cheap --curr-clean ...
  # main agent: past selves + exploiters as league opponents. Pass fixed
  # ckpt_*.pt files, not latest.pt: league nets are cached per path.
  python -m train2.azero4 --out runs/v11 --init runs/v10/latest.pt \
      --extra-league runs/x1/ckpt_<time>.pt ...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from dominion.engine2 import Game, N_ACTIONS, NAMES
from train2.net4 import PolicyNetV4
from train2.selfplay4 import CAPPED_LOSS, selfplay_batch, _commit_onehot

EPOCHS_EQ = 4          # each new position is trained on ~this many times
SV_MIX_DEFAULT = 0.5
AUX_W = 0.2


def train_step(net, opt, buf, n_new, batch=512, sv_mix=0.5, aux_w=AUX_W,
               wts=None):
    if len(buf) < batch:
        return 0.0, 0.0, 0
    steps = max(8, (n_new * EPOCHS_EQ) // batch)
    pl = vl = 0.0
    for _ in range(steps):
        if wts is not None:     # --late-green-weight: weighted, with replacement
            b = [buf[j] for j in random.choices(range(len(buf)), weights=wts, k=batch)]
        else:
            b = random.sample(buf, batch)
        obs = torch.tensor(np.stack([x[0] for x in b]))
        mask = torch.tensor(np.stack([x[1] for x in b]))
        pi = torch.tensor(np.stack([x[2] for x in b]))
        z = torch.tensor([x[3] for x in b], dtype=torch.float32)
        sv = torch.tensor([x[4] for x in b], dtype=torch.float32)
        is_full = torch.tensor([x[5] for x in b], dtype=torch.bool)
        aux_t = torch.tensor(np.stack([x[6] for x in b]),
                             dtype=torch.float32)
        # buffer records from before commitments have no 8th field
        commit = _commit_onehot([x[7] if len(x) > 7 else -1 for x in b])

        logits, v = net(obs, mask, commit)
        # policy: only full-budget search targets carry gradient
        logp = F.log_softmax(logits, -1)
        ploss_all = -(pi * logp).sum(-1)
        denom = is_full.float().sum().clamp(min=1.0)
        ploss = (ploss_all * is_full.float()).sum() / denom
        # value: mix outcome with root search value (same grounding signal)
        target = (1.0 - sv_mix) * z + sv_mix * sv
        vloss = F.mse_loss(torch.tanh(v), target.clamp(-1.0, 1.0))
        loss = ploss + vloss
        if aux_w > 0:
            loss = loss + aux_w * F.mse_loss(net.aux_out, aux_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        pl += ploss.item()
        vl += vloss.item()
    return pl / steps, vl / steps, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--sims", type=int, default=64,
                    help="FULL search budget; Gumbel works small (32-96)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--buffer", type=int, default=60_000)
    ap.add_argument("--late-green-weight", type=float, default=1.0,
                    help="sampling weight for positions from games where the winner gained "
                         "their first Victory card (not Curse) on a later turn than the loser")
    ap.add_argument("--late-green-gap", type=int, default=1,
                    help="minimum turns later the winner must green for --late-green-weight")
    ap.add_argument("--out", default="runs/v9")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch-games", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--curr", type=float, default=0.25,
                    help="P(seat gets 2 uniform-random opening buys)")
    ap.add_argument("--proj", type=float, default=0.0,
                    help="P(seat is a PROJECT seat): gets a uniform-random "
                         "1-3 kingdom-card combo force-bought during the "
                         "build window. Never a policy target.")
    ap.add_argument("--mid-curr", type=float, default=0.0,
                    help="for curriculum seats: P(per buy decision, turns "
                         "2-12) of a forced uniform-random buy. Sustained "
                         "exploration injector; never a policy target.")
    ap.add_argument("--piles", default="10")
    ap.add_argument("--league-p", type=float, default=0.3)
    ap.add_argument("--league-size", type=int, default=6)
    ap.add_argument("--sv-mix", type=float, default=SV_MIX_DEFAULT,
                    help="value target = (1-m)*outcome + m*search_value")
    ap.add_argument("--aux", type=float, default=AUX_W,
                    help="aux prediction-head loss weight (0 disables)")
    ap.add_argument("--init", default="",
                    help="warm-start weights for a NEW run (ignored when "
                         "resuming from <out>/latest.pt)")
    ap.add_argument("--exploit", default="",
                    help="EXPLOITER mode: comma-separated frozen checkpoints. "
                         "Every game is learner vs one of these (greedy), "
                         "so the only reward is beating them. Overrides the "
                         "past-self league and forces league_p=1.")
    ap.add_argument("--extra-league", default="",
                    help="comma-separated checkpoints (e.g. exploiters) "
                         "added to the past-self league as opponents")
    ap.add_argument("--proj-max-copies", type=int, default=None,
                    help="cap on forced project copies (default 3)")
    ap.add_argument("--proj-cheap", action="store_true",
                    help="force a project buy only when coins <= cost+1, "
                         "so the experiment displaces little")
    ap.add_argument("--curr-clean", action="store_true",
                    help="random-opening buys never pick Curse or Copper")
    ap.add_argument("--commit-p", type=float, default=0.0,
                    help="P(learner seat commits to owning one copy of a "
                         "bandit-chosen kingdom card by turn 4; the net is "
                         "told via its commit input)")
    ap.add_argument("--commit-hide", type=float, default=0.0,
                    help="P(committed seat is not told): the card is still "
                         "bought, but the net sees no commit input, so the "
                         "ordinary policy learns what owning it is worth")
    ap.add_argument("--require-card", default="",
                    help="DIAGNOSTIC: card name forced onto every board")
    ap.add_argument("--commit-card", default="",
                    help="DIAGNOSTIC: every committed seat commits to this "
                         "card (with --commit-p 1.0: every learner seat)")
    ap.add_argument("--deep-sims", type=int, default=0,
                    help="sims for buy decisions in each learner seat's "
                         "first --deep-turns turns, every legal buy a root "
                         "candidate (0 = off)")
    ap.add_argument("--deep-turns", type=int, default=2)
    ap.add_argument("--commit-mirror", action="store_true",
                    help="--commit-p is per GAME: both learner seats commit "
                         "to the same card (mirror), instead of per seat")
    ap.add_argument("--start-p", type=float, default=0.0,
                    help="P(each seat starts with a random deck: 0-10 "
                         "Coppers, 0-5 Estates, 0-2 random non-Victory "
                         "kingdom cards) instead of 7 Copper + 3 Estate")
    ap.add_argument("--turn-search", action="store_true",
                    help="hybrid turn-level search: the current turn branches "
                         "on every decision (play order searched as before); "
                         "later turns branch only at buys/gains/trashes/"
                         "choices, with their action phases filled in by the "
                         "policy, so the search looks several turns ahead")
    ap.add_argument("--commit-hide-adaptive", action="store_true",
                    help="only hide commitments to cards whose committed "
                         "win rate (slow average, >=30 games) has reached "
                         "the uncommitted baseline")
    ap.add_argument("--commit-hide-threshold", type=float, default=None,
                    help="with --commit-hide-adaptive: FIXED win-rate bar a "
                         "card's committed play must reach to be hidden. "
                         "Default (unset) uses the live uncommitted baseline, "
                         "which can spiral: hidden handicapped games weaken "
                         "ordinary play, the baseline falls, more cards hide")
    ap.add_argument("--commit-floor", type=float, default=0.03,
                    help="minimum sampling weight per card, added to its "
                         "learning progress, so no commitment is starved")
    a = ap.parse_args()
    exploit = [p for p in a.exploit.split(",") if p.strip()]
    extra = [p for p in a.extra_league.split(",") if p.strip()]
    explore = {"proj_cheap": a.proj_cheap, "curr_clean": a.curr_clean,
               "commit_p": a.commit_p, "commit_hide": a.commit_hide,
               "turn_search": a.turn_search, "start_p": a.start_p,
               "commit_mirror": a.commit_mirror,
               "deep_sims": a.deep_sims, "deep_turns": a.deep_turns,
               "require_card": (NAMES.index(a.require_card)
                                if a.require_card else None),
               "commit_card": (NAMES.index(a.commit_card)
                               if a.commit_card else None)}
    if a.proj_max_copies is not None:
        explore["proj_max"] = a.proj_max_copies

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    net = PolicyNetV4()
    latest = os.path.join(a.out, "latest.pt")
    state_path = os.path.join(a.out, "trainer_state.pt")
    if os.path.exists(latest):
        net.load_state_dict(torch.load(latest, weights_only=True))
        print("resumed latest.pt", flush=True)
    elif a.init:
        net.load_state_dict(torch.load(a.init, weights_only=True))
        print(f"warm-started from {a.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
    buf: deque = deque(maxlen=a.buffer)
    bufw: deque = deque(maxlen=a.buffer)   # per-position sampling weight
    start_iter = 1
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        try:
            opt.load_state_dict(st["opt"])
            for rec in st["buf"]:
                buf.append(rec)
                bufw.append(1.0)
            start_iter = st.get("iter", 0) + 1
            print(f"resumed optimizer + {len(buf)} buffer positions "
                  f"(iter {start_iter})", flush=True)
        except Exception as e:
            print(f"warn: could not restore trainer state ({e})", flush=True)

    log = os.path.join(a.out, "log.csv")
    if not os.path.exists(log):
        open(log, "w").write(
            "iter,positions,steps,ploss,vloss,Random,BigMoney,BMSmithy,"
            "Rush,ChapelMoney,secs\n")

    stats_csv = os.path.join(a.out, "stats.csv")
    if not os.path.exists(stats_csv):
        open(stats_csv, "w").write(
            "iter,games,learner_trashes_per_game,league_games,"
            "league_winrate\n")

    # commit sampler: LEARNING PROGRESS per card. Two running averages of
    # committed-seat results (fast ~20 games, slow ~100 games); progress =
    # max(0, fast - slow). Weight = floor + progress: improving commitments
    # get practice, flat or declining ones fall back to the floor.
    FAST, SLOW = 0.05, 0.01
    lp_path = os.path.join(a.out, "commit_lp.json")
    lp = {}                              # card -> {"n", "fast", "slow"}
    if os.path.exists(lp_path):
        lp = {int(k): v for k, v in json.load(open(lp_path)).items()}
    commit_csv = os.path.join(a.out, "commit_lp.csv")
    if a.commit_p > 0 and not os.path.exists(commit_csv):
        open(commit_csv, "w").write(
            "iter,card,games,winrate,fast,slow,weight\n")

    use_csv = os.path.join(a.out, "commit_use.csv")
    if a.commit_p > 0 and not os.path.exists(use_csv):
        open(use_csv, "w").write(
            "iter,card,games,plays_per_game,trash_curse,trash_estate,"
            "trash_copper,trash_silver_gold,trash_other\n")
    TRASH_GROUPS = (("curse", (3,)), ("estate", (4,)), ("copper", (0,)),
                    ("silver_gold", (1, 2)))

    def commit_weights():
        # key -1 is the uncommitted baseline: tracked, never sampled.
        # Only IMPROVEMENT counts: cards start at the baseline, and a
        # commitment that just loses drifts down (fast < slow); rewarding
        # |fast - slow| would mistake that settling for progress.
        return {c: a.commit_floor + max(0.0, s["fast"] - s["slow"])
                for c, s in lp.items() if c >= 0}

    def update_bandit(it, results):
        per, use = {}, {}
        for _, w, _, _, st in results:
            for seat in (0, 1):
                if seat == st["opp_seat"]:
                    continue
                c = st["commit"][seat]
                r = 0.5 if w is None else (1.0 if w == seat else 0.0)
                per.setdefault(c, []).append(r)
                if c >= 0:
                    use.setdefault(c, []).append(
                        (st["plays"][seat], st["trashed"][seat]))
        with open(use_csv, "a") as f:
            for c, games in sorted(use.items()):
                n = len(games)
                tot = {}
                for _, tr in games:
                    for card, k in tr.items():
                        tot[card] = tot.get(card, 0) + k
                grp = [sum(tot.get(x, 0) for x in xs) / n
                       for _, xs in TRASH_GROUPS]
                other = sum(tot.values()) / n - sum(grp)
                f.write(f"{it},{NAMES[c]},{n},"
                        f"{sum(p for p, _ in games) / n:.2f},"
                        + ",".join(f"{g:.2f}" for g in grp)
                        + f",{other:.2f}\n")
        base = lp.get(-1, {}).get("slow", 0.5)
        for c, rs in sorted(per.items()):   # -1 (baseline) first
            init = float(np.mean(rs)) if c < 0 and -1 not in lp else base
            s = lp.setdefault(c, {"n": 0, "fast": init, "slow": init})
            for r in rs:
                s["n"] += 1
                s["fast"] += FAST * (r - s["fast"])
                s["slow"] += SLOW * (r - s["slow"])
        json.dump(lp, open(lp_path, "w"))
        wts = commit_weights()
        with open(commit_csv, "a") as f:
            for c, rs in sorted(per.items()):
                nm = NAMES[c] if c >= 0 else "none"
                s = lp.get(c, {"fast": float("nan"), "slow": float("nan")})
                f.write(f"{it},{nm},{len(rs)},{np.mean(rs):.3f},"
                        f"{s['fast']:.3f},{s['slow']:.3f},"
                        f"{wts.get(c, float('nan')):.3f}\n")
        top = sorted(wts.items(), key=lambda kv: -kv[1])[:3]
        return " ".join(f"{NAMES[c]}:{v:.2f}" for c, v in top)

    def current_league():
        if exploit:
            return exploit
        cks = sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")))
        cks = cks[:: max(1, len(cks) // a.league_size)][-a.league_size:]
        return (cks + extra) or None

    piles = tuple(int(x) for x in str(a.piles).split(","))
    t0 = time.time()
    seed = int(time.time()) % 10**6 * 100
    for it in range(start_iter, start_iter + a.iters):
        net.eval()
        league = current_league()
        explore["commit_w"] = commit_weights()
        if a.commit_hide_adaptive:
            base = (a.commit_hide_threshold
                    if a.commit_hide_threshold is not None
                    else lp.get(-1, {}).get("slow", 1.0))
            explore["commit_hide_cards"] = [
                c for c, s in lp.items()
                if c >= 0 and s["n"] >= 30 and s["slow"] >= base]
        from train2.selfplay4 import parallel_selfplay
        results = parallel_selfplay(net, a.games_per_iter, a.sims,
                                    seed0=seed, workers=a.workers,
                                    batch_games=a.batch_games,
                                    curr_p=a.curr, piles=piles,
                                    mid_p=a.mid_curr, proj_p=a.proj,
                                    league=league,
                                    league_p=(1.0 if exploit else
                                              a.league_p if league else 0.0),
                                    explore=explore)
        seed += a.games_per_iter
        n_pos = 0
        lg = [(w, st["opp_seat"]) for _, w, _, _, st in results
              if st["opp_seat"] >= 0]
        lwin = (sum(1.0 if w == 1 - s else 0.5 if w is None else 0.0
                    for w, s in lg) / len(lg)) if lg else float("nan")
        tpg = sum(st["trashes"] for *_, st in results) / max(len(results), 1)
        depth = (sum(st["depth"] for *_, st in results)
                 / max(len(results), 1))
        dd = [sum(st["deep_depth"][i] for *_, st in results) for i in (0, 1)]
        deep_txt = f" (openings {dd[0] / dd[1]:.1f}t)" if dd[1] else ""
        with open(stats_csv, "a") as f:
            f.write(f"{it},{len(results)},{tpg:.3f},{len(lg)},{lwin:.3f}\n")
        top_commits = update_bandit(it, results) if a.commit_p > 0 else ""
        n_lg = 0
        for recs, w, margin, vps, st in results:
            gr = st.get("green", [None, None])
            lg_game = (w in (0, 1) and (gr[w] or 99) - (gr[1 - w] or 99) >= a.late_green_gap)
            n_lg += lg_game
            pw = a.late_green_weight if lg_game else 1.0
            for obs, mask, pi, mover, root_v, is_full, commit in recs:
                if w == CAPPED_LOSS:
                    z = -1.0
                elif w is None:
                    z = 0.0
                else:
                    z = 1.0 if w == mover else -1.0
                sv = root_v
                aux = np.array([math.tanh(margin[mover] / 20.0),
                                vps[mover] / 50.0], dtype=np.float32)
                buf.append((obs, mask, pi, z, sv, is_full, aux, commit))
                bufw.append(pw)
            n_pos += len(recs)
        net.train()
        pl, vl, steps = train_step(net, opt, buf, n_pos,
                                   sv_mix=a.sv_mix, aux_w=a.aux,
                                   wts=list(bufw) if a.late_green_weight != 1.0 else None)
        torch.save(net.state_dict(), latest)
        torch.save({"opt": opt.state_dict(), "buf": list(buf), "iter": it},
                   state_path)
        line = (f"iter {it:3d}  +{n_pos} pos (buf {len(buf)})  "
                f"steps {steps:3d}  ploss {pl:.3f}  vloss {vl:.3f}  "
                f"trash/gm {tpg:.2f}  lookahead {depth:.1f}t{deep_txt}"
                + (f"  vs-league {lwin:.0%} ({len(lg)})" if lg else "")
                + f"  late-green {n_lg / max(len(results), 1):.0%}"
                + (f"  top-commits {top_commits}" if top_commits else ""))
        if it % a.eval_every == 0 or it == a.iters:
            from train2.eval4 import evaluate
            net.eval()
            ev = evaluate(net, n_games=100, seed=12345)
            line += "  eval " + " ".join(f"{k}:{v*100:.0f}%"
                                         for k, v in ev.items())
            torch.save(net.state_dict(),
                       os.path.join(a.out, f"ckpt_{int(time.time())}.pt"))
            with open(log, "a") as f:
                f.write(f"{it},{n_pos},{steps},{pl:.3f},{vl:.3f}," +
                        ",".join(f"{v:.3f}" for v in ev.values()) +
                        f",{time.time()-t0:.0f}\n")
        print(line, flush=True)
    print(f"done {(time.time()-t0)/60:.1f} min", flush=True)



if __name__ == "__main__":
    main()