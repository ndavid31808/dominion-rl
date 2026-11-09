"""monitor4: hands-off training monitor for a v4 run.

Every cycle (default 12 min) it:
  1. plays a short raw-only Elo batch over the run's checkpoints,
     refits the ladder, and appends it to <run>/monitor_elo.log
  2. for any NEW checkpoint since the last cycle, measures buys/game
     per card (greedy, fixed seeds -> comparable across checkpoints)
     and appends a row to <run>/buy_table.csv, printing the cumulative
     table so the strategy's evolution is visible at a glance

Read-only instrumentation: never touches training, never selects.

  python -m train2.monitor4 --run-dir runs/v9                # forever
  python -m train2.monitor4 --run-dir runs/v9 --cycles 4     # then exit
  python -m train2.monitor4 --run-dir runs/v9 --interval 600 --elo-seconds 90

Run it niced so it yields CPU to training:
  nohup nice -n 15 python -m train2.monitor4 --run-dir runs/v9 \
      > monitor.out 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import torch

from dominion.engine2 import (Game, KINGDOM_POOL, N_CARDS, NAMES,
                              BUY_BASE, GAIN_BASE)
from train2.elo4 import play_batch, fit_elo, _load_net, GreedyNetBot

BUY_SEED0 = 33_000_000      # fixed -> identical boards for every checkpoint


def buys_per_game(net, n=60):
    """Average buys+gains of each card per game (seat 0, greedy),
    over n fixed-seed random kingdoms. Comparable across checkpoints."""
    bot = GreedyNetBot(net)
    acq = np.zeros(N_CARDS)
    for i in range(n):
        g = Game(seed=BUY_SEED0 + i)
        steps = 0
        while not g.done:
            cur = g.current
            a = bot.act(g)
            if cur == 0:
                if BUY_BASE <= a < BUY_BASE + N_CARDS:
                    acq[a - BUY_BASE] += 1
                elif GAIN_BASE <= a < GAIN_BASE + N_CARDS:
                    acq[a - GAIN_BASE] += 1
            g.step(a)
            steps += 1
            if steps > 6000:
                break
    return acq / n


def print_buy_table(csv_path, top_k=14):
    rows = list(csv.reader(open(csv_path)))
    if len(rows) < 2:
        return
    header, data = rows[0], rows[1:]
    cards = header[1:]
    # rank cards by their buy rate in the LATEST row, show top_k
    latest = [float(x) for x in data[-1][1:]]
    order = sorted(range(len(cards)), key=lambda i: -latest[i])[:top_k]
    name_w = max(len(cards[i]) for i in order) + 1
    print("\nBUYS/GAME BY CHECKPOINT (top cards by latest):")
    print(" " * name_w + " ".join(f"{r[0][-6:]:>7s}" for r in data))
    for i in order:
        vals = " ".join(f"{float(r[1 + i]):7.1f}" for r in data)
        print(f"{cards[i]:{name_w}s}{vals}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=720,
                    help="seconds between cycles (default 12 min)")
    ap.add_argument("--elo-seconds", type=float, default=75,
                    help="length of each raw-only Elo batch")
    ap.add_argument("--buy-games", type=int, default=60,
                    help="fixed-seed games per checkpoint for the buy table")
    ap.add_argument("--cycles", type=int, default=0,
                    help="stop after this many cycles (0 = run forever)")
    a = ap.parse_args()

    buy_csv = os.path.join(a.run_dir, "buy_table.csv")
    elo_log = os.path.join(a.run_dir, "monitor_elo.log")
    if not os.path.exists(buy_csv):
        with open(buy_csv, "w", newline="") as f:
            csv.writer(f).writerow(["checkpoint"] + list(NAMES))
    seen = {r[0] for r in csv.reader(open(buy_csv))} - {"checkpoint"}

    cycle = 0
    while True:
        cycle += 1
        stamp = time.strftime("%H:%M:%S")
        print(f"\n===== monitor cycle {cycle} @ {stamp} =====", flush=True)

        # --- 1. Elo: focused batch on the NEWEST immutable checkpoint ---
        try:
            cks = sorted(glob.glob(os.path.join(a.run_dir, "ckpt_*.pt")))
            focus = None
            if cks:
                newest_tag = os.path.basename(cks[-1]) \
                    .replace("ckpt_", "").replace(".pt", "")
                focus = f"{newest_tag}-raw"
            play_batch(a.elo_seconds, seed0=int(time.time()) % 10**6 * 100,
                       run_dir=a.run_dir, sims=(), focus=focus,
                       include_latest=False)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ladder = fit_elo()
            print(buf.getvalue(), flush=True)
            if focus and ladder and focus in ladder:
                r, se, ng = ladder[focus]
                print(f">>> NEWEST CKPT {focus}: {r:.0f} ±{se:.0f} Elo "
                      f"({ng} games)", flush=True)
            with open(elo_log, "a") as f:
                f.write(f"\n===== {stamp} cycle {cycle} "
                        f"(focus={focus}) =====\n{buf.getvalue()}")
        except Exception as e:
            print(f"[monitor] elo cycle failed: {e}", flush=True)

        # --- 2. Buy table: measure any new checkpoints ---
        try:
            cks = sorted(glob.glob(os.path.join(a.run_dir, "ckpt_*.pt")))
            new = [p for p in cks
                   if os.path.basename(p).replace("ckpt_", "")
                                         .replace(".pt", "") not in seen]
            for p in new:
                tag = os.path.basename(p).replace("ckpt_", "") \
                                         .replace(".pt", "")
                net = _load_net(p)
                rates = buys_per_game(net, n=a.buy_games)
                with open(buy_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [tag] + [f"{x:.2f}" for x in rates])
                seen.add(tag)
                print(f"[monitor] buy rates logged for ckpt {tag}",
                      flush=True)
            print_buy_table(buy_csv)
        except Exception as e:
            print(f"[monitor] buy-table cycle failed: {e}", flush=True)

        if a.cycles and cycle >= a.cycles:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()