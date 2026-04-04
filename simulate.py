"""
Rank-strategy simulator.

Simulates "what would rank_today.py have selected on a given date/month?"
then validates whether those coins actually produced profitable trades that
day using the real strategy logic (4H breakout, entry/stop/target, leverage).

Two modes:
  --date  YYYY-MM-DD   Single-day simulation (full per-symbol breakdown)
  --month YYYY-MM      Full-month simulation (per-day table + month summary)

How it works (both modes):
  1. Pre-fetch every (symbol × day) result into an in-memory cache.
  2. For each simulation day D:
       - Weekly  ranking window : [D-weekly_days  ..  D-1]
       - Monthly ranking window : [D-monthly_days ..  D-1]
       - Apply same filter as rank_today (positive in both, >= min_trades weekly)
       - Score = 0.70 × weekly_pnl + 0.30 × monthly_pnl
  3. Check actual strategy trades on day D for the ranked symbols.

Usage:
    python simulate.py --date  2026-03-20
    python simulate.py --month 2026-02
    python simulate.py --date  2026-03-20 --leverage 20 --workers 8
    python simulate.py --month 2026-03   --weekly-days 5 --monthly-days 14
"""

import sys
import argparse
import calendar
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date

from main import (
    clamp_leverage,
    DEFAULT_LEVERAGE,
    NY_TZ,
)
from backtest_month import (
    DEFAULT_SYMBOLS,
    backtest_symbol,
    _iter_days,
)
from rank_today import W_WEIGHT, M_WEIGHT

_print_lock = threading.Lock()

# Default preset from the best row in march_2026_setups.csv.
BEST_SETUP = {
    "weekly_days": 7,
    "monthly_days": 21,
    "min_trades": 1,
    "min_weekly_wr": 0.0,
    "min_monthly_wr": 0.0,
    "min_weekly_nlr": 50.0,
    "min_monthly_nlr": 40.0,
    "top_n": 1,
    "max_trades_per_day": 0,
    "be_stop": False,
    "be_trigger_r": 1.0,
    "target_r": 2.0,
}


def _safe_print(msg):
    with _print_lock:
        print(msg, flush=True)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _fetch_one(
    sym,
    d,
    today_midnight_ny,
    leverage,
    enable_break_even,
    break_even_trigger_r,
    target_r,
    max_trades_per_day,
):
    """Run single-day backtest for one symbol. Returns result dict or None."""
    return backtest_symbol(
        sym,
        [d],
        today_midnight_ny,
        leverage,
        enable_break_even,
        break_even_trigger_r,
        target_r,
        max_trades_per_day,
    )


def build_cache(
    symbols,
    date_range,
    today_midnight_ny,
    leverage,
    workers,
    enable_break_even,
    break_even_trigger_r,
    target_r,
    max_trades_per_day,
):
    """
    Pre-fetch all (symbol, day) combinations in parallel.
    Returns: {symbol: {date: result_dict_or_None}}

    Fetching once and reusing is far more efficient than re-running ranking
    windows repeatedly (especially in month mode).
    """
    cache = {sym: {} for sym in symbols}
    tasks = [(sym, d) for sym in symbols for d in date_range]
    total = len(tasks)
    done_count = [0]  # list for mutable closure in thread

    print(
        f"  Pre-fetching {total} day×symbol combinations "
        f"({len(symbols)} symbols × {len(date_range)} days)...\n"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                sym,
                d,
                today_midnight_ny,
                leverage,
                enable_break_even,
                break_even_trigger_r,
                target_r,
                max_trades_per_day,
            ): (sym, d)
            for sym, d in tasks
        }
        for future in as_completed(futures):
            sym, d = futures[future]
            with _print_lock:
                done_count[0] += 1
                current = done_count[0]
            try:
                cache[sym][d] = future.result()
            except Exception:
                cache[sym][d] = None

            if current % 200 == 0 or current == total:
                _safe_print(f"  Fetch progress: {current}/{total}")

    print()
    return cache


# ── Window aggregation ────────────────────────────────────────────────────────

def _aggregate(cache, symbol, date_list):
    """
    Aggregate cached single-day results over date_list.
    Returns a result dict (same shape as backtest_symbol output) or None if
    no trades exist in the window.
    """
    total_pnl = 0.0
    trades = wins = breakevens = losses = liqs = 0
    market = None

    for d in date_list:
        r = cache[symbol].get(d)
        if r is None:
            continue
        total_pnl += r["total_pnl"]
        trades    += r["trades"]
        wins      += r["wins"]
        breakevens += r.get("breakevens", 0)
        losses    += r["losses"]
        liqs      += r["liquidations"]
        if market is None:
            market = r["market"]

    if trades == 0:
        return None

    return {
        "symbol":       symbol,
        "market":       market or "?",
        "total_pnl":    total_pnl,
        "trades":       trades,
        "wins":         wins,
        "breakevens":   breakevens,
        "losses":       losses,
        "win_rate":     wins / trades * 100,
        "non_loss_rate": (wins + breakevens) / trades * 100,
        "liquidations": liqs,
    }


# ── Ranking (mirrors rank_today.py) ──────────────────────────────────────────

def _rank_for_day(
    cache,
    symbols,
    sim_date,
    weekly_days,
    monthly_days,
    min_trades,
    min_weekly_wr,
    min_monthly_wr,
    min_weekly_nlr,
    min_monthly_nlr,
    top_n,
):
    """
    Compute ranked shortlist as-of sim_date using cached results.
    Windows end on sim_date-1 (yesterday relative to the simulation day).
    """
    yesterday     = sim_date - timedelta(days=1)
    weekly_dates  = list(_iter_days(yesterday - timedelta(days=weekly_days - 1),  yesterday))
    monthly_dates = list(_iter_days(yesterday - timedelta(days=monthly_days - 1), yesterday))

    ranked = []
    for sym in symbols:
        w = _aggregate(cache, sym, weekly_dates)
        m = _aggregate(cache, sym, monthly_dates)

        if w is None or m is None:
            continue
        if w["total_pnl"] <= 0 or m["total_pnl"] <= 0:
            continue
        if w["trades"] < min_trades:
            continue
        if w["win_rate"] < min_weekly_wr or m["win_rate"] < min_monthly_wr:
            continue
        if w["non_loss_rate"] < min_weekly_nlr or m["non_loss_rate"] < min_monthly_nlr:
            continue

        score = W_WEIGHT * w["total_pnl"] + M_WEIGHT * m["total_pnl"]
        ranked.append({
            "symbol":         sym,
            "market":         w["market"],
            "score":          score,
            "weekly_pnl":     w["total_pnl"],
            "weekly_trades":  w["trades"],
            "weekly_wr":      w["win_rate"],
            "weekly_nlr":     w["non_loss_rate"],
            "monthly_pnl":    m["total_pnl"],
            "monthly_wr":     m["win_rate"],
            "monthly_nlr":    m["non_loss_rate"],
        })

    ranked.sort(key=lambda r: r["score"], reverse=True)
    if top_n > 0:
        ranked = ranked[:top_n]
    return ranked


# ── Single-date simulation ────────────────────────────────────────────────────

def simulate_date(
    sim_date,
    symbols,
    weekly_days,
    monthly_days,
    min_trades,
    min_weekly_wr,
    min_monthly_wr,
    min_weekly_nlr,
    min_monthly_nlr,
    top_n,
    leverage,
    workers,
    enable_break_even,
    break_even_trigger_r,
    target_r,
    max_trades_per_day,
    daily_loss_limit=-25.0,
):
    today_midnight_ny = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch range: far enough back to cover the monthly ranking window + sim_date itself
    fetch_start = sim_date - timedelta(days=monthly_days)
    fetch_end   = sim_date
    date_range  = list(_iter_days(fetch_start, fetch_end))

    print(f"\n{'=' * 72}")
    print(f"  SIMULATOR  |  Date: {sim_date}  |  {len(symbols)} symbols  |  {leverage:.1f}x lev")
    print(f"  Ranking: {weekly_days}d ({W_WEIGHT:.0%} weight) + {monthly_days}d ({M_WEIGHT:.0%} weight)")
    print(
        f"  Filters: min trades {min_trades}, min WR {min_weekly_wr:.1f}%/{min_monthly_wr:.1f}%, "
        f"min NLR {min_weekly_nlr:.1f}%/{min_monthly_nlr:.1f}%, "
        f"top {top_n if top_n > 0 else 'all'}, max trades/day {max_trades_per_day or 'unlimited'}"
    )
    if enable_break_even:
        print(f"  Risk mgmt: break-even ON at {break_even_trigger_r:.2f}R, target {target_r:.2f}R")
    else:
        print(f"  Risk mgmt: break-even OFF, target {target_r:.2f}R")
    if daily_loss_limit < 0:
        print(f"  Daily stop: stop new symbols when day PnL would breach {daily_loss_limit:.2f}%")
    else:
        print("  Daily stop: disabled")
    print(f"{'=' * 72}\n")

    cache = build_cache(
        symbols,
        date_range,
        today_midnight_ny,
        leverage,
        workers,
        enable_break_even,
        break_even_trigger_r,
        target_r,
        max_trades_per_day,
    )

    ranked = _rank_for_day(
        cache,
        symbols,
        sim_date,
        weekly_days,
        monthly_days,
        min_trades,
        min_weekly_wr,
        min_monthly_wr,
        min_weekly_nlr,
        min_monthly_nlr,
        top_n,
    )

    # ── Ranked shortlist ──────────────────────────────────────────────────────
    print(f"{'─' * 72}")
    print(f"  RANKED SYMBOLS for {sim_date}  ({len(ranked)} selected)")
    print(f"{'─' * 72}")
    if not ranked:
        print("  No symbols met the ranking criteria.")
    else:
        print(
            f"  {'Symbol':<22} {'Score':>7}  "
            f"{'7d PnL':>8} {'7d WR':>6} {'7d NLR':>7}  "
            f"{'30d PnL':>8} {'30d WR':>6} {'30d NLR':>7}"
        )
        print(f"  {'-' * 74}")
        for r in ranked:
            print(
                f"  {r['symbol']:<22} {r['score']:>+6.2f}%  "
                f"{r['weekly_pnl']:>+7.2f}% {r['weekly_wr']:>5.1f}% {r['weekly_nlr']:>6.1f}%  "
                f"{r['monthly_pnl']:>+7.2f}% {r['monthly_wr']:>5.1f}% {r['monthly_nlr']:>6.1f}%"
            )

    # ── Actual trades on sim_date ─────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  ACTUAL TRADES on {sim_date}  (for ranked symbols)")
    print(f"{'─' * 72}")

    day_pnl = day_trades = day_wins = day_bes = day_losses = day_liqs = 0

    if not ranked:
        print("  (no ranked symbols to validate)\n")
        return

    print(
        f"  {'Symbol':<22} {'PnL':>8} {'Trades':>7} "
        f"{'W':>4} {'L':>4} {'Liq':>5} {'WR':>6}"
    )
    print(f"  {'-' * 60}")

    for r in ranked:
        result = cache[r["symbol"]].get(sim_date)
        if result is None:
            print(f"  {r['symbol']:<22}  — no trades")
            continue

        # Stop before adding this symbol if it would breach the daily loss limit.
        if daily_loss_limit < 0 and (day_pnl + result["total_pnl"]) <= daily_loss_limit:
            print(f"  {r['symbol']:<22}  skipped (daily loss limit reached)")
            break

        pnl = result["total_pnl"]
        t   = result["trades"]
        w   = result["wins"]
        be  = result.get("breakevens", 0)
        l   = result["losses"]
        liq = result["liquidations"]
        wr  = result["win_rate"]
        day_pnl    += pnl
        day_trades += t
        day_wins   += w
        day_bes    += be
        day_losses += l
        day_liqs   += liq
        print(
            f"  {r['symbol']:<22} {pnl:>+7.2f}% {t:>7} "
            f"{w:>4} {l:>4} {liq:>5} {wr:>5.1f}%"
        )

    print(f"\n  {'Summary':}")
    if day_trades > 0:
        print(
            f"  Total  {day_pnl:>+7.2f}%  |  {day_trades} trades  "
            f"{day_wins}W/{day_bes}BE/{day_losses}L  Liq {day_liqs}  "
            f"WR {day_wins/day_trades*100:.1f}% NLR {(day_wins + day_bes)/day_trades*100:.1f}%"
        )
    else:
        print("  No trades on this day for any ranked symbol.")
    print()


# ── Full-month simulation ─────────────────────────────────────────────────────

def simulate_month(
    year,
    month,
    symbols,
    weekly_days,
    monthly_days,
    min_trades,
    min_weekly_wr,
    min_monthly_wr,
    min_weekly_nlr,
    min_monthly_nlr,
    top_n,
    leverage,
    workers,
    enable_break_even,
    break_even_trigger_r,
    target_r,
    max_trades_per_day,
    daily_loss_limit=-25.0,
):
    today_midnight_ny = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today = today_midnight_ny.date()

    days_in_month = calendar.monthrange(year, month)[1]
    month_dates   = [date(year, month, d) for d in range(1, days_in_month + 1)]
    month_dates   = [d for d in month_dates if d < today]  # skip future / today

    if not month_dates:
        print("ERROR: no completed days in that month yet.")
        sys.exit(1)

    # Fetch range covers ranking windows before the month + all days of the month
    fetch_start = month_dates[0] - timedelta(days=monthly_days)
    fetch_end   = month_dates[-1]
    date_range  = list(_iter_days(fetch_start, fetch_end))

    period_label = f"{year}-{month:02d}"
    print(f"\n{'=' * 72}")
    print(f"  SIMULATOR  |  Month: {period_label}  |  {len(symbols)} symbols  |  {leverage:.1f}x lev")
    print(
        f"  Ranking: {weekly_days}d ({W_WEIGHT:.0%} weight) + {monthly_days}d ({M_WEIGHT:.0%} weight)"
    )
    print(
        f"  Filters: min trades {min_trades}, min WR {min_weekly_wr:.1f}%/{min_monthly_wr:.1f}%, "
        f"min NLR {min_weekly_nlr:.1f}%/{min_monthly_nlr:.1f}%, "
        f"top {top_n if top_n > 0 else 'all'}, max trades/day {max_trades_per_day or 'unlimited'}"
    )
    if enable_break_even:
        print(f"  Risk mgmt: break-even ON at {break_even_trigger_r:.2f}R, target {target_r:.2f}R")
    else:
        print(f"  Risk mgmt: break-even OFF, target {target_r:.2f}R")
    print(f"  Simulating {len(month_dates)} completed trading days")
    print(f"{'=' * 72}\n")

    cache = build_cache(
        symbols,
        date_range,
        today_midnight_ny,
        leverage,
        workers,
        enable_break_even,
        break_even_trigger_r,
        target_r,
        max_trades_per_day,
    )

    # ── Per-day loop ──────────────────────────────────────────────────────────
    month_pnl = month_trades = month_wins = month_bes = month_losses = month_liqs = 0
    days_with_ranked = days_with_trades = 0
    day_rows = []

    for sim_date in month_dates:
        ranked = _rank_for_day(
            cache,
            symbols,
            sim_date,
            weekly_days,
            monthly_days,
            min_trades,
            min_weekly_wr,
            min_monthly_wr,
            min_weekly_nlr,
            min_monthly_nlr,
            top_n,
        )

        if not ranked:
            day_rows.append((sim_date, 0, 0, 0, 0, 0, 0, 0.0))
            continue

        days_with_ranked += 1
        day_pnl = day_trades = day_wins = day_bes = day_losses = day_liqs = 0

        for r in ranked:
                result = cache[r["symbol"]].get(sim_date)
                if result is None:
                    continue
            
                # Check if daily loss limit would be breached (stop trading for this day)
                if daily_loss_limit < 0 and (day_pnl + result["total_pnl"]) <= daily_loss_limit:
                    break
            
                day_pnl    += result["total_pnl"]
                day_trades += result["trades"]
                day_wins   += result["wins"]
                day_bes    += result.get("breakevens", 0)
                day_losses += result["losses"]
                day_liqs   += result["liquidations"]

        if day_trades > 0:
            days_with_trades += 1

        month_pnl    += day_pnl
        month_trades += day_trades
        month_wins   += day_wins
        month_bes    += day_bes
        month_losses += day_losses
        month_liqs   += day_liqs

        day_rows.append((sim_date, len(ranked), day_trades, day_wins, day_bes, day_losses, day_liqs, day_pnl))

    # ── Month summary ─────────────────────────────────────────────────────────
    print(f"{'=' * 72}")
    print(f"  SIMULATION RESULTS — {period_label}")
    print(f"{'=' * 72}")
    print(f"  Days simulated        : {len(month_dates)}")
    print(f"  Days with ranked syms : {days_with_ranked}")
    print(f"  Days with trades      : {days_with_trades}")
    print(f"  Total trades          : {month_trades}")

    if month_trades > 0:
        print(
            f"  Wins / BE / Losses    : {month_wins}W / {month_bes}BE / {month_losses}L  "
            f"(WR {month_wins / month_trades * 100:.1f}% | NLR {(month_wins + month_bes) / month_trades * 100:.1f}%)"
        )
        print(f"  Liquidations          : {month_liqs}")
        print(f"  Total leveraged PnL   : {month_pnl:+.2f}%")
        print(f"  Avg daily PnL         : {month_pnl / len(month_dates):+.2f}%")

    # ── Per-day table ─────────────────────────────────────────────────────────
    print(f"\n  {'Date':<12} {'Ranked':>6} {'Trades':>7} {'W':>4} {'BE':>4} {'L':>4} {'Liq':>5} {'Day PnL':>9}")
    print(f"  {'-' * 58}")

    for sim_date, n_ranked, d_trades, d_wins, d_bes, d_losses, d_liqs, d_pnl in day_rows:
        if n_ranked == 0:
            print(f"  {sim_date!s:<12}  {'—':>6}  no ranked symbols")
        elif d_trades == 0:
            print(f"  {sim_date!s:<12} {n_ranked:>6}  {'—':>7}  no trades")
        else:
            print(
                f"  {sim_date!s:<12} {n_ranked:>6} {d_trades:>7} "
                f"{d_wins:>4} {d_bes:>4} {d_losses:>4} {d_liqs:>5} {d_pnl:>+8.2f}%"
            )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate rank_today selection and validate actual results"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date",  help="Single-day simulation: YYYY-MM-DD")
    group.add_argument("--month", help="Full-month simulation: YYYY-MM")

    parser.add_argument(
        "--weekly-days",  type=int, default=BEST_SETUP["weekly_days"],
        help=f"Weekly ranking window in days (default: {BEST_SETUP['weekly_days']})"
    )
    parser.add_argument(
        "--monthly-days", type=int, default=BEST_SETUP["monthly_days"],
        help=f"Monthly ranking window in days (default: {BEST_SETUP['monthly_days']})"
    )
    parser.add_argument(
        "--min-trades",   type=int, default=BEST_SETUP["min_trades"],
        help=f"Min weekly trades to qualify for ranking (default: {BEST_SETUP['min_trades']})"
    )
    parser.add_argument(
        "--min-weekly-wr", type=float, default=BEST_SETUP["min_weekly_wr"],
        help=f"Minimum weekly target-win rate percent (default: {BEST_SETUP['min_weekly_wr']})"
    )
    parser.add_argument(
        "--min-monthly-wr", type=float, default=BEST_SETUP["min_monthly_wr"],
        help=f"Minimum monthly target-win rate percent (default: {BEST_SETUP['min_monthly_wr']})"
    )
    parser.add_argument(
        "--min-weekly-nlr", type=float, default=BEST_SETUP["min_weekly_nlr"],
        help=f"Minimum weekly non-loss rate percent (default: {BEST_SETUP['min_weekly_nlr']})"
    )
    parser.add_argument(
        "--min-monthly-nlr", type=float, default=BEST_SETUP["min_monthly_nlr"],
        help=f"Minimum monthly non-loss rate percent (default: {BEST_SETUP['min_monthly_nlr']})"
    )
    parser.add_argument(
        "--top-n", type=int, default=BEST_SETUP["top_n"],
        help=f"Keep only top N ranked symbols per day (default: {BEST_SETUP['top_n']}, 0 = unlimited)"
    )
    parser.add_argument(
        "--symbols",      nargs="+", default=None,
        help="Override symbol list (default: built-in universe)"
    )
    parser.add_argument(
        "--workers",      type=int, default=6,
        help="Parallel fetch threads (default: 6)"
    )
    parser.add_argument(
        "--leverage",     type=float, default=DEFAULT_LEVERAGE,
        help=f"Leverage for PnL calculations (default: {DEFAULT_LEVERAGE})"
    )
    parser.add_argument(
        "--max-trades-per-day", type=int, default=BEST_SETUP["max_trades_per_day"],
        help=f"Cap counted trades per symbol/day in backtests (default: {BEST_SETUP['max_trades_per_day']}, 0 = unlimited)"
    )
    parser.add_argument(
        "--be-stop", action="store_true", default=BEST_SETUP["be_stop"],
        help="Enable break-even stop (move stop to entry after favorable move reaches trigger R)."
    )
    parser.add_argument(
        "--no-be-stop", action="store_false", dest="be_stop",
        help="Disable break-even stop even when default preset enables it."
    )
    parser.add_argument(
        "--be-trigger-r", type=float, default=BEST_SETUP["be_trigger_r"],
        help=f"R-multiple to arm break-even stop when --be-stop is enabled (default: {BEST_SETUP['be_trigger_r']})."
    )
    parser.add_argument(
        "--target-r", type=float, default=BEST_SETUP["target_r"],
        help=f"Profit target in R-multiple from entry (default: {BEST_SETUP['target_r']})"
    )
    parser.add_argument(
        "--daily-loss-limit", type=float, default=-25.0,
        help="Stop trading for the day if daily loss reaches this % (default: -25.0). Set to 0 to disable."
    )

    args = parser.parse_args()
    symbols  = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_SYMBOLS
    leverage = clamp_leverage(args.leverage)
    max_trades_per_day = args.max_trades_per_day if args.max_trades_per_day > 0 else None

    if args.date:
        try:
            sim_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: --date must be YYYY-MM-DD")
            sys.exit(1)
        if sim_date >= datetime.now(NY_TZ).date():
            print("ERROR: --date must be a past date (before today in NY time)")
            sys.exit(1)
        simulate_date(
            sim_date, symbols,
            args.weekly_days,
            args.monthly_days,
            args.min_trades,
            args.min_weekly_wr,
            args.min_monthly_wr,
            args.min_weekly_nlr,
            args.min_monthly_nlr,
            args.top_n,
            leverage,
            args.workers,
            args.be_stop,
            args.be_trigger_r,
            args.target_r,
            max_trades_per_day,
            args.daily_loss_limit,
        )
    else:
        try:
            month_dt = datetime.strptime(args.month, "%Y-%m")
        except ValueError:
            print("ERROR: --month must be YYYY-MM")
            sys.exit(1)
        simulate_month(
            month_dt.year,
            month_dt.month,
            symbols,
            args.weekly_days,
            args.monthly_days,
            args.min_trades,
            args.min_weekly_wr,
            args.min_monthly_wr,
            args.min_weekly_nlr,
            args.min_monthly_nlr,
            args.top_n,
            leverage,
            args.workers,
            args.be_stop,
            args.be_trigger_r,
            args.target_r,
            max_trades_per_day,
            args.daily_loss_limit,
        )


if __name__ == "__main__":
    main()
