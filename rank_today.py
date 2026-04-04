"""
Daily symbol ranker for the scalping strategy.

Runs two backtests for every symbol in the universe:
  - Weekly  window: last N completed trading days  (default 7)
  - Monthly window: last M completed trading days  (default 30)

Scores each symbol:  score = W_WEIGHT * weekly_pnl + M_WEIGHT * monthly_pnl
Keeps only symbols that are profitable in BOTH windows.
Prints a ranked "trade today" shortlist.

Usage:
    python rank_today.py
    python rank_today.py --leverage 20 --workers 8
    python rank_today.py --weekly-days 5 --monthly-days 20
    python rank_today.py --symbols BTCUSDT_PERP ETHUSDT_PERP SOLUSDT_PERP
"""

import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from main import (
    clamp_leverage,
    DEFAULT_LEVERAGE,
    NY_TZ,
)
from backtest_month import (
    DEFAULT_SYMBOLS,
    backtest_symbol,
    _iter_days,
    _safe_print,
)

# Scoring weights (must sum to 1.0)
W_WEIGHT = 0.70   # recent 7-day weight
M_WEIGHT = 0.30   # longer 30-day weight

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


def run_backtest_window(
    symbols,
    trading_dates,
    today_midnight_ny,
    leverage,
    workers,
    label,
    enable_break_even,
    break_even_trigger_r,
    target_r,
    max_trades_per_day,
):
    """Run backtest for all symbols over the given date list. Returns {symbol: result_dict}."""
    results = {}
    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                backtest_symbol,
                sym,
                trading_dates,
                today_midnight_ny,
                leverage,
                enable_break_even,
                break_even_trigger_r,
                target_r,
                max_trades_per_day,
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            completed += 1
            sym = futures[future]
            try:
                result = future.result()
            except Exception as e:
                _safe_print(f"  [{label}] [{completed:>3}/{total}] {sym:<22} ERROR: {e}")
                continue

            if result is not None:
                results[sym] = result
                _safe_print(
                    f"  [{label}] [{completed:>3}/{total}] {sym:<22} {result['total_pnl']:+.2f}%  "
                    f"({result['trades']} trades, {result['wins']}W/{result['losses']}L)"
                )
            else:
                _safe_print(f"  [{label}] [{completed:>3}/{total}] {sym:<22} no trades")

    return results


def main():
    parser = argparse.ArgumentParser(description="Daily trade shortlist ranker")
    parser.add_argument(
        "--weekly-days",
        type=int,
        default=BEST_SETUP["weekly_days"],
        help=f"Number of completed days for the weekly window (default: {BEST_SETUP['weekly_days']})",
    )
    parser.add_argument(
        "--monthly-days",
        type=int,
        default=BEST_SETUP["monthly_days"],
        help=f"Number of completed days for the monthly window (default: {BEST_SETUP['monthly_days']})",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbol list. Default: built-in universe.",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel worker threads per window (default: 6)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=DEFAULT_LEVERAGE,
        help=f"Leverage for PnL calculations (default: {DEFAULT_LEVERAGE})",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=BEST_SETUP["min_trades"],
        help=f"Minimum total trades in weekly window to qualify (default: {BEST_SETUP['min_trades']})",
    )
    parser.add_argument(
        "--min-weekly-wr",
        type=float,
        default=BEST_SETUP["min_weekly_wr"],
        help=f"Minimum weekly target-win rate percent (default: {BEST_SETUP['min_weekly_wr']})",
    )
    parser.add_argument(
        "--min-monthly-wr",
        type=float,
        default=BEST_SETUP["min_monthly_wr"],
        help=f"Minimum monthly target-win rate percent (default: {BEST_SETUP['min_monthly_wr']})",
    )
    parser.add_argument(
        "--min-weekly-nlr",
        type=float,
        default=BEST_SETUP["min_weekly_nlr"],
        help=f"Minimum weekly non-loss rate percent (default: {BEST_SETUP['min_weekly_nlr']})",
    )
    parser.add_argument(
        "--min-monthly-nlr",
        type=float,
        default=BEST_SETUP["min_monthly_nlr"],
        help=f"Minimum monthly non-loss rate percent (default: {BEST_SETUP['min_monthly_nlr']})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=BEST_SETUP["top_n"],
        help=f"Keep only top N ranked symbols (default: {BEST_SETUP['top_n']}, 0 = unlimited)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=BEST_SETUP["max_trades_per_day"],
        help=f"Cap counted trades per symbol/day in backtests (default: {BEST_SETUP['max_trades_per_day']}, 0 = unlimited)",
    )
    parser.add_argument(
        "--be-stop",
        action="store_true",
        default=BEST_SETUP["be_stop"],
        help="Enable break-even stop (move stop to entry after favorable move reaches trigger R).",
    )
    parser.add_argument(
        "--no-be-stop",
        action="store_false",
        dest="be_stop",
        help="Disable break-even stop even when default preset enables it.",
    )
    parser.add_argument(
        "--be-trigger-r",
        type=float,
        default=BEST_SETUP["be_trigger_r"],
        help=f"R-multiple to arm break-even stop when --be-stop is enabled (default: {BEST_SETUP['be_trigger_r']}).",
    )
    parser.add_argument(
        "--target-r",
        type=float,
        default=BEST_SETUP["target_r"],
        help=f"Profit target in R-multiple from entry (default: {BEST_SETUP['target_r']}).",
    )
    args = parser.parse_args()

    now_ny = datetime.now(NY_TZ)
    today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    leverage = clamp_leverage(args.leverage)
    symbols = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_SYMBOLS
    max_trades_per_day = args.max_trades_per_day if args.max_trades_per_day > 0 else None

    # Both windows end yesterday (last completed day)
    yesterday = now_ny.date() - timedelta(days=1)
    weekly_start  = yesterday - timedelta(days=args.weekly_days - 1)
    monthly_start = yesterday - timedelta(days=args.monthly_days - 1)

    weekly_dates  = list(_iter_days(weekly_start,  yesterday))
    monthly_dates = list(_iter_days(monthly_start, yesterday))

    print(f"\n{'=' * 72}")
    print(f"  Trade Ranker  |  Today: {now_ny.date().isoformat()}  |  {len(symbols)} symbols  |  {leverage:.1f}x lev")
    print(f"  Weekly  window : {weekly_start}  →  {yesterday}  ({len(weekly_dates)} days)")
    print(f"  Monthly window : {monthly_start}  →  {yesterday}  ({len(monthly_dates)} days)")
    print(f"  Score = {W_WEIGHT:.0%} × weekly_pnl  +  {M_WEIGHT:.0%} × monthly_pnl")
    print(
        f"  Filters: min trades {args.min_trades}, min WR {args.min_weekly_wr:.1f}%/{args.min_monthly_wr:.1f}%, "
        f"min NLR {args.min_weekly_nlr:.1f}%/{args.min_monthly_nlr:.1f}%, "
        f"top {args.top_n if args.top_n > 0 else 'all'}, max trades/day {max_trades_per_day or 'unlimited'}"
    )
    if args.be_stop:
        print(f"  Risk mgmt: break-even ON at {args.be_trigger_r:.2f}R, target {args.target_r:.2f}R")
    else:
        print(f"  Risk mgmt: break-even OFF, target {args.target_r:.2f}R")
    print(f"{'=' * 72}\n")

    # ── Weekly scan ──────────────────────────────────────────────────────────
    print(f"  --- Weekly scan ({args.weekly_days}d) ---")
    weekly_results = run_backtest_window(
        symbols,
        weekly_dates,
        today_midnight_ny,
        leverage,
        args.workers,
        "7d",
        args.be_stop,
        args.be_trigger_r,
        args.target_r,
        max_trades_per_day,
    )

    # ── Monthly scan ─────────────────────────────────────────────────────────
    print(f"\n  --- Monthly scan ({args.monthly_days}d) ---")
    monthly_results = run_backtest_window(
        symbols,
        monthly_dates,
        today_midnight_ny,
        leverage,
        args.workers,
        "30d",
        args.be_stop,
        args.be_trigger_r,
        args.target_r,
        max_trades_per_day,
    )

    # ── Score & filter ───────────────────────────────────────────────────────
    ranked = []
    for sym in symbols:
        w = weekly_results.get(sym)
        m = monthly_results.get(sym)

        if w is None or m is None:
            continue  # no data in one of the windows
        if w["total_pnl"] <= 0 or m["total_pnl"] <= 0:
            continue  # must be profitable in BOTH
        if w["trades"] < args.min_trades:
            continue  # not enough recent activity
        if w["win_rate"] < args.min_weekly_wr or m["win_rate"] < args.min_monthly_wr:
            continue
        if w["non_loss_rate"] < args.min_weekly_nlr or m["non_loss_rate"] < args.min_monthly_nlr:
            continue

        score = W_WEIGHT * w["total_pnl"] + M_WEIGHT * m["total_pnl"]
        ranked.append({
            "symbol":       sym,
            "market":       w["market"],
            "score":        score,
            "weekly_pnl":   w["total_pnl"],
            "weekly_trades": w["trades"],
            "weekly_wins":  w["wins"],
            "weekly_losses": w["losses"],
            "weekly_wr":    w["win_rate"],
            "weekly_nlr":   w["non_loss_rate"],
            "monthly_pnl":  m["total_pnl"],
            "monthly_trades": m["trades"],
            "monthly_wr":   m["win_rate"],
            "monthly_nlr":  m["non_loss_rate"],
        })

    ranked.sort(key=lambda r: r["score"], reverse=True)
    if args.top_n > 0:
        ranked = ranked[:args.top_n]

    # ── Print shortlist ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  TRADE TODAY SHORTLIST  —  {now_ny.date().isoformat()}  ({len(ranked)} symbols)")
    print(
        f"  Criteria: positive in both windows, >={args.min_trades} weekly trades, "
        f"WR >={args.min_weekly_wr:.1f}%/{args.min_monthly_wr:.1f}%, "
        f"NLR >={args.min_weekly_nlr:.1f}%/{args.min_monthly_nlr:.1f}%"
    )
    print(f"{'=' * 72}")

    if not ranked:
        print("  No symbols meet the criteria today. Consider relaxing --min-trades.\n")
        return

    header = (
        f"  {'#':<4} {'Symbol':<22} {'Score':>7}  "
        f"{'7d PnL':>8} {'7d W/L':>7} {'7d WR':>6} {'7d NLR':>7}  "
        f"{'30d PnL':>8} {'30d WR':>6} {'30d NLR':>7}"
    )
    print(header)
    print(f"  {'-' * 84}")

    for rank, r in enumerate(ranked, 1):
        weekly_wl = f"{r['weekly_wins']}W/{r['weekly_losses']}L"
        print(
            f"  {rank:<4} {r['symbol']:<22} {r['score']:>+6.2f}%  "
            f"{r['weekly_pnl']:>+7.2f}% {weekly_wl:>7} {r['weekly_wr']:>5.1f}% {r['weekly_nlr']:>6.1f}%  "
            f"{r['monthly_pnl']:>+7.2f}% {r['monthly_wr']:>5.1f}% {r['monthly_nlr']:>6.1f}%"
        )

    print(f"\n  Top picks (copy-paste):")
    print(f"  " + "  ".join(r["symbol"] for r in ranked[:10]))
    print()


if __name__ == "__main__":
    main()
