"""
Multi-symbol monthly backtest for the scalping strategy.

Scans a large built-in coin list (or a custom list via --symbols),
runs the strategy for every trading day in the target month, then
prints ONLY the symbols with a positive total PnL, ranked best-first.

Usage:
    python backtest_month.py
    python backtest_month.py --month 2026-02
    python backtest_month.py --month 2026-02 --symbols BTCUSDT_PERP ETHUSDT_PERP
    python backtest_month.py --workers 8
"""

import sys
import argparse
import calendar
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta, date

# Import shared logic from main.py (safe – UI code is guarded by __name__ == "__main__")
from main import (
    normalize_symbol_and_market,
    run_strategy_positions_only,
    calculate_leveraged_pnl_pct,
    clamp_leverage,
    DEFAULT_LEVERAGE,
    NY_TZ,
    Client,
)
from kline_disk_cache import (
    get_klines_cached as get_klines,
    get_first_4h_candle_levels_cached as get_first_4h_candle_levels,
)

# ── Default symbol universe ──────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    # Large-cap
    "BTCUSDT_PERP", "ETHUSDT_PERP", "BNBUSDT_PERP", "SOLUSDT_PERP",
    "XRPUSDT_PERP", "DOGEUSDT_PERP", "ADAUSDT_PERP", "AVAXUSDT_PERP",
    "DOTUSDT_PERP", "LINKUSDT_PERP", "LTCUSDT_PERP", "BCHUSDT_PERP",
    "UNIUSDT_PERP", "ATOMUSDT_PERP", "ETCUSDT_PERP", "XLMUSDT_PERP",
    "VETUSDT_PERP", "FILUSDT_PERP", "TRXUSDT_PERP", "EOSUSDT_PERP",
    # Mid-cap DeFi / L1 / L2
    "AAVEUSDT_PERP", "SUSHIUSDT_PERP", "COMPUSDT_PERP", "MKRUSDT_PERP",
    "SNXUSDT_PERP", "CRVUSDT_PERP", "YFIUSDT_PERP", "1INCHUSDT_PERP",
    "RUNEUSDT_PERP", "NEARUSDT_PERP", "FTMUSDT_PERP", "ALGOUSDT_PERP",
    "ICPUSDT_PERP", "SANDUSDT_PERP", "MANAUSDT_PERP", "AXSUSDT_PERP",
    "GALAUSDT_PERP", "ENJUSDT_PERP", "CHZUSDT_PERP", "SHIBUSDT_PERP",
    "FLOKIUSDT_PERP", "PEPEUSDT_PERP", "WIFUSDT_PERP", "BONKUSDT_PERP",
    # Layer 2 / infra
    "MATICUSDT_PERP", "OPUSDT_PERP", "ARBUSDT_PERP", "STXUSDT_PERP",
    "LDOUSDT_PERP", "IMXUSDT_PERP", "APTUSDT_PERP", "SUIUSDT_PERP",
    "SEIUSDT_PERP", "INJUSDT_PERP", "TIAUSDT_PERP", "JUPUSDT_PERP",
    "WUSDT_PERP", "STRKUSDT_PERP", "ZKUSDT_PERP", "TONUSDT_PERP",
    # Meme / narrative
    "FETUSDT_PERP", "RENDERUSDT_PERP", "WLDUSDT_PERP", "PENDLEUSDT_PERP",
    "EIGENUSDT_PERP", "ENAUSDT_PERP", "REZUSDT_PERP", "NOTUSDT_PERP",
    "BIGTIMEUSDT_PERP", "AGLDUSDT_PERP", "HBARUSDT_PERP", "IOTAUSDT_PERP",
    "ZECUSDT_PERP", "DASHUSDT_PERP", "NEOUSDT_PERP", "ONTUSDT_PERP",
    "QTUMUSDT_PERP", "IOTXUSDT_PERP", "ZILUSDT_PERP", "BATUSDT_PERP",
]

_print_lock = threading.Lock()


def _safe_print(msg):
    with _print_lock:
        print(msg, flush=True)


def _iter_days(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def backtest_symbol(
    display_symbol,
    trading_dates,
    today_midnight_ny,
    leverage,
    enable_break_even=False,
    break_even_trigger_r=1.0,
    target_r=2.0,
    max_trades_per_day=None,
):
    """Run full-month backtest for one symbol. Returns a result dict or None."""
    try:
        symbol_for_api, market = normalize_symbol_and_market(display_symbol)
    except Exception as e:
        _safe_print(f"  [SKIP] {display_symbol}: normalize error – {e}")
        return None

    all_trades = []
    days_with_trades = 0

    for d in trading_dates:
        date_midnight_ny = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=NY_TZ)
        if date_midnight_ny >= today_midnight_ny:
            break  # skip future days

        try:
            open_level, high_level = get_first_4h_candle_levels(
                symbol_for_api, market, date_midnight_ny
            )
            if open_level is None or high_level is None:
                continue

            start_str = date_midnight_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
            end_str = (date_midnight_ny + timedelta(days=1)).astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
            klines = get_klines(
                symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE,
                start_str, end_str, date_midnight_ny=date_midnight_ny,
            )
            if not klines:
                continue

            closed = [
                t for t in run_strategy_positions_only(
                    klines,
                    open_level,
                    high_level,
                    date_midnight_ny,
                    leverage=leverage,
                    enable_break_even=enable_break_even,
                    break_even_trigger_r=break_even_trigger_r,
                    target_r=target_r,
                )
                if t["result"] != "open"
            ]
            if max_trades_per_day is not None and max_trades_per_day > 0:
                closed = closed[:max_trades_per_day]
            if closed:
                all_trades.extend(closed)
                days_with_trades += 1

        except Exception:
            continue  # skip bad days, keep going

    if not all_trades:
        return None

    total_pnl = 0.0
    wins = 0
    breakevens = 0
    liquidation_count = 0
    for t in all_trades:
        pnl = calculate_leveraged_pnl_pct(
            t["entry"], t["exit_price"], t["side"], t.get("leverage", leverage)
        )
        total_pnl += pnl
        if t["result"] == "target":
            wins += 1
        if t["result"] == "breakeven":
            breakevens += 1
        if t["result"] == "liquidation":
            liquidation_count += 1

    losses = len(all_trades) - wins - breakevens
    win_rate = wins / len(all_trades) * 100
    non_loss_rate = (wins + breakevens) / len(all_trades) * 100

    return {
        "symbol": display_symbol,
        "market": market,
        "total_pnl": total_pnl,
        "trades": len(all_trades),
        "wins": wins,
        "breakevens": breakevens,
        "losses": losses,
        "win_rate": win_rate,
        "non_loss_rate": non_loss_rate,
        "days_with_trades": days_with_trades,
        "liquidations": liquidation_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol monthly backtest")
    parser.add_argument(
        "--month",
        help="Month as YYYY-MM (default: last full month)",
        default=None,
    )
    parser.add_argument(
        "--start-date",
        help="Start date as YYYY-MM-DD (overrides --month when used with --end-date)",
        default=None,
    )
    parser.add_argument(
        "--end-date",
        help="End date as YYYY-MM-DD (inclusive, overrides --month when used with --start-date)",
        default=None,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbol list (space-separated). Default: built-in universe.",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel worker threads (default: 6). Higher = faster but may hit rate limits.",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=DEFAULT_LEVERAGE,
        help="Cross leverage used for liquidation and PnL calculations (default: 20).",
    )
    parser.add_argument(
        "--be-stop",
        action="store_true",
        help="Enable break-even stop (move stop to entry after favorable move reaches trigger R).",
    )
    parser.add_argument(
        "--be-trigger-r",
        type=float,
        default=1.0,
        help="R-multiple to arm break-even stop when --be-stop is enabled (default: 1.0).",
    )
    parser.add_argument(
        "--target-r",
        type=float,
        default=2.0,
        help="Profit target in R-multiple from entry (default: 2.0).",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=0,
        help="Cap number of trades counted per symbol/day (0 = unlimited).",
    )
    args = parser.parse_args()

    now_ny = datetime.now(NY_TZ)
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        print("ERROR: provide both --start-date and --end-date for range backtest")
        sys.exit(1)

    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: --start-date and --end-date must be YYYY-MM-DD")
            sys.exit(1)
        if end_date < start_date:
            print("ERROR: --end-date must be on or after --start-date")
            sys.exit(1)
        trading_dates = list(_iter_days(start_date, end_date))
        period_label = f"{start_date.isoformat()} to {end_date.isoformat()}"
    else:
        # ── Resolve target month ─────────────────────────────────────────────
        if args.month:
            try:
                month_dt = datetime.strptime(args.month, "%Y-%m")
            except ValueError:
                print(f"ERROR: --month must be in YYYY-MM format, got '{args.month}'")
                sys.exit(1)
            year, month = month_dt.year, month_dt.month
        else:
            first_of_this_month = now_ny.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month_end = first_of_this_month - timedelta(days=1)
            year, month = last_month_end.year, last_month_end.month

        days_in_month = calendar.monthrange(year, month)[1]
        trading_dates = [date(year, month, day) for day in range(1, days_in_month + 1)]
        period_label = f"{year}-{month:02d}"

    today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    symbols = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_SYMBOLS
    leverage = clamp_leverage(args.leverage)
    max_trades_per_day = args.max_trades_per_day if args.max_trades_per_day > 0 else None

    print(f"\n{'=' * 72}")
    print(
        f"  Multi-Symbol Backtest  |  {period_label}  |  {len(symbols)} symbols  |  "
        f"{args.workers} workers  |  {leverage:.1f}x"
    )
    if args.be_stop:
        print(
            f"  Risk mgmt: break-even ON (trigger {args.be_trigger_r:.2f}R), "
            f"target {args.target_r:.2f}R, max trades/day {max_trades_per_day or 'unlimited'}"
        )
    else:
        print(
            f"  Risk mgmt: break-even OFF, target {args.target_r:.2f}R, "
            f"max trades/day {max_trades_per_day or 'unlimited'}"
        )
    print(f"{'=' * 72}")
    print(f"  Scanning... (this may take a few minutes)\n")

    results = []
    completed = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                backtest_symbol,
                sym,
                trading_dates,
                today_midnight_ny,
                leverage,
                args.be_stop,
                args.be_trigger_r,
                args.target_r,
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
                _safe_print(f"  [{completed:>3}/{total}] {sym:<22} ERROR: {e}")
                continue

            if result is not None:
                status = (
                    f"{result['total_pnl']:+.2f}%  "
                    f"({result['trades']} trades, {result['wins']}W/{result['losses']}L, Liq {result['liquidations']})"
                )
                results.append(result)
            else:
                status = "no trades"

            _safe_print(f"  [{completed:>3}/{total}] {sym:<22} {status}")

    # ── Filter & sort ────────────────────────────────────────────────────────
    positive = sorted(
        [r for r in results if r["total_pnl"] > 0],
        key=lambda r: r["total_pnl"],
        reverse=True,
    )
    negative = sorted(
        [r for r in results if r["total_pnl"] <= 0],
        key=lambda r: r["total_pnl"],
    )

    # ── Print final tables ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  RESULTS  {period_label}  —  Profitable  ({len(positive)} of {len(results)} symbols with trades)")
    print(f"{'=' * 72}")

    if not positive:
        print("  No symbols had a positive PnL this month.")
    else:
        print(
            f"  {'#':<4} {'Symbol':<22} {'Market':<7} {'LevPnL':>9} {'Trades':>7} {'W':>4} {'BE':>4} {'L':>4} {'Liq':>5} {'WR':>7} {'NLR':>7} {'Days':>5}"
        )
        print(f"  {'-' * 76}")
        for rank, r in enumerate(positive, 1):
            print(
                f"  {rank:<4} {r['symbol']:<22} {r['market'].upper():<7} "
                f"{r['total_pnl']:>+8.2f}% "
                f"{r['trades']:>7} "
                f"{r['wins']:>4} "
                f"{r['breakevens']:>4} "
                f"{r['losses']:>4} "
                f"{r['liquidations']:>5} "
                f"{r['win_rate']:>6.1f}% "
                f"{r['non_loss_rate']:>6.1f}% "
                f"{r['days_with_trades']:>5}"
            )

    if negative:
        print(f"\n{'=' * 72}")
        print(f"  Unprofitable  ({len(negative)} symbols)")
        print(f"{'=' * 72}")
        print(
            f"  {'Symbol':<22} {'Market':<7} {'LevPnL':>9} {'Trades':>7} {'W':>4} {'BE':>4} {'L':>4} {'Liq':>5} {'WR':>7} {'NLR':>7}"
        )
        print(f"  {'-' * 72}")
        for r in negative:
            print(
                f"  {r['symbol']:<22} {r['market'].upper():<7} "
                f"{r['total_pnl']:>+8.2f}% "
                f"{r['trades']:>7} "
                f"{r['wins']:>4} "
                f"{r['breakevens']:>4} "
                f"{r['losses']:>4} "
                f"{r['liquidations']:>5} "
                f"{r['win_rate']:>6.1f}% "
                f"{r['non_loss_rate']:>6.1f}%"
            )

    no_trade_count = len(symbols) - len(results)
    print(f"\n  Symbols with no trades this month: {no_trade_count}")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
