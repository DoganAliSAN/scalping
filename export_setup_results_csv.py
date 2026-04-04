import argparse
import calendar
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from backtest_month import DEFAULT_SYMBOLS
from main import NY_TZ
from simulate import _rank_for_day, build_cache


def month_completed_dates(year, month, today):
    days = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    return [d for d in days if d < today]


def parse_month(month_str):
    try:
        year, month = map(int, month_str.split("-"))
    except Exception as exc:
        raise ValueError("--month must be YYYY-MM") from exc
    if month < 1 or month > 12:
        raise ValueError("--month must be YYYY-MM")
    return year, month


def run_account_month(
    cache,
    symbols,
    month_dates,
    weekly_days,
    monthly_days,
    min_trades,
    min_weekly_wr,
    min_monthly_wr,
    min_weekly_nlr,
    min_monthly_nlr,
    top_n,
    start_balance,
    position_size,
):
    balance = start_balance
    peak = balance
    max_dd = 0.0

    total_trades = 0
    wins = 0
    breakevens = 0
    losses = 0
    liquidations = 0

    ranked_days = 0
    traded_days = 0

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

        if ranked:
            ranked_days += 1

        # Capital-constrained slots for this day.
        slots = int(balance // position_size)
        if slots <= 0:
            balance = 0.0
            break

        selected = ranked[:slots] if len(ranked) > slots else ranked
        if not selected:
            continue

        day_pnl_usd = 0.0
        day_has_trade = False

        for r in selected:
            day_result = cache[r["symbol"]].get(sim_date)
            if day_result is None:
                continue

            day_pnl_usd += (day_result["total_pnl"] / 100.0) * position_size
            total_trades += day_result["trades"]
            wins += day_result["wins"]
            breakevens += day_result.get("breakevens", 0)
            losses += day_result["losses"]
            liquidations += day_result["liquidations"]
            day_has_trade = day_has_trade or (day_result["trades"] > 0)

        if day_has_trade:
            traded_days += 1

        balance += day_pnl_usd
        if balance < 0:
            balance = 0.0

        if balance > peak:
            peak = balance
        if peak > 0:
            dd = (peak - balance) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        if balance <= 0:
            break

    if total_trades == 0:
        win_rate = 0.0
        nlr = 0.0
    else:
        win_rate = wins / total_trades * 100.0
        nlr = (wins + breakevens) / total_trades * 100.0

    return {
        "final_balance": balance,
        "pnl_usd": balance - start_balance,
        "return_pct": ((balance / start_balance) - 1.0) * 100.0 if start_balance > 0 else 0.0,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "wins": wins,
        "breakevens": breakevens,
        "losses": losses,
        "liquidations": liquidations,
        "win_rate": win_rate,
        "non_loss_rate": nlr,
        "ranked_days": ranked_days,
        "traded_days": traded_days,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export setup parameters and account-aware results to CSV"
    )
    parser.add_argument("--month", default="2026-03", help="Month to evaluate (YYYY-MM)")
    parser.add_argument("--start-balance", type=float, default=800.0, help="Starting account balance in USD")
    parser.add_argument("--position-size", type=float, default=100.0, help="USD allocated per symbol position")
    parser.add_argument("--leverage", type=float, default=20.0, help="Leverage passed to strategy backtests")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count for cache build")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="CSV output path (default: setup_results_<YYYY-MM>.csv in current directory)",
    )
    args = parser.parse_args()

    if args.start_balance <= 0:
        raise SystemExit("ERROR: --start-balance must be > 0")
    if args.position_size <= 0:
        raise SystemExit("ERROR: --position-size must be > 0")

    year, month = parse_month(args.month)

    now_ny = datetime.now(NY_TZ)
    today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    today = today_midnight_ny.date()

    month_dates = month_completed_dates(year, month, today)
    if not month_dates:
        raise SystemExit("ERROR: no completed dates in selected month")

    max_monthly_window = 45
    fetch_start = month_dates[0] - timedelta(days=max_monthly_window)
    fetch_end = month_dates[-1]
    date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

    csv_path = Path(args.csv_out) if args.csv_out else Path(f"setup_results_{year:04d}-{month:02d}.csv")

    symbols = DEFAULT_SYMBOLS

    profiles = [
        {"be": False, "be_trigger_r": 1.0, "target_r": 2.0, "max_tpd": None},
        {"be": True, "be_trigger_r": 0.8, "target_r": 1.0, "max_tpd": None},
        {"be": True, "be_trigger_r": 1.0, "target_r": 1.0, "max_tpd": None},
        {"be": True, "be_trigger_r": 1.0, "target_r": 1.5, "max_tpd": 3},
        {"be": True, "be_trigger_r": 1.0, "target_r": 2.0, "max_tpd": 3},
    ]

    weekly_opts = [3, 5, 7, 10]
    monthly_opts = [14, 21, 30, 45]
    min_trades_opts = [1, 2, 3]
    min_wr_opts = [0, 10, 20, 30]
    min_nlr_opts = [40, 50, 60]
    top_n_opts = [1, 2, 3, 5, 8, 0]

    header = [
        "month",
        "profile_be_stop",
        "profile_be_trigger_r",
        "profile_target_r",
        "profile_max_trades_per_day",
        "weekly_days",
        "monthly_days",
        "min_trades",
        "min_weekly_wr",
        "min_monthly_wr",
        "min_weekly_nlr",
        "min_monthly_nlr",
        "top_n",
        "start_balance",
        "position_size",
        "leverage",
        "final_balance",
        "pnl_usd",
        "return_pct",
        "max_drawdown_pct",
        "total_trades",
        "wins",
        "breakevens",
        "losses",
        "liquidations",
        "win_rate",
        "non_loss_rate",
        "ranked_days",
        "traded_days",
    ]

    tested = 0
    best_row = None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for profile in profiles:
            print(
                f"\n=== cache be={profile['be']} trigger={profile['be_trigger_r']} "
                f"target={profile['target_r']} max_tpd={profile['max_tpd']} ==="
            )
            cache = build_cache(
                symbols,
                date_range,
                today_midnight_ny,
                args.leverage,
                args.workers,
                profile["be"],
                profile["be_trigger_r"],
                profile["target_r"],
                profile["max_tpd"],
            )

            for wd in weekly_opts:
                for md in monthly_opts:
                    if md < wd:
                        continue
                    for min_trades in min_trades_opts:
                        for min_weekly_wr in min_wr_opts:
                            for min_monthly_wr in min_wr_opts:
                                for min_weekly_nlr in min_nlr_opts:
                                    for min_monthly_nlr in min_nlr_opts:
                                        for top_n in top_n_opts:
                                            tested += 1

                                            result = run_account_month(
                                                cache,
                                                symbols,
                                                month_dates,
                                                wd,
                                                md,
                                                min_trades,
                                                min_weekly_wr,
                                                min_monthly_wr,
                                                min_weekly_nlr,
                                                min_monthly_nlr,
                                                top_n,
                                                args.start_balance,
                                                args.position_size,
                                            )

                                            row = {
                                                "month": f"{year:04d}-{month:02d}",
                                                "profile_be_stop": profile["be"],
                                                "profile_be_trigger_r": profile["be_trigger_r"],
                                                "profile_target_r": profile["target_r"],
                                                "profile_max_trades_per_day": profile["max_tpd"] if profile["max_tpd"] is not None else 0,
                                                "weekly_days": wd,
                                                "monthly_days": md,
                                                "min_trades": min_trades,
                                                "min_weekly_wr": min_weekly_wr,
                                                "min_monthly_wr": min_monthly_wr,
                                                "min_weekly_nlr": min_weekly_nlr,
                                                "min_monthly_nlr": min_monthly_nlr,
                                                "top_n": top_n,
                                                "start_balance": args.start_balance,
                                                "position_size": args.position_size,
                                                "leverage": args.leverage,
                                                **result,
                                            }
                                            writer.writerow(row)

                                            if best_row is None or row["final_balance"] > best_row["final_balance"]:
                                                best_row = row

                                            if tested % 5000 == 0:
                                                print(f"  Tested {tested} setups...")
                                                f.flush()

    print(f"\nCSV written: {csv_path}")
    print(f"Total setups exported: {tested}")
    if best_row:
        print(
            "Best final balance: "
            f"{best_row['final_balance']:.2f} USD "
            f"(PnL {best_row['pnl_usd']:+.2f}, Return {best_row['return_pct']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
