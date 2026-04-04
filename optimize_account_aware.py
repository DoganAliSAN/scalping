from datetime import datetime, timedelta, date
import argparse
import calendar

from simulate import build_cache, _rank_for_day
from backtest_month import DEFAULT_SYMBOLS
from main import NY_TZ


def month_completed_dates(year, month, today):
    days = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    return [d for d in days if d < today]


def to_cmd(row, year, month):
    p = row["profile"]
    max_tpd = p["max_tpd"] if p["max_tpd"] is not None else 0
    parts = [
        f"python simulate.py --month {year:04d}-{month:02d}",
        f"--weekly-days {row['weekly_days']}",
        f"--monthly-days {row['monthly_days']}",
        f"--min-trades {row['min_trades']}",
        f"--min-weekly-wr {row['min_weekly_wr']}",
        f"--min-monthly-wr {row['min_monthly_wr']}",
        f"--min-weekly-nlr {row['min_weekly_nlr']}",
        f"--min-monthly-nlr {row['min_monthly_nlr']}",
        f"--top-n {row['top_n']}",
        f"--max-trades-per-day {max_tpd}",
        f"--target-r {p['target_r']}",
    ]
    if p["be"]:
        parts.append("--be-stop")
        parts.append(f"--be-trigger-r {p['be_trigger_r']}")
    else:
        parts.append("--no-be-stop")
    return " ".join(parts)


def better(a, b):
    if b is None:
        return True
    if a["final_balance"] != b["final_balance"]:
        return a["final_balance"] > b["final_balance"]
    if a["max_drawdown_pct"] != b["max_drawdown_pct"]:
        return a["max_drawdown_pct"] < b["max_drawdown_pct"]
    return a["total_trades"] > b["total_trades"]


def main():
    parser = argparse.ArgumentParser(description="Account-aware setup optimizer")
    parser.add_argument("--month", default="2026-03", help="Month to optimize: YYYY-MM")
    parser.add_argument("--start-balance", type=float, default=800.0, help="Starting account balance in USD")
    parser.add_argument("--position-size", type=float, default=100.0, help="USD used per symbol position")
    parser.add_argument("--leverage", type=float, default=20.0, help="Leverage passed to backtests")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for cache build")
    parser.add_argument("--min-trades-valid", type=int, default=20, help="Minimum trades for robust winner")
    args = parser.parse_args()

    try:
        year, month = map(int, args.month.split("-"))
    except Exception:
        raise SystemExit("ERROR: --month must be YYYY-MM")

    if args.position_size <= 0:
        raise SystemExit("ERROR: --position-size must be > 0")

    symbols = DEFAULT_SYMBOLS
    now_ny = datetime.now(NY_TZ)
    today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    today = today_midnight_ny.date()

    month_dates = month_completed_dates(year, month, today)
    if not month_dates:
        raise SystemExit("ERROR: no completed dates for the selected month")

    max_monthly_window = 45
    fetch_start = month_dates[0] - timedelta(days=max_monthly_window)
    fetch_end = month_dates[-1]
    date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

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

    best_overall = None
    best_valid = None
    tested = 0

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
                                        balance = args.start_balance
                                        peak = balance
                                        max_dd = 0.0

                                        total_trades = 0
                                        wins = 0
                                        bes = 0
                                        losses = 0
                                        ranked_days = 0
                                        traded_days = 0

                                        for sim_date in month_dates:
                                            ranked = _rank_for_day(
                                                cache,
                                                symbols,
                                                sim_date,
                                                wd,
                                                md,
                                                min_trades,
                                                min_weekly_wr,
                                                min_monthly_wr,
                                                min_weekly_nlr,
                                                min_monthly_nlr,
                                                top_n,
                                            )
                                            if ranked:
                                                ranked_days += 1

                                            # Account-aware capacity: each symbol position needs one slot.
                                            slots = int(balance // args.position_size)
                                            if slots <= 0:
                                                balance = 0.0
                                                break

                                            selected = ranked[:slots] if len(ranked) > slots else ranked
                                            if not selected:
                                                continue

                                            day_pnl_usd = 0.0
                                            day_had_trades = False
                                            for r in selected:
                                                day_result = cache[r["symbol"]].get(sim_date)
                                                if day_result is None:
                                                    continue
                                                day_pnl_usd += (day_result["total_pnl"] / 100.0) * args.position_size
                                                total_trades += day_result["trades"]
                                                wins += day_result["wins"]
                                                bes += day_result.get("breakevens", 0)
                                                losses += day_result["losses"]
                                                day_had_trades = day_had_trades or (day_result["trades"] > 0)

                                            if day_had_trades:
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
                                            continue

                                        row = {
                                            "profile": profile,
                                            "weekly_days": wd,
                                            "monthly_days": md,
                                            "min_trades": min_trades,
                                            "min_weekly_wr": min_weekly_wr,
                                            "min_monthly_wr": min_monthly_wr,
                                            "min_weekly_nlr": min_weekly_nlr,
                                            "min_monthly_nlr": min_monthly_nlr,
                                            "top_n": top_n,
                                            "total_trades": total_trades,
                                            "wins": wins,
                                            "breakevens": bes,
                                            "losses": losses,
                                            "win_rate": wins / total_trades * 100.0,
                                            "non_loss_rate": (wins + bes) / total_trades * 100.0,
                                            "start_balance": args.start_balance,
                                            "final_balance": balance,
                                            "pnl_usd": balance - args.start_balance,
                                            "return_pct": ((balance / args.start_balance) - 1.0) * 100.0,
                                            "max_drawdown_pct": max_dd,
                                            "ranked_days": ranked_days,
                                            "traded_days": traded_days,
                                        }

                                        if better(row, best_overall):
                                            best_overall = row

                                        if total_trades >= args.min_trades_valid and better(row, best_valid):
                                            best_valid = row

                                        if tested % 5000 == 0:
                                            print(f"  Tested {tested} setups so far...")

    print("\n=== ACCOUNT-AWARE BEST (overall) ===")
    print(best_overall)
    if best_overall:
        print(to_cmd(best_overall, year, month))

    print("\n=== ACCOUNT-AWARE BEST (valid min trades) ===")
    print(best_valid)
    if best_valid:
        print(to_cmd(best_valid, year, month))


if __name__ == "__main__":
    main()
