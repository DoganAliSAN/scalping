import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from backtest_month import DEFAULT_SYMBOLS
from export_setup_results_csv import month_completed_dates, parse_month, run_account_month
from main import NY_TZ
from simulate import build_cache

KEY_FIELDS = [
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
]


def setup_key(row):
    return tuple(row[k] for k in KEY_FIELDS)


def load_month_csv(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    month = rows[0].get("month", path.stem)
    keyed = {setup_key(r): r for r in rows}
    return month, keyed


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Take top setups by profitable-month count from monthly CSVs, then "
            "re-simulate them with chained month-to-month compounding balance."
        )
    )
    p.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="Monthly CSV files (e.g. setup_results_2025-01.csv ... setup_results_2025-12.csv)",
    )
    p.add_argument("--top", type=int, default=12, help="Top N candidate setups to chain-test")
    p.add_argument("--start-balance", type=float, default=800.0, help="Initial balance before first month")
    p.add_argument("--position-size", type=float, default=100.0, help="USD allocation per position")
    p.add_argument("--leverage", type=float, default=20.0, help="Leverage for strategy backtests")
    p.add_argument("--workers", type=int, default=8, help="Parallel worker count for cache builds")
    return p.parse_args()


def rank_candidates(monthly):
    common = set(monthly[0][1].keys())
    for _month, keyed, _path in monthly[1:]:
        common &= set(keyed.keys())

    if not common:
        raise RuntimeError("no common setup keys across provided CSV files")

    ranked = []
    for key in common:
        profitable_months = 0
        total_pnl = 0.0
        worst_pnl = None

        for _month, keyed, _path in monthly:
            row = keyed[key]
            pnl = float(row["pnl_usd"])
            total_pnl += pnl
            if pnl > 0:
                profitable_months += 1
            if worst_pnl is None or pnl < worst_pnl:
                worst_pnl = pnl

        ranked.append(
            {
                "key": key,
                "profitable_months": profitable_months,
                "month_count": len(monthly),
                "sum_pnl_usd": total_pnl,
                "worst_month_pnl_usd": worst_pnl,
            }
        )

    ranked.sort(
        key=lambda x: (x["profitable_months"], x["sum_pnl_usd"], x["worst_month_pnl_usd"]),
        reverse=True,
    )
    return ranked


def month_cache_for_profile(
    cache_store,
    month_str,
    profile,
    leverage,
    workers,
    today_midnight_ny,
    symbols,
):
    key = (
        month_str,
        profile["be_stop"],
        profile["be_trigger_r"],
        profile["target_r"],
        profile["max_tpd"],
    )
    if key in cache_store:
        return cache_store[key]

    year, month = parse_month(month_str)
    month_dates = month_completed_dates(year, month, today_midnight_ny.date())
    if not month_dates:
        raise RuntimeError(f"no completed dates in month: {month_str}")

    max_monthly_window = 45
    fetch_start = month_dates[0] - timedelta(days=max_monthly_window)
    fetch_end = month_dates[-1]
    date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

    cache = build_cache(
        symbols,
        date_range,
        today_midnight_ny,
        leverage,
        workers,
        profile["be_stop"],
        profile["be_trigger_r"],
        profile["target_r"],
        profile["max_tpd"],
    )

    payload = {"cache": cache, "month_dates": month_dates}
    cache_store[key] = payload
    return payload


def main():
    args = parse_args()

    if args.start_balance <= 0:
        raise SystemExit("ERROR: --start-balance must be > 0")
    if args.position_size <= 0:
        raise SystemExit("ERROR: --position-size must be > 0")

    monthly = []
    for file_name in args.csv_files:
        path = Path(file_name)
        if not path.exists():
            raise SystemExit(f"ERROR: file not found: {path}")
        month, keyed = load_month_csv(path)
        monthly.append((month, keyed, path))

    baseline_ranked = rank_candidates(monthly)
    candidates = baseline_ranked[: args.top]

    now_ny = datetime.now(NY_TZ)
    today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    symbols = DEFAULT_SYMBOLS

    print("=" * 80)
    print("CHAINED MONTH-BY-MONTH TEST (COMPOUNDING BALANCE)")
    print("=" * 80)
    print(f"Months: {', '.join([m for m, _k, _p in monthly])}")
    print(f"Candidates from baseline ranking: top {len(candidates)}")

    cache_store = {}
    chained_results = []

    for i, cand in enumerate(candidates, 1):
        key = cand["key"]

        profile = {
            "be_stop": key[0] == "True",
            "be_trigger_r": float(key[1]),
            "target_r": float(key[2]),
            "max_tpd": int(key[3]),
        }
        if profile["max_tpd"] <= 0:
            profile["max_tpd"] = None

        setup = {
            "weekly_days": int(key[4]),
            "monthly_days": int(key[5]),
            "min_trades": int(key[6]),
            "min_weekly_wr": float(key[7]),
            "min_monthly_wr": float(key[8]),
            "min_weekly_nlr": float(key[9]),
            "min_monthly_nlr": float(key[10]),
            "top_n": int(key[11]),
        }

        balance = args.start_balance
        month_rows = []
        profitable_months = 0

        for month_str, _keyed, _path in monthly:
            payload = month_cache_for_profile(
                cache_store,
                month_str,
                profile,
                args.leverage,
                args.workers,
                today_midnight_ny,
                symbols,
            )
            result = run_account_month(
                payload["cache"],
                symbols,
                payload["month_dates"],
                setup["weekly_days"],
                setup["monthly_days"],
                setup["min_trades"],
                setup["min_weekly_wr"],
                setup["min_monthly_wr"],
                setup["min_weekly_nlr"],
                setup["min_monthly_nlr"],
                setup["top_n"],
                balance,
                args.position_size,
            )
            month_pnl = result["pnl_usd"]
            if month_pnl > 0:
                profitable_months += 1
            balance = result["final_balance"]
            month_rows.append(
                {
                    "month": month_str,
                    "start": result["final_balance"] - result["pnl_usd"],
                    "pnl": month_pnl,
                    "end": result["final_balance"],
                    "max_dd": result["max_drawdown_pct"],
                }
            )

            if balance <= 0:
                break

        chained_results.append(
            {
                "rank_source": i,
                "key": key,
                "profile": profile,
                "setup": setup,
                "baseline_profitable_months": cand["profitable_months"],
                "baseline_sum_pnl_usd": cand["sum_pnl_usd"],
                "chained_profitable_months": profitable_months,
                "chained_months_tested": len(month_rows),
                "final_balance": balance,
                "year_pnl_usd": balance - args.start_balance,
                "month_rows": month_rows,
            }
        )

    chained_results.sort(
        key=lambda x: (x["final_balance"], x["chained_profitable_months"], x["year_pnl_usd"]),
        reverse=True,
    )

    print("\nTop results by TRUE year-end balance:")
    for j, row in enumerate(chained_results, 1):
        print(
            f"  {j:>2}. final {row['final_balance']:.2f} | "
            f"year pnl {row['year_pnl_usd']:+.2f} | "
            f"chained months+ {row['chained_profitable_months']}/{row['chained_months_tested']} | "
            f"source-rank {row['rank_source']}"
        )

    best = chained_results[0]
    print("\n" + "-" * 80)
    print("Best chained setup details")
    print("-" * 80)
    print("Profile:")
    print(
        f"  be_stop={best['profile']['be_stop']}, "
        f"be_trigger_r={best['profile']['be_trigger_r']}, "
        f"target_r={best['profile']['target_r']}, "
        f"max_tpd={best['profile']['max_tpd']}"
    )
    print("Setup:")
    for name in [
        "weekly_days",
        "monthly_days",
        "min_trades",
        "min_weekly_wr",
        "min_monthly_wr",
        "min_weekly_nlr",
        "min_monthly_nlr",
        "top_n",
    ]:
        print(f"  {name}={best['setup'][name]}")

    print("Monthly path (chained):")
    for m in best["month_rows"]:
        print(
            f"  {m['month']}: start {m['start']:.2f} -> "
            f"pnl {m['pnl']:+.2f} -> end {m['end']:.2f} "
            f"(dd {m['max_dd']:.2f}%)"
        )


if __name__ == "__main__":
    main()
