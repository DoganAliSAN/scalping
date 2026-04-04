import argparse
import csv
from pathlib import Path

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


def key_to_dict(key):
    return {k: v for k, v in zip(KEY_FIELDS, key)}


def load_month_csv(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None, {}
    month = rows[0].get("month", path.stem)
    keyed = {setup_key(r): r for r in rows}
    return month, keyed


def parse_args():
    p = argparse.ArgumentParser(
        description="Find best setup across multiple monthly setup CSV files"
    )
    p.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="Monthly CSV files (e.g. setup_results_2025-01.csv ... setup_results_2025-12.csv)",
    )
    p.add_argument(
        "--require-all-profitable",
        action="store_true",
        help="Only report setups with pnl_usd > 0 in every provided month",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top setups to print (default: 10)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    monthly = []
    for f in args.csv_files:
        path = Path(f)
        if not path.exists():
            raise SystemExit(f"ERROR: file not found: {path}")
        month, keyed = load_month_csv(path)
        if not keyed:
            raise SystemExit(f"ERROR: empty CSV: {path}")
        monthly.append((month, keyed, path))

    months = [m for m, _k, _p in monthly]
    common = set(monthly[0][1].keys())
    for _m, keyed, _p in monthly[1:]:
        common &= set(keyed.keys())

    if not common:
        raise SystemExit("ERROR: no common setup keys across provided CSV files")

    results = []
    for k in common:
        month_stats = []
        total_pnl = 0.0
        total_return = 0.0
        profitable_months = 0
        worst_pnl = None

        for month, keyed, _p in monthly:
            r = keyed[k]
            pnl = float(r["pnl_usd"])
            ret = float(r["return_pct"])
            final_balance = float(r["final_balance"])
            dd = float(r["max_drawdown_pct"])

            month_stats.append(
                {
                    "month": month,
                    "pnl_usd": pnl,
                    "return_pct": ret,
                    "final_balance": final_balance,
                    "max_drawdown_pct": dd,
                }
            )

            total_pnl += pnl
            total_return += ret
            if pnl > 0:
                profitable_months += 1
            if worst_pnl is None or pnl < worst_pnl:
                worst_pnl = pnl

        all_profitable = profitable_months == len(monthly)
        if args.require_all_profitable and not all_profitable:
            continue

        avg_return = total_return / len(monthly)
        results.append(
            {
                "key": k,
                "setup": key_to_dict(k),
                "months": month_stats,
                "total_pnl_usd": total_pnl,
                "avg_return_pct": avg_return,
                "worst_month_pnl_usd": worst_pnl,
                "profitable_months": profitable_months,
                "month_count": len(monthly),
                "all_profitable": all_profitable,
            }
        )

    if not results:
        print("No setups matched the selected criteria.")
        return

    # Ranking priority:
    # 1) more profitable months, 2) higher total PnL, 3) higher worst-month PnL
    results.sort(
        key=lambda x: (
            x["profitable_months"],
            x["total_pnl_usd"],
            x["worst_month_pnl_usd"],
        ),
        reverse=True,
    )

    best = results[0]

    print("=" * 72)
    print("BEST SETUP ACROSS PROVIDED MONTHS")
    print("=" * 72)
    print(f"Months: {', '.join(months)}")
    print(f"Profitable months: {best['profitable_months']}/{best['month_count']}")
    print(f"Total PnL USD: {best['total_pnl_usd']:+.2f}")
    print(f"Average return: {best['avg_return_pct']:+.2f}%")
    print(f"Worst month PnL: {best['worst_month_pnl_usd']:+.2f}")
    print(f"All months profitable: {best['all_profitable']}")
    print("Setup:")
    for k, v in best["setup"].items():
        print(f"  {k}: {v}")

    print("\nPer-month performance for best setup:")
    for ms in best["months"]:
        print(
            f"  {ms['month']}: pnl {ms['pnl_usd']:+.2f} USD | "
            f"return {ms['return_pct']:+.2f}% | "
            f"final {ms['final_balance']:.2f} | dd {ms['max_drawdown_pct']:.2f}%"
        )

    print("\nTop setups:")
    for i, r in enumerate(results[: args.top], 1):
        print(
            f"  {i:>2}. months+ {r['profitable_months']}/{r['month_count']} | "
            f"sum {r['total_pnl_usd']:+.2f} USD | "
            f"avg {r['avg_return_pct']:+.2f}% | "
            f"worst {r['worst_month_pnl_usd']:+.2f}"
        )


if __name__ == "__main__":
    main()
