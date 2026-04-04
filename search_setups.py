from datetime import datetime, timedelta, date
import calendar

from simulate import build_cache, _rank_for_day
from backtest_month import DEFAULT_SYMBOLS
from main import NY_TZ

YEAR = 2026
MONTH = 3
symbols = DEFAULT_SYMBOLS
workers = 8
leverage = 20.0
weekly_days = 7
monthly_days = 30

now_ny = datetime.now(NY_TZ)
today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
today = today_midnight_ny.date()

month_dates = [date(YEAR, MONTH, d) for d in range(1, calendar.monthrange(YEAR, MONTH)[1] + 1)]
month_dates = [d for d in month_dates if d < today]

fetch_start = month_dates[0] - timedelta(days=monthly_days)
fetch_end = month_dates[-1]
date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

risk_profiles = [
    {"name": "baseline", "be": False, "be_r": 1.0, "max_tpd": None},
    {"name": "be_1.0_tpd3", "be": True, "be_r": 1.0, "max_tpd": 3},
    {"name": "be_0.8_tpd3", "be": True, "be_r": 0.8, "max_tpd": 3},
    {"name": "be_1.2_tpd3", "be": True, "be_r": 1.2, "max_tpd": 3},
    {"name": "be_1.0_tpd2", "be": True, "be_r": 1.0, "max_tpd": 2},
]

min_trades_opts = [1, 2, 3, 4]
wr_opts = [0, 20, 30, 40, 50, 60, 70]
top_n_opts = [1, 2, 3, 5, 8, 0]

all_results = []

for rp in risk_profiles:
    print(f"\n=== building cache for {rp['name']} ===")
    cache = build_cache(
        symbols,
        date_range,
        today_midnight_ny,
        leverage,
        workers,
        rp["be"],
        rp["be_r"],
        rp["max_tpd"],
    )

    profile_best_any = None
    profile_best_20 = None
    profile_best_50 = None

    for min_trades in min_trades_opts:
        for min_weekly_wr in wr_opts:
            for min_monthly_wr in wr_opts:
                for top_n in top_n_opts:
                    total_trades = 0
                    wins = 0
                    losses = 0
                    total_pnl = 0.0
                    selected_days = 0

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
                            top_n,
                        )
                        if ranked:
                            selected_days += 1
                        for r in ranked:
                            day_res = cache[r["symbol"]].get(sim_date)
                            if day_res is None:
                                continue
                            total_trades += day_res["trades"]
                            wins += day_res["wins"]
                            losses += day_res["losses"]
                            total_pnl += day_res["total_pnl"]

                    if total_trades == 0:
                        continue

                    wr = wins / total_trades * 100
                    row = {
                        "profile": rp["name"],
                        "be": rp["be"],
                        "be_r": rp["be_r"],
                        "max_tpd": rp["max_tpd"],
                        "min_trades": min_trades,
                        "min_weekly_wr": min_weekly_wr,
                        "min_monthly_wr": min_monthly_wr,
                        "top_n": top_n,
                        "total_trades": total_trades,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": wr,
                        "total_pnl": total_pnl,
                        "days_with_ranked": selected_days,
                    }
                    all_results.append(row)

                    if profile_best_any is None or wr > profile_best_any["win_rate"]:
                        profile_best_any = row
                    if total_trades >= 20:
                        if profile_best_20 is None or wr > profile_best_20["win_rate"]:
                            profile_best_20 = row
                    if total_trades >= 50:
                        if profile_best_50 is None or wr > profile_best_50["win_rate"]:
                            profile_best_50 = row

    print("best_any:", profile_best_any)
    print("best_>=20:", profile_best_20)
    print("best_>=50:", profile_best_50)

all_results_sorted = sorted(all_results, key=lambda x: x["win_rate"], reverse=True)
print("\n=== global top 20 by win rate ===")
for r in all_results_sorted[:20]:
    print(r)

print("\n=== global top 20 by win rate with >=20 trades ===")
filtered20 = [r for r in all_results_sorted if r["total_trades"] >= 20]
for r in filtered20[:20]:
    print(r)

print("\n=== global top 20 by win rate with >=50 trades ===")
filtered50 = [r for r in all_results_sorted if r["total_trades"] >= 50]
for r in filtered50[:20]:
    print(r)

pass70_any = [r for r in all_results if r["win_rate"] >= 70]
pass70_20 = [r for r in pass70_any if r["total_trades"] >= 20]
pass70_50 = [r for r in pass70_any if r["total_trades"] >= 50]

print(f"\ncount winrate>=70 any-trades: {len(pass70_any)}")
print(f"count winrate>=70 with >=20 trades: {len(pass70_20)}")
print(f"count winrate>=70 with >=50 trades: {len(pass70_50)}")
