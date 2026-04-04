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

now_ny = datetime.now(NY_TZ)
today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
today = today_midnight_ny.date()
month_dates = [date(YEAR, MONTH, d) for d in range(1, calendar.monthrange(YEAR, MONTH)[1] + 1)]
month_dates = [d for d in month_dates if d < today]

max_monthly_days = 45
fetch_start = month_dates[0] - timedelta(days=max_monthly_days)
fetch_end = month_dates[-1]
date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

# Use best risk profile candidate from previous sweep
cache = build_cache(symbols, date_range, today_midnight_ny, leverage, workers, True, 0.8, 3)

weekly_opts = [3, 5, 7, 10, 14]
monthly_opts = [14, 21, 30, 45]
min_trades_opts = [1, 2, 3]
wr_opts = [0, 20, 30, 40, 50]
top_n_opts = [1, 2, 3, 5, 8, 0]

best_any = None
best_20 = None
best_50 = None
count = 0

for wd in weekly_opts:
    for md in monthly_opts:
        if md < wd:
            continue
        for mt in min_trades_opts:
            for wwr in wr_opts:
                for mwr in wr_opts:
                    for top_n in top_n_opts:
                        count += 1
                        trades = wins = losses = 0
                        pnl = 0.0
                        days = 0
                        for sim_date in month_dates:
                            ranked = _rank_for_day(cache, symbols, sim_date, wd, md, mt, wwr, mwr, top_n)
                            if ranked:
                                days += 1
                            for r in ranked:
                                dr = cache[r["symbol"]].get(sim_date)
                                if dr is None:
                                    continue
                                trades += dr["trades"]
                                wins += dr["wins"]
                                losses += dr["losses"]
                                pnl += dr["total_pnl"]
                        if trades == 0:
                            continue
                        wr = wins / trades * 100
                        row = {
                            "wd": wd,
                            "md": md,
                            "mt": mt,
                            "wwr": wwr,
                            "mwr": mwr,
                            "top_n": top_n,
                            "trades": trades,
                            "wins": wins,
                            "losses": losses,
                            "wr": wr,
                            "pnl": pnl,
                            "days": days,
                        }
                        if best_any is None or row["wr"] > best_any["wr"]:
                            best_any = row
                        if trades >= 20 and (best_20 is None or row["wr"] > best_20["wr"]):
                            best_20 = row
                        if trades >= 50 and (best_50 is None or row["wr"] > best_50["wr"]):
                            best_50 = row

print("tested", count, "combos")
print("best_any", best_any)
print("best_20", best_20)
print("best_50", best_50)
print("hit70_any", best_any is not None and best_any["wr"] >= 70)
print("hit70_20", best_20 is not None and best_20["wr"] >= 70)
print("hit70_50", best_50 is not None and best_50["wr"] >= 70)
