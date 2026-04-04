from datetime import datetime, timedelta, date
import calendar

from simulate import build_cache, _rank_for_day
from backtest_month import DEFAULT_SYMBOLS
from main import NY_TZ

START_BAL = 800.0
POSITION_SIZE = 100.0
YEAR = 2026
MONTH = 3

params = dict(
    weekly_days=3,
    monthly_days=21,
    min_trades=1,
    min_weekly_wr=30.0,
    min_monthly_wr=10.0,
    min_weekly_nlr=60.0,
    min_monthly_nlr=40.0,
    top_n=0,
    leverage=20.0,
    workers=8,
    enable_be=True,
    be_trigger_r=1.0,
    target_r=1.5,
    max_trades_per_day=3,
    daily_loss_limit=-10.0,
)

now_ny = datetime.now(NY_TZ)
today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
today = today_midnight_ny.date()

month_dates = [date(YEAR, MONTH, d) for d in range(1, calendar.monthrange(YEAR, MONTH)[1] + 1)]
month_dates = [d for d in month_dates if d < today]
fetch_start = month_dates[0] - timedelta(days=params["monthly_days"])
fetch_end = month_dates[-1]
date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

cache = build_cache(
    DEFAULT_SYMBOLS,
    date_range,
    today_midnight_ny,
    params["leverage"],
    params["workers"],
    params["enable_be"],
    params["be_trigger_r"],
    params["target_r"],
    params["max_trades_per_day"],
)

balance = START_BAL
peak = START_BAL
max_dd = 0.0

month_trades = month_wins = month_bes = month_losses = 0

print("date,ranked,slots_used,day_pnl_pct,day_pnl_usd,equity")
for sim_date in month_dates:
    ranked = _rank_for_day(
        cache,
        DEFAULT_SYMBOLS,
        sim_date,
        params["weekly_days"],
        params["monthly_days"],
        params["min_trades"],
        params["min_weekly_wr"],
        params["min_monthly_wr"],
        params["min_weekly_nlr"],
        params["min_monthly_nlr"],
        params["top_n"],
    )

    slots = int(balance // POSITION_SIZE)
    if slots <= 0:
        balance = 0.0
        print(f"{sim_date},0,0,0.00,0.00,{balance:.2f}")
        break

    selected = ranked[:slots] if slots < len(ranked) else ranked

    day_pnl_pct = 0.0
    day_pnl_usd = 0.0
    used = 0

    for r in selected:
        result = cache[r["symbol"]].get(sim_date)
        if result is None:
            continue

        if params["daily_loss_limit"] < 0 and (day_pnl_pct + result["total_pnl"]) <= params["daily_loss_limit"]:
            break

        day_pnl_pct += result["total_pnl"]
        day_pnl_usd += (result["total_pnl"] / 100.0) * POSITION_SIZE
        used += 1

        month_trades += result["trades"]
        month_wins += result["wins"]
        month_bes += result.get("breakevens", 0)
        month_losses += result["losses"]

    balance += day_pnl_usd
    if balance < 0:
        balance = 0.0

    if balance > peak:
        peak = balance
    dd = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
    if dd > max_dd:
        max_dd = dd

    print(f"{sim_date},{len(ranked)},{used},{day_pnl_pct:+.2f},{day_pnl_usd:+.2f},{balance:.2f}")

ret_pct = (balance / START_BAL - 1.0) * 100.0 if START_BAL > 0 else 0.0
wr = (month_wins / month_trades * 100.0) if month_trades else 0.0
nlr = ((month_wins + month_bes) / month_trades * 100.0) if month_trades else 0.0

print("---")
print(f"FINAL_BALANCE={balance:.2f}")
print(f"PNL_USD={balance - START_BAL:+.2f}")
print(f"RETURN_PCT={ret_pct:+.2f}")
print(f"MAX_DRAWDOWN_PCT={max_dd:.2f}")
print(f"TRADES={month_trades}")
print(f"WINS_BE_LOSSES={month_wins}/{month_bes}/{month_losses}")
print(f"WR={wr:.2f}")
print(f"NLR={nlr:.2f}")
