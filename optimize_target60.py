from datetime import datetime, timedelta, date
import calendar

from simulate import build_cache, _rank_for_day
from backtest_month import DEFAULT_SYMBOLS
from main import NY_TZ

YEAR = 2026
MONTH = 3
LEVERAGE = 20.0
SYMBOLS = DEFAULT_SYMBOLS
WORKERS = 8

TARGET_NLR = 60.0
MIN_TRADES_FOR_VALID = 20

now_ny = datetime.now(NY_TZ)
today_midnight_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
today = today_midnight_ny.date()

month_dates = [date(YEAR, MONTH, d) for d in range(1, calendar.monthrange(YEAR, MONTH)[1] + 1)]
month_dates = [d for d in month_dates if d < today]

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

best_any = None
best_valid = None
best_valid_over_target = None

for profile in profiles:
    print(f"\n=== cache be={profile['be']} trigger={profile['be_trigger_r']} target={profile['target_r']} max_tpd={profile['max_tpd']} ===")
    cache = build_cache(
        SYMBOLS,
        date_range,
        today_midnight_ny,
        LEVERAGE,
        WORKERS,
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
                                    total_trades = wins = bes = losses = 0
                                    total_pnl = 0.0
                                    ranked_days = 0
                                    for sim_date in month_dates:
                                        ranked = _rank_for_day(
                                            cache,
                                            SYMBOLS,
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
                                        for r in ranked:
                                            day_result = cache[r["symbol"]].get(sim_date)
                                            if day_result is None:
                                                continue
                                            total_trades += day_result["trades"]
                                            wins += day_result["wins"]
                                            bes += day_result.get("breakevens", 0)
                                            losses += day_result["losses"]
                                            total_pnl += day_result["total_pnl"]

                                    if total_trades == 0:
                                        continue

                                    win_rate = wins / total_trades * 100
                                    nlr = (wins + bes) / total_trades * 100

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
                                        "trades": total_trades,
                                        "wins": wins,
                                        "breakevens": bes,
                                        "losses": losses,
                                        "win_rate": win_rate,
                                        "non_loss_rate": nlr,
                                        "pnl": total_pnl,
                                        "ranked_days": ranked_days,
                                    }

                                    if best_any is None or nlr > best_any["non_loss_rate"]:
                                        best_any = row

                                    if total_trades >= MIN_TRADES_FOR_VALID:
                                        if best_valid is None or nlr > best_valid["non_loss_rate"]:
                                            best_valid = row
                                        if nlr >= TARGET_NLR:
                                            if best_valid_over_target is None or row["pnl"] > best_valid_over_target["pnl"]:
                                                best_valid_over_target = row


def to_cmd(r):
    p = r["profile"]
    max_tpd = p["max_tpd"] if p["max_tpd"] is not None else 0
    parts = [
        "python simulate.py --month 2026-03",
        f"--weekly-days {r['weekly_days']}",
        f"--monthly-days {r['monthly_days']}",
        f"--min-trades {r['min_trades']}",
        f"--min-weekly-wr {r['min_weekly_wr']}",
        f"--min-monthly-wr {r['min_monthly_wr']}",
        f"--min-weekly-nlr {r['min_weekly_nlr']}",
        f"--min-monthly-nlr {r['min_monthly_nlr']}",
        f"--top-n {r['top_n']}",
        f"--max-trades-per-day {max_tpd}",
        f"--target-r {p['target_r']}",
    ]
    if p["be"]:
        parts.append("--be-stop")
        parts.append(f"--be-trigger-r {p['be_trigger_r']}")
    return " ".join(parts)

print("\n=== BEST ANY (by NLR) ===")
print(best_any)
if best_any:
    print(to_cmd(best_any))

print("\n=== BEST VALID >=20 TRADES (by NLR) ===")
print(best_valid)
if best_valid:
    print(to_cmd(best_valid))

print("\n=== BEST VALID >=20 TRADES AND >=60% NLR (highest PnL) ===")
print(best_valid_over_target)
if best_valid_over_target:
    print(to_cmd(best_valid_over_target))
