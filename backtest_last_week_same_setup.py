from datetime import datetime, timedelta
import subprocess

from main import NY_TZ

# Same setup as your last run
BASE_CMD = [
    "python",
    "simulate.py",
    "--weekly-days",
    "3",
    "--monthly-days",
    "21",
    "--min-trades",
    "1",
    "--min-weekly-wr",
    "30",
    "--min-monthly-wr",
    "10",
    "--min-weekly-nlr",
    "60",
    "--min-monthly-nlr",
    "40",
    "--top-n",
    "0",
    "--leverage",
    "20",
    "--workers",
    "8",
    "--be-stop",
    "--be-trigger-r",
    "1.0",
    "--target-r",
    "1.5",
    "--max-trades-per-day",
    "3",
    "--daily-loss-limit",
    "-10",
]


def parse_single_date_output(stdout_text):
    summary_no_trades = "No trades on this day for any ranked symbol." in stdout_text
    day_pnl = 0.0
    trades = 0

    if not summary_no_trades:
        for line in stdout_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Total"):
                # Example: Total   -16.73%  |  3 trades  1W/0BE/2L ...
                parts = stripped.split("|")
                if len(parts) >= 2:
                    left = parts[0].replace("Total", "").strip().replace("%", "")
                    right = parts[1].strip()
                    try:
                        day_pnl = float(left)
                    except ValueError:
                        day_pnl = 0.0
                    try:
                        trades = int(right.split()[0])
                    except Exception:
                        trades = 0
                break

    return day_pnl, trades, summary_no_trades


def main():
    now_ny = datetime.now(NY_TZ)
    today_ny = now_ny.date()
    # Last 7 completed days (exclude today)
    dates = [today_ny - timedelta(days=i) for i in range(7, 0, -1)]

    rows = []
    for d in dates:
        cmd = BASE_CMD + ["--date", d.isoformat()]
        p = subprocess.run(cmd, capture_output=True, text=True)
        day_pnl, trades, no_trades = parse_single_date_output(p.stdout)
        rows.append((d.isoformat(), day_pnl, trades, "NO_TRADES" if no_trades else "TRADED"))

    print("DATE,DAY_PNL_PCT,TRADES,STATUS")
    for d, pnl, trades, status in rows:
        print(f"{d},{pnl:+.2f},{trades},{status}")

    week_pnl = sum(r[1] for r in rows)
    week_trades = sum(r[2] for r in rows)
    traded_days = sum(1 for r in rows if r[2] > 0)

    print("---")
    print(f"WEEK_TOTAL_PNL_PCT={week_pnl:+.2f}")
    print(f"WEEK_TOTAL_TRADES={week_trades}")
    print(f"WEEK_TRADED_DAYS={traded_days}/7")
    print(f"WEEK_AVG_DAILY_PNL_PCT={week_pnl/7:+.2f}")


if __name__ == "__main__":
    main()
