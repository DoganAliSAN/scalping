"""
Step 1: Mark high & lows of the first 4 hour candle of the day (UTC-4) on the 5min chart.
important: first 4 hour candle should be closed 
Step 2: After breakout of the range high or low we will wait for price to reenter the range 
If price breaks range high it means to enter short when reenter
If price breaks range low it means to enter long when reenter
Step 3: Enter a after reenter and candle close  put a stop loss to edge of the break 
this means if its a high break out of range the top winkle of that break should be 
our stop loss however if the range is broken more than 1 hour and the price difference 
between range and the high point of the breakout is too high we should not be entering that position
this goes for both high and low breakouts.
And for the profit at 2 times the stoploss size

When a position is entered other signals should be ignored until the position is closed.
"""

from binance import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox, Button
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import time

from binance_live import get_binance_client

NY_TZ = timezone(timedelta(hours=-4))  # UTC-4 (New York / EDT)

client = get_binance_client()
WATCHLIST_CACHE_PATH = Path(__file__).with_name("watchlist_cache.json")
PAPER_LIVE_STATE_PATH = Path(__file__).with_name("paper_live_logs") / "state.json"
PAPER_LIVE_OUTPUT_DIR = Path(__file__).with_name("paper_live_logs")
PAPER_LIVE_CURRENT_POSITIONS_PATH = PAPER_LIVE_OUTPUT_DIR / "current_positions.json"
PAPER_LIVE_EVENTS_PATH = PAPER_LIVE_OUTPUT_DIR / "events.jsonl"
CHART_DATA_REFRESH_SEC = 3.0
MONITOR_ROTATING_REFRESH_MS = 3000
STATUS_REFRESH_MS = 3000
DEFAULT_LEVERAGE = 20.0
MIN_LEVERAGE = 1.0
MAX_LEVERAGE = 125.0

# Default live-app strategy preset aligned with the current live setup.
DEFAULT_ENABLE_BREAK_EVEN = True
DEFAULT_BREAK_EVEN_TRIGGER_R = 1.0
DEFAULT_TARGET_R = 1.5


def clamp_leverage(leverage):
    try:
        lev = float(leverage)
    except (TypeError, ValueError):
        lev = DEFAULT_LEVERAGE
    return max(MIN_LEVERAGE, min(MAX_LEVERAGE, lev))


def calculate_liquidation_price(entry_price, side, leverage):
    lev = clamp_leverage(leverage)
    if side == "long":
        return entry_price * (1 - (1 / lev))
    return entry_price * (1 + (1 / lev))


def calculate_price_pnl_pct(entry_price, exit_price, side):
    if side == "long":
        return ((exit_price - entry_price) / entry_price) * 100
    return ((entry_price - exit_price) / entry_price) * 100


def calculate_leveraged_pnl_pct(entry_price, exit_price, side, leverage):
    return calculate_price_pnl_pct(entry_price, exit_price, side) * clamp_leverage(leverage)


def dedupe_symbols(raw_symbols):
    symbols = []
    for raw_symbol in raw_symbols:
        sym = str(raw_symbol).strip().upper()
        if sym:
            symbols.append(sym)
    return list(dict.fromkeys(symbols))


def load_watchlist_cache(default_symbol):
    if WATCHLIST_CACHE_PATH.exists():
        try:
            raw = json.loads(WATCHLIST_CACHE_PATH.read_text(encoding="utf-8"))
            symbols = dedupe_symbols(raw.get("symbols", []))
            if symbols:
                return symbols
        except Exception:
            pass
    return [default_symbol.upper()]


def save_watchlist_cache(symbols):
    try:
        payload = {"symbols": [s.upper() for s in symbols]}
        WATCHLIST_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal: app should continue even if cache write fails.
        pass


def load_live_selected_symbols(state_path=PAPER_LIVE_STATE_PATH):
    if not state_path.exists():
        return [], f"Missing live state file: {state_path.name}"

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"Failed to read {state_path.name}: {exc}"

    live_day = raw.get("live_day")
    if not isinstance(live_day, dict):
        return [], f"No live_day data in {state_path.name}"

    selected_symbols = []
    selected_rows = live_day.get("selected", [])
    if isinstance(selected_rows, list):
        for row in selected_rows:
            if isinstance(row, dict):
                selected_symbols.append(row.get("symbol", ""))
            else:
                selected_symbols.append(row)

    symbols = dedupe_symbols(selected_symbols)
    if not symbols:
        snapshot = live_day.get("last_snapshot")
        if isinstance(snapshot, dict):
            symbols = dedupe_symbols(snapshot.get("selected_symbols", []))

    if not symbols:
        return [], f"No selected symbols in {state_path.name}"

    return symbols, ""


def normalize_symbol_and_market(raw_symbol):
    s = raw_symbol.strip().upper()
    # USDT-M perpetual user input often comes as BTCUSDT_PERP, API expects BTCUSDT.
    if s.endswith("USDT_PERP"):
        return s.replace("_PERP", ""), "usdm"
    # COIN-M perpetual stays as BTCUSD_PERP.
    if s.endswith("USD_PERP"):
        return s, "coinm"
    # Treat plain USDT pairs as USDT-M futures by default.
    if s.endswith("USDT"):
        return s, "usdm"
    # Fallback for any other symbol format.
    return s, "spot"


def get_klines(symbol, market, interval, start_str, end_str=None):
    if market == "usdm":
        # Use historical helper so start_str is correctly applied on futures.
        if hasattr(client, "futures_historical_klines"):
            return client.futures_historical_klines(symbol, interval, start_str, end_str)

        # Fallback for older client versions: convert start_str -> startTime (ms).
        start_dt = datetime.strptime(start_str, "%d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        kw = dict(symbol=symbol, interval=interval, startTime=start_ms, limit=1500)
        if end_str:
            kw["endTime"] = int(datetime.strptime(end_str, "%d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return client.futures_klines(**kw)

    if market == "coinm":
        if hasattr(client, "futures_coin_historical_klines"):
            return client.futures_coin_historical_klines(symbol, interval, start_str, end_str)

        start_dt = datetime.strptime(start_str, "%d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        kw = dict(symbol=symbol, interval=interval, startTime=start_ms, limit=1500)
        if end_str:
            kw["endTime"] = int(datetime.strptime(end_str, "%d %b %Y %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return client.futures_coin_klines(**kw)

    return client.get_historical_klines(symbol, interval, start_str, end_str) if end_str else client.get_historical_klines(symbol, interval, start_str)


def get_first_4h_candle_levels(symbol, market, date_midnight_ny=None):
    """Return (low, high) of the first closed 4H candle of the given day (UTC-4)."""
    if date_midnight_ny is None:
        date_midnight_ny = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_ny = date_midnight_ny + timedelta(days=1)
    # Convert to UTC for Binance API
    today_midnight_utc = date_midnight_ny.astimezone(timezone.utc)
    start_str = today_midnight_utc.strftime("%d %b %Y %H:%M:%S")
    end_str = next_day_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
    klines_4h = get_klines(symbol, market, Client.KLINE_INTERVAL_4HOUR, start_str, end_str)
    # For historical dates use end-of-day as reference; for today use now.
    now_ny = datetime.now(NY_TZ)
    if date_midnight_ny.date() < now_ny.date():
        ref_ms = int(next_day_ny.astimezone(timezone.utc).timestamp() * 1000)
    else:
        ref_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    closed = [k for k in klines_4h if k[6] < ref_ms]  # k[6] = candle close time
    if not closed:
        return None, None
    first = closed[0]
    return float(first[3]), float(first[2])  # low, high


def draw_candlesticks(ax, klines):
    for i, kline in enumerate(klines):
        o, h, l, c = (
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
        )
        color = "#26a69a" if c >= o else "#ef5350"
        # Wick: full high-low range
        ax.vlines(i, l, h, color=color, linewidth=0.8)
        # Body: open-close range (tiny offset for doji)
        body_top = max(o, c)
        body_bottom = min(o, c)
        if body_top == body_bottom:
            body_top += o * 0.0001
        ax.vlines(i, body_bottom, body_top, color=color, linewidth=3)


def get_close_position(close_price, open_level, high_level):
    if open_level is None or high_level is None:
        return "inside"
    if close_price < open_level:
        return "below"
    if close_price > high_level:
        return "above"
    return "inside"


def get_marker_y(close_price, position, open_level, high_level):
    if position == "above":
        return close_price + high_level * 0.0005
    if position == "below":
        return close_price - open_level * 0.0005
    return close_price


def run_strategy_positions_only(
    klines,
    open_level,
    high_level,
    today_midnight_ny,
    leverage=DEFAULT_LEVERAGE,
    enable_break_even=False,
    break_even_trigger_r=1.0,
    target_r=2.0,
):
    """Run strategy logic for one day and return all trades (closed + possibly open)."""
    trades = []
    active_trade = None
    active_break = None
    one_hour_ms = 60 * 60 * 1000
    prev_position = None

    closed_klines = klines[:-1] if len(klines) > 1 else []
    first_4h_close_ny = today_midnight_ny + timedelta(hours=4)
    first_4h_close_ms = int(first_4h_close_ny.astimezone(timezone.utc).timestamp() * 1000)

    for i, kline in enumerate(closed_klines):
        close_price = float(kline[4])
        high_price = float(kline[2])
        low_price = float(kline[3])
        candle_time_ms = int(kline[0])
        position = get_close_position(close_price, open_level, high_level)

        if candle_time_ms < first_4h_close_ms:
            prev_position = position
            continue

        if active_trade is not None:
            if i <= active_trade["entry_index"]:
                prev_position = position
                continue

            if enable_break_even and not active_trade.get("break_even_armed", False):
                trigger_distance = active_trade["initial_risk"] * break_even_trigger_r
                if active_trade["side"] == "long":
                    favorable_move = high_price - active_trade["entry"]
                    if favorable_move >= trigger_distance:
                        active_trade["stop"] = max(active_trade["stop"], active_trade["entry"])
                        active_trade["break_even_armed"] = True
                else:
                    favorable_move = active_trade["entry"] - low_price
                    if favorable_move >= trigger_distance:
                        active_trade["stop"] = min(active_trade["stop"], active_trade["entry"])
                        active_trade["break_even_armed"] = True

            if active_trade["side"] == "long":
                liquidation_hit = low_price <= active_trade["liquidation_price"]
                stop_hit = low_price <= active_trade["stop"]
                target_hit = high_price >= active_trade["target"]
            else:
                liquidation_hit = high_price >= active_trade["liquidation_price"]
                stop_hit = high_price >= active_trade["stop"]
                target_hit = low_price <= active_trade["target"]

            if liquidation_hit or stop_hit or target_hit:
                if liquidation_hit:
                    result = "liquidation"
                    exit_price = active_trade["liquidation_price"]
                elif stop_hit:
                    exit_price = active_trade["stop"]
                    result = "breakeven" if active_trade.get("break_even_armed", False) else "stop"
                else:
                    result = "target"
                    exit_price = active_trade["target"]
                trades.append(
                    {
                        **active_trade,
                        "exit_index": i,
                        "exit_price": exit_price,
                        "result": result,
                    }
                )
                active_trade = None

            prev_position = position
            continue

        if active_break is None:
            if position in {"above", "below"} and prev_position == "inside":
                active_break = {
                    "index": i,
                    "time_ms": candle_time_ms,
                    "position": position,
                    "stop": high_price if position == "above" else low_price,
                    "timed_out": False,
                }
            prev_position = position
            continue

        if position == active_break["position"] == "above":
            active_break["stop"] = max(active_break["stop"], high_price)
        elif position == active_break["position"] == "below":
            active_break["stop"] = min(active_break["stop"], low_price)

        if position == "inside":
            within_hour = candle_time_ms - active_break["time_ms"] <= one_hour_ms
            if within_hour and not active_break["timed_out"]:
                if active_break["position"] == "above":
                    side = "short"
                    entry = close_price
                    stop = active_break["stop"]
                    risk = stop - entry
                    target = entry - (target_r * risk)
                else:
                    side = "long"
                    entry = close_price
                    stop = active_break["stop"]
                    risk = entry - stop
                    target = entry + (target_r * risk)

                if risk > 0:
                    liquidation_price = calculate_liquidation_price(entry, side, leverage)
                    active_trade = {
                        "side": side,
                        "entry_index": i,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "initial_risk": risk,
                        "break_even_armed": False,
                        "liquidation_price": liquidation_price,
                        "leverage": clamp_leverage(leverage),
                    }
            active_break = None
            prev_position = position
            continue

        if (
            not active_break["timed_out"]
            and candle_time_ms - active_break["time_ms"] > one_hour_ms
        ):
            active_break["timed_out"] = True

        if position != active_break["position"]:
            active_break = {
                "index": i,
                "time_ms": candle_time_ms,
                "position": position,
                "stop": high_price if position == "above" else low_price,
                "timed_out": False,
            }

        prev_position = position

    if active_trade is not None and closed_klines:
        trades.append(
            {
                **active_trade,
                "exit_index": len(closed_klines) - 1,
                "exit_price": float(closed_klines[-1][4]),
                "result": "open",
            }
        )

    return trades


def get_symbol_positions_summary(
    display_symbol,
    date_midnight_ny=None,
    leverage=DEFAULT_LEVERAGE,
    enable_break_even=DEFAULT_ENABLE_BREAK_EVEN,
    break_even_trigger_r=DEFAULT_BREAK_EVEN_TRIGGER_R,
    target_r=DEFAULT_TARGET_R,
):
    if date_midnight_ny is None:
        date_midnight_ny = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    symbol_for_api, market = normalize_symbol_and_market(display_symbol)
    open_level, high_level = get_first_4h_candle_levels(symbol_for_api, market, date_midnight_ny)

    start_str = date_midnight_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
    now_ny = datetime.now(NY_TZ)
    if date_midnight_ny.date() < now_ny.date():
        # Historical date: limit klines to that day only
        next_day_ny = date_midnight_ny + timedelta(days=1)
        end_str = next_day_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
        klines = get_klines(symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE, start_str, end_str)
    else:
        klines = get_klines(symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE, start_str)
    trades = run_strategy_positions_only(
        klines,
        open_level,
        high_level,
        date_midnight_ny,
        leverage=leverage,
        enable_break_even=enable_break_even,
        break_even_trigger_r=break_even_trigger_r,
        target_r=target_r,
    )

    closed = [t for t in trades if t["result"] != "open"]
    open_trade = next((t for t in reversed(trades) if t["result"] == "open"), None)

    wins = 0
    pnl_pct_total = 0.0
    leveraged_pnl_pct_total = 0.0
    liquidation_count = 0
    for t in closed:
        pnl_pct = calculate_price_pnl_pct(t["entry"], t["exit_price"], t["side"])
        leveraged_pnl_pct = calculate_leveraged_pnl_pct(
            t["entry"], t["exit_price"], t["side"], t.get("leverage", leverage)
        )
        pnl_pct_total += pnl_pct
        leveraged_pnl_pct_total += leveraged_pnl_pct
        if t["result"] == "target":
            wins += 1
        if t["result"] == "liquidation":
            liquidation_count += 1

    return {
        "display_symbol": display_symbol,
        "api_symbol": symbol_for_api,
        "market": market,
        "open_level": open_level,
        "high_level": high_level,
        "trades": trades,
        "closed_count": len(closed),
        "wins": wins,
        "pnl_pct_total": pnl_pct_total,
        "leveraged_pnl_pct_total": leveraged_pnl_pct_total,
        "liquidation_count": liquidation_count,
        "open_trade": open_trade,
    }


def launch_positions_monitor(
    initial_symbol,
    initial_leverage=DEFAULT_LEVERAGE,
    enable_break_even=DEFAULT_ENABLE_BREAK_EVEN,
    break_even_trigger_r=DEFAULT_BREAK_EVEN_TRIGGER_R,
    target_r=DEFAULT_TARGET_R,
    open_symbol_callback=None,
):
    cached_symbols = load_watchlist_cache(initial_symbol)
    _today = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    monitor_state = {
        "manual_symbols": list(cached_symbols),
        "symbols": list(cached_symbols),
        "selected": cached_symbols[0] if cached_symbols else None,
        "rows": {},
        "cache": {},
        "next_symbol_idx": 0,
        "selected_date": _today,
        "leverage": clamp_leverage(initial_leverage),
        "enable_break_even": bool(enable_break_even),
        "break_even_trigger_r": float(break_even_trigger_r),
        "target_r": float(target_r),
        "source_mode": "manual",
        "source_error": "",
        "row_open_buttons": [],
        "row_open_axes": [],
        "open_symbol_callback": open_symbol_callback,
    }

    def get_source_label():
        if monitor_state["source_mode"] == "manual":
            return "My Added"
        return "state.json Selected"

    def set_active_symbols(symbols):
        new_symbols = dedupe_symbols(symbols)
        previous_symbols = monitor_state["symbols"]
        monitor_state["symbols"] = new_symbols
        monitor_state["cache"] = {sym: monitor_state["cache"].get(sym) for sym in new_symbols}
        if monitor_state["selected"] not in new_symbols:
            monitor_state["selected"] = new_symbols[0] if new_symbols else None
        if not new_symbols or monitor_state["next_symbol_idx"] >= len(new_symbols):
            monitor_state["next_symbol_idx"] = 0
        return new_symbols != previous_symbols

    def sync_symbol_source():
        if monitor_state["source_mode"] == "manual":
            monitor_state["source_error"] = ""
            return set_active_symbols(monitor_state["manual_symbols"])

        live_symbols, source_error = load_live_selected_symbols()
        monitor_state["source_error"] = source_error
        return set_active_symbols(live_symbols)

    fig_m = plt.figure(figsize=(10, 6))
    fig_m.patch.set_facecolor("#111319")
    ax_list = fig_m.add_axes([0.03, 0.12, 0.44, 0.8])
    ax_detail = fig_m.add_axes([0.50, 0.12, 0.47, 0.8])
    ax_list.set_facecolor("#131722")
    ax_detail.set_facecolor("#131722")
    ax_list.axis("off")
    ax_detail.axis("off")
    fig_m.suptitle("Positions Monitor", color="white", fontsize=13)

    status_text = fig_m.text(0.03, 0.95, "", color="#9e9e9e", fontsize=9)

    source_ax = fig_m.add_axes([0.73, 0.935, 0.24, 0.045])
    source_btn = Button(source_ax, "", color="#2a2e39", hovercolor="#3a4152")
    source_btn.label.set_color("white")

    def update_source_button_label():
        source_btn.label.set_text(f"Source: {get_source_label()}")

    update_source_button_label()

    input_ax = fig_m.add_axes([0.03, 0.03, 0.2, 0.055])
    input_box = TextBox(
        input_ax,
        "Symbol",
        initial=initial_symbol.upper(),
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    input_ax.set_facecolor("#1e2130")
    input_box.label.set_color("white")
    input_box.text_disp.set_color("white")
    if hasattr(input_box, "cursor") and input_box.cursor is not None:
        input_box.cursor.set_color("white")

    def on_monitor_text_change(_text):
        input_box.text_disp.set_color("white")
        if hasattr(input_box, "cursor") and input_box.cursor is not None:
            input_box.cursor.set_color("white")

    input_box.on_text_change(on_monitor_text_change)

    add_ax = fig_m.add_axes([0.24, 0.03, 0.09, 0.055])
    add_btn = Button(add_ax, "Add", color="#2a2e39", hovercolor="#3a4152")
    rem_ax = fig_m.add_axes([0.34, 0.03, 0.09, 0.055])
    rem_btn = Button(rem_ax, "Remove", color="#2a2e39", hovercolor="#3a4152")
    ref_ax = fig_m.add_axes([0.44, 0.03, 0.09, 0.055])
    ref_btn = Button(ref_ax, "Refresh", color="#2a2e39", hovercolor="#3a4152")
    add_btn.label.set_color("white")
    rem_btn.label.set_color("white")
    ref_btn.label.set_color("white")

    date_ax = fig_m.add_axes([0.55, 0.03, 0.14, 0.055])
    date_box = TextBox(
        date_ax,
        "Date",
        initial=datetime.now(NY_TZ).strftime("%Y-%m-%d"),
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    date_ax.set_facecolor("#1e2130")
    date_box.label.set_color("white")
    date_box.text_disp.set_color("white")
    if hasattr(date_box, "cursor") and date_box.cursor is not None:
        date_box.cursor.set_color("white")

    def on_date_text_change(_text):
        date_box.text_disp.set_color("white")
        if hasattr(date_box, "cursor") and date_box.cursor is not None:
            date_box.cursor.set_color("white")

    date_box.on_text_change(on_date_text_change)

    go_ax = fig_m.add_axes([0.70, 0.03, 0.06, 0.055])
    go_btn = Button(go_ax, "Go", color="#2a2e39", hovercolor="#3a4152")
    go_btn.label.set_color("white")

    lev_ax = fig_m.add_axes([0.77, 0.03, 0.09, 0.055])
    lev_box = TextBox(
        lev_ax,
        "Lev",
        initial=f"{monitor_state['leverage']:.1f}",
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    lev_ax.set_facecolor("#1e2130")
    lev_box.label.set_color("white")
    lev_box.text_disp.set_color("white")
    if hasattr(lev_box, "cursor") and lev_box.cursor is not None:
        lev_box.cursor.set_color("white")

    def on_lev_text_change(_text):
        lev_box.text_disp.set_color("white")
        if hasattr(lev_box, "cursor") and lev_box.cursor is not None:
            lev_box.cursor.set_color("white")

    lev_box.on_text_change(on_lev_text_change)

    lev_set_ax = fig_m.add_axes([0.87, 0.03, 0.10, 0.055])
    lev_set_btn = Button(lev_set_ax, "Set Lev", color="#2a2e39", hovercolor="#3a4152")
    lev_set_btn.label.set_color("white")

    def clear_row_open_buttons():
        monitor_state["row_open_buttons"].clear()
        for row_ax in monitor_state["row_open_axes"]:
            try:
                fig_m.delaxes(row_ax)
            except Exception:
                pass
        monitor_state["row_open_axes"] = []

    def open_symbol_in_chart(sym):
        monitor_state["selected"] = sym
        callback = monitor_state.get("open_symbol_callback")
        if callable(callback):
            callback(sym)
            status_text.set_text(f"Opening {sym} on main chart")
        else:
            status_text.set_text("Chart open action is not available")
        render_from_cache()

    def render_from_cache():
        ax_list.clear()
        ax_detail.clear()
        ax_list.set_facecolor("#131722")
        ax_detail.set_facecolor("#131722")
        ax_list.axis("off")
        ax_detail.axis("off")
        monitor_state["rows"] = {}
        clear_row_open_buttons()
        source_label = get_source_label()

        if not monitor_state["symbols"]:
            ax_list.text(
                0.02,
                0.95,
                f"No symbols in {source_label}.",
                color="#9e9e9e",
                fontsize=10,
                transform=ax_list.transAxes,
            )
            if monitor_state["source_error"]:
                ax_list.text(
                    0.02,
                    0.88,
                    monitor_state["source_error"],
                    color="#FF8A65",
                    fontsize=9,
                    transform=ax_list.transAxes,
                )

        y = 0.95
        for idx, sym in enumerate(monitor_state["symbols"]):
            data = monitor_state["cache"].get(sym)
            if data is None:
                line = f"{sym:14} | loading..."
                color = "#9e9e9e"
            else:
                if data["open_trade"] is not None:
                    ot = data["open_trade"]
                    line = (
                        f"{sym:14} | OPEN {ot['side'].upper():5} | "
                        f"E {ot['entry']:.4f} | Liq {ot['liquidation_price']:.4f}"
                    )
                    color = "#7CFFB2"
                elif data["closed_count"] > 0:
                    line = (
                        f"{sym:14} | CLOSED {data['closed_count']:2d} | "
                        f"LevPnL {data['leveraged_pnl_pct_total']:+.2f}% | Liq {data['liquidation_count']}"
                    )
                    color = "#B0BEC5"
                else:
                    line = f"{sym:14} | NO POSITIONS TODAY"
                    color = "#90A4AE"

            if sym == monitor_state["selected"]:
                line = "▶ " + line
            else:
                line = "  " + line

            txt = ax_list.text(
                0.02,
                y,
                line,
                color=color,
                fontsize=10,
                fontfamily="monospace",
                transform=ax_list.transAxes,
                picker=True,
            )
            monitor_state["rows"][txt] = sym

            if callable(monitor_state.get("open_symbol_callback")):
                button_bottom = 0.12 + (0.8 * (y - 0.03))
                if button_bottom >= 0.12:
                    open_ax = fig_m.add_axes([0.39, button_bottom, 0.07, 0.038])
                    open_btn = Button(open_ax, "Chart", color="#2a2e39", hovercolor="#3a4152")
                    open_btn.label.set_color("white")
                    open_btn.on_clicked(lambda _event, symbol=sym: open_symbol_in_chart(symbol))
                    monitor_state["row_open_axes"].append(open_ax)
                    monitor_state["row_open_buttons"].append(open_btn)

            y -= 0.07

        selected_symbol = monitor_state["selected"]
        selected_data = monitor_state["cache"].get(selected_symbol) if selected_symbol else None
        date_label = monitor_state["selected_date"].strftime("%Y-%m-%d")
        ax_detail.text(
            0.02,
            0.95,
            f"{selected_symbol or source_label} — {date_label}",
            color="white",
            fontsize=11,
            transform=ax_detail.transAxes,
        )

        if selected_symbol is None:
            detail_text = monitor_state["source_error"] or f"No symbols available from {source_label}."
            ax_detail.text(0.02, 0.86, detail_text, color="#9e9e9e", transform=ax_detail.transAxes)
        elif selected_data is None:
            ax_detail.text(0.02, 0.86, "Loading...", color="#9e9e9e", transform=ax_detail.transAxes)
        else:
            ax_detail.text(
                0.02,
                0.88,
                (
                    f"Market: {selected_data['market'].upper()} | Lev: {monitor_state['leverage']:.1f}x | "
                    f"Closed: {selected_data['closed_count']} | Wins: {selected_data['wins']} | "
                    f"Liq: {selected_data['liquidation_count']} | Day LevPnL: {selected_data['leveraged_pnl_pct_total']:+.2f}%"
                ),
                color="#cfd8dc",
                fontsize=9,
                transform=ax_detail.transAxes,
            )

            y2 = 0.80
            trades = selected_data["trades"]
            if not trades:
                ax_detail.text(0.02, y2, "No positions yet today.", color="#90A4AE", transform=ax_detail.transAxes)
            else:
                for idx, t in enumerate(trades[-12:], start=max(1, len(trades) - 11)):
                    pnl_pct = calculate_price_pnl_pct(t["entry"], t["exit_price"], t["side"])
                    lev_pnl_pct = calculate_leveraged_pnl_pct(
                        t["entry"], t["exit_price"], t["side"], t.get("leverage", monitor_state["leverage"])
                    )
                    result = t["result"].upper()
                    if result == "TARGET":
                        color = "#7CFFB2"
                    elif result == "LIQUIDATION":
                        color = "#FF1744"
                    elif result == "STOP":
                        color = "#FF8A65"
                    else:
                        color = "#90A4AE"
                    line = (
                        f"#{idx:02d} {t['side'].upper():5} | "
                        f"E {t['entry']:.4f}  S {t['stop']:.4f}  T {t['target']:.4f}  L {t['liquidation_price']:.4f} | "
                        f"{result:11} | Px {pnl_pct:+.2f}% | Lev {lev_pnl_pct:+.2f}%"
                    )
                    ax_detail.text(0.02, y2, line, color=color, fontsize=9, fontfamily="monospace", transform=ax_detail.transAxes)
                    y2 -= 0.055

        fig_m.canvas.draw_idle()

    def refresh_data(_frame=None, force_all=False):
        source_changed = sync_symbol_source()
        if source_changed:
            force_all = True

        if not monitor_state["symbols"]:
            monitor_state["cache"] = {}
            render_from_cache()
            date_lbl = monitor_state["selected_date"].strftime("%Y-%m-%d")
            source_note = monitor_state["source_error"] or f"No symbols in {get_source_label()}"
            status_text.set_text(f"[{date_lbl}] {get_source_label()} | {source_note}")
            fig_m.canvas.draw_idle()
            return

        # Skip scheduled auto-refresh for historical dates; explicit refreshes still run.
        now_ny = datetime.now(NY_TZ)
        if not force_all and monitor_state["selected_date"].date() < now_ny.date():
            return

        cache = dict(monitor_state["cache"])
        if force_all:
            symbols_to_update = list(monitor_state["symbols"])
        else:
            idx = monitor_state["next_symbol_idx"] % len(monitor_state["symbols"])
            symbols_to_update = [monitor_state["symbols"][idx]]
            monitor_state["next_symbol_idx"] = (idx + 1) % len(monitor_state["symbols"])

        sel_date = monitor_state["selected_date"]
        for sym in symbols_to_update:
            try:
                cache[sym] = get_symbol_positions_summary(
                    sym,
                    sel_date,
                    leverage=monitor_state["leverage"],
                    enable_break_even=monitor_state["enable_break_even"],
                    break_even_trigger_r=monitor_state["break_even_trigger_r"],
                    target_r=monitor_state["target_r"],
                )
            except Exception as e:
                cache[sym] = {
                    "display_symbol": sym,
                    "market": "?",
                    "closed_count": 0,
                    "wins": 0,
                    "pnl_pct_total": 0.0,
                    "leveraged_pnl_pct_total": 0.0,
                    "liquidation_count": 0,
                    "open_trade": None,
                    "trades": [],
                    "error": str(e),
                }

        monitor_state["cache"] = cache
        render_from_cache()
        updated = ", ".join(symbols_to_update)
        date_lbl = monitor_state["selected_date"].strftime("%Y-%m-%d")
        status_text.set_text(
            f"[{date_lbl}] {get_source_label()} | {monitor_state['leverage']:.1f}x | Updated {updated} at {datetime.now(NY_TZ).strftime('%H:%M:%S')} (NY)"
        )
        fig_m.canvas.draw_idle()

    def on_pick(event):
        sym = monitor_state["rows"].get(event.artist)
        if sym:
            monitor_state["selected"] = sym
            render_from_cache()

    def on_add(_event):
        if monitor_state["source_mode"] != "manual":
            status_text.set_text("Switch source to My Added to edit your watchlist")
            fig_m.canvas.draw_idle()
            return
        sym = input_box.text.strip().upper()
        if not sym:
            return
        if sym not in monitor_state["manual_symbols"]:
            monitor_state["manual_symbols"].append(sym)
            monitor_state["selected"] = sym
            save_watchlist_cache(monitor_state["manual_symbols"])
            refresh_data(force_all=True)

    def on_remove(_event):
        if monitor_state["source_mode"] != "manual":
            status_text.set_text("Switch source to My Added to edit your watchlist")
            fig_m.canvas.draw_idle()
            return
        sym = input_box.text.strip().upper()
        if sym in monitor_state["manual_symbols"] and len(monitor_state["manual_symbols"]) > 1:
            monitor_state["manual_symbols"].remove(sym)
            if monitor_state["selected"] == sym:
                monitor_state["selected"] = monitor_state["manual_symbols"][0]
            save_watchlist_cache(monitor_state["manual_symbols"])
            refresh_data(force_all=True)

    def on_toggle_source(_event):
        if monitor_state["source_mode"] == "manual":
            monitor_state["source_mode"] = "live_selected"
        else:
            monitor_state["source_mode"] = "manual"
        monitor_state["cache"] = {}
        monitor_state["next_symbol_idx"] = 0
        update_source_button_label()
        refresh_data(force_all=True)

    def on_go(_event):
        raw = date_box.text.strip()
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=NY_TZ)
        except ValueError:
            status_text.set_text("Invalid date format. Use YYYY-MM-DD")
            fig_m.canvas.draw_idle()
            return
        monitor_state["selected_date"] = parsed
        monitor_state["cache"] = {}   # clear stale data from previous date
        monitor_state["next_symbol_idx"] = 0
        refresh_data(force_all=True)

    def on_set_leverage(_event=None):
        raw = lev_box.text.strip()
        try:
            lev = clamp_leverage(float(raw))
        except ValueError:
            status_text.set_text("Invalid leverage. Use a number like 20 or 20.0")
            fig_m.canvas.draw_idle()
            return
        monitor_state["leverage"] = lev
        lev_box.set_val(f"{lev:.1f}")
        monitor_state["cache"] = {}
        monitor_state["next_symbol_idx"] = 0
        refresh_data(force_all=True)

    add_btn.on_clicked(on_add)
    rem_btn.on_clicked(on_remove)
    ref_btn.on_clicked(lambda _e: refresh_data(force_all=True))
    source_btn.on_clicked(on_toggle_source)
    go_btn.on_clicked(on_go)
    lev_set_btn.on_clicked(on_set_leverage)
    lev_box.on_submit(lambda _t: on_set_leverage())
    fig_m.canvas.mpl_connect("pick_event", on_pick)

    # Keep strong references so widgets stay interactive.
    fig_m._monitor_ui_refs = {
        "input_box": input_box,
        "on_monitor_text_change": on_monitor_text_change,
        "add_btn": add_btn,
        "rem_btn": rem_btn,
        "ref_btn": ref_btn,
        "source_btn": source_btn,
        "on_toggle_source": on_toggle_source,
        "date_box": date_box,
        "on_date_text_change": on_date_text_change,
        "go_btn": go_btn,
        "lev_box": lev_box,
        "on_lev_text_change": on_lev_text_change,
        "lev_set_btn": lev_set_btn,
        "on_add": on_add,
        "on_remove": on_remove,
        "on_go": on_go,
        "on_set_leverage": on_set_leverage,
        "on_pick": on_pick,
        "refresh_data": refresh_data,
    }

    refresh_data(force_all=True)
    ani_monitor = animation.FuncAnimation(
        fig_m,
        refresh_data,
        interval=MONITOR_ROTATING_REFRESH_MS,
        cache_frame_data=False,
    )
    return fig_m, ani_monitor


def launch_paper_live_status_window():
    fig_s = plt.figure(figsize=(10, 6))
    fig_s.patch.set_facecolor("#111319")
    ax_status = fig_s.add_axes([0.03, 0.08, 0.94, 0.86])
    ax_status.set_facecolor("#131722")
    ax_status.axis("off")
    fig_s.suptitle("Paper Live Logger Status", color="white", fontsize=13)

    status_text = ax_status.text(
        0.02,
        0.98,
        "Loading...",
        transform=ax_status.transAxes,
        va="top",
        color="#CFD8DC",
        fontsize=10,
        fontfamily="monospace",
    )

    def fmt_float(value):
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "?"

    def parse_iso_or_none(iso_str):
        if not iso_str:
            return None
        try:
            return datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        except Exception:
            return None

    def last_event_line(events_path):
        if not events_path.exists():
            return "No events file yet"
        try:
            with events_path.open("rb") as f:
                f.seek(0, 2)
                end = f.tell()
                if end <= 0:
                    return "No events logged yet"
                cursor = end - 1
                while cursor >= 0:
                    f.seek(cursor)
                    ch = f.read(1)
                    if ch == b"\n" and cursor < end - 1:
                        break
                    cursor -= 1
                start = max(0, cursor + 1)
                f.seek(start)
                raw = f.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return "No events logged yet"
            payload = json.loads(raw)
            ts = payload.get("ts_utc", "?")
            evt = payload.get("event", "?")
            date = payload.get("date", "?")
            return f"{ts} | {evt} | date={date}"
        except Exception as exc:
            return f"Failed to read events: {exc}"

    def refresh_status(_frame=None):
        lines = []
        now_utc = datetime.now(timezone.utc)
        lines.append(f"Updated: {now_utc.isoformat()}")
        lines.append("")

        state_data = None
        snapshot_data = None

        if PAPER_LIVE_STATE_PATH.exists():
            try:
                state_data = json.loads(PAPER_LIVE_STATE_PATH.read_text(encoding="utf-8"))
                lines.append(f"State File: OK ({PAPER_LIVE_STATE_PATH.name})")
            except Exception as exc:
                lines.append(f"State File: ERROR ({exc})")
        else:
            lines.append(f"State File: MISSING ({PAPER_LIVE_STATE_PATH.name})")

        if PAPER_LIVE_CURRENT_POSITIONS_PATH.exists():
            try:
                snapshot_data = json.loads(PAPER_LIVE_CURRENT_POSITIONS_PATH.read_text(encoding="utf-8"))
                lines.append(f"Snapshot File: OK ({PAPER_LIVE_CURRENT_POSITIONS_PATH.name})")
            except Exception as exc:
                lines.append(f"Snapshot File: ERROR ({exc})")
        else:
            lines.append(f"Snapshot File: MISSING ({PAPER_LIVE_CURRENT_POSITIONS_PATH.name})")

        lines.append("")

        if isinstance(state_data, dict):
            last_processed = state_data.get("last_processed_date", "?")
            balance = fmt_float(state_data.get("balance", 0.0))
            peak = fmt_float(state_data.get("equity_peak", 0.0))
            dd = fmt_float(state_data.get("max_drawdown_pct", 0.0))
            processed_days = state_data.get("processed_days", "?")
            lines.append("Account State")
            lines.append(f"  Balance: {balance} | Peak: {peak} | MaxDD: {dd}%")
            lines.append(f"  Last Processed Day: {last_processed} | Processed Days: {processed_days}")

            live_day = state_data.get("live_day")
            if isinstance(live_day, dict):
                live_date = live_day.get("date", "?")
                live_loss_hit = "YES" if live_day.get("daily_loss_limit_reached") else "NO"
                selected = live_day.get("selected", [])
                selected_symbols = [r.get("symbol", "?") for r in selected if isinstance(r, dict)]
                lines.append(f"  Live Day: {live_date} | Loss Limit Hit: {live_loss_hit}")
                lines.append(f"  Selected Symbols ({len(selected_symbols)}): {', '.join(selected_symbols) if selected_symbols else '-'}")
            else:
                lines.append("  Live Day: not initialized")

        lines.append("")

        if isinstance(snapshot_data, dict):
            gen_at = snapshot_data.get("generated_at_utc")
            gen_dt = parse_iso_or_none(gen_at)
            staleness = "?"
            freshness = "UNKNOWN"
            if gen_dt is not None:
                age_sec = max(0.0, (now_utc - gen_dt).total_seconds())
                staleness = f"{age_sec:.0f}s"
                freshness = "RUNNING" if age_sec <= 600 else "STALE"

            est_balance = fmt_float(snapshot_data.get("estimated_balance", 0.0))
            realized_usd = fmt_float(snapshot_data.get("realized_pnl_usd", 0.0))
            realized_pct = fmt_float(snapshot_data.get("realized_pnl_pct", 0.0))
            open_positions = snapshot_data.get("open_positions", []) or []
            selected_symbols = snapshot_data.get("selected_symbols", []) or []
            loss_limit_hit = "YES" if snapshot_data.get("daily_loss_limit_reached") else "NO"

            lines.append("Live Snapshot")
            lines.append(f"  Date: {snapshot_data.get('date', '?')} | Status: {freshness} | Snapshot Age: {staleness}")
            lines.append(f"  Est Balance: {est_balance} | Realized: {realized_usd} USD ({realized_pct}%) | Loss Limit Hit: {loss_limit_hit}")
            lines.append(f"  Tracked: {len(selected_symbols)} | Open Positions: {len(open_positions)}")

            if open_positions:
                lines.append("  Open Positions")
                for pos in open_positions[:8]:
                    symbol = pos.get("symbol", "?")
                    side = str(pos.get("side", "?")).upper()
                    entry = fmt_float(pos.get("entry_price", 0.0))
                    mark = fmt_float(pos.get("mark_price", 0.0))
                    upnl = fmt_float(pos.get("unrealized_pnl_pct", 0.0))
                    lines.append(f"    {symbol:14} {side:5} | E {entry} | M {mark} | uPnL {upnl}%")

        lines.append("")
        lines.append("Last Event")
        lines.append(f"  {last_event_line(PAPER_LIVE_EVENTS_PATH)}")

        status_text.set_text("\n".join(lines))
        fig_s.canvas.draw_idle()

    ref_ax = fig_s.add_axes([0.85, 0.02, 0.12, 0.05])
    ref_btn = Button(ref_ax, "Refresh", color="#2a2e39", hovercolor="#3a4152")
    ref_btn.label.set_color("white")
    ref_btn.on_clicked(lambda _e: refresh_status())

    fig_s._status_ui_refs = {
        "refresh_status": refresh_status,
        "refresh_button": ref_btn,
    }

    refresh_status()
    return fig_s, None


def live_5min_chart(
    symbol,
    leverage=DEFAULT_LEVERAGE,
    enable_break_even=DEFAULT_ENABLE_BREAK_EVEN,
    break_even_trigger_r=DEFAULT_BREAK_EVEN_TRIGGER_R,
    target_r=DEFAULT_TARGET_R,
):
    symbol_for_api, market = normalize_symbol_and_market(symbol)
    _today_chart = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    open_level, high_level = get_first_4h_candle_levels(symbol_for_api, market, _today_chart)
    state = {
        "display_symbol": symbol.upper(),
        "symbol_for_api": symbol_for_api,
        "market": market,
        "open_level": open_level,
        "high_level": high_level,
        "klines_cache": None,
        "last_klines_fetch_ts": 0.0,
        "selected_date": _today_chart,
        "leverage": clamp_leverage(leverage),
        "enable_break_even": bool(enable_break_even),
        "break_even_trigger_r": float(break_even_trigger_r),
        "target_r": float(target_r),
        "reset_view": True,
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#131722")

    def update(frame):
        previous_xlim = None
        previous_ylim = None
        restore_view = False
        if not state["reset_view"] and ax.has_data():
            previous_xlim = ax.get_xlim()
            previous_ylim = ax.get_ylim()
            restore_view = True

        ax.clear()
        ax.set_facecolor("#131722")

        today_midnight_ny = state["selected_date"]
        now_ny = datetime.now(NY_TZ)
        is_historical = today_midnight_ny.date() < now_ny.date()

        start_str = today_midnight_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
        now_ts = time.time()
        if (
            state["klines_cache"] is None
            or (not is_historical and (now_ts - state["last_klines_fetch_ts"]) >= CHART_DATA_REFRESH_SEC)
        ):
            if is_historical:
                next_day_ny = today_midnight_ny + timedelta(days=1)
                end_str = next_day_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
                state["klines_cache"] = get_klines(
                    state["symbol_for_api"],
                    state["market"],
                    Client.KLINE_INTERVAL_5MINUTE,
                    start_str,
                    end_str,
                )
            else:
                state["klines_cache"] = get_klines(
                    state["symbol_for_api"],
                    state["market"],
                    Client.KLINE_INTERVAL_5MINUTE,
                    start_str,
                )
            state["last_klines_fetch_ts"] = now_ts

        klines = state["klines_cache"]

        # Build list of NY-time labels from candle open timestamps (k[0] = open time ms)
        times_ny = [
            datetime.fromtimestamp(k[0] / 1000, tz=NY_TZ) for k in klines
        ]

        draw_candlesticks(ax, klines)
        n = len(klines)

        # Current price = close of last candle
        current_price = float(klines[-1][4]) if klines else None

        if state["open_level"] is not None:
            ax.axhline(
                y=state["open_level"],
                color="#2196F3",
                linestyle="--",
                linewidth=1.2,
                label=f"4H Low: {state['open_level']:.2f}",
            )
        if state["high_level"] is not None:
            ax.axhline(
                y=state["high_level"],
                color="#FF9800",
                linestyle="--",
                linewidth=1.2,
                label=f"4H High (Peak): {state['high_level']:.2f}",
            )

        # Mark each outside break and the return inside.
        # Rules:
        #   - Only CLOSED candles count (exclude the last/live candle whose close is still changing).
        #   - A break is triggered only when the close crosses a level.
        #   - A return is triggered only when the close is back inside the range.
        #   - Each continuous outside run gets at most ONE break marker (timed_out flag prevents
        #     a second yellow diamond from being emitted for the same run after 1h).
        #   - Break detection is drawn immediately (blue diamond), even before return.
        #   - Paired (≤1h): blue diamond + green circle.
        #   - Unpaired (>1h): yellow diamond + orange circle when it eventually returns.
        break_detect_xs, break_detect_ys = [], []
        pair_return_xs, pair_return_ys = [], []
        single_break_xs, single_break_ys = [], []
        late_return_xs, late_return_ys = [], []
        active_break = None  # dict: index, time_ms, position, y, stop, timed_out
        active_trade = None  # dict: side, entry_index, entry, stop, target
        trades = []
        one_hour_ms = 60 * 60 * 1000

        # Exclude the last kline – it is the currently open candle and its close
        # price changes tick-by-tick, causing false inside/outside transitions.
        closed_klines = klines[:-1] if len(klines) > 1 else []

        # Strategy starts only after the first 4H candle is closed (04:00 NY).
        first_4h_close_ny = today_midnight_ny + timedelta(hours=4)
        first_4h_close_ms = int(first_4h_close_ny.astimezone(timezone.utc).timestamp() * 1000)
        prev_position = None

        for i, kline in enumerate(closed_klines):
            close_price = float(kline[4])
            high_price = float(kline[2])
            low_price = float(kline[3])
            candle_time_ms = int(kline[0])
            position = get_close_position(close_price, state["open_level"], state["high_level"])

            if candle_time_ms < first_4h_close_ms:
                prev_position = position
                continue

            # While a trade is open, ignore all new signal generation.
            # Only track whether this candle closes the position (TP/SL).
            if active_trade is not None:
                if i <= active_trade["entry_index"]:
                    prev_position = position
                    continue

                if state["enable_break_even"] and not active_trade.get("break_even_armed", False):
                    trigger_distance = active_trade["initial_risk"] * state["break_even_trigger_r"]
                    if active_trade["side"] == "long":
                        favorable_move = high_price - active_trade["entry"]
                        if favorable_move >= trigger_distance:
                            active_trade["stop"] = max(active_trade["stop"], active_trade["entry"])
                            active_trade["break_even_armed"] = True
                    else:
                        favorable_move = active_trade["entry"] - low_price
                        if favorable_move >= trigger_distance:
                            active_trade["stop"] = min(active_trade["stop"], active_trade["entry"])
                            active_trade["break_even_armed"] = True

                stop_hit = False
                target_hit = False
                if active_trade["side"] == "long":
                    liquidation_hit = low_price <= active_trade["liquidation_price"]
                    stop_hit = low_price <= active_trade["stop"]
                    target_hit = high_price >= active_trade["target"]
                else:
                    liquidation_hit = high_price >= active_trade["liquidation_price"]
                    stop_hit = high_price >= active_trade["stop"]
                    target_hit = low_price <= active_trade["target"]

                if liquidation_hit or stop_hit or target_hit:
                    # Use conservative fill ordering: liquidation, then stop, then target.
                    if liquidation_hit:
                        result = "liquidation"
                        exit_price = active_trade["liquidation_price"]
                    elif stop_hit:
                        result = "breakeven" if active_trade.get("break_even_armed", False) else "stop"
                        exit_price = active_trade["stop"]
                    else:
                        result = "target"
                        exit_price = active_trade["target"]
                    trades.append(
                        {
                            **active_trade,
                            "exit_index": i,
                            "exit_price": exit_price,
                            "result": result,
                        }
                    )
                    active_trade = None
                prev_position = position
                continue

            if active_break is None:
                # Break must be a close transition from inside -> outside.
                # If price is already outside at strategy start, wait for re-entry first.
                if position in {"above", "below"} and prev_position == "inside":
                    break_detect_xs.append(i)
                    break_detect_ys.append(get_marker_y(close_price, position, state["open_level"], state["high_level"]))
                    active_break = {
                        "index": i,
                        "time_ms": candle_time_ms,
                        "position": position,
                        "y": get_marker_y(close_price, position, state["open_level"], state["high_level"]),
                        "stop": high_price if position == "above" else low_price,
                        "timed_out": False,
                    }
                prev_position = position
                continue  # nothing to do while inside and no active break

            # ── active_break is set ──────────────────────────────────────────
            # Keep stop at the true extreme wick while price remains outside.
            # For above-break (future short), stop must follow highest high.
            # For below-break (future long), stop must follow lowest low.
            if position == active_break["position"] == "above":
                active_break["stop"] = max(active_break["stop"], high_price)
            elif position == active_break["position"] == "below":
                active_break["stop"] = min(active_break["stop"], low_price)

            if position == "inside":
                # Price confirmed back inside on a closed candle close.
                within_hour = candle_time_ms - active_break["time_ms"] <= one_hour_ms
                if within_hour and not active_break["timed_out"]:
                    pair_return_xs.append(i)
                    pair_return_ys.append(close_price)

                    # Enter trade on confirmed return-inside close.
                    # Break above range -> short. Break below range -> long.
                    if active_break["position"] == "above":
                        side = "short"
                        entry = close_price
                        stop = active_break["stop"]
                        risk = stop - entry
                        target = entry - (state["target_r"] * risk)
                    else:
                        side = "long"
                        entry = close_price
                        stop = active_break["stop"]
                        risk = entry - stop
                        target = entry + (state["target_r"] * risk)

                    if risk > 0:
                        liquidation_price = calculate_liquidation_price(entry, side, state["leverage"])
                        active_trade = {
                            "side": side,
                            "entry_index": i,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "initial_risk": risk,
                            "break_even_armed": False,
                            "liquidation_price": liquidation_price,
                            "leverage": state["leverage"],
                        }
                else:
                    # Emit yellow only if not already emitted for this run.
                    if not active_break["timed_out"]:
                        single_break_xs.append(active_break["index"])
                        single_break_ys.append(active_break["y"])
                    late_return_xs.append(i)
                    late_return_ys.append(close_price)
                active_break = None
                prev_position = position
                continue

            # Still outside – mark as single once after 1 h (only once per run).
            if (
                not active_break["timed_out"]
                and candle_time_ms - active_break["time_ms"] > one_hour_ms
            ):
                single_break_xs.append(active_break["index"])
                single_break_ys.append(active_break["y"])
                active_break["timed_out"] = True

            # If direction flipped (crossed entire range in one candle), reset tracker.
            if position != active_break["position"]:
                break_detect_xs.append(i)
                break_detect_ys.append(get_marker_y(close_price, position, state["open_level"], state["high_level"]))
                active_break = {
                    "index": i,
                    "time_ms": candle_time_ms,
                    "position": position,
                    "y": get_marker_y(close_price, position, state["open_level"], state["high_level"]),
                    "stop": high_price if position == "above" else low_price,
                    "timed_out": False,
                }

            prev_position = position

        # Still-active break at end of closed data – emit yellow if time exceeded.
        if active_break is not None and not active_break["timed_out"]:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if current_time_ms - active_break["time_ms"] > one_hour_ms:
                single_break_xs.append(active_break["index"])
                single_break_ys.append(active_break["y"])

        # Open trade is projected to the latest closed candle.
        if active_trade is not None and closed_klines:
            trades.append(
                {
                    **active_trade,
                    "exit_index": len(closed_klines) - 1,
                    "exit_price": float(closed_klines[-1][4]),
                    "result": "open",
                }
            )

        if break_detect_xs:
            ax.scatter(
                break_detect_xs,
                break_detect_ys,
                marker="D",
                color="#00E5FF",
                s=64,
                zorder=5,
                label="Break detected",
            )
        if pair_return_xs:
            ax.scatter(
                pair_return_xs,
                pair_return_ys,
                marker="o",
                color="#7CFFB2",
                s=54,
                zorder=5,
                label="Return inside within 1h",
            )
        if single_break_xs:
            ax.scatter(
                single_break_xs,
                single_break_ys,
                marker="D",
                color="#FFD54F",
                s=64,
                zorder=5,
                label="Break unresolved after 1h",
            )
        if late_return_xs:
            ax.scatter(
                late_return_xs,
                late_return_ys,
                marker="o",
                color="#FF8A65",
                s=54,
                zorder=5,
                label="Late return inside",
            )

        # Draw position overlay (TradingView-like): entry, stop, target + risk/reward bands.
        entry_label_added = False
        stop_label_added = False
        target_label_added = False
        liq_label_added = False
        exit_label_added = False
        for trade in trades:
            x0 = trade["entry_index"]
            x1 = trade["exit_index"] if trade["exit_index"] > x0 else x0 + 0.2

            risk_low = min(trade["entry"], trade["stop"])
            risk_high = max(trade["entry"], trade["stop"])
            reward_low = min(trade["entry"], trade["target"])
            reward_high = max(trade["entry"], trade["target"])

            ax.fill_between([x0, x1], risk_low, risk_high, color="#FF5252", alpha=0.16, zorder=2)
            ax.fill_between([x0, x1], reward_low, reward_high, color="#00E676", alpha=0.14, zorder=2)

            ax.hlines(
                trade["entry"], x0, x1,
                colors="#ECEFF1", linewidth=1.2,
                label="Entry" if not entry_label_added else None,
            )
            entry_label_added = True

            ax.hlines(
                trade["stop"], x0, x1,
                colors="#FF5252", linewidth=1.0, linestyles="--",
                label="Stop Loss" if not stop_label_added else None,
            )
            stop_label_added = True

            ax.hlines(
                trade["target"], x0, x1,
                colors="#00E676", linewidth=1.0, linestyles="--",
                label=f"Take Profit ({state['target_r']:.2f}R)" if not target_label_added else None,
            )
            target_label_added = True

            ax.hlines(
                trade["liquidation_price"], x0, x1,
                colors="#FF1744", linewidth=1.0, linestyles=":",
                label="Liquidation Price" if not liq_label_added else None,
            )
            liq_label_added = True

            ax.scatter(
                [x0], [trade["entry"]], marker=">", s=70, color="#ECEFF1", zorder=6
            )
            if trade["result"] == "target":
                exit_color = "#00E676"
            elif trade["result"] == "liquidation":
                exit_color = "#FF1744"
            else:
                exit_color = "#FF5252"
            if trade["result"] == "open":
                exit_color = "#90A4AE"
            ax.scatter(
                [x1], [trade["exit_price"]], marker="X", s=70, color=exit_color,
                zorder=6,
                label="Trade Exit" if not exit_label_added and trade["result"] != "open" else None,
            )
            if trade["result"] != "open":
                exit_label_added = True
                pnl_pct = calculate_price_pnl_pct(trade["entry"], trade["exit_price"], trade["side"])
                lev_pnl_pct = calculate_leveraged_pnl_pct(
                    trade["entry"], trade["exit_price"], trade["side"], trade.get("leverage", state["leverage"])
                )

                pnl_sign = "+" if pnl_pct >= 0 else ""
                ax.annotate(
                    f"{trade['result'].upper()}\nPx {pnl_sign}{pnl_pct:.2f}% | Lev {lev_pnl_pct:+.2f}%",
                    xy=(x1, trade["exit_price"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    color=exit_color,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#131722", ec=exit_color, alpha=0.85),
                    zorder=7,
                )

        default_xlim = (-1, max(n, 1))
        if restore_view and previous_xlim is not None and previous_ylim is not None:
            min_x, max_x = default_xlim
            left = max(min_x, min(previous_xlim[0], max_x))
            right = max(min_x, min(previous_xlim[1], max_x))
            if right - left > 1e-9:
                ax.set_xlim(left, right)
            else:
                ax.set_xlim(*default_xlim)
            ax.set_ylim(previous_ylim)
        else:
            ax.set_xlim(*default_xlim)
        state["reset_view"] = False

        # X-axis: show time labels every ~30 min (every 6 candles at 5-min interval)
        step = 6
        tick_positions = list(range(0, n, step))
        tick_labels = [times_ny[i].strftime("%H:%M") for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        date_lbl = today_midnight_ny.strftime("%Y-%m-%d")
        chart_mode = "Historical" if is_historical else "Live"
        ax.set_title(
            f"{state['display_symbol']}  |  5-Min {chart_mode} Chart  {date_lbl}  |  Lev {state['leverage']:.1f}x  (UTC-4 / NY)",
            color="white",
            fontsize=13,
            pad=10,
        )
        ax.set_xlabel("Time (NY)", color="#9e9e9e")
        ax.set_ylabel("Price (USDT)", color="#9e9e9e")
        ax.tick_params(colors="#9e9e9e")
        legend_title = f"Price: {current_price:.2f}" if current_price is not None else None
        handles, labels = ax.get_legend_handles_labels()
        legend_items = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
        legend = None
        if legend_items:
            legend = ax.legend(
                [h for h, _ in legend_items],
                [l for _, l in legend_items],
                loc="upper left",
                facecolor="#1e2130",
                labelcolor="white",
                fontsize=9,
                framealpha=0.9,
                title=legend_title,
                title_fontsize=10,
            )
        if legend is not None and legend.get_title() is not None:
            legend.get_title().set_color("white")
        ax.grid(True, color="#2a2e39", linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2e39")

    update(0)
    ani = animation.FuncAnimation(
        fig, update, interval=1_000, cache_frame_data=False
    )

    # ── Ruler tool ──────────────────────────────────────────────────────────
    # Left-click once  → set start point (yellow dot)
    # Left-click again → complete ruler (line + % label); start resets
    # Right-click      → clear ruler
    ruler = {"start": None, "end": None}
    ruler_artists = []  # track drawn artists so they survive redraws

    def draw_ruler():
        for a in ruler_artists:
            try:
                a.remove()
            except Exception:
                pass
        ruler_artists.clear()

        if ruler["start"] is None:
            fig.canvas.draw_idle()
            return

        sx, sy = ruler["start"]
        dot = ax.plot(sx, sy, "o", color="#FFFF00", ms=7, zorder=10)[0]
        ruler_artists.append(dot)

        if ruler["end"] is not None:
            ex, ey = ruler["end"]
            line = ax.plot([sx, ex], [sy, ey], color="#FFFF00",
                           linewidth=1.4, linestyle="-", zorder=10)[0]
            ruler_artists.append(line)

            pct = (ey - sy) / sy * 100
            sign = "+" if pct >= 0 else ""
            label = ax.annotate(
                f"{sign}{pct:.3f}%\n{ey:.2f}",
                xy=(ex, ey),
                xytext=(8, 8),
                textcoords="offset points",
                color="#FFFF00",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#131722", ec="#FFFF00", alpha=0.85),
                zorder=11,
            )
            ruler_artists.append(label)

            dot2 = ax.plot(ex, ey, "o", color="#FFFF00", ms=7, zorder=10)[0]
            ruler_artists.append(dot2)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 3:  # right-click → clear
            ruler["start"] = None
            ruler["end"] = None
            draw_ruler()
            return
        if event.button == 1:  # left-click
            if ruler["start"] is None:
                ruler["start"] = (event.xdata, event.ydata)
                ruler["end"] = None
            else:
                ruler["end"] = (event.xdata, event.ydata)
                draw_ruler()
                ruler["start"] = None  # reset for next measurement
                return
            draw_ruler()

    # Re-draw ruler after every live update so it survives ax.clear()
    def update_with_ruler(frame):
        update(frame)
        draw_ruler()

    ani.func = update_with_ruler
    fig.canvas.mpl_connect("button_press_event", on_click)

    # In-app symbol switcher
    status_text = fig.text(
        0.02,
        0.965,
        "",
        ha="left",
        va="top",
        fontsize=9,
        color="#9e9e9e",
    )

    symbol_ax = fig.add_axes([0.12, 0.01, 0.2, 0.04])
    symbol_box = TextBox(
        symbol_ax,
        "Symbol",
        initial=state["display_symbol"],
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    symbol_ax.set_facecolor("#1e2130")
    symbol_box.label.set_color("white")
    symbol_box.text_disp.set_color("white")
    if hasattr(symbol_box, "cursor") and symbol_box.cursor is not None:
        symbol_box.cursor.set_color("white")

    def on_chart_text_change(_text):
        symbol_box.text_disp.set_color("white")
        if hasattr(symbol_box, "cursor") and symbol_box.cursor is not None:
            symbol_box.cursor.set_color("white")

    symbol_box.on_text_change(on_chart_text_change)

    load_ax = fig.add_axes([0.33, 0.01, 0.08, 0.04])
    load_btn = Button(load_ax, "Load", color="#2a2e39", hovercolor="#3a4152")
    load_btn.label.set_color("white")

    def apply_symbol(new_symbol):
        typed = new_symbol.strip().upper()
        if not typed:
            status_text.set_text("Enter a symbol, e.g. BTCUSDT_PERP")
            fig.canvas.draw_idle()
            return

        try:
            new_api_symbol, new_market = normalize_symbol_and_market(typed)
            new_open, new_high = get_first_4h_candle_levels(new_api_symbol, new_market, state["selected_date"])
            state["display_symbol"] = typed
            state["symbol_for_api"] = new_api_symbol
            state["market"] = new_market
            state["open_level"] = new_open
            state["high_level"] = new_high
            state["klines_cache"] = None
            state["last_klines_fetch_ts"] = 0.0
            state["reset_view"] = True
            ruler["start"] = None
            ruler["end"] = None
            status_text.set_text(f"Loaded {typed} ({new_market.upper()})")
            update_with_ruler(0)
        except Exception as e:
            status_text.set_text(f"Failed to load {typed}: {e}")
            fig.canvas.draw_idle()

    symbol_box.on_submit(apply_symbol)

    def on_load_click(event):
        apply_symbol(symbol_box.text)

    load_btn.on_clicked(on_load_click)

    aux_windows = {
        "monitor_fig": None,
        "monitor_ani": None,
        "monitor_hidden": False,
        "status_fig": None,
        "status_ani": None,
        "status_hidden": False,
    }

    def is_open(fig_obj):
        if fig_obj is None:
            return False
        try:
            return plt.fignum_exists(fig_obj.number)
        except Exception:
            return False

    def safe_stop_animation(ani_obj):
        if ani_obj is None:
            return
        try:
            if hasattr(ani_obj, "event_source") and ani_obj.event_source is not None:
                ani_obj.event_source.stop()
        except Exception:
            pass

    def safe_start_animation(ani_obj):
        if ani_obj is None:
            return
        try:
            if hasattr(ani_obj, "event_source") and ani_obj.event_source is not None:
                ani_obj.event_source.start()
        except Exception:
            pass

    def hide_figure(fig_obj):
        if fig_obj is None:
            return
        try:
            manager = getattr(fig_obj.canvas, "manager", None)
            window = getattr(manager, "window", None)
            if window is not None:
                if hasattr(window, "withdraw"):
                    window.withdraw()
                    return
                if hasattr(window, "setVisible"):
                    window.setVisible(False)
                    return
                if hasattr(window, "iconify"):
                    window.iconify()
                    return
            fig_obj.set_visible(False)
        except Exception:
            pass

    def show_figure(fig_obj):
        if fig_obj is None:
            return
        try:
            manager = getattr(fig_obj.canvas, "manager", None)
            window = getattr(manager, "window", None)
            if window is not None:
                if hasattr(window, "deiconify"):
                    window.deiconify()
                if hasattr(window, "setVisible"):
                    window.setVisible(True)
                if hasattr(window, "lift"):
                    window.lift()
            fig_obj.set_visible(True)
            fig_obj.show()
            fig_obj.canvas.draw_idle()
        except Exception:
            pass

    def update_toggle_labels():
        mon_on = is_open(aux_windows["monitor_fig"]) and not aux_windows.get("monitor_hidden", False)
        stat_on = is_open(aux_windows["status_fig"]) and not aux_windows.get("status_hidden", False)
        monitor_toggle_btn.label.set_text("Monitor: ON" if mon_on else "Monitor: OFF")
        status_toggle_btn.label.set_text("Status: ON" if stat_on else "Status: OFF")
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    def bind_aux_close(fig_obj, kind):
        if fig_obj is None:
            return

        def on_aux_close(_event):
            if kind == "monitor":
                aux_windows["monitor_fig"] = None
                aux_windows["monitor_ani"] = None
                aux_windows["monitor_hidden"] = False
            else:
                aux_windows["status_fig"] = None
                aux_windows["status_ani"] = None
                aux_windows["status_hidden"] = False
            update_toggle_labels()

        fig_obj.canvas.mpl_connect("close_event", on_aux_close)

    def toggle_monitor_window(_event=None):
        try:
            if is_open(aux_windows["monitor_fig"]):
                if aux_windows.get("monitor_hidden", False):
                    show_figure(aux_windows["monitor_fig"])
                    safe_start_animation(aux_windows.get("monitor_ani"))
                    aux_windows["monitor_hidden"] = False
                    status_text.set_text("Positions monitor opened")
                else:
                    safe_stop_animation(aux_windows.get("monitor_ani"))
                    hide_figure(aux_windows["monitor_fig"])
                    aux_windows["monitor_hidden"] = True
                    status_text.set_text("Positions monitor hidden")
            else:
                monitor_fig, monitor_ani = launch_positions_monitor(
                    state["display_symbol"],
                    initial_leverage=state["leverage"],
                    enable_break_even=state["enable_break_even"],
                    break_even_trigger_r=state["break_even_trigger_r"],
                    target_r=state["target_r"],
                    open_symbol_callback=lambda sym: symbol_box.set_val(sym),
                )
                aux_windows["monitor_fig"] = monitor_fig
                aux_windows["monitor_ani"] = monitor_ani
                aux_windows["monitor_hidden"] = False
                bind_aux_close(monitor_fig, "monitor")
                show_figure(monitor_fig)
                status_text.set_text("Positions monitor opened")
        except Exception as exc:
            status_text.set_text(f"Failed to toggle monitor: {exc}")
        update_toggle_labels()

    def toggle_status_window(_event=None):
        try:
            if is_open(aux_windows["status_fig"]):
                if aux_windows.get("status_hidden", False):
                    show_figure(aux_windows["status_fig"])
                    aux_windows["status_hidden"] = False
                    status_text.set_text("Paper live status window opened")
                else:
                    hide_figure(aux_windows["status_fig"])
                    aux_windows["status_hidden"] = True
                    status_text.set_text("Paper live status window hidden")
            else:
                status_fig, status_ani = launch_paper_live_status_window()
                aux_windows["status_fig"] = status_fig
                aux_windows["status_ani"] = status_ani
                aux_windows["status_hidden"] = False
                bind_aux_close(status_fig, "status")
                show_figure(status_fig)
                status_text.set_text("Paper live status window opened")
        except Exception as exc:
            status_text.set_text(f"Failed to toggle status: {exc}")
        update_toggle_labels()

    # Date picker for chart
    date_chart_ax = fig.add_axes([0.44, 0.01, 0.15, 0.04])
    date_chart_box = TextBox(
        date_chart_ax,
        "Date",
        initial=state["selected_date"].strftime("%Y-%m-%d"),
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    date_chart_ax.set_facecolor("#1e2130")
    date_chart_box.label.set_color("white")
    date_chart_box.text_disp.set_color("white")
    if hasattr(date_chart_box, "cursor") and date_chart_box.cursor is not None:
        date_chart_box.cursor.set_color("white")

    def on_date_chart_text_change(_text):
        date_chart_box.text_disp.set_color("white")
        if hasattr(date_chart_box, "cursor") and date_chart_box.cursor is not None:
            date_chart_box.cursor.set_color("white")

    date_chart_box.on_text_change(on_date_chart_text_change)

    go_chart_ax = fig.add_axes([0.60, 0.01, 0.05, 0.04])
    go_chart_btn = Button(go_chart_ax, "Go", color="#2a2e39", hovercolor="#3a4152")
    go_chart_btn.label.set_color("white")

    lev_chart_ax = fig.add_axes([0.67, 0.01, 0.10, 0.04])
    lev_chart_box = TextBox(
        lev_chart_ax,
        "Lev",
        initial=f"{state['leverage']:.1f}",
        color="#1e2130",
        hovercolor="#2a2e39",
    )
    lev_chart_ax.set_facecolor("#1e2130")
    lev_chart_box.label.set_color("white")
    lev_chart_box.text_disp.set_color("white")
    if hasattr(lev_chart_box, "cursor") and lev_chart_box.cursor is not None:
        lev_chart_box.cursor.set_color("white")

    def on_lev_chart_text_change(_text):
        lev_chart_box.text_disp.set_color("white")
        if hasattr(lev_chart_box, "cursor") and lev_chart_box.cursor is not None:
            lev_chart_box.cursor.set_color("white")

    lev_chart_box.on_text_change(on_lev_chart_text_change)

    lev_chart_set_ax = fig.add_axes([0.78, 0.01, 0.09, 0.04])
    lev_chart_set_btn = Button(lev_chart_set_ax, "Set Lev", color="#2a2e39", hovercolor="#3a4152")
    lev_chart_set_btn.label.set_color("white")

    def apply_date_chart(_event=None):
        raw = date_chart_box.text.strip()
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=NY_TZ)
        except ValueError:
            status_text.set_text("Invalid date format. Use YYYY-MM-DD")
            fig.canvas.draw_idle()
            return
        try:
            new_open, new_high = get_first_4h_candle_levels(
                state["symbol_for_api"], state["market"], parsed
            )
            state["selected_date"] = parsed
            state["open_level"] = new_open
            state["high_level"] = new_high
            state["klines_cache"] = None
            state["last_klines_fetch_ts"] = 0.0
            state["reset_view"] = True
            ruler["start"] = None
            ruler["end"] = None
            status_text.set_text(f"Showing {state['display_symbol']} on {parsed.strftime('%Y-%m-%d')}")
            update_with_ruler(0)
        except Exception as e:
            status_text.set_text(f"Failed to load date {raw}: {e}")
            fig.canvas.draw_idle()

    date_chart_box.on_submit(lambda _t: apply_date_chart())
    go_chart_btn.on_clicked(apply_date_chart)

    def apply_chart_leverage(_event=None):
        raw = lev_chart_box.text.strip()
        try:
            lev = clamp_leverage(float(raw))
        except ValueError:
            status_text.set_text("Invalid leverage. Use a number like 20 or 20.0")
            fig.canvas.draw_idle()
            return
        state["leverage"] = lev
        lev_chart_box.set_val(f"{lev:.1f}")
        state["klines_cache"] = None
        state["last_klines_fetch_ts"] = 0.0
        ruler["start"] = None
        ruler["end"] = None
        status_text.set_text(f"Set leverage to {lev:.1f}x")
        update_with_ruler(0)

    lev_chart_set_btn.on_clicked(apply_chart_leverage)
    lev_chart_box.on_submit(lambda _t: apply_chart_leverage())

    monitor_toggle_ax = fig.add_axes([0.875, 0.01, 0.06, 0.04])
    monitor_toggle_btn = Button(monitor_toggle_ax, "Monitor: OFF", color="#2a2e39", hovercolor="#3a4152")
    monitor_toggle_btn.label.set_color("white")

    status_toggle_ax = fig.add_axes([0.94, 0.01, 0.055, 0.04])
    status_toggle_btn = Button(status_toggle_ax, "Status: OFF", color="#2a2e39", hovercolor="#3a4152")
    status_toggle_btn.label.set_color("white")

    monitor_toggle_btn.on_clicked(toggle_monitor_window)
    status_toggle_btn.on_clicked(toggle_status_window)
    update_toggle_labels()

    # Keep strong references so widgets stay interactive.
    fig._chart_ui_refs = {
        "symbol_box": symbol_box,
        "on_chart_text_change": on_chart_text_change,
        "load_btn": load_btn,
        "apply_symbol": apply_symbol,
        "on_load_click": on_load_click,
        "date_chart_box": date_chart_box,
        "on_date_chart_text_change": on_date_chart_text_change,
        "go_chart_btn": go_chart_btn,
        "lev_chart_box": lev_chart_box,
        "on_lev_chart_text_change": on_lev_chart_text_change,
        "lev_chart_set_btn": lev_chart_set_btn,
        "apply_chart_leverage": apply_chart_leverage,
        "apply_date_chart": apply_date_chart,
        "aux_windows": aux_windows,
        "toggle_monitor_window": toggle_monitor_window,
        "toggle_status_window": toggle_status_window,
        "update_toggle_labels": update_toggle_labels,
        "monitor_toggle_btn": monitor_toggle_btn,
        "status_toggle_btn": status_toggle_btn,
        "status_text": status_text,
    }

    fig.text(
        0.5, 0.002,
        "Ruler: left-click = set start | left-click again = measure | right-click = clear",
        ha="center", fontsize=8, color="#9e9e9e",
    )
    # ────────────────────────────────────────────────────────────────────────

    plt.subplots_adjust(bottom=0.1, top=0.93)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "BTCUSDT_PERP"

    if len(sys.argv) > 2:
        try:
            leverage = clamp_leverage(float(sys.argv[2]))
        except ValueError:
            leverage = DEFAULT_LEVERAGE
    else:
        leverage = DEFAULT_LEVERAGE

    live_5min_chart(symbol, leverage=leverage)
