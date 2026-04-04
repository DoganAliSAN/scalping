from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from binance import Client

from binance_live import (
    BinanceLiveError,
    close_futures_position,
    get_futures_account_balance_detail,
    get_futures_open_positions,
    get_live_runtime_summary,
    open_new_futures_position,
)
from live_runner import get_live_runner_status, start_live_runner, stop_live_runner

from main import (
    NY_TZ,
    PAPER_LIVE_STATE_PATH,
    calculate_leveraged_pnl_pct,
    calculate_price_pnl_pct,
    clamp_leverage,
    get_first_4h_candle_levels,
    get_klines,
    get_symbol_positions_summary,
    load_live_selected_symbols,
    load_watchlist_cache,
    normalize_symbol_and_market,
)

PAPER_LIVE_OUTPUT_DIR = Path(__file__).with_name("paper_live_logs")
PAPER_LIVE_CURRENT_POSITIONS_PATH = PAPER_LIVE_OUTPUT_DIR / "current_positions.json"
PAPER_LIVE_EVENTS_PATH = PAPER_LIVE_OUTPUT_DIR / "events.jsonl"
PAPER_LIVE_DAYS_DIR = PAPER_LIVE_OUTPUT_DIR / "days"


@st.cache_data(ttl=10)
def load_day_klines(display_symbol: str, date_str: str):
    selected_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=NY_TZ)
    symbol_for_api, market = normalize_symbol_and_market(display_symbol)
    start_str = selected_date.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")

    now_ny = datetime.now(NY_TZ)
    if selected_date.date() < now_ny.date():
        end_str = (selected_date + timedelta(days=1)).astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
        klines = get_klines(symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE, start_str, end_str)
    else:
        klines = get_klines(symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE, start_str)

    return selected_date, symbol_for_api, market, klines or []


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=5)
def load_binance_snapshot():
    runtime = get_live_runtime_summary()
    try:
        account = get_futures_account_balance_detail()
        positions = get_futures_open_positions()
        return {
            "runtime": runtime,
            "account": account,
            "positions": positions,
            "error": "",
        }
    except Exception as exc:
        return {
            "runtime": runtime,
            "account": None,
            "positions": [],
            "error": str(exc),
        }


def last_event_line(path: Path) -> str:
    if not path.exists():
        return "No events file"
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end <= 0:
                return "No events"
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
        payload = json.loads(raw)
        return f"{payload.get('ts_utc', '?')} | {payload.get('event', '?')} | date={payload.get('date', '?')}"
    except Exception as exc:
        return f"Failed to read last event: {exc}"


def load_rules_source(state: dict | None):
    """
    Resolve setup rules from the most relevant day summary file.
    Priority:
    1) state.last_processed_date day file
    2) latest day file in paper_live_logs/days
    """
    if not PAPER_LIVE_DAYS_DIR.exists():
        return None, "No day files found"

    if isinstance(state, dict):
        d = state.get("last_processed_date")
        if d:
            p = PAPER_LIVE_DAYS_DIR / f"{d}.json"
            if p.exists():
                payload = read_json(p)
                if isinstance(payload, dict):
                    return payload, f"rules source: {p.name}"

    day_files = sorted(PAPER_LIVE_DAYS_DIR.glob("*.json"))
    if not day_files:
        return None, "No day files found"

    p = day_files[-1]
    payload = read_json(p)
    if isinstance(payload, dict):
        return payload, f"rules source: {p.name}"
    return None, "Unable to read day rules"


def build_chart(display_symbol: str, date_str: str, leverage: float):
    selected_date, _, _, klines = load_day_klines(display_symbol, date_str)
    if not klines:
        st.warning("No kline data returned for the selected symbol/date.")
        return

    times = [datetime.fromtimestamp(k[0] / 1000, tz=NY_TZ) for k in klines]
    opens = [float(k[1]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=times,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name=display_symbol,
            )
        ]
    )

    low_4h, high_4h = get_first_4h_candle_levels(*normalize_symbol_and_market(display_symbol), selected_date)
    if low_4h is not None:
        fig.add_hline(y=low_4h, line_dash="dash", line_color="#2196F3", annotation_text="4H Low")
    if high_4h is not None:
        fig.add_hline(y=high_4h, line_dash="dash", line_color="#FF9800", annotation_text="4H High")

    summary = get_symbol_positions_summary(
        display_symbol,
        selected_date,
        leverage=leverage,
    )

    trades = summary.get("trades", [])
    closed_times = times[:-1] if len(times) > 1 else times

    def idx_to_time(index: int):
        if closed_times:
            i = max(0, min(int(index), len(closed_times) - 1))
            return closed_times[i]
        return times[-1]

    legend_flags = {
        "entry": True,
        "stop": True,
        "target": True,
        "liq": True,
        "risk": True,
        "reward": True,
        "entry_marker": True,
        "exit_marker": True,
    }

    for t in trades:
        x0 = idx_to_time(t.get("entry_index", 0))
        x1 = idx_to_time(t.get("exit_index", t.get("entry_index", 0)))

        entry = float(t["entry"])
        stop = float(t["stop"])
        target = float(t["target"])
        liq = float(t["liquidation_price"])
        exit_price = float(t["exit_price"])

        risk_low, risk_high = min(entry, stop), max(entry, stop)
        reward_low, reward_high = min(entry, target), max(entry, target)

        # Risk / reward blocks
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=risk_low,
            y1=risk_high,
            line_width=0,
            fillcolor="rgba(255,82,82,0.18)",
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=reward_low,
            y1=reward_high,
            line_width=0,
            fillcolor="rgba(0,230,118,0.16)",
            layer="below",
        )

        if legend_flags["risk"]:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color="rgba(255,82,82,0.35)", symbol="square"),
                    name="Risk Zone",
                )
            )
            legend_flags["risk"] = False

        if legend_flags["reward"]:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color="rgba(0,230,118,0.3)", symbol="square"),
                    name="Reward Zone",
                )
            )
            legend_flags["reward"] = False

        # Level lines
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[entry, entry],
                mode="lines",
                line=dict(color="#ECEFF1", width=1.6),
                name="Entry",
                showlegend=legend_flags["entry"],
                hovertemplate="Entry %{y:.6f}<extra></extra>",
            )
        )
        legend_flags["entry"] = False

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[stop, stop],
                mode="lines",
                line=dict(color="#FF5252", width=1.2, dash="dash"),
                name="Stop",
                showlegend=legend_flags["stop"],
                hovertemplate="Stop %{y:.6f}<extra></extra>",
            )
        )
        legend_flags["stop"] = False

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[target, target],
                mode="lines",
                line=dict(color="#00E676", width=1.2, dash="dash"),
                name="Target",
                showlegend=legend_flags["target"],
                hovertemplate="Target %{y:.6f}<extra></extra>",
            )
        )
        legend_flags["target"] = False

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[liq, liq],
                mode="lines",
                line=dict(color="#FF1744", width=1.0, dash="dot"),
                name="Liquidation",
                showlegend=legend_flags["liq"],
                hovertemplate="Liq %{y:.6f}<extra></extra>",
            )
        )
        legend_flags["liq"] = False

        # Entry / exit markers
        fig.add_trace(
            go.Scatter(
                x=[x0],
                y=[entry],
                mode="markers",
                marker=dict(size=10, color="#ECEFF1", symbol="triangle-right"),
                name="Entry Marker",
                showlegend=legend_flags["entry_marker"],
                hovertemplate=f"{t['side'].upper()} Entry {{y:.6f}}<extra></extra>",
            )
        )
        legend_flags["entry_marker"] = False

        result = str(t.get("result", "")).lower()
        if result == "target":
            exit_color = "#00E676"
        elif result == "liquidation":
            exit_color = "#FF1744"
        elif result == "open":
            exit_color = "#90A4AE"
        else:
            exit_color = "#FF8A65"

        px = calculate_price_pnl_pct(entry, exit_price, t["side"])
        lev = calculate_leveraged_pnl_pct(entry, exit_price, t["side"], t.get("leverage", leverage))
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[exit_price],
                mode="markers+text",
                text=[f"{str(t.get('result', '')).upper()} | Px {px:+.2f}% | Lev {lev:+.2f}%"],
                textposition="top right",
                marker=dict(size=10, color=exit_color, symbol="x"),
                name="Exit Marker",
                showlegend=legend_flags["exit_marker"],
                hovertemplate="Exit %{y:.6f}<extra></extra>",
            )
        )
        legend_flags["exit_marker"] = False

    open_trade = summary.get("open_trade")
    if open_trade:
        fig.add_hline(y=float(open_trade["entry"]), line_color="#ECEFF1", annotation_text="Open Entry")
        fig.add_hline(y=float(open_trade["stop"]), line_dash="dot", line_color="#FF5252", annotation_text="Open Stop")
        fig.add_hline(y=float(open_trade["target"]), line_dash="dot", line_color="#00E676", annotation_text="Open Target")

    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_title="Time (NY)",
        yaxis_title="Price",
        title=f"{display_symbol} | 5m Chart | {date_str} | {leverage:.1f}x",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    if trades:
        rows = []
        for idx, t in enumerate(trades, start=1):
            px = calculate_price_pnl_pct(t["entry"], t["exit_price"], t["side"])
            lev = calculate_leveraged_pnl_pct(t["entry"], t["exit_price"], t["side"], t.get("leverage", leverage))
            rows.append(
                {
                    "#": idx,
                    "side": t["side"],
                    "result": t["result"],
                    "entry": round(float(t["entry"]), 6),
                    "exit": round(float(t["exit_price"]), 6),
                    "stop": round(float(t["stop"]), 6),
                    "target": round(float(t["target"]), 6),
                    "pnl_pct": round(px, 2),
                    "lev_pnl_pct": round(lev, 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_monitor(source_mode: str, manual_symbols: list[str], leverage: float, date_str: str):
    selected_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=NY_TZ)
    if source_mode == "Live Selected":
        symbols, source_error = load_live_selected_symbols()
        if source_error:
            st.info(source_error)
    else:
        symbols = manual_symbols

    if not symbols:
        st.info("No symbols to monitor.")
        return

    rows = []
    for sym in symbols:
        try:
            s = get_symbol_positions_summary(sym, selected_date, leverage=leverage)
            ot = s.get("open_trade")
            rows.append(
                {
                    "symbol": sym,
                    "status": "OPEN" if ot else ("CLOSED" if s.get("closed_count", 0) > 0 else "NONE"),
                    "side": (ot.get("side", "") if ot else ""),
                    "entry": (round(float(ot.get("entry", 0.0)), 6) if ot else None),
                    "lev_pnl_pct": round(float(s.get("leveraged_pnl_pct_total", 0.0)), 2),
                    "closed": int(s.get("closed_count", 0)),
                    "liq": int(s.get("liquidation_count", 0)),
                }
            )
        except Exception as exc:
            rows.append({"symbol": sym, "status": "ERROR", "side": "", "entry": None, "lev_pnl_pct": 0.0, "closed": 0, "liq": 0, "error": str(exc)})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_binance_live_status(default_symbol: str, default_leverage: float):
    st.subheader("Binance Live")

    runner_status = get_live_runner_status()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Runner", "RUNNING" if runner_status.get("running") else "STOPPED")
    r2.metric("Runner Mode", str(runner_status.get("execution_mode", "stopped")).upper())
    r3.metric("Runner PID", runner_status.get("pid") or "-")
    r4.metric("Managed", "YES" if runner_status.get("managed") else "NO")

    cta1, cta2 = st.columns(2)
    if cta1.button("Activate Live Binance Execution", use_container_width=True):
        try:
            result = start_live_runner(execution_mode="binance", poll_seconds=300)
            st.cache_data.clear()
            st.success(f"Started Binance live runner with PID {result['pid']}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to start Binance live execution: {exc}")

    if cta2.button("Deactivate Live Execution", use_container_width=True):
        try:
            result = stop_live_runner(force=True)
            st.cache_data.clear()
            if result.get("stopped"):
                st.success(result.get("message", "Live execution stopped."))
            else:
                st.info(result.get("message", "No live runner process found."))
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to stop live execution: {exc}")

    if runner_status.get("command"):
        st.caption(f"Runner command: {runner_status['command']}")
    if runner_status.get("log_path"):
        st.caption(f"Runner log: {runner_status['log_path']}")

    runtime = get_live_runtime_summary()
    if not runtime.get("credentials_loaded"):
        st.warning("BINANCE_API_KEY and BINANCE_API_SECRET are not loaded. Live trading and account reads are disabled.")
    else:
        default_trade_symbol, default_market = normalize_symbol_and_market(default_symbol)
        manual_symbol_default = default_trade_symbol if default_market == "usdm" else "BTCUSDT"

        c1, c2, c3, c4 = st.columns(4)
        manual_symbol = c1.text_input("Manual Trade Symbol", value=manual_symbol_default, key="manual_trade_symbol").strip().upper()
        manual_side = c2.selectbox("Manual Side", options=["long", "short"], key="manual_trade_side")
        manual_size = c3.number_input(
            "Manual Margin Size (USD)",
            min_value=1.0,
            value=float(runtime.get("position_size_usd", 33.0)),
            step=1.0,
            key="manual_trade_size",
        )
        manual_leverage = c4.number_input(
            "Manual Leverage",
            min_value=1.0,
            max_value=125.0,
            value=float(runtime.get("leverage", default_leverage)),
            step=1.0,
            key="manual_trade_leverage",
        )

        b1, b2 = st.columns(2)
        if b1.button("Open Test Position", use_container_width=True):
            try:
                symbol_for_api, market = normalize_symbol_and_market(manual_symbol)
                if market != "usdm":
                    raise BinanceLiveError("Manual trading is only enabled for USDT-M futures symbols.")
                open_new_futures_position(
                    symbol=symbol_for_api,
                    side=manual_side,
                    margin_usd=float(manual_size),
                    leverage=float(manual_leverage),
                    source="dashboard_manual",
                )
                st.cache_data.clear()
                st.success(f"Opened {manual_side.upper()} test position on {symbol_for_api}.")
            except Exception as exc:
                st.error(f"Failed to open position: {exc}")

        if b2.button("Close Symbol Position", use_container_width=True):
            try:
                symbol_for_api, market = normalize_symbol_and_market(manual_symbol)
                if market != "usdm":
                    raise BinanceLiveError("Manual close is only enabled for USDT-M futures symbols.")
                close_futures_position(symbol_for_api, source="dashboard_manual", allow_missing=False)
                st.cache_data.clear()
                st.success(f"Closed Binance position for {symbol_for_api}.")
            except Exception as exc:
                st.error(f"Failed to close position: {exc}")

    snapshot = load_binance_snapshot()
    runtime = snapshot.get("runtime", {})
    st.caption(
        "Mode: Binance USDT-M futures"
        f" | Testnet: {'YES' if runtime.get('testnet') else 'NO'}"
        f" | Default size: {float(runtime.get('position_size_usd', 0.0)):.2f}"
        f" | Max positions: {int(runtime.get('max_open_positions', 0) or 0)}"
    )

    if snapshot.get("error"):
        st.error(snapshot["error"])
        return

    account = snapshot.get("account") or {}
    positions = snapshot.get("positions") or []

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Available Balance", f"{float(account.get('available_balance', 0.0)):.2f} {account.get('margin_asset', 'USDT')}")
    c6.metric("Wallet Balance", f"{float(account.get('wallet_balance', 0.0)):.2f}")
    c7.metric("Unrealized PnL", f"{float(account.get('total_unrealized_profit', 0.0)):+.2f}")
    c8.metric("Open Binance Positions", len(positions))

    if positions:
        st.markdown("Real Binance positions")
        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
    else:
        st.info("No open Binance USDT-M positions.")

    assets = account.get("assets", []) if isinstance(account, dict) else []
    if assets:
        st.markdown("Binance futures balances")
        st.dataframe(pd.DataFrame(assets), use_container_width=True, hide_index=True)


def build_paper_live_status(default_symbol: str, default_leverage: float):
    state = read_json(PAPER_LIVE_STATE_PATH)
    snap = read_json(PAPER_LIVE_CURRENT_POSITIONS_PATH)

    build_binance_live_status(default_symbol, default_leverage)

    day_payload, rules_source = load_rules_source(state if isinstance(state, dict) else None)

    c1, c2, c3 = st.columns(3)
    if isinstance(state, dict):
        c1.metric("Balance", f"{float(state.get('balance', 0.0)):.2f}")
        c2.metric("Processed Days", int(state.get("processed_days", 0)))
        c3.metric("Last Processed", state.get("last_processed_date", "-"))
    else:
        c1.metric("Balance", "-")
        c2.metric("Processed Days", "-")
        c3.metric("Last Processed", "-")

    if isinstance(snap, dict):
        c4, c5, c6 = st.columns(3)
        c4.metric("Snapshot Date", snap.get("date", "-"))
        c5.metric("Open Positions", len(snap.get("open_positions", []) or []))
        c6.metric("Realized PnL %", f"{float(snap.get('realized_pnl_pct', 0.0)):.2f}")

        open_positions = snap.get("open_positions", []) or []
        if open_positions:
            st.subheader("Strategy Open Positions")
            st.dataframe(pd.DataFrame(open_positions), use_container_width=True, hide_index=True)

        binance_positions = snap.get("binance_open_positions", []) or []
        if binance_positions:
            st.subheader("Snapshot Binance Positions")
            st.dataframe(pd.DataFrame(binance_positions), use_container_width=True, hide_index=True)

        if snap.get("binance_error"):
            st.warning(f"Binance snapshot error: {snap.get('binance_error')}")

    st.subheader("Setup Rules")
    if isinstance(day_payload, dict):
        settings = day_payload.get("settings", {}) if isinstance(day_payload.get("settings", {}), dict) else {}
        if settings:
            rules_row = {
                "weekly_days": settings.get("weekly_days"),
                "monthly_days": settings.get("monthly_days"),
                "min_trades": settings.get("min_trades"),
                "min_weekly_wr": settings.get("min_weekly_wr"),
                "min_monthly_wr": settings.get("min_monthly_wr"),
                "min_weekly_nlr": settings.get("min_weekly_nlr"),
                "min_monthly_nlr": settings.get("min_monthly_nlr"),
                "top_n": settings.get("top_n"),
                "leverage": settings.get("leverage"),
                "be_stop": settings.get("be_stop"),
                "be_trigger_r": settings.get("be_trigger_r"),
                "target_r": settings.get("target_r"),
                "max_trades_per_day": settings.get("max_trades_per_day"),
                "daily_loss_limit": settings.get("daily_loss_limit"),
                "position_size": settings.get("position_size"),
                "start_balance": settings.get("start_balance"),
                "max_open_positions": settings.get("max_open_positions"),
                "execution_mode": settings.get("execution_mode"),
            }

            if isinstance(snap, dict):
                rules_row["live_daily_loss_limit"] = snap.get("daily_loss_limit")
                rules_row["live_snapshot_date"] = snap.get("date")

            if isinstance(state, dict):
                live_day = state.get("live_day")
                if isinstance(live_day, dict):
                    rules_row["live_slots"] = live_day.get("slots")

            st.dataframe(pd.DataFrame([rules_row]), use_container_width=True, hide_index=True)
            st.caption(rules_source)
        else:
            st.info("No setup settings found in the selected day file.")
            st.caption(rules_source)
    else:
        st.info("Setup rules are not available yet.")
        st.caption(rules_source)

    if isinstance(state, dict):
        live_day = state.get("live_day")
        if isinstance(live_day, dict):
            st.subheader("Current Setups")

            selected = live_day.get("selected", [])
            ranked = live_day.get("ranked", [])
            symbol_states = live_day.get("symbol_states", {})

            c7, c8, c9 = st.columns(3)
            c7.metric("Live Day", live_day.get("date", "-"))
            c8.metric("Selected Setups", len(selected) if isinstance(selected, list) else 0)
            c9.metric("Ranked Candidates", len(ranked) if isinstance(ranked, list) else 0)

            if isinstance(selected, list) and selected:
                selected_rows = []
                for row in selected:
                    if not isinstance(row, dict):
                        continue
                    selected_rows.append(
                        {
                            "symbol": row.get("symbol", ""),
                            "score": round(float(row.get("score", 0.0)), 3),
                            "weekly_pnl": round(float(row.get("weekly_pnl", 0.0)), 2),
                            "monthly_pnl": round(float(row.get("monthly_pnl", 0.0)), 2),
                            "weekly_wr": round(float(row.get("weekly_wr", 0.0)), 2),
                            "monthly_wr": round(float(row.get("monthly_wr", 0.0)), 2),
                        }
                    )
                if selected_rows:
                    st.markdown("Selected setup details")
                    st.dataframe(pd.DataFrame(selected_rows), use_container_width=True, hide_index=True)

            if isinstance(symbol_states, dict) and symbol_states:
                state_rows = []
                for sym, data in symbol_states.items():
                    if not isinstance(data, dict):
                        continue
                    state_rows.append(
                        {
                            "symbol": sym,
                            "status": data.get("status", ""),
                            "skipped_reason": data.get("skipped_reason", ""),
                            "closed_trades": len(data.get("closed_trade_keys", []) or []),
                            "has_open_position": bool(data.get("open_trade_key")),
                            "realized_pnl_pct": round(float(data.get("realized_pnl_pct", 0.0)), 2),
                            "realized_pnl_usd": round(float(data.get("realized_pnl_usd", 0.0)), 2),
                        }
                    )
                if state_rows:
                    st.markdown("Per-symbol live status")
                    st.dataframe(pd.DataFrame(state_rows), use_container_width=True, hide_index=True)

            if isinstance(ranked, list) and ranked:
                with st.expander("Show full ranked candidates"):
                    ranked_rows = []
                    for row in ranked:
                        if not isinstance(row, dict):
                            continue
                        ranked_rows.append(
                            {
                                "symbol": row.get("symbol", ""),
                                "score": round(float(row.get("score", 0.0)), 3),
                                "weekly_pnl": round(float(row.get("weekly_pnl", 0.0)), 2),
                                "monthly_pnl": round(float(row.get("monthly_pnl", 0.0)), 2),
                                "weekly_wr": round(float(row.get("weekly_wr", 0.0)), 2),
                                "monthly_wr": round(float(row.get("monthly_wr", 0.0)), 2),
                            }
                        )
                    if ranked_rows:
                        st.dataframe(pd.DataFrame(ranked_rows), use_container_width=True, hide_index=True)

    st.caption(last_event_line(PAPER_LIVE_EVENTS_PATH))


def main():
    st.set_page_config(page_title="Scalping Dashboard", layout="wide")
    st.title("Scalping Local Web Dashboard")

    default_symbol = "BTCUSDT_PERP"
    watchlist = load_watchlist_cache(default_symbol)

    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Chart Symbol", value=watchlist[0] if watchlist else default_symbol).strip().upper()
        today_ny = datetime.now(NY_TZ).strftime("%Y-%m-%d")
        date_str = st.text_input("Date (YYYY-MM-DD)", value=today_ny).strip()
        leverage = clamp_leverage(st.number_input("Leverage", min_value=1.0, max_value=125.0, value=20.0, step=1.0))
        source_mode = st.radio("Monitor Source", ["Manual", "Live Selected"], horizontal=True)

        manual_raw = st.text_area(
            "Manual symbols (one per line)",
            value="\n".join(watchlist),
            height=180,
            disabled=(source_mode != "Manual"),
        )
        manual_symbols = [s.strip().upper() for s in manual_raw.splitlines() if s.strip()]

    tab_chart, tab_monitor, tab_status = st.tabs(["Chart", "Positions Monitor", "Paper Live Status"])

    with tab_chart:
        build_chart(symbol, date_str, leverage)

    with tab_monitor:
        build_monitor(source_mode, manual_symbols, leverage, date_str)

    with tab_status:
        build_paper_live_status(symbol, leverage)


if __name__ == "__main__":
    main()
