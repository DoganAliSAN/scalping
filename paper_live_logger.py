"""
Paper/live trader/logger for the scalping setup.

This script:
1) ranks symbols using the same ranking logic as simulate/rank_today,
2) tracks intraday strategy state with account constraints,
3) can mirror open/close events to Binance USDT-M futures,
4) writes detailed logs (day summary + symbol stats + per-trade details).

Modes:
- One-shot for a specific date:
    python paper_live_logger.py --date 2026-03-27
- Daemon mode (processes yesterday once per NY day):
    python paper_live_logger.py --loop --poll-seconds 300
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from binance import Client

from main import (
    NY_TZ,
    get_klines,
    get_symbol_positions_summary,
    normalize_symbol_and_market,
    run_strategy_positions_only,
    calculate_leveraged_pnl_pct,
)
from kline_disk_cache import (
    get_first_4h_candle_levels_cached,
    get_klines_cached,
)
from backtest_month import DEFAULT_SYMBOLS
from binance_live import (
    close_futures_position,
    get_futures_account_balance_detail,
    get_futures_open_positions,
    has_binance_credentials,
    open_new_futures_position,
)
from simulate import build_cache, _rank_for_day


DEFAULT_SETUP = {
    "weekly_days": 3,
    "monthly_days": 21,
    "min_trades": 1,
    "min_weekly_wr": 30.0,
    "min_monthly_wr": 10.0,
    "min_weekly_nlr": 60.0,
    "min_monthly_nlr": 40.0,
    "top_n": 0,
    "leverage": 20.0,
    "workers": 8,
    "be_stop": True,
    "be_trigger_r": 1.0,
    "target_r": 1.5,
    "max_trades_per_day": 3,
    "daily_loss_limit": -10.0,
}


def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def ny_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=NY_TZ)


def to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def to_iso_ny(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(NY_TZ).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_state(state_path: Path, start_balance: float, position_size: float) -> dict[str, Any]:
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if "balance" in data and "last_processed_date" in data:
                data.setdefault("start_balance", float(start_balance))
                data.setdefault("position_size", float(position_size))
                data.setdefault("equity_peak", float(data.get("balance", start_balance)))
                data.setdefault("max_drawdown_pct", 0.0)
                data.setdefault("processed_days", 0)
                data.setdefault("live_day", None)
                return data
        except Exception:
            pass
    return {
        "balance": float(start_balance),
        "start_balance": float(start_balance),
        "position_size": float(position_size),
        "last_processed_date": None,
        "equity_peak": float(start_balance),
        "max_drawdown_pct": 0.0,
        "processed_days": 0,
        "live_day": None,
    }


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    ensure_dir(state_path.parent)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


@dataclass
class TradeLog:
    symbol: str
    market: str
    trade_index: int
    side: str
    result: str
    leverage: float
    entry_price: float
    stop_price: float
    target_price: float
    liquidation_price: float
    exit_price: float
    entry_time_utc: str
    entry_time_ny: str
    exit_time_utc: str
    exit_time_ny: str
    pnl_pct: float
    pnl_usd: float


@dataclass
class SymbolDayLog:
    symbol: str
    market: str
    ranked_score: float
    ranked_weekly_pnl: float
    ranked_weekly_wr: float
    ranked_weekly_nlr: float
    ranked_monthly_pnl: float
    ranked_monthly_wr: float
    ranked_monthly_nlr: float
    trade_count: int
    wins: int
    breakevens: int
    losses: int
    liquidations: int
    total_pnl_pct: float
    total_pnl_usd: float
    skipped_reason: str


class ExecutionAdapter:
    mode = "base"

    def on_entry(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def on_exit(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class PaperExecutionAdapter(ExecutionAdapter):
    mode = "paper"

    def __init__(self, output_dir: Path):
        self.path = output_dir / "execution_events.jsonl"

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        ensure_dir(self.path.parent)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "event": event,
                        "execution_mode": self.mode,
                        **payload,
                    }
                )
                + "\n"
            )

    def on_entry(self, payload: dict[str, Any]) -> None:
        self._write("entry_signal", payload)

    def on_exit(self, payload: dict[str, Any]) -> None:
        self._write("exit_signal", payload)


class BinanceExecutionAdapter(PaperExecutionAdapter):
    mode = "binance"

    def __init__(self, output_dir: Path, position_size: float, leverage: float, max_open_positions: int):
        super().__init__(output_dir)
        self.position_size = float(position_size)
        self.leverage = float(leverage)
        self.max_open_positions = int(max_open_positions)

    def on_entry(self, payload: dict[str, Any]) -> None:
        position = payload.get("position", {})
        symbol = str(payload.get("symbol", "")).upper()
        try:
            execution_result = open_new_futures_position(
                symbol=symbol,
                side=str(position.get("side", "")),
                margin_usd=self.position_size,
                leverage=self.leverage,
                stop_price=float(position["stop_price"]) if position.get("stop_price") is not None else None,
                take_profit_price=float(position["target_price"]) if position.get("target_price") is not None else None,
                max_open_positions=self.max_open_positions,
                source="strategy",
            )
            self._write("entry_filled", {**payload, "binance": execution_result})
        except Exception as exc:
            self._write("entry_error", {**payload, "error": str(exc)})
            raise

    def on_exit(self, payload: dict[str, Any]) -> None:
        symbol = str(payload.get("symbol", "")).upper()
        try:
            execution_result = close_futures_position(symbol, source="strategy", allow_missing=True)
            self._write("exit_filled", {**payload, "binance": execution_result})
        except Exception as exc:
            self._write("exit_error", {**payload, "error": str(exc)})
            raise


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def fetch_binance_live_snapshot() -> dict[str, Any]:
    try:
        account = get_futures_account_balance_detail()
        positions = get_futures_open_positions()
        return {
            "account": account,
            "open_positions": positions,
            "error": "",
        }
    except Exception as exc:
        return {
            "account": None,
            "open_positions": [],
            "error": str(exc),
        }


def make_trade_key(symbol: str, trade: dict[str, Any]) -> str:
    return "|".join(
        [
            symbol,
            str(trade.get("side", "?")),
            str(int(trade.get("entry_index", -1))),
            f"{float(trade.get('entry', 0.0)):.8f}",
        ]
    )


def get_day_klines(display_symbol: str, sim_date: date) -> tuple[str, str, list[Any]]:
    symbol_for_api, market = normalize_symbol_and_market(display_symbol)
    day_start_ny = ny_midnight(sim_date)
    start_str = day_start_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")

    if sim_date < now_ny().date():
        end_str = (day_start_ny + timedelta(days=1)).astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
        klines = get_klines_cached(
            symbol_for_api,
            market,
            Client.KLINE_INTERVAL_5MINUTE,
            start_str,
            end_str,
            date_midnight_ny=day_start_ny,
        )
    else:
        klines = get_klines(symbol_for_api, market, Client.KLINE_INTERVAL_5MINUTE, start_str)

    return symbol_for_api, market, klines or []


def apply_trade_limit(trades: list[dict[str, Any]], max_trades_per_day: int | None) -> list[dict[str, Any]]:
    if max_trades_per_day is None or max_trades_per_day <= 0:
        return trades

    closed = []
    open_trade = None
    for trade in trades:
        if trade.get("result") == "open":
            open_trade = trade
            continue
        if len(closed) < max_trades_per_day:
            closed.append(trade)

    if open_trade is not None and len(closed) < max_trades_per_day:
        closed.append(open_trade)

    return closed


def trade_to_log(
    display_symbol: str,
    market: str,
    klines: list[Any],
    trade: dict[str, Any],
    trade_index: int,
    leverage: float,
    position_size: float,
) -> TradeLog:
    closed_klines = klines[:-1] if len(klines) > 1 else []
    entry_i = int(trade.get("entry_index", -1))
    exit_i = int(trade.get("exit_index", -1))

    if 0 <= entry_i < len(closed_klines):
        entry_ms = int(closed_klines[entry_i][0])
    else:
        entry_ms = int(closed_klines[0][0]) if closed_klines else 0

    if trade.get("result") == "open":
        if closed_klines:
            exit_ms = int(closed_klines[-1][6])
        elif klines:
            exit_ms = int(klines[-1][6])
        else:
            exit_ms = 0
    elif 0 <= exit_i < len(closed_klines):
        exit_ms = int(closed_klines[exit_i][6])
    else:
        exit_ms = int(closed_klines[-1][6]) if closed_klines else 0

    exit_price = float(trade.get("exit_price", trade.get("entry", 0.0)))
    pnl_pct = calculate_leveraged_pnl_pct(
        float(trade["entry"]),
        exit_price,
        trade["side"],
        float(trade.get("leverage", leverage)),
    )
    pnl_usd = (pnl_pct / 100.0) * position_size

    return TradeLog(
        symbol=display_symbol,
        market=market,
        trade_index=trade_index,
        side=trade["side"],
        result=trade["result"],
        leverage=float(trade.get("leverage", leverage)),
        entry_price=float(trade["entry"]),
        stop_price=float(trade["stop"]),
        target_price=float(trade["target"]),
        liquidation_price=float(trade["liquidation_price"]),
        exit_price=exit_price,
        entry_time_utc=to_iso_utc(entry_ms) if entry_ms else "",
        entry_time_ny=to_iso_ny(entry_ms) if entry_ms else "",
        exit_time_utc=to_iso_utc(exit_ms) if exit_ms else "",
        exit_time_ny=to_iso_ny(exit_ms) if exit_ms else "",
        pnl_pct=float(pnl_pct),
        pnl_usd=float(pnl_usd),
    )


def open_trade_snapshot(
    display_symbol: str,
    market: str,
    klines: list[Any],
    trade: dict[str, Any],
    ranked_row: dict[str, Any],
    leverage: float,
) -> dict[str, Any]:
    mark_price = float(klines[-1][4]) if klines else float(trade["entry"])
    unrealized_pct = calculate_leveraged_pnl_pct(
        float(trade["entry"]),
        mark_price,
        trade["side"],
        float(trade.get("leverage", leverage)),
    )
    return {
        "symbol": display_symbol,
        "market": market,
        "trade_key": make_trade_key(display_symbol, trade),
        "side": trade["side"],
        "entry_price": float(trade["entry"]),
        "stop_price": float(trade["stop"]),
        "target_price": float(trade["target"]),
        "liquidation_price": float(trade["liquidation_price"]),
        "mark_price": mark_price,
        "unrealized_pnl_pct": unrealized_pct,
        "entry_index": int(trade.get("entry_index", -1)),
        "ranked_score": float(ranked_row["score"]),
        "weekly_pnl": float(ranked_row["weekly_pnl"]),
        "monthly_pnl": float(ranked_row["monthly_pnl"]),
    }


def initialize_live_day(args, state: dict[str, Any], sim_date: date) -> dict[str, Any]:
    symbols = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_SYMBOLS
    max_tpd = args.max_trades_per_day if args.max_trades_per_day > 0 else None
    today_midnight_ny = now_ny().replace(hour=0, minute=0, second=0, microsecond=0)
    fetch_start = sim_date - timedelta(days=args.monthly_days)
    fetch_end = sim_date - timedelta(days=1)
    if fetch_end < fetch_start:
        fetch_end = fetch_start
    date_range = [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]

    cache = build_cache(
        symbols,
        date_range,
        today_midnight_ny,
        args.leverage,
        args.workers,
        args.be_stop,
        args.be_trigger_r,
        args.target_r,
        max_tpd,
    )
    ranked = _rank_for_day(
        cache,
        symbols,
        sim_date,
        args.weekly_days,
        args.monthly_days,
        args.min_trades,
        args.min_weekly_wr,
        args.min_monthly_wr,
        args.min_weekly_nlr,
        args.min_monthly_nlr,
        args.top_n,
    )

    balance_before = float(state["balance"])
    slots = min(int(balance_before // float(args.position_size)), int(args.max_open_positions))
    selected = ranked[:slots] if slots > 0 else []

    return {
        "date": sim_date.isoformat(),
        "balance_before": balance_before,
        "slots": slots,
        "ranked": ranked,
        "selected": selected,
        "symbol_states": {},
        "daily_loss_limit_reached": False,
        "daily_loss_limit_logged": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def process_live_day(
    args,
    state: dict[str, Any],
    sim_date: date,
    output_dir: Path,
    execution: ExecutionAdapter,
) -> dict[str, Any]:
    live_day = state.get("live_day")
    if not isinstance(live_day, dict) or live_day.get("date") != sim_date.isoformat():
        live_day = initialize_live_day(args, state, sim_date)
        state["live_day"] = live_day

    symbol_states: dict[str, Any] = live_day.setdefault("symbol_states", {})
    selected = live_day.get("selected", [])
    run_id = f"{sim_date.isoformat()}_live_{int(time.time())}"
    max_tpd = args.max_trades_per_day if args.max_trades_per_day > 0 else None

    realized_pnl_pct = 0.0
    realized_pnl_usd = 0.0
    open_positions: list[dict[str, Any]] = []
    pending_symbols: list[dict[str, Any]] = []

    live_events_path = output_dir / "events.jsonl"

    for ranked_row in selected:
        symbol = ranked_row["symbol"]
        symbol_state = symbol_states.setdefault(
            symbol,
            {
                "closed_trade_keys": [],
                "closed_trade_logs": [],
                "open_trade_key": None,
                "open_trade": None,
                "realized_pnl_pct": 0.0,
                "realized_pnl_usd": 0.0,
                "status": "watching",
                "skipped_reason": "",
            },
        )

        realized_pnl_pct += float(symbol_state.get("realized_pnl_pct", 0.0))
        realized_pnl_usd += float(symbol_state.get("realized_pnl_usd", 0.0))

        if (
            args.daily_loss_limit < 0
            and realized_pnl_pct <= args.daily_loss_limit
            and not symbol_state.get("closed_trade_keys")
            and not symbol_state.get("open_trade_key")
        ):
            symbol_state["status"] = "blocked"
            symbol_state["skipped_reason"] = "daily_loss_limit_reached"
            live_day["daily_loss_limit_reached"] = True
            pending_symbols.append(
                {
                    "symbol": symbol,
                    "market": ranked_row.get("market", "?"),
                    "status": "blocked",
                    "skipped_reason": "daily_loss_limit_reached",
                    "ranked_score": float(ranked_row["score"]),
                }
            )
            continue

        summary = get_symbol_positions_summary(
            symbol,
            ny_midnight(sim_date),
            leverage=args.leverage,
            enable_break_even=args.be_stop,
            break_even_trigger_r=args.be_trigger_r,
            target_r=args.target_r,
        )

        trades = apply_trade_limit(list(summary.get("trades", [])), max_tpd)
        closed = [t for t in trades if t.get("result") != "open"]
        open_trade = next((t for t in reversed(trades) if t.get("result") == "open"), None)
        _, market, klines = get_day_klines(symbol, sim_date)

        for idx, trade in enumerate(closed, start=1):
            trade_key = make_trade_key(symbol, trade)
            if trade_key in symbol_state["closed_trade_keys"]:
                continue

            trade_log = asdict(trade_to_log(symbol, market, klines, trade, idx, args.leverage, args.position_size))
            symbol_state["closed_trade_keys"].append(trade_key)
            symbol_state["closed_trade_logs"].append(trade_log)
            symbol_state["realized_pnl_pct"] = float(symbol_state.get("realized_pnl_pct", 0.0)) + float(trade_log["pnl_pct"])
            symbol_state["realized_pnl_usd"] = float(symbol_state.get("realized_pnl_usd", 0.0)) + float(trade_log["pnl_usd"])

            if symbol_state.get("open_trade_key") == trade_key:
                symbol_state["open_trade_key"] = None
                symbol_state["open_trade"] = None

            exit_payload = {
                "run_id": run_id,
                "date": sim_date.isoformat(),
                "symbol": symbol,
                "market": market,
                "trade_key": trade_key,
                "trade": trade_log,
            }
            execution.on_exit(exit_payload)
            append_jsonl(
                live_events_path,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "trade_closed_live",
                    **exit_payload,
                },
            )

        if open_trade is not None:
            open_snapshot = open_trade_snapshot(symbol, market, klines, open_trade, ranked_row, args.leverage)
            open_key = open_snapshot["trade_key"]
            if symbol_state.get("open_trade_key") != open_key and open_key not in symbol_state["closed_trade_keys"]:
                entry_payload = {
                    "run_id": run_id,
                    "date": sim_date.isoformat(),
                    "symbol": symbol,
                    "market": market,
                    "trade_key": open_key,
                    "position": open_snapshot,
                }
                execution.on_entry(entry_payload)
                append_jsonl(
                    live_events_path,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "trade_open_live",
                        **entry_payload,
                    },
                )

            symbol_state["open_trade_key"] = open_key
            symbol_state["open_trade"] = open_snapshot
            symbol_state["status"] = "open"
            symbol_state["skipped_reason"] = ""
            open_positions.append(open_snapshot)
        else:
            symbol_state["open_trade_key"] = None
            symbol_state["open_trade"] = None
            if symbol_state.get("closed_trade_keys"):
                symbol_state["status"] = "closed"
            else:
                symbol_state["status"] = "watching"
            symbol_state["skipped_reason"] = symbol_state.get("skipped_reason", "")

        pending_symbols.append(
            {
                "symbol": symbol,
                "market": market,
                "status": symbol_state["status"],
                "skipped_reason": symbol_state.get("skipped_reason", ""),
                "ranked_score": float(ranked_row["score"]),
                "closed_trades": len(symbol_state.get("closed_trade_keys", [])),
                "realized_pnl_pct": float(symbol_state.get("realized_pnl_pct", 0.0)),
                "has_open_position": bool(symbol_state.get("open_trade_key")),
            }
        )

    realized_pnl_pct = sum(float(v.get("realized_pnl_pct", 0.0)) for v in symbol_states.values())
    realized_pnl_usd = sum(float(v.get("realized_pnl_usd", 0.0)) for v in symbol_states.values())
    live_day["daily_loss_limit_reached"] = args.daily_loss_limit < 0 and realized_pnl_pct <= args.daily_loss_limit
    live_day["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    if live_day["daily_loss_limit_reached"] and not live_day.get("daily_loss_limit_logged"):
        append_jsonl(
            live_events_path,
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": "daily_loss_limit_reached_live",
                "date": sim_date.isoformat(),
                "realized_pnl_pct": realized_pnl_pct,
                "limit_pct": args.daily_loss_limit,
            },
        )
        live_day["daily_loss_limit_logged"] = True

    live_snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": sim_date.isoformat(),
        "execution_mode": execution.mode,
        "balance_before": float(live_day["balance_before"]),
        "estimated_balance": float(live_day["balance_before"]) + realized_pnl_usd,
        "realized_pnl_usd": realized_pnl_usd,
        "realized_pnl_pct": realized_pnl_pct,
        "daily_loss_limit": args.daily_loss_limit,
        "daily_loss_limit_reached": bool(live_day["daily_loss_limit_reached"]),
        "slots": int(live_day["slots"]),
        "max_open_positions": int(args.max_open_positions),
        "selected_symbols": [row["symbol"] for row in selected],
        "open_positions": open_positions,
        "pending_symbols": pending_symbols,
    }

    if execution.mode == "binance":
        exchange_snapshot = fetch_binance_live_snapshot()
        live_snapshot["binance_account"] = exchange_snapshot["account"]
        live_snapshot["binance_open_positions"] = exchange_snapshot["open_positions"]
        live_snapshot["binance_error"] = exchange_snapshot["error"]

    append_jsonl(
        live_events_path,
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "live_snapshot",
            "date": sim_date.isoformat(),
            "estimated_balance": live_snapshot["estimated_balance"],
            "realized_pnl_usd": realized_pnl_usd,
            "realized_pnl_pct": realized_pnl_pct,
            "open_positions": len(open_positions),
            "tracked_symbols": len(selected),
        },
    )

    snapshot_path = output_dir / "current_positions.json"
    ensure_dir(snapshot_path.parent)
    snapshot_path.write_text(json.dumps(live_snapshot, indent=2), encoding="utf-8")

    live_day["last_snapshot"] = live_snapshot
    state["live_day"] = live_day
    return live_snapshot


def compute_symbol_trades_for_day(
    display_symbol: str,
    sim_date: date,
    leverage: float,
    enable_break_even: bool,
    break_even_trigger_r: float,
    target_r: float,
    max_trades_per_day: int | None,
    position_size: float,
) -> tuple[dict[str, Any] | None, list[TradeLog]]:
    """
    Compute closed trades for one symbol/day and return:
    - summary dict compatible with backtest-style keys
    - detailed trade logs list
    """
    symbol_for_api, market = normalize_symbol_and_market(display_symbol)
    day_start_ny = ny_midnight(sim_date)

    low_4h, high_4h = get_first_4h_candle_levels_cached(symbol_for_api, market, day_start_ny)
    if low_4h is None or high_4h is None:
        return None, []

    start_str = day_start_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
    end_str = (day_start_ny + timedelta(days=1)).astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")
    klines = get_klines_cached(
        symbol_for_api,
        market,
        Client.KLINE_INTERVAL_5MINUTE,
        start_str,
        end_str,
        date_midnight_ny=day_start_ny,
    )
    if not klines:
        return None, []

    all_trades = run_strategy_positions_only(
        klines,
        low_4h,
        high_4h,
        day_start_ny,
        leverage=leverage,
        enable_break_even=enable_break_even,
        break_even_trigger_r=break_even_trigger_r,
        target_r=target_r,
    )

    closed = [t for t in all_trades if t.get("result") != "open"]
    if max_trades_per_day is not None and max_trades_per_day > 0:
        closed = closed[:max_trades_per_day]
    if not closed:
        return None, []

    closed_klines = klines[:-1] if len(klines) > 1 else []

    wins = 0
    bes = 0
    losses = 0
    liqs = 0
    total_pnl_pct = 0.0
    trade_logs: list[TradeLog] = []

    for idx, t in enumerate(closed, start=1):
        pnl_pct = calculate_leveraged_pnl_pct(
            t["entry"],
            t["exit_price"],
            t["side"],
            t.get("leverage", leverage),
        )
        pnl_usd = (pnl_pct / 100.0) * position_size
        total_pnl_pct += pnl_pct

        if t["result"] == "target":
            wins += 1
        elif t["result"] == "breakeven":
            bes += 1
        else:
            losses += 1
            if t["result"] == "liquidation":
                liqs += 1

        entry_i = int(t["entry_index"])
        exit_i = int(t["exit_index"])
        if 0 <= entry_i < len(closed_klines):
            entry_ms = int(closed_klines[entry_i][0])
        else:
            entry_ms = int(closed_klines[0][0]) if closed_klines else 0

        if 0 <= exit_i < len(closed_klines):
            exit_ms = int(closed_klines[exit_i][6])
        else:
            exit_ms = int(closed_klines[-1][6]) if closed_klines else 0

        trade_logs.append(
            TradeLog(
                symbol=display_symbol,
                market=market,
                trade_index=idx,
                side=t["side"],
                result=t["result"],
                leverage=float(t.get("leverage", leverage)),
                entry_price=float(t["entry"]),
                stop_price=float(t["stop"]),
                target_price=float(t["target"]),
                liquidation_price=float(t["liquidation_price"]),
                exit_price=float(t["exit_price"]),
                entry_time_utc=to_iso_utc(entry_ms) if entry_ms else "",
                entry_time_ny=to_iso_ny(entry_ms) if entry_ms else "",
                exit_time_utc=to_iso_utc(exit_ms) if exit_ms else "",
                exit_time_ny=to_iso_ny(exit_ms) if exit_ms else "",
                pnl_pct=float(pnl_pct),
                pnl_usd=float(pnl_usd),
            )
        )

    summary = {
        "symbol": display_symbol,
        "market": market,
        "total_pnl": total_pnl_pct,
        "trades": len(closed),
        "wins": wins,
        "breakevens": bes,
        "losses": losses,
        "liquidations": liqs,
    }
    return summary, trade_logs


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def build_date_range_for_ranking(sim_date: date, monthly_days: int) -> list[date]:
    fetch_start = sim_date - timedelta(days=monthly_days)
    fetch_end = sim_date
    return [fetch_start + timedelta(days=i) for i in range((fetch_end - fetch_start).days + 1)]


def process_day(
    args,
    state: dict[str, Any],
    sim_date: date,
    output_dir: Path,
    *,
    commit_state: bool = True,
) -> dict[str, Any]:
    today_midnight_ny = now_ny().replace(hour=0, minute=0, second=0, microsecond=0)

    symbols = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_SYMBOLS
    max_tpd = args.max_trades_per_day if args.max_trades_per_day > 0 else None

    live_day = state.get("live_day")
    use_live_day_reconcile = (
        commit_state
        and isinstance(live_day, dict)
        and live_day.get("date") == sim_date.isoformat()
        and isinstance(live_day.get("symbol_states"), dict)
        and bool(live_day.get("symbol_states"))
    )

    if use_live_day_reconcile:
        ranked = live_day.get("ranked", []) if isinstance(live_day.get("ranked"), list) else []
        selected = live_day.get("selected", []) if isinstance(live_day.get("selected"), list) else []
    else:
        date_range = build_date_range_for_ranking(sim_date, args.monthly_days)
        cache = build_cache(
            symbols,
            date_range,
            today_midnight_ny,
            args.leverage,
            args.workers,
            args.be_stop,
            args.be_trigger_r,
            args.target_r,
            max_tpd,
        )

        ranked = _rank_for_day(
            cache,
            symbols,
            sim_date,
            args.weekly_days,
            args.monthly_days,
            args.min_trades,
            args.min_weekly_wr,
            args.min_monthly_wr,
            args.min_weekly_nlr,
            args.min_monthly_nlr,
            args.top_n,
        )

    run_kind = "final" if commit_state else "live"
    run_id = f"{sim_date.isoformat()}_{run_kind}_{int(time.time())}"
    balance_before = float(state["balance"])
    slots = int(balance_before // float(args.position_size))
    if not use_live_day_reconcile:
        selected = ranked[: min(slots, int(args.max_open_positions))] if slots > 0 else []

    day_pnl_pct = 0.0
    day_pnl_usd = 0.0
    symbol_rows: list[SymbolDayLog] = []
    trade_rows: list[dict[str, Any]] = []

    day_trades = day_wins = day_bes = day_losses = day_liqs = 0

    # Reconcile final day accounting from live-day state when available.
    # This keeps finalized balance consistent with intraday realized PnL.

    if use_live_day_reconcile:
        if isinstance(live_day.get("ranked"), list):
            ranked = live_day.get("ranked", ranked)
        if isinstance(live_day.get("selected"), list):
            selected = live_day.get("selected", selected)

        selected_map: dict[str, dict[str, Any]] = {}
        for row in selected:
            if isinstance(row, dict) and row.get("symbol"):
                selected_map[str(row["symbol"])] = row

        symbol_states: dict[str, Any] = live_day.get("symbol_states", {})

        for symbol, sym_state_raw in symbol_states.items():
            if not isinstance(sym_state_raw, dict):
                continue

            sym_state = sym_state_raw
            srow = selected_map.get(symbol, {})
            logs = sym_state.get("closed_trade_logs", [])
            if not isinstance(logs, list):
                logs = []

            wins = 0
            bes = 0
            losses = 0
            liqs = 0
            for lg in logs:
                if not isinstance(lg, dict):
                    continue
                result = str(lg.get("result", "")).lower()
                if result == "target":
                    wins += 1
                elif result == "breakeven":
                    bes += 1
                elif result:
                    losses += 1
                    if result == "liquidation":
                        liqs += 1

            symbol_pnl_pct = float(sym_state.get("realized_pnl_pct", 0.0))
            symbol_pnl_usd = float(sym_state.get("realized_pnl_usd", 0.0))

            skipped_reason = str(sym_state.get("skipped_reason", "") or "")
            if not logs and not skipped_reason and str(sym_state.get("status", "")) == "watching":
                skipped_reason = "no_closed_trades"

            market = srow.get("market") if isinstance(srow, dict) else None
            if not market and logs and isinstance(logs[0], dict):
                market = logs[0].get("market")

            symbol_rows.append(
                SymbolDayLog(
                    symbol=symbol,
                    market=str(market or "?"),
                    ranked_score=float(srow.get("score", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_weekly_pnl=float(srow.get("weekly_pnl", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_weekly_wr=float(srow.get("weekly_wr", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_weekly_nlr=float(srow.get("weekly_nlr", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_monthly_pnl=float(srow.get("monthly_pnl", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_monthly_wr=float(srow.get("monthly_wr", 0.0)) if isinstance(srow, dict) else 0.0,
                    ranked_monthly_nlr=float(srow.get("monthly_nlr", 0.0)) if isinstance(srow, dict) else 0.0,
                    trade_count=len(logs),
                    wins=wins,
                    breakevens=bes,
                    losses=losses,
                    liquidations=liqs,
                    total_pnl_pct=symbol_pnl_pct,
                    total_pnl_usd=symbol_pnl_usd,
                    skipped_reason=skipped_reason,
                )
            )

            day_pnl_pct += symbol_pnl_pct
            day_pnl_usd += symbol_pnl_usd
            day_trades += len(logs)
            day_wins += wins
            day_bes += bes
            day_losses += losses
            day_liqs += liqs

            for lg in logs:
                if not isinstance(lg, dict):
                    continue
                row = dict(lg)
                row.update(
                    {
                        "run_id": run_id,
                        "date": sim_date.isoformat(),
                        "position_size_usd": float(args.position_size),
                    }
                )
                trade_rows.append(row)

    if not use_live_day_reconcile:
        for r in selected:
            summary, trade_logs = compute_symbol_trades_for_day(
                r["symbol"],
                sim_date,
                args.leverage,
                args.be_stop,
                args.be_trigger_r,
                args.target_r,
                max_tpd,
                args.position_size,
            )

            if summary is None:
                symbol_rows.append(
                    SymbolDayLog(
                        symbol=r["symbol"],
                        market=r.get("market", "?"),
                        ranked_score=float(r["score"]),
                        ranked_weekly_pnl=float(r["weekly_pnl"]),
                        ranked_weekly_wr=float(r["weekly_wr"]),
                        ranked_weekly_nlr=float(r["weekly_nlr"]),
                        ranked_monthly_pnl=float(r["monthly_pnl"]),
                        ranked_monthly_wr=float(r["monthly_wr"]),
                        ranked_monthly_nlr=float(r["monthly_nlr"]),
                        trade_count=0,
                        wins=0,
                        breakevens=0,
                        losses=0,
                        liquidations=0,
                        total_pnl_pct=0.0,
                        total_pnl_usd=0.0,
                        skipped_reason="no_closed_trades",
                    )
                )
                continue

            symbol_pnl_pct = float(summary["total_pnl"])
            symbol_pnl_usd = (symbol_pnl_pct / 100.0) * float(args.position_size)

            if args.daily_loss_limit < 0 and (day_pnl_pct + symbol_pnl_pct) <= args.daily_loss_limit:
                symbol_rows.append(
                    SymbolDayLog(
                        symbol=r["symbol"],
                        market=summary["market"],
                        ranked_score=float(r["score"]),
                        ranked_weekly_pnl=float(r["weekly_pnl"]),
                        ranked_weekly_wr=float(r["weekly_wr"]),
                        ranked_weekly_nlr=float(r["weekly_nlr"]),
                        ranked_monthly_pnl=float(r["monthly_pnl"]),
                        ranked_monthly_wr=float(r["monthly_wr"]),
                        ranked_monthly_nlr=float(r["monthly_nlr"]),
                        trade_count=0,
                        wins=0,
                        breakevens=0,
                        losses=0,
                        liquidations=0,
                        total_pnl_pct=0.0,
                        total_pnl_usd=0.0,
                        skipped_reason="daily_loss_limit_reached",
                    )
                )
                break

            day_pnl_pct += symbol_pnl_pct
            day_pnl_usd += symbol_pnl_usd
            day_trades += int(summary["trades"])
            day_wins += int(summary["wins"])
            day_bes += int(summary["breakevens"])
            day_losses += int(summary["losses"])
            day_liqs += int(summary["liquidations"])

            symbol_rows.append(
                SymbolDayLog(
                    symbol=r["symbol"],
                    market=summary["market"],
                    ranked_score=float(r["score"]),
                    ranked_weekly_pnl=float(r["weekly_pnl"]),
                    ranked_weekly_wr=float(r["weekly_wr"]),
                    ranked_weekly_nlr=float(r["weekly_nlr"]),
                    ranked_monthly_pnl=float(r["monthly_pnl"]),
                    ranked_monthly_wr=float(r["monthly_wr"]),
                    ranked_monthly_nlr=float(r["monthly_nlr"]),
                    trade_count=int(summary["trades"]),
                    wins=int(summary["wins"]),
                    breakevens=int(summary["breakevens"]),
                    losses=int(summary["losses"]),
                    liquidations=int(summary["liquidations"]),
                    total_pnl_pct=symbol_pnl_pct,
                    total_pnl_usd=symbol_pnl_usd,
                    skipped_reason="",
                )
            )

            for t in trade_logs:
                row = asdict(t)
                row.update(
                    {
                        "run_id": run_id,
                        "date": sim_date.isoformat(),
                        "position_size_usd": float(args.position_size),
                    }
                )
                trade_rows.append(row)

    balance_after = balance_before + day_pnl_usd
    if balance_after < 0:
        balance_after = 0.0

    equity_peak = max(float(state.get("equity_peak", balance_before)), balance_after)
    drawdown = ((equity_peak - balance_after) / equity_peak * 100.0) if equity_peak > 0 else 0.0

    if commit_state:
        state["balance"] = balance_after
        state["processed_days"] = int(state.get("processed_days", 0)) + 1
        state["last_processed_date"] = sim_date.isoformat()
        state["equity_peak"] = equity_peak
        if drawdown > float(state.get("max_drawdown_pct", 0.0)):
            state["max_drawdown_pct"] = drawdown

    day_summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": sim_date.isoformat(),
        "mode": args.execution_mode,
        "run_kind": run_kind,
        "is_final": bool(commit_state),
        "settings": {
            "weekly_days": args.weekly_days,
            "monthly_days": args.monthly_days,
            "min_trades": args.min_trades,
            "min_weekly_wr": args.min_weekly_wr,
            "min_monthly_wr": args.min_monthly_wr,
            "min_weekly_nlr": args.min_weekly_nlr,
            "min_monthly_nlr": args.min_monthly_nlr,
            "top_n": args.top_n,
            "leverage": args.leverage,
            "workers": args.workers,
            "be_stop": args.be_stop,
            "be_trigger_r": args.be_trigger_r,
            "target_r": args.target_r,
            "max_trades_per_day": args.max_trades_per_day,
            "daily_loss_limit": args.daily_loss_limit,
            "start_balance": args.start_balance,
            "position_size": args.position_size,
            "max_open_positions": args.max_open_positions,
            "execution_mode": args.execution_mode,
        },
        "account": {
            "balance_before": balance_before,
            "balance_after": balance_after,
            "day_pnl_usd": day_pnl_usd,
            "day_return_on_balance_pct": ((day_pnl_usd / balance_before) * 100.0) if balance_before > 0 else 0.0,
            "equity_peak": float(state.get("equity_peak", equity_peak)) if commit_state else equity_peak,
            "max_drawdown_pct": float(state.get("max_drawdown_pct", 0.0)) if commit_state else max(float(state.get("max_drawdown_pct", 0.0)), drawdown),
            "slots": slots,
        },
        "ranking": {
            "ranked_count": len(ranked),
            "selected_count": len(selected),
            "selected_symbols": [r["symbol"] for r in selected],
        },
        "day_stats": {
            "day_pnl_pct": day_pnl_pct,
            "trades": day_trades,
            "wins": day_wins,
            "breakevens": day_bes,
            "losses": day_losses,
            "liquidations": day_liqs,
            "win_rate_pct": (day_wins / day_trades * 100.0) if day_trades > 0 else 0.0,
            "non_loss_rate_pct": ((day_wins + day_bes) / day_trades * 100.0) if day_trades > 0 else 0.0,
        },
        "ranked_symbols": ranked,
        "symbol_results": [asdict(s) for s in symbol_rows],
        "trade_results": trade_rows,
    }

    day_path = output_dir / "days" / f"{sim_date.isoformat()}.json"
    ensure_dir(day_path.parent)
    day_path.write_text(json.dumps(day_summary, indent=2), encoding="utf-8")

    if commit_state:
        append_csv(
            output_dir / "summary.csv",
            [
                {
                    "run_id": run_id,
                    "date": sim_date.isoformat(),
                    "balance_before": f"{balance_before:.2f}",
                    "balance_after": f"{balance_after:.2f}",
                    "day_pnl_usd": f"{day_pnl_usd:.2f}",
                    "day_pnl_pct": f"{day_pnl_pct:.2f}",
                    "ranked_count": len(ranked),
                    "selected_count": len(selected),
                    "trades": day_trades,
                    "wins": day_wins,
                    "breakevens": day_bes,
                    "losses": day_losses,
                    "liquidations": day_liqs,
                    "max_drawdown_pct": f"{state['max_drawdown_pct']:.4f}",
                }
            ],
        )

        append_csv(output_dir / "symbol_results.csv", [asdict(s) | {"run_id": run_id, "date": sim_date.isoformat()} for s in symbol_rows])
        append_csv(output_dir / "trades.csv", trade_rows)

    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "day_processed" if commit_state else "live_snapshot",
            "run_id": run_id,
            "date": sim_date.isoformat(),
            "balance_after": balance_after,
            "day_pnl_usd": day_pnl_usd,
            "trades": day_trades,
        }) + "\n")

    return day_summary


def parse_args():
    p = argparse.ArgumentParser(description="Paper/live trader/logger with optional Binance execution")
    p.add_argument("--date", default=None, help="Process specific date YYYY-MM-DD once")
    p.add_argument("--loop", action="store_true", help="Run as daemon, settle completed days, and track the current NY day live")
    p.add_argument("--poll-seconds", type=int, default=300, help="Loop sleep interval in seconds (default: 300)")
    p.add_argument("--output-dir", default="paper_live_logs", help="Output directory for logs")
    p.add_argument("--state-file", default=None, help="Optional explicit state JSON path (default: <output-dir>/state.json)")
    p.add_argument("--execution-mode", default="binance", choices=["paper", "binance"], help="Execution adapter mode")

    p.add_argument("--start-balance", type=float, default=100.0)
    p.add_argument("--position-size", type=float, default=33.0)
    p.add_argument("--max-open-positions", type=int, default=3)

    p.add_argument("--weekly-days", type=int, default=DEFAULT_SETUP["weekly_days"])
    p.add_argument("--monthly-days", type=int, default=DEFAULT_SETUP["monthly_days"])
    p.add_argument("--min-trades", type=int, default=DEFAULT_SETUP["min_trades"])
    p.add_argument("--min-weekly-wr", type=float, default=DEFAULT_SETUP["min_weekly_wr"])
    p.add_argument("--min-monthly-wr", type=float, default=DEFAULT_SETUP["min_monthly_wr"])
    p.add_argument("--min-weekly-nlr", type=float, default=DEFAULT_SETUP["min_weekly_nlr"])
    p.add_argument("--min-monthly-nlr", type=float, default=DEFAULT_SETUP["min_monthly_nlr"])
    p.add_argument("--top-n", type=int, default=DEFAULT_SETUP["top_n"])

    p.add_argument("--leverage", type=float, default=DEFAULT_SETUP["leverage"])
    p.add_argument("--workers", type=int, default=DEFAULT_SETUP["workers"])
    p.add_argument("--be-stop", action="store_true", default=DEFAULT_SETUP["be_stop"])
    p.add_argument("--no-be-stop", action="store_false", dest="be_stop")
    p.add_argument("--be-trigger-r", type=float, default=DEFAULT_SETUP["be_trigger_r"])
    p.add_argument("--target-r", type=float, default=DEFAULT_SETUP["target_r"])
    p.add_argument("--max-trades-per-day", type=int, default=DEFAULT_SETUP["max_trades_per_day"])
    p.add_argument("--daily-loss-limit", type=float, default=DEFAULT_SETUP["daily_loss_limit"])

    p.add_argument("--symbols", nargs="+", default=None)
    return p.parse_args()


def parse_date_or_none(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def print_day_brief(day_summary: dict[str, Any]) -> None:
    d = day_summary["date"]
    acct = day_summary["account"]
    stats = day_summary["day_stats"]
    run_kind = day_summary.get("run_kind", "final")
    print("=" * 72)
    print(f"PAPER DAY {d} [{run_kind.upper()}]")
    print("=" * 72)
    print(f"Balance: {acct['balance_before']:.2f} -> {acct['balance_after']:.2f} | PnL {acct['day_pnl_usd']:+.2f} USD")
    print(f"Day PnL: {stats['day_pnl_pct']:+.2f}% | Trades {stats['trades']} | W/BE/L {stats['wins']}/{stats['breakevens']}/{stats['losses']}")
    print(f"Ranked: {day_summary['ranking']['ranked_count']} | Selected: {day_summary['ranking']['selected_count']}")
    print()


def print_live_brief(live_summary: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"PAPER DAY {live_summary['date']} [LIVE]")
    print("=" * 72)
    print(
        f"Est Balance: {live_summary['estimated_balance']:.2f} | "
        f"Realized PnL {live_summary['realized_pnl_usd']:+.2f} USD ({live_summary['realized_pnl_pct']:+.2f}%)"
    )
    print(
        f"Open Positions: {len(live_summary['open_positions'])} | "
        f"Tracked Symbols: {len(live_summary['selected_symbols'])} | "
        f"Loss Limit Hit: {'YES' if live_summary['daily_loss_limit_reached'] else 'NO'}"
    )
    print()


def build_execution_adapter(args, output_dir: Path) -> ExecutionAdapter:
    if args.execution_mode == "paper":
        return PaperExecutionAdapter(output_dir)
    if not has_binance_credentials():
        raise SystemExit("Binance execution mode requires BINANCE_API_KEY and BINANCE_API_SECRET.")
    return BinanceExecutionAdapter(
        output_dir,
        position_size=args.position_size,
        leverage=args.leverage,
        max_open_positions=args.max_open_positions,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    state_path = Path(args.state_file) if args.state_file else output_dir / "state.json"

    state = load_state(state_path, args.start_balance, args.position_size)
    execution: ExecutionAdapter = build_execution_adapter(args, output_dir)

    one_date = parse_date_or_none(args.date)
    if one_date is not None:
        summary = process_day(args, state, one_date, output_dir)
        save_state(state_path, state)
        print_day_brief(summary)
        return

    if not args.loop:
        raise SystemExit("Provide --date YYYY-MM-DD for one-shot or use --loop for daemon mode.")

    print("Starting paper live logger loop...")
    print(f"Output dir: {output_dir}")
    print(f"State file: {state_path}")
    print(f"Execution mode: {execution.mode}")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            ny_today = now_ny().date()
            target = ny_today - timedelta(days=1)
            last_processed = state.get("last_processed_date")
            if last_processed != target.isoformat():
                summary = process_day(args, state, target, output_dir, commit_state=True)
                if isinstance(state.get("live_day"), dict) and state["live_day"].get("date") == target.isoformat():
                    state["live_day"] = None
                save_state(state_path, state)
                print_day_brief(summary)

            live_summary = process_live_day(args, state, ny_today, output_dir, execution)
            save_state(state_path, state)
            print_live_brief(live_summary)

            time.sleep(max(30, args.poll_seconds))
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            with (output_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "error",
                    "error": str(e),
                }) + "\n")
            print(f"[WARN] loop error: {e}")
            time.sleep(max(30, args.poll_seconds))


if __name__ == "__main__":
    main()
