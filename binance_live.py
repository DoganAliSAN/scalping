from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
from typing import Any

from binance import Client
from binance.exceptions import BinanceAPIException


DEFAULT_POSITION_SIZE_USD = 33.0
DEFAULT_MAX_OPEN_POSITIONS = 3
DEFAULT_FUTURES_LEVERAGE = 20.0
DEFAULT_MARGIN_ASSET = "USDT"


class BinanceLiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class BinanceLiveConfig:
    api_key: str | None
    api_secret: str | None
    testnet: bool
    position_size_usd: float
    max_open_positions: int
    leverage: float
    margin_asset: str


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def load_binance_live_config() -> BinanceLiveConfig:
    return BinanceLiveConfig(
        api_key=os.getenv("BINANCE_API_KEY") or None,
        api_secret=os.getenv("BINANCE_API_SECRET") or None,
        testnet=_env_flag("BINANCE_TESTNET", default=False),
        position_size_usd=_env_float("BINANCE_POSITION_SIZE_USD", DEFAULT_POSITION_SIZE_USD),
        max_open_positions=max(1, _env_int("BINANCE_MAX_OPEN_POSITIONS", DEFAULT_MAX_OPEN_POSITIONS)),
        leverage=max(1.0, _env_float("BINANCE_FUTURES_LEVERAGE", DEFAULT_FUTURES_LEVERAGE)),
        margin_asset=(os.getenv("BINANCE_MARGIN_ASSET") or DEFAULT_MARGIN_ASSET).strip().upper(),
    )


def has_binance_credentials(config: BinanceLiveConfig | None = None) -> bool:
    cfg = config or load_binance_live_config()
    return bool(cfg.api_key and cfg.api_secret)


@lru_cache(maxsize=2)
def _build_client(api_key: str | None, api_secret: str | None, testnet: bool) -> Client:
    return Client(api_key, api_secret, testnet=testnet)


def get_binance_client(*, require_credentials: bool = False) -> Client:
    cfg = load_binance_live_config()
    if require_credentials and not has_binance_credentials(cfg):
        raise BinanceLiveError(
            "Missing Binance credentials. Set BINANCE_API_KEY and BINANCE_API_SECRET in the environment."
        )
    return _build_client(cfg.api_key, cfg.api_secret, cfg.testnet)


def get_live_runtime_summary() -> dict[str, Any]:
    cfg = load_binance_live_config()
    return {
        "credentials_loaded": has_binance_credentials(cfg),
        "testnet": cfg.testnet,
        "position_size_usd": cfg.position_size_usd,
        "max_open_positions": cfg.max_open_positions,
        "leverage": cfg.leverage,
        "margin_asset": cfg.margin_asset,
    }


def _round_down_to_step(value: float, step: str) -> Decimal:
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(str(step))
    if decimal_step <= 0:
        return decimal_value
    return (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN) * decimal_step


def _decimal_to_string(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _get_symbol_info(symbol: str) -> dict[str, Any]:
    client = get_binance_client()
    exchange_info = client.futures_exchange_info()
    for item in exchange_info.get("symbols", []):
        if item.get("symbol") == symbol:
            return item
    raise BinanceLiveError(f"USDT-M futures symbol not found on Binance: {symbol}")


def _get_filter(symbol_info: dict[str, Any], filter_type: str) -> dict[str, Any]:
    for item in symbol_info.get("filters", []):
        if item.get("filterType") == filter_type:
            return item
    return {}


def format_quantity(symbol: str, quantity: float) -> str:
    symbol_info = _get_symbol_info(symbol)
    lot_filter = _get_filter(symbol_info, "MARKET_LOT_SIZE") or _get_filter(symbol_info, "LOT_SIZE")
    step_size = str(lot_filter.get("stepSize", "1"))
    min_qty = Decimal(str(lot_filter.get("minQty", "0")))
    rounded = _round_down_to_step(quantity, step_size)
    if rounded < min_qty:
        raise BinanceLiveError(
            f"Calculated quantity {quantity} for {symbol} is below exchange minimum {min_qty}."
        )
    return _decimal_to_string(rounded)


def format_price(symbol: str, price: float) -> str:
    symbol_info = _get_symbol_info(symbol)
    price_filter = _get_filter(symbol_info, "PRICE_FILTER")
    tick_size = str(price_filter.get("tickSize", "0.01"))
    rounded = _round_down_to_step(price, tick_size)
    return _decimal_to_string(rounded)


def get_mark_price(symbol: str) -> float:
    client = get_binance_client()
    payload = client.futures_mark_price(symbol=symbol)
    return float(payload["markPrice"])


def get_futures_open_positions(symbol: str | None = None) -> list[dict[str, Any]]:
    client = get_binance_client(require_credentials=True)
    raw_positions = client.futures_position_information(symbol=symbol) if symbol else client.futures_position_information()
    rows: list[dict[str, Any]] = []
    for item in raw_positions:
        position_amt = float(item.get("positionAmt", 0.0))
        if abs(position_amt) < 1e-12:
            continue
        entry_price = float(item.get("entryPrice", 0.0))
        mark_price = float(item.get("markPrice", 0.0))
        notional = abs(float(item.get("notional", 0.0)))
        rows.append(
            {
                "symbol": item.get("symbol", ""),
                "side": "LONG" if position_amt > 0 else "SHORT",
                "qty": abs(position_amt),
                "position_amt": position_amt,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "break_even_price": float(item.get("breakEvenPrice", 0.0)),
                "liquidation_price": float(item.get("liquidationPrice", 0.0)),
                "leverage": int(float(item.get("leverage", 0.0) or 0.0)),
                "margin_type": item.get("marginType", ""),
                "isolated_margin": float(item.get("isolatedMargin", 0.0)),
                "notional": notional,
                "unrealized_pnl": float(item.get("unRealizedProfit", 0.0)),
                "percentage": float(item.get("percentage", 0.0)) if item.get("percentage") not in (None, "") else None,
                "update_time": int(item.get("updateTime", 0) or 0),
            }
        )
    return rows


def get_open_position(symbol: str) -> dict[str, Any] | None:
    rows = get_futures_open_positions(symbol=symbol)
    return rows[0] if rows else None


def get_futures_account_balance_detail() -> dict[str, Any]:
    client = get_binance_client(require_credentials=True)
    cfg = load_binance_live_config()
    account = client.futures_account()
    balances = client.futures_account_balance()

    selected_asset = next(
        (row for row in balances if row.get("asset") == cfg.margin_asset),
        None,
    )
    assets = []
    for row in balances:
        wallet_balance = float(row.get("balance", 0.0))
        available_balance = float(row.get("availableBalance", 0.0))
        cross_wallet = float(row.get("crossWalletBalance", 0.0))
        if wallet_balance > 0 or available_balance > 0 or cross_wallet > 0 or row.get("asset") == cfg.margin_asset:
            assets.append(
                {
                    "asset": row.get("asset", ""),
                    "wallet_balance": wallet_balance,
                    "available_balance": available_balance,
                    "cross_wallet_balance": cross_wallet,
                    "cross_unpnl": float(row.get("crossUnPnl", 0.0)),
                    "update_time": int(row.get("updateTime", 0) or 0),
                }
            )

    return {
        "margin_asset": cfg.margin_asset,
        "can_trade": bool(account.get("canTrade", False)),
        "fee_tier": int(account.get("feeTier", 0) or 0),
        "available_balance": float(selected_asset.get("availableBalance", 0.0)) if selected_asset else 0.0,
        "wallet_balance": float(selected_asset.get("balance", 0.0)) if selected_asset else 0.0,
        "cross_wallet_balance": float(selected_asset.get("crossWalletBalance", 0.0)) if selected_asset else 0.0,
        "total_wallet_balance": float(account.get("totalWalletBalance", 0.0)),
        "total_margin_balance": float(account.get("totalMarginBalance", 0.0)),
        "total_unrealized_profit": float(account.get("totalUnrealizedProfit", 0.0)),
        "total_initial_margin": float(account.get("totalInitialMargin", 0.0)),
        "total_open_order_initial_margin": float(account.get("totalOpenOrderInitialMargin", 0.0)),
        "total_position_initial_margin": float(account.get("totalPositionInitialMargin", 0.0)),
        "max_withdraw_amount": float(account.get("maxWithdrawAmount", 0.0)),
        "assets": assets,
    }


def _cancel_symbol_orders(client: Client, symbol: str) -> None:
    for cancel_call in (
        lambda: client.futures_cancel_all_open_orders(symbol=symbol),
        lambda: client.futures_cancel_all_algo_open_orders(symbol=symbol),
    ):
        try:
            cancel_call()
        except BinanceAPIException as exc:
            msg = str(exc).lower()
            if "unknown order" in msg or "no order" in msg or "does not exist" in msg:
                continue
            raise


def _ensure_capacity(margin_usd: float, max_open_positions: int) -> None:
    account = get_futures_account_balance_detail()
    open_positions = get_futures_open_positions()
    if len(open_positions) >= max_open_positions:
        raise BinanceLiveError(
            f"Refusing to open a new position: {len(open_positions)} positions already open, max is {max_open_positions}."
        )
    if float(account.get("available_balance", 0.0)) < float(margin_usd):
        raise BinanceLiveError(
            f"Available {account.get('margin_asset', 'USDT')} balance is below requested margin size {margin_usd:.2f}."
        )


def open_new_futures_position(
    *,
    symbol: str,
    side: str,
    margin_usd: float,
    leverage: float,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    max_open_positions: int | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    client = get_binance_client(require_credentials=True)
    cfg = load_binance_live_config()
    trade_side = side.strip().lower()
    if trade_side not in {"long", "short"}:
        raise BinanceLiveError(f"Unsupported futures side: {side}")

    if get_open_position(symbol):
        raise BinanceLiveError(f"A Binance position is already open for {symbol}.")

    _ensure_capacity(float(margin_usd), int(max_open_positions or cfg.max_open_positions))

    leverage_value = max(1, int(round(float(leverage))))
    client.futures_change_leverage(symbol=symbol, leverage=leverage_value)

    mark_price = get_mark_price(symbol)
    quantity = format_quantity(symbol, (float(margin_usd) * leverage_value) / mark_price)
    order_side = "BUY" if trade_side == "long" else "SELL"
    exit_side = "SELL" if trade_side == "long" else "BUY"

    entry_order = client.futures_create_order(
        symbol=symbol,
        side=order_side,
        type="MARKET",
        quantity=quantity,
    )

    stop_order = None
    if stop_price is not None:
        stop_order = client.futures_create_order(
            symbol=symbol,
            side=exit_side,
            type="STOP_MARKET",
            stopPrice=format_price(symbol, stop_price),
            closePosition="true",
            workingType="MARK_PRICE",
        )

    take_profit_order = None
    if take_profit_price is not None:
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=exit_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=format_price(symbol, take_profit_price),
            closePosition="true",
            workingType="MARK_PRICE",
        )

    return {
        "source": source,
        "symbol": symbol,
        "side": trade_side,
        "margin_usd": float(margin_usd),
        "configured_leverage": leverage_value,
        "entry_order": entry_order,
        "stop_order": stop_order,
        "take_profit_order": take_profit_order,
        "position": get_open_position(symbol),
    }


def close_futures_position(symbol: str, *, source: str = "manual", allow_missing: bool = False) -> dict[str, Any]:
    client = get_binance_client(require_credentials=True)
    position = get_open_position(symbol)
    if not position:
        if allow_missing:
            return {
                "source": source,
                "symbol": symbol,
                "status": "already_closed",
            }
        raise BinanceLiveError(f"No open Binance USDT-M position for {symbol}.")

    _cancel_symbol_orders(client, symbol)
    close_side = "SELL" if float(position["position_amt"]) > 0 else "BUY"
    quantity = format_quantity(symbol, abs(float(position["position_amt"])))
    close_order = client.futures_create_order(
        symbol=symbol,
        side=close_side,
        type="MARKET",
        quantity=quantity,
        reduceOnly="true",
    )

    return {
        "source": source,
        "symbol": symbol,
        "status": "closed",
        "close_order": close_order,
        "position_before_close": position,
        "position_after_close": get_open_position(symbol),
    }