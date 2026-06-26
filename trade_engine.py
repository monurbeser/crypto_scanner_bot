
import os
import time
import hmac
import hashlib
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://fapi.binance.com"

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

ORDER_MODE = os.getenv("BINANCE_ORDER_MODE", "TEST").upper()
POSITION_MODE = os.getenv("POSITION_MODE", "ONE_WAY").upper()
MARGIN_TYPE = os.getenv("MARGIN_TYPE", "ISOLATED").upper()

DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "3"))
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "10"))
DEFAULT_NOTIONAL_USDT = float(os.getenv("DEFAULT_NOTIONAL_USDT", "10"))


def _headers():
    if not BINANCE_API_KEY:
        raise RuntimeError("BINANCE_API_KEY missing")
    return {"X-MBX-APIKEY": BINANCE_API_KEY}


def _sign(params: dict) -> dict:
    if not BINANCE_SECRET_KEY:
        raise RuntimeError("BINANCE_SECRET_KEY missing")
    query_string = urlencode(params)
    signature = hmac.new(
        BINANCE_SECRET_KEY.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = signature
    return params


def public_get(path: str, params: dict | None = None):
    r = requests.get(BASE_URL + path, params=params or {}, timeout=20)
    return r.status_code, r.json()


def signed_request(method: str, path: str, params: dict | None = None):
    base_params = params.copy() if params else {}
    base_params["timestamp"] = int(time.time() * 1000)
    base_params["recvWindow"] = 5000
    signed_params = _sign(base_params)
    url = BASE_URL + path
    if method.upper() == "GET":
        r = requests.get(url, headers=_headers(), params=signed_params, timeout=20)
    elif method.upper() == "POST":
        r = requests.post(url, headers=_headers(), params=signed_params, timeout=20)
    else:
        raise ValueError(f"Unsupported method: {method}")
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    return r.status_code, data


def get_mark_price(symbol: str) -> float:
    status, data = public_get("/fapi/v1/ticker/price", {"symbol": symbol})
    if status != 200:
        raise RuntimeError(f"Price error {status}: {data}")
    return float(data["price"])


def get_symbol_rules(symbol: str) -> dict:
    status, data = public_get("/fapi/v1/exchangeInfo")
    if status != 200:
        raise RuntimeError(f"exchangeInfo error {status}: {data}")
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            rules = {
                "quantityPrecision": int(s.get("quantityPrecision", 3)),
                "pricePrecision": int(s.get("pricePrecision", 2)),
                "minQty": Decimal("0"),
                "stepSize": Decimal("0.001"),
            }
            for f in s.get("filters", []):
                if f.get("filterType") in ["LOT_SIZE", "MARKET_LOT_SIZE"]:
                    rules["minQty"] = Decimal(str(f.get("minQty", "0")))
                    rules["stepSize"] = Decimal(str(f.get("stepSize", "0.001")))
            return rules
    raise RuntimeError(f"Symbol not found: {symbol}")


def floor_to_step(value, step: Decimal) -> Decimal:
    value_decimal = Decimal(str(value))
    return (value_decimal / step).to_integral_value(rounding=ROUND_DOWN) * step


def normalize_quantity(symbol: str, quantity) -> str:
    rules = get_symbol_rules(symbol)
    qty = floor_to_step(abs(Decimal(str(quantity))), rules["stepSize"])
    if qty <= 0:
        raise RuntimeError(f"Quantity is zero. Symbol={symbol}, quantity={quantity}")
    if qty < rules["minQty"]:
        raise RuntimeError(f"Quantity below minQty. Symbol={symbol}, qty={qty}, minQty={rules['minQty']}")
    return format(qty.normalize(), "f")


def calculate_quantity(symbol: str, notional_usdt: float) -> str:
    price = get_mark_price(symbol)
    raw_qty = notional_usdt / price
    return normalize_quantity(symbol, raw_qty)


def set_margin_type(symbol: str, margin_type: str = MARGIN_TYPE):
    status, data = signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
    if isinstance(data, dict) and data.get("code") == -4046:
        return 200, {"msg": "Margin type already set"}
    return status, data


def set_leverage(symbol: str, leverage: int):
    leverage = max(1, min(int(leverage), MAX_LEVERAGE))
    return signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def get_position_amt(symbol: str, direction: str | None = None) -> Decimal:
    status, data = signed_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
    if status != 200:
        raise RuntimeError(f"Position risk error {status}: {data}")
    direction = direction.upper() if direction else None
    for p in data:
        if p.get("symbol") != symbol:
            continue
        amt = Decimal(str(p.get("positionAmt", "0")))
        if POSITION_MODE == "HEDGE" and direction:
            if p.get("positionSide") == direction:
                return amt
        if POSITION_MODE != "HEDGE":
            return amt
    return Decimal("0")


def place_futures_market_order(symbol: str, direction: str, notional_usdt: float | None = None, leverage: int | None = None):
    symbol = symbol.upper()
    direction = direction.upper()
    if direction not in ["LONG", "SHORT"]:
        raise ValueError("direction must be LONG or SHORT")
    notional_usdt = float(notional_usdt or DEFAULT_NOTIONAL_USDT)
    leverage = int(leverage or DEFAULT_LEVERAGE)
    side = "BUY" if direction == "LONG" else "SELL"
    quantity = calculate_quantity(symbol, notional_usdt)
    margin_status, margin_data = set_margin_type(symbol, MARGIN_TYPE)
    leverage_status, leverage_data = set_leverage(symbol, leverage)
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "newOrderRespType": "RESULT",
    }
    if POSITION_MODE == "HEDGE":
        params["positionSide"] = direction
    endpoint = "/fapi/v1/order/test" if ORDER_MODE == "TEST" else "/fapi/v1/order"
    order_status, order_data = signed_request("POST", endpoint, params)
    return {
        "mode": ORDER_MODE,
        "endpoint": endpoint,
        "symbol": symbol,
        "direction": direction,
        "side": side,
        "quantity": quantity,
        "notional_usdt": notional_usdt,
        "leverage": leverage,
        "margin_status": margin_status,
        "margin_data": margin_data,
        "leverage_status": leverage_status,
        "leverage_data": leverage_data,
        "order_status": order_status,
        "order_data": order_data,
    }


def close_futures_market_position(symbol: str, direction: str, quantity=None):
    symbol = symbol.upper()
    direction = direction.upper()
    if direction not in ["LONG", "SHORT"]:
        raise ValueError("direction must be LONG or SHORT")
    if quantity is None:
        amt = get_position_amt(symbol, direction)
        if amt == 0:
            raise RuntimeError(f"No open position found for {symbol} {direction}")
        quantity = abs(amt)
    quantity = normalize_quantity(symbol, quantity)
    close_side = "SELL" if direction == "LONG" else "BUY"
    params = {
        "symbol": symbol,
        "side": close_side,
        "type": "MARKET",
        "quantity": quantity,
        "newOrderRespType": "RESULT",
    }
    if POSITION_MODE == "HEDGE":
        params["positionSide"] = direction
    else:
        params["reduceOnly"] = "true"
    endpoint = "/fapi/v1/order/test" if ORDER_MODE == "TEST" else "/fapi/v1/order"
    order_status, order_data = signed_request("POST", endpoint, params)
    return {
        "mode": ORDER_MODE,
        "endpoint": endpoint,
        "symbol": symbol,
        "direction": direction,
        "close_side": close_side,
        "quantity": quantity,
        "order_status": order_status,
        "order_data": order_data,
    }
