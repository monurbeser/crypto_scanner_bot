import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

PAPER_INITIAL_EQUITY = float(os.getenv("PAPER_INITIAL_EQUITY", "1000"))
PAPER_RISK_PER_TRADE = float(os.getenv("PAPER_RISK_PER_TRADE", "0.05"))
PAPER_STATE_FILE = Path(os.getenv("PAPER_STATE_FILE", "paper_state.json"))
PAPER_TRADES_FILE = Path(os.getenv("PAPER_TRADES_FILE", "paper_trades.json"))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_paper_state():
    state = _read_json(PAPER_STATE_FILE, None)
    if not state:
        state = {
            "initial_equity": PAPER_INITIAL_EQUITY,
            "cash": PAPER_INITIAL_EQUITY,
            "realized_pnl": 0.0,
            "open_positions": {},
            "last_update": now_iso(),
        }
        save_paper_state(state)
    return state


def save_paper_state(state):
    state["last_update"] = now_iso()
    _write_json(PAPER_STATE_FILE, state)


def load_paper_trades():
    return _read_json(PAPER_TRADES_FILE, [])


def save_paper_trades(trades):
    _write_json(PAPER_TRADES_FILE, trades)


def get_trade_notional(equity: float, leverage: float) -> float:
    margin = max(0.0, equity * PAPER_RISK_PER_TRADE)
    return margin * max(1.0, leverage)


def open_paper_trade(symbol, side, result, order_result=None, order_note=None):
    state = load_paper_state()
    trades = load_paper_trades()

    if symbol in state.get("open_positions", {}):
        return {
            "ok": False,
            "msg": f"Paper trade already open for {symbol}",
            "trade": state["open_positions"][symbol],
        }

    plan = result.get("plan", {})
    entry = float(result["price"])
    leverage = float(plan.get("leverage", 1))
    equity = float(state.get("cash", PAPER_INITIAL_EQUITY))
    notional = get_trade_notional(max(equity, 0), leverage)
    qty = notional / entry if entry > 0 else 0.0

    trade_id = str(uuid.uuid4())[:8]
    trade = {
        "id": trade_id,
        "symbol": symbol,
        "side": side,
        "status": "OPEN",
        "open_time": now_iso(),
        "close_time": None,
        "entry_price": entry,
        "exit_price": None,
        "qty": qty,
        "notional_usdt": notional,
        "leverage": leverage,
        "margin_used": notional / leverage if leverage else notional,
        "sl": float(plan.get("stop_loss", 0)),
        "tp1": float(plan.get("tp1", 0)),
        "tp2": float(plan.get("tp2", 0)),
        "tp3": float(plan.get("tp3", 0)),
        "bull_score": int(result.get("bull_score", 0)),
        "bear_score": int(result.get("bear_score", 0)),
        "er": float(result.get("er", 0)),
        "z": float(result.get("z", 0)),
        "atr_pct": float(plan.get("atr_pct", 0)),
        "order_mode": (order_result or {}).get("mode"),
        "order_status": (order_result or {}).get("order_status"),
        "order_endpoint": (order_result or {}).get("endpoint"),
        "order_note": order_note,
        "realized_pnl": None,
        "realized_pnl_pct_on_margin": None,
        "exit_reason": None,
    }

    state.setdefault("open_positions", {})[symbol] = trade
    trades.append(trade)

    save_paper_state(state)
    save_paper_trades(trades)

    return {"ok": True, "msg": "Paper trade opened", "trade": trade}


def close_paper_trade(symbol, exit_price, reason="EXIT", close_order_result=None):
    state = load_paper_state()
    trades = load_paper_trades()

    open_positions = state.setdefault("open_positions", {})
    trade = open_positions.get(symbol)

    if not trade:
        return {"ok": False, "msg": f"No paper position for {symbol}", "trade": None}

    exit_price = float(exit_price)
    entry = float(trade["entry_price"])
    qty = float(trade["qty"])
    side = trade["side"]
    margin_used = float(trade.get("margin_used", 0))

    if side == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    pnl_pct_on_margin = (pnl / margin_used * 100) if margin_used else 0.0

    trade["status"] = "CLOSED"
    trade["close_time"] = now_iso()
    trade["exit_price"] = exit_price
    trade["realized_pnl"] = pnl
    trade["realized_pnl_pct_on_margin"] = pnl_pct_on_margin
    trade["exit_reason"] = reason

    if close_order_result:
        trade["close_order_mode"] = close_order_result.get("mode")
        trade["close_order_status"] = close_order_result.get("order_status")
        trade["close_order_endpoint"] = close_order_result.get("endpoint")

    open_positions.pop(symbol, None)

    for i, t in enumerate(trades):
        if t.get("id") == trade.get("id"):
            trades[i] = trade
            break

    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    state["cash"] = float(state.get("initial_equity", PAPER_INITIAL_EQUITY)) + float(state.get("realized_pnl", 0.0))

    save_paper_state(state)
    save_paper_trades(trades)

    return {"ok": True, "msg": "Paper trade closed", "trade": trade}
