import os
import json
import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

BINANCE_BASE_URL = "https://api.binance.com"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_AI_COMMENT = os.getenv("USE_AI_COMMENT", "false").lower() == "true"

SCAN_INTERVAL = os.getenv("SCAN_INTERVAL", "5m")
HTF_INTERVAL = os.getenv("HTF_INTERVAL", "30m")
TOP_N = int(os.getenv("TOP_N", "100"))

ENTRY_SCORE = int(os.getenv("ENTRY_SCORE", "7"))
EXIT_SCORE = int(os.getenv("EXIT_SCORE", "4"))

MIN_ER = float(os.getenv("MIN_ER", "0.18"))
MIN_ADX = float(os.getenv("MIN_ADX", "15"))
MIN_24H_QUOTE_VOLUME = float(os.getenv("MIN_24H_QUOTE_VOLUME", "10000000"))

KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "250"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "300"))

MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "20"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP1_R_MULT = float(os.getenv("TP1_R_MULT", "1.0"))
TP2_R_MULT = float(os.getenv("TP2_R_MULT", "2.0"))
TP3_R_MULT = float(os.getenv("TP3_R_MULT", "3.0"))

STATE_FILE = "signals_state.json"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or chat id missing.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        print("Telegram error:", r.text)


def get_top_symbols():
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    data = requests.get(url, timeout=30).json()

    rows = []
    for item in data:
        symbol = item.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        if any(x in symbol for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"]):
            continue

        quote_volume = float(item.get("quoteVolume", 0))
        if quote_volume < MIN_24H_QUOTE_VOLUME:
            continue

        rows.append({"symbol": symbol, "quoteVolume": quote_volume})

    rows = sorted(rows, key=lambda x: x["quoteVolume"], reverse=True)
    return rows[:TOP_N]


def get_klines(symbol, interval, limit=250):
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params, timeout=30).json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = df[col].astype(float)

    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df, length=14):
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def dmi_adx(df, length=14, smoothing=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = true_range(df)
    atr_val = tr.ewm(alpha=1 / length, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / smoothing, adjust=False).mean()

    return plus_di, minus_di, adx


def efficiency_ratio(close, length=20):
    change = (close - close.shift(length)).abs()
    volatility = close.diff().abs().rolling(length).sum()
    return change / volatility


def kama(close, er_length=20, fast=2, slow=30):
    er = efficiency_ratio(close, er_length).fillna(0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    result = [close.iloc[0]]
    for i in range(1, len(close)):
        result.append(result[-1] + sc.iloc[i] * (close.iloc[i] - result[-1]))

    return pd.Series(result, index=close.index)


def z_momentum(close, lookback=12, z_len=50):
    past = close.shift(lookback)
    log_ret = (close / past).apply(lambda x: math.log(x) if x and x > 0 else 0)
    mean = log_ret.rolling(z_len).mean()
    std = log_ret.rolling(z_len).std()
    return (log_ret - mean) / std


def grade_signal(bull_score):
    if bull_score >= 10:
        return "A+"
    if bull_score == 9:
        return "A"
    if bull_score >= 7:
        return "B"
    return "C"


def suggest_leverage(atr_pct, bull_score, er_value, z_value):
    if atr_pct <= 0:
        return 2

    if atr_pct >= 2.5:
        base = 2
    elif atr_pct >= 1.8:
        base = 3
    elif atr_pct >= 1.2:
        base = 5
    elif atr_pct >= 0.8:
        base = 7
    elif atr_pct >= 0.5:
        base = 10
    else:
        base = 15

    if bull_score >= 10 and er_value >= 0.35 and z_value >= 1.2:
        base += 2
    elif bull_score <= 7:
        base -= 2

    base = max(2, min(base, MAX_LEVERAGE))
    return int(base)


def build_trade_plan(price, atr_value, atr_pct, bull_score, er_value, z_value):
    sl_distance = atr_value * SL_ATR_MULT
    stop_loss = price - sl_distance

    tp1 = price + sl_distance * TP1_R_MULT
    tp2 = price + sl_distance * TP2_R_MULT
    tp3 = price + sl_distance * TP3_R_MULT

    leverage = suggest_leverage(atr_pct, bull_score, er_value, z_value)

    liquidation_buffer_pct = 100 / leverage
    sl_pct = (sl_distance / price) * 100

    if sl_pct > liquidation_buffer_pct * 0.45:
        leverage = max(2, int(45 / sl_pct))

    leverage = max(2, min(leverage, MAX_LEVERAGE))

    return {
        "leverage": leverage,
        "entry": price,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl_pct": sl_pct,
        "atr_pct": atr_pct,
    }


def get_ai_comment(result, plan):
    if not USE_AI_COMMENT or not openai_client:
        return None

    prompt = f"""
You are a crypto futures risk analyst. Give a short Turkish trading note.
Do not promise profit. Do not use hype. Mention risk clearly.
Signal data:
Symbol: {result['symbol']}
Price: {result['price']}
Bull score: {result['bull_score']}/10
ER: {result['er']:.2f}
Z momentum: {result['z']:.2f}
HTF bull: {result['htf_bull']}
ATR OK: {result['atr_ok']}
ATR %: {plan['atr_pct']:.2f}
Suggested leverage: {plan['leverage']}x
Stop loss: {plan['stop_loss']}
TP1: {plan['tp1']}
TP2: {plan['tp2']}
TP3: {plan['tp3']}

Return maximum 2 short sentences in Turkish.
"""

    try:
        response = openai_client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
            max_output_tokens=120,
        )
        return response.output_text.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return None


def analyze_symbol(symbol):
    df = get_klines(symbol, SCAN_INTERVAL, KLINE_LIMIT)
    htf = get_klines(symbol, HTF_INTERVAL, KLINE_LIMIT)

    close = df["close"]

    ema_fast = ema(close, 21)
    ema_slow = ema(close, 55)
    ema_macro = ema(close, 200)

    kama_line = kama(close, 20, 2, 30)
    er = efficiency_ratio(close, 20)
    z = z_momentum(close, 12, 50)
    rsi_val = rsi(close, 14)
    _, _, macd_hist = macd(close, 12, 26, 9)
    plus_di, minus_di, adx_val = dmi_adx(df, 14, 14)

    atr_val = atr(df, 14)
    atr_pct = atr_val / close * 100
    atr_avg = atr_pct.rolling(50).mean()

    htf_fast = ema(htf["close"], 21)
    htf_slow = ema(htf["close"], 55)

    last = -1
    prev = -2

    price = close.iloc[last]
    price_above_kama = close.iloc[last] > kama_line.iloc[last]
    price_below_kama = close.iloc[last] < kama_line.iloc[last]

    kama_green = kama_line.iloc[last] > kama_line.iloc[prev]
    kama_red = kama_line.iloc[last] < kama_line.iloc[prev]

    htf_bull = htf_fast.iloc[prev] > htf_slow.iloc[prev]
    htf_bear = htf_fast.iloc[prev] < htf_slow.iloc[prev]

    atr_regime_ok = (
        atr_avg.iloc[last] > 0
        and atr_pct.iloc[last] > atr_avg.iloc[last] * 0.55
        and atr_pct.iloc[last] < atr_avg.iloc[last] * 3.0
    )

    bull_score = 0
    bull_score += 1 if price_above_kama else 0
    bull_score += 1 if kama_green else 0
    bull_score += 1 if ema_fast.iloc[last] > ema_slow.iloc[last] else 0
    bull_score += 1 if close.iloc[last] > ema_macro.iloc[last] else 0
    bull_score += 1 if htf_bull else 0
    bull_score += 1 if z.iloc[last] > 0 else 0
    bull_score += 1 if rsi_val.iloc[last] > 50 else 0
    bull_score += 1 if macd_hist.iloc[last] > 0 and macd_hist.iloc[last] > macd_hist.iloc[prev] else 0
    bull_score += 1 if adx_val.iloc[last] > MIN_ADX and plus_di.iloc[last] > minus_di.iloc[last] else 0
    bull_score += 1 if er.iloc[last] >= MIN_ER and atr_regime_ok else 0

    strict_buy = (
        bull_score >= ENTRY_SCORE
        and htf_bull
        and er.iloc[last] >= MIN_ER
        and atr_regime_ok
        and price_above_kama
        and kama_green
    )

    sell_ready = (
        bull_score <= EXIT_SCORE
        or price_below_kama
        or kama_red
        or htf_bear
    )

    result = {
        "symbol": symbol,
        "price": float(price),
        "bull_score": int(bull_score),
        "er": float(er.iloc[last]),
        "z": float(z.iloc[last]),
        "htf_bull": bool(htf_bull),
        "htf_bear": bool(htf_bear),
        "atr_ok": bool(atr_regime_ok),
        "atr": float(atr_val.iloc[last]),
        "atr_pct": float(atr_pct.iloc[last]),
        "price_above_kama": bool(price_above_kama),
        "kama_green": bool(kama_green),
        "strict_buy": bool(strict_buy),
        "sell_ready": bool(sell_ready),
    }

    plan = build_trade_plan(
        price=result["price"],
        atr_value=result["atr"],
        atr_pct=result["atr_pct"],
        bull_score=result["bull_score"],
        er_value=result["er"],
        z_value=result["z"],
    )

    result["plan"] = plan
    return result


def format_signal(result, quote_volume):
    symbol = result["symbol"].replace("USDT", "")
    grade = grade_signal(result["bull_score"])
    volume_m = quote_volume / 1_000_000
    plan = result["plan"]

    ai_comment = get_ai_comment(result, plan)

    msg = (
        f"🟢 <b>{symbol} BUY [{grade}]</b>\n"
        f"Price: <b>{result['price']:.6g}</b>\n"
        f"TF: <b>{SCAN_INTERVAL}</b> | HTF: <b>{HTF_INTERVAL}</b>\n"
        f"Bull: <b>{result['bull_score']}/10</b> | ER: <b>{result['er']:.2f}</b> | Z: <b>{result['z']:.2f}</b>\n"
        f"HTF: <b>{'BULL' if result['htf_bull'] else 'BEAR'}</b> | ATR: <b>{'OK' if result['atr_ok'] else 'BAD'}</b> | KAMA: <b>{'GREEN' if result['kama_green'] else 'RED'}</b>\n"
        f"24h Vol: <b>{volume_m:.1f}M USDT</b>\n\n"
        f"⚙️ <b>Futures Plan</b>\n"
        f"Leverage: <b>{plan['leverage']}x</b>\n"
        f"Entry: <b>{plan['entry']:.6g}</b>\n"
        f"SL: <b>{plan['stop_loss']:.6g}</b> (-{plan['sl_pct']:.2f}%)\n"
        f"TP1: <b>{plan['tp1']:.6g}</b>\n"
        f"TP2: <b>{plan['tp2']:.6g}</b>\n"
        f"TP3: <b>{plan['tp3']:.6g}</b>\n"
    )

    if ai_comment:
        msg += f"\n🤖 <b>AI Note</b>\n{ai_comment}"

    return msg


def run_once():
    print(f"[{datetime.now(timezone.utc).isoformat()}] scanning...")

    state = load_state()
    top_symbols = get_top_symbols()
    messages = []

    for item in top_symbols:
        symbol = item["symbol"]
        quote_volume = item["quoteVolume"]

        try:
            result = analyze_symbol(symbol)
        except Exception as e:
            print(f"{symbol} error: {e}")
            continue

        old_state = state.get(symbol, "FLAT")

        if result["strict_buy"] and old_state == "FLAT":
            messages.append(format_signal(result, quote_volume))
            state[symbol] = "LONG"
            state[symbol + "_entry_price"] = result["price"]
            state[symbol + "_entry_time"] = datetime.now(timezone.utc).isoformat()
            state[symbol + "_leverage"] = result["plan"]["leverage"]
            state[symbol + "_sl"] = result["plan"]["stop_loss"]
            state[symbol + "_tp1"] = result["plan"]["tp1"]
            state[symbol + "_tp2"] = result["plan"]["tp2"]
            state[symbol + "_tp3"] = result["plan"]["tp3"]

        elif old_state == "LONG" and result["sell_ready"]:
            state[symbol] = "FLAT"
            for key in ["_entry_price", "_entry_time", "_leverage", "_sl", "_tp1", "_tp2", "_tp3"]:
                state.pop(symbol + key, None)

        print(
            symbol,
            "state:", state.get(symbol, "FLAT"),
            "bull:", result["bull_score"],
            "er:", round(result["er"], 2),
            "z:", round(result["z"], 2),
            "lev:", result["plan"]["leverage"],
            "buy:", result["strict_buy"],
        )

        time.sleep(0.08)

    save_state(state)

    if messages:
        header = (
            f"🚨 <b>BINANCE STRICT BUY SCANNER</b>\n"
            f"TF: <b>{SCAN_INTERVAL}</b> | HTF: <b>{HTF_INTERVAL}</b>\n"
            f"Risk model: <b>ATR-based leverage + ATR SL/TP</b>\n\n"
        )
        send_telegram(header + "\n\n────────────\n\n".join(messages))
    else:
        print("No BUY signal.")


def main():
    while True:
        try:
            run_once()
        except Exception as e:
            print("Main loop error:", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
