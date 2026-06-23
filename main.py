import asyncio
import html
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SCAN_INTERVAL = os.getenv("SCAN_INTERVAL", "5m")
HTF_INTERVAL = os.getenv("HTF_INTERVAL", "30m")
TOP_N = int(os.getenv("TOP_N", "100"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "300"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))
STATE_FILE = Path(os.getenv("STATE_FILE", "signals_state.json"))
SEND_SELL_ALERTS = os.getenv("SEND_SELL_ALERTS", "false").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

ENTRY_SCORE = int(os.getenv("ENTRY_SCORE", "7"))
EXIT_SCORE = int(os.getenv("EXIT_SCORE", "4"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "10"))
USE_CONFIRMED_HTF = os.getenv("USE_CONFIRMED_HTF", "true").lower() == "true"

EMA_FAST_LEN = int(os.getenv("EMA_FAST_LEN", "21"))
EMA_SLOW_LEN = int(os.getenv("EMA_SLOW_LEN", "55"))
EMA_MACRO_LEN = int(os.getenv("EMA_MACRO_LEN", "200"))
KAMA_ER_LEN = int(os.getenv("KAMA_ER_LEN", "20"))
KAMA_FAST = int(os.getenv("KAMA_FAST", "2"))
KAMA_SLOW = int(os.getenv("KAMA_SLOW", "30"))
ROC_LEN = int(os.getenv("ROC_LEN", "12"))
Z_LEN = int(os.getenv("Z_LEN", "50"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
DI_LEN = int(os.getenv("DI_LEN", "14"))
ADX_SMOOTH = int(os.getenv("ADX_SMOOTH", "14"))
ADX_MIN = float(os.getenv("ADX_MIN", "15.0"))
ER_MIN = float(os.getenv("ER_MIN", "0.18"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
ATR_REGIME_LEN = int(os.getenv("ATR_REGIME_LEN", "50"))
ATR_LOW_RATIO = float(os.getenv("ATR_LOW_RATIO", "0.55"))
ATR_HIGH_RATIO = float(os.getenv("ATR_HIGH_RATIO", "3.0"))

EXCLUDE_SYMBOL_KEYWORDS = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
EXCLUDE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "TRY", "BRL", "AEUR"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("crypto-scanner")


@dataclass
class SignalResult:
    symbol: str
    price: float
    bull_score: int
    bear_score: int
    htf: str
    er: float
    atr_ok: bool
    price_above_kama: bool
    kama_green: bool
    z_mom: float
    strict_buy_ready: bool
    buy_signal: bool
    sell_signal: bool
    state: str
    closed_at: int


def interval_to_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(interval[:-1]) * units[interval[-1]]


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        logger.warning("State file could not be read. Starting with empty state.")
        return {}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def dmi_adx(df: pd.DataFrame, length: int, smooth: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr_wilder = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_wilder.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / smooth, adjust=False).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)


def efficiency_ratio(series: pd.Series, length: int) -> pd.Series:
    change = (series - series.shift(length)).abs()
    volatility = series.diff().abs().rolling(length).sum()
    return (change / volatility.replace(0, np.nan)).fillna(0)


def kama(series: pd.Series, er_len: int, fast_len: int, slow_len: int) -> pd.Series:
    er_val = efficiency_ratio(series, er_len)
    fast_sc = 2.0 / (fast_len + 1.0)
    slow_sc = 2.0 / (slow_len + 1.0)
    sc = (er_val * (fast_sc - slow_sc) + slow_sc) ** 2

    values = np.full(len(series), np.nan)
    for i, price in enumerate(series.to_numpy(dtype=float)):
        if i == 0 or np.isnan(values[i - 1]):
            values[i] = price
        else:
            values[i] = values[i - 1] + sc.iloc[i] * (price - values[i - 1])
    return pd.Series(values, index=series.index)


def build_df(klines: List[List[Any]]) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(klines, columns=cols)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
    return df.dropna().reset_index(drop=True)


def drop_open_candle(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    # Binance returns the currently-forming candle as the last row. We only use fully closed candles.
    now_ms = int(time.time() * 1000)
    interval_ms = interval_to_ms(interval)
    if len(df) > 0 and df.iloc[-1]["open_time"] + interval_ms > now_ms:
        return df.iloc[:-1].copy()
    return df.copy()


def analyze_ltf(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    min_needed = max(EMA_MACRO_LEN, ATR_REGIME_LEN, Z_LEN, KAMA_ER_LEN, MACD_SLOW + MACD_SIGNAL, DI_LEN + ADX_SMOOTH) + 5
    if len(df) < min_needed:
        return None

    close = df["close"]
    ema_fast = ema(close, EMA_FAST_LEN)
    ema_slow = ema(close, EMA_SLOW_LEN)
    ema_macro = ema(close, EMA_MACRO_LEN)
    k = kama(close, KAMA_ER_LEN, KAMA_FAST, KAMA_SLOW)
    er_val = efficiency_ratio(close, KAMA_ER_LEN)

    past_close = close.shift(ROC_LEN)
    log_ret = np.log(close / past_close).replace([np.inf, -np.inf], np.nan).fillna(0)
    ret_mean = log_ret.rolling(Z_LEN).mean()
    ret_std = log_ret.rolling(Z_LEN).std()
    z_mom = ((log_ret - ret_mean) / ret_std.replace(0, np.nan)).fillna(0)

    rsi_val = rsi(close, RSI_LEN)
    _, _, macd_hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    plus_di, minus_di, adx_val = dmi_adx(df, DI_LEN, ADX_SMOOTH)
    atr_val = atr(df, ATR_LEN)
    atr_pct = atr_val / close.replace(0, np.nan) * 100
    atr_avg = atr_pct.rolling(ATR_REGIME_LEN).mean()

    i = len(df) - 1
    price_above_kama = close.iloc[i] > k.iloc[i]
    price_below_kama = close.iloc[i] < k.iloc[i]
    kama_green = k.iloc[i] > k.iloc[i - 1]
    kama_red = k.iloc[i] < k.iloc[i - 1]
    macro_bull = close.iloc[i] > ema_macro.iloc[i]
    macro_bear = close.iloc[i] < ema_macro.iloc[i]
    atr_ok = bool(atr_avg.iloc[i] > 0 and atr_pct.iloc[i] > atr_avg.iloc[i] * ATR_LOW_RATIO and atr_pct.iloc[i] < atr_avg.iloc[i] * ATR_HIGH_RATIO)

    local_bull = 0
    local_bull += int(price_above_kama)
    local_bull += int(kama_green)
    local_bull += int(ema_fast.iloc[i] > ema_slow.iloc[i])
    local_bull += int(macro_bull)
    local_bull += int(z_mom.iloc[i] > 0)
    local_bull += int(rsi_val.iloc[i] > 50)
    local_bull += int(macd_hist.iloc[i] > 0 and macd_hist.iloc[i] > macd_hist.iloc[i - 1])
    local_bull += int(adx_val.iloc[i] > ADX_MIN and plus_di.iloc[i] > minus_di.iloc[i])
    local_bull += int(er_val.iloc[i] >= ER_MIN and atr_ok)

    local_bear = 0
    local_bear += int(price_below_kama)
    local_bear += int(kama_red)
    local_bear += int(ema_fast.iloc[i] < ema_slow.iloc[i])
    local_bear += int(macro_bear)
    local_bear += int(z_mom.iloc[i] < 0)
    local_bear += int(rsi_val.iloc[i] < 50)
    local_bear += int(macd_hist.iloc[i] < 0 and macd_hist.iloc[i] < macd_hist.iloc[i - 1])
    local_bear += int(adx_val.iloc[i] > ADX_MIN and minus_di.iloc[i] > plus_di.iloc[i])
    local_bear += int(er_val.iloc[i] >= ER_MIN and atr_ok)

    return {
        "price": float(close.iloc[i]),
        "local_bull": int(local_bull),
        "local_bear": int(local_bear),
        "er": float(er_val.iloc[i]),
        "atr_ok": atr_ok,
        "price_above_kama": bool(price_above_kama),
        "kama_green": bool(kama_green),
        "z_mom": float(z_mom.iloc[i]),
        "closed_at": int(df.iloc[i]["close_time"]),
    }


def analyze_htf(df: pd.DataFrame) -> str:
    min_needed = max(EMA_FAST_LEN, EMA_SLOW_LEN) + 3
    if len(df) < min_needed:
        return "NEUTRAL"
    close = df["close"]
    fast = ema(close, EMA_FAST_LEN)
    slow = ema(close, EMA_SLOW_LEN)
    idx = len(df) - 2 if USE_CONFIRMED_HTF and len(df) >= 2 else len(df) - 1
    diff = fast.iloc[idx] - slow.iloc[idx]
    if diff > 0:
        return "BULL"
    if diff < 0:
        return "BEAR"
    return "NEUTRAL"


async def fetch_json(session: aiohttp.ClientSession, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BINANCE_BASE_URL}{path}"
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                text = await resp.text()
                if resp.status == 429:
                    wait = 5 + attempt * 5
                    logger.warning("Rate limited by Binance. Waiting %s seconds.", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return json.loads(text)
        except Exception as exc:
            if attempt == 2:
                raise
            await asyncio.sleep(1 + attempt * 2)
    raise RuntimeError("Unreachable")


async def get_tradable_usdt_symbols(session: aiohttp.ClientSession) -> set[str]:
    data = await fetch_json(session, "/api/v3/exchangeInfo")
    symbols = set()
    for item in data.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if not item.get("isSpotTradingAllowed", True):
            continue
        base = item.get("baseAsset", "")
        symbol = item.get("symbol", "")
        if base in EXCLUDE_BASES:
            continue
        if any(k in symbol for k in EXCLUDE_SYMBOL_KEYWORDS):
            continue
        symbols.add(symbol)
    return symbols


async def get_top_usdt_symbols(session: aiohttp.ClientSession) -> List[str]:
    tradable = await get_tradable_usdt_symbols(session)
    tickers = await fetch_json(session, "/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if symbol not in tradable:
            continue
        try:
            quote_volume = float(t.get("quoteVolume", "0"))
            last_price = float(t.get("lastPrice", "0"))
        except ValueError:
            continue
        if quote_volume <= 0 or last_price <= 0:
            continue
        rows.append((symbol, quote_volume))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:TOP_N]]


async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str) -> pd.DataFrame:
    data = await fetch_json(session, "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": KLINE_LIMIT})
    return drop_open_candle(build_df(data), interval)


async def analyze_symbol(session: aiohttp.ClientSession, symbol: str, state: Dict[str, Any], sem: asyncio.Semaphore) -> Optional[SignalResult]:
    async with sem:
        try:
            ltf_df, htf_df = await asyncio.gather(fetch_klines(session, symbol, SCAN_INTERVAL), fetch_klines(session, symbol, HTF_INTERVAL))
            ltf = analyze_ltf(ltf_df)
            if ltf is None:
                return None
            htf = analyze_htf(htf_df)
        except Exception as exc:
            logger.warning("%s skipped: %s", symbol, exc)
            return None

    bull_score = int(ltf["local_bull"] + (1 if htf == "BULL" else 0))
    bear_score = int(ltf["local_bear"] + (1 if htf == "BEAR" else 0))
    strict_buy_ready = bool(bull_score >= ENTRY_SCORE and htf == "BULL" and ltf["er"] >= ER_MIN and ltf["atr_ok"] and ltf["price_above_kama"] and ltf["kama_green"])

    item = state.setdefault(symbol, {"state": "FLAT", "last_signal_close_time": 0, "last_seen_close_time": 0})
    previous_state = item.get("state", "FLAT")
    last_signal_close_time = int(item.get("last_signal_close_time", 0))
    closed_at = int(ltf["closed_at"])
    interval_ms = interval_to_ms(SCAN_INTERVAL)
    cooldown_ok = closed_at - last_signal_close_time >= COOLDOWN_BARS * interval_ms

    buy_signal = strict_buy_ready and previous_state == "FLAT" and cooldown_ok
    sell_signal = previous_state == "LONG" and cooldown_ok and (bull_score <= EXIT_SCORE or not ltf["price_above_kama"] or not ltf["kama_green"] or htf == "BEAR")

    if buy_signal:
        item["state"] = "LONG"
        item["last_signal_close_time"] = closed_at
    elif sell_signal:
        item["state"] = "FLAT"
        item["last_signal_close_time"] = closed_at

    item["last_seen_close_time"] = closed_at
    item["last_price"] = ltf["price"]
    item["bull_score"] = bull_score
    item["htf"] = htf

    return SignalResult(
        symbol=symbol,
        price=float(ltf["price"]),
        bull_score=bull_score,
        bear_score=bear_score,
        htf=htf,
        er=float(ltf["er"]),
        atr_ok=bool(ltf["atr_ok"]),
        price_above_kama=bool(ltf["price_above_kama"]),
        kama_green=bool(ltf["kama_green"]),
        z_mom=float(ltf["z_mom"]),
        strict_buy_ready=strict_buy_ready,
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        state=item["state"],
        closed_at=closed_at,
    )


def format_buy_message(results: List[SignalResult]) -> str:
    lines = ["🚨 <b>BINANCE STRICT BUY SCANNER</b>", f"TF: <b>{SCAN_INTERVAL}</b> | HTF: <b>{HTF_INTERVAL}</b> | Top: <b>{TOP_N}</b>", ""]
    for r in results:
        base = r.symbol.replace("USDT", "")
        lines.append(f"🟢 <b>{html.escape(base)}</b> BUY")
        lines.append(f"Price: <code>{r.price:g}</code> | Bull: <b>{r.bull_score}/10</b> | ER: <b>{r.er:.2f}</b> | Z: <b>{r.z_mom:.2f}</b>")
        lines.append(f"HTF: <b>{r.htf}</b> | ATR: <b>{'OK' if r.atr_ok else 'BAD'}</b> | KAMA: <b>{'GREEN' if r.kama_green else 'NO'}</b>")
        lines.append("")
    return "\n".join(lines).strip()


def format_sell_message(results: List[SignalResult]) -> str:
    lines = ["🔴 <b>BINANCE SELL / EXIT</b>", f"TF: <b>{SCAN_INTERVAL}</b> | HTF: <b>{HTF_INTERVAL}</b>", ""]
    for r in results:
        base = r.symbol.replace("USDT", "")
        lines.append(f"🔴 <b>{html.escape(base)}</b> EXIT | Price: <code>{r.price:g}</code> | Bull: {r.bull_score}/10 | HTF: {r.htf}")
    return "\n".join(lines).strip()


async def send_telegram(session: aiohttp.ClientSession, text: str) -> None:
    if DRY_RUN:
        logger.info("DRY RUN Telegram message:\n%s", text)
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:3900], "parse_mode": "HTML", "disable_web_page_preview": True}
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as resp:
        body = await resp.text()
        if resp.status >= 300:
            raise RuntimeError(f"Telegram error {resp.status}: {body}")


async def scan_once(session: aiohttp.ClientSession, state: Dict[str, Any]) -> None:
    symbols = await get_top_usdt_symbols(session)
    logger.info("Scanning %s symbols. First 10: %s", len(symbols), ", ".join(symbols[:10]))

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [analyze_symbol(session, symbol, state, sem) for symbol in symbols]
    results = [r for r in await asyncio.gather(*tasks) if r is not None]

    buys = [r for r in results if r.buy_signal]
    sells = [r for r in results if r.sell_signal]
    ready = [r for r in results if r.strict_buy_ready]

    logger.info("Scan done. BUY=%s SELL=%s READY=%s LONG=%s", len(buys), len(sells), len(ready), sum(1 for r in results if r.state == "LONG"))

    if buys:
        await send_telegram(session, format_buy_message(buys))
    if SEND_SELL_ALERTS and sells:
        await send_telegram(session, format_sell_message(sells))

    save_state(state)


async def main() -> None:
    logger.info("Starting Binance STRICT scanner. scan=%s htf=%s top=%s dry_run=%s", SCAN_INTERVAL, HTF_INTERVAL, TOP_N, DRY_RUN)
    state = load_state()
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "strict-crypto-scanner/1.0"}) as session:
        while True:
            start = time.time()
            try:
                await scan_once(session, state)
            except Exception as exc:
                logger.exception("Scan failed: %s", exc)
            elapsed = time.time() - start
            sleep_for = max(5, POLL_SECONDS - elapsed)
            await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
