import os
import time
import hmac
import hashlib
from datetime import datetime, timezone

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FUTURES_BASE_URL = "https://fapi.binance.com"


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="Crypto Scanner Pro",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f9fafb;
        font-size: 26px;
        font-weight: 700;
    }
    .ok {
        color: #22c55e;
        font-weight: 700;
    }
    .bad {
        color: #ef4444;
        font-weight: 700;
    }
    .warn {
        color: #f59e0b;
        font-weight: 700;
    }
    .small {
        color: #9ca3af;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# BINANCE HELPERS
# =========================

def sign_params(params: dict) -> dict:
    if not BINANCE_SECRET_KEY:
        raise ValueError("BINANCE_SECRET_KEY missing")

    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        BINANCE_SECRET_KEY.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    params["signature"] = signature
    return params


def futures_get_public(path: str, params: dict | None = None):
    url = FUTURES_BASE_URL + path
    r = requests.get(url, params=params or {}, timeout=20)
    return r.status_code, r.json()


def futures_get_signed(path: str, params: dict | None = None):
    if not BINANCE_API_KEY:
        raise ValueError("BINANCE_API_KEY missing")

    base_params = params.copy() if params else {}
    base_params["timestamp"] = int(time.time() * 1000)
    base_params["recvWindow"] = 5000

    signed_params = sign_params(base_params)

    headers = {
        "X-MBX-APIKEY": BINANCE_API_KEY
    }

    url = FUTURES_BASE_URL + path
    r = requests.get(url, headers=headers, params=signed_params, timeout=20)

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}

    return r.status_code, data


def send_telegram_test():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram token veya chat id eksik."

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "✅ Crypto Scanner Pro VPS dashboard Telegram test başarılı."
    }

    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)


def status_badge(condition: bool, ok_text="Connected", bad_text="Missing"):
    if condition:
        return f"<span class='ok'>● {ok_text}</span>"
    return f"<span class='bad'>● {bad_text}</span>"


def card(title, value, subtitle=None):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="small">{subtitle or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UI HEADER
# =========================

st.title("📊 Crypto Scanner Pro")
st.caption("Windows VPS Binance Futures bağlantı ve risk kontrol paneli")

st.divider()


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Kontroller")

    refresh = st.button("🔄 Yenile", use_container_width=True)

    st.caption("Bu panel emir göndermez. Sadece okuma ve bağlantı testi yapar.")

    if st.button("📨 Telegram Test", use_container_width=True):
        ok, msg = send_telegram_test()
        if ok:
            st.success("Telegram test mesajı gönderildi.")
        else:
            st.error(msg)

    st.divider()

    st.write("VPS zamanı:")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# =========================
# ENV STATUS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Binance API Key</div>
            <div>{status_badge(bool(BINANCE_API_KEY), "OK", "Missing")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Binance Secret</div>
            <div>{status_badge(bool(BINANCE_SECRET_KEY), "OK", "Missing")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Telegram</div>
            <div>{status_badge(bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID), "OK", "Missing")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">OpenAI</div>
            <div>{status_badge(bool(OPENAI_API_KEY), "OK", "Missing")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =========================
# BINANCE PUBLIC TEST
# =========================

st.subheader("1) Binance Public Server Test")

server_status, server_data = futures_get_public("/fapi/v1/time")

if server_status == 200:
    server_time_ms = server_data.get("serverTime")
    server_dt = datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        card("Binance Futures API", "ONLINE", "Public endpoint erişilebilir")

    with col_b:
        card("Server Time UTC", server_dt.strftime("%Y-%m-%d %H:%M:%S"))

    with col_c:
        local_ms = int(time.time() * 1000)
        diff_ms = abs(local_ms - server_time_ms)
        card("Time Difference", f"{diff_ms} ms", "5000 ms altı ideal")
else:
    st.error("Binance public server time alınamadı.")
    st.json(server_data)

st.divider()


# =========================
# SIGNED ACCOUNT TEST
# =========================

st.subheader("2) Binance Futures Account Test")

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    st.warning("BINANCE_API_KEY veya BINANCE_SECRET_KEY eksik. .env dosyasını kontrol et.")
    st.stop()

try:
    account_status, account_data = futures_get_signed("/fapi/v2/account")
except Exception as e:
    st.error(f"Account test hatası: {e}")
    st.stop()

if account_status != 200:
    st.error("Binance Futures account endpoint başarısız.")
    st.code(f"HTTP Status: {account_status}")
    st.json(account_data)

    st.warning(
        "Muhtemel nedenler: API key yanlış, secret yanlış, Futures yetkisi kapalı, "
        "IP whitelist yanlış, futures hesabı aktif değil veya server time farkı yüksek."
    )
    st.stop()


# =========================
# WALLET
# =========================

total_wallet_balance = float(account_data.get("totalWalletBalance", 0))
available_balance = float(account_data.get("availableBalance", 0))
total_unrealized_profit = float(account_data.get("totalUnrealizedProfit", 0))
total_margin_balance = float(account_data.get("totalMarginBalance", 0))

col1, col2, col3, col4 = st.columns(4)

with col1:
    card("Total Wallet Balance", f"{total_wallet_balance:,.2f} USDT")

with col2:
    card("Available Balance", f"{available_balance:,.2f} USDT")

with col3:
    card("Unrealized PnL", f"{total_unrealized_profit:,.2f} USDT")

with col4:
    card("Margin Balance", f"{total_margin_balance:,.2f} USDT")

st.success("Binance Futures signed account connection başarılı.")

st.divider()


# =========================
# ASSETS
# =========================

st.subheader("3) Futures Assets")

assets = account_data.get("assets", [])
asset_rows = []

for a in assets:
    wallet = float(a.get("walletBalance", 0))
    available = float(a.get("availableBalance", 0))
    unrealized = float(a.get("unrealizedProfit", 0))

    if wallet != 0 or available != 0 or unrealized != 0:
        asset_rows.append({
            "asset": a.get("asset"),
            "walletBalance": wallet,
            "availableBalance": available,
            "unrealizedProfit": unrealized,
            "marginBalance": float(a.get("marginBalance", 0)),
        })

if asset_rows:
    df_assets = pd.DataFrame(asset_rows)
    st.dataframe(df_assets, use_container_width=True)
else:
    st.info("Futures asset bakiyesi görünmüyor veya tüm değerler sıfır.")


# =========================
# POSITIONS
# =========================

st.divider()
st.subheader("4) Open Futures Positions")

try:
    pos_status, pos_data = futures_get_signed("/fapi/v2/positionRisk")
except Exception as e:
    st.error(f"Position test hatası: {e}")
    st.stop()

if pos_status != 200:
    st.error("Position endpoint başarısız.")
    st.code(f"HTTP Status: {pos_status}")
    st.json(pos_data)
else:
    open_positions = []

    for p in pos_data:
        position_amt = float(p.get("positionAmt", 0))

        if position_amt != 0:
            entry_price = float(p.get("entryPrice", 0))
            mark_price = float(p.get("markPrice", 0))
            unrealized_profit = float(p.get("unRealizedProfit", 0))
            leverage = p.get("leverage")
            side = "LONG" if position_amt > 0 else "SHORT"

            open_positions.append({
                "symbol": p.get("symbol"),
                "side": side,
                "positionAmt": position_amt,
                "entryPrice": entry_price,
                "markPrice": mark_price,
                "unRealizedProfit": unrealized_profit,
                "leverage": leverage,
                "marginType": p.get("marginType"),
                "liquidationPrice": float(p.get("liquidationPrice", 0)),
            })

    if open_positions:
        df_pos = pd.DataFrame(open_positions)
        st.dataframe(df_pos, use_container_width=True)
    else:
        st.info("Açık futures pozisyonu yok.")


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Bu dashboard sadece bağlantı, bakiye ve pozisyon okuma testi yapar. "
    "Bu sürümde emir gönderme fonksiyonu yoktur."
)
