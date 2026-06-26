import os
import json
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PAPER_STATE_FILE = Path(os.getenv("PAPER_STATE_FILE", "paper_state.json"))
PAPER_TRADES_FILE = Path(os.getenv("PAPER_TRADES_FILE", "paper_trades.json"))
FUTURES_BASE_URL = os.getenv("BINANCE_MARKET_DATA_URL", "https://fapi.binance.com")


st.set_page_config(
    page_title="Paper Futures Portfolio",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    .metric-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0,0,0,0.22);
    }
    .metric-title { color: #9ca3af; font-size: 13px; }
    .metric-value { color: #f9fafb; font-size: 26px; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def get_mark_price(symbol):
    try:
        r = requests.get(
            f"{FUTURES_BASE_URL}/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=10,
        )
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        return None
    return None


def fmt_money(x):
    try:
        return f"{float(x):,.2f} USDT"
    except Exception:
        return "-"


def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div style="color:#9ca3af;font-size:12px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("🧪 Paper Futures Portfolio")
st.caption("TEST order ve sinyal performansı için 1000 USDT sanal portföy paneli.")

state = read_json(PAPER_STATE_FILE, {
    "initial_equity": 1000.0,
    "cash": 1000.0,
    "realized_pnl": 0.0,
    "open_positions": {},
})
trades = read_json(PAPER_TRADES_FILE, [])

open_positions = state.get("open_positions", {})

mark_prices = {}
for symbol in open_positions.keys():
    mark_prices[symbol] = get_mark_price(symbol)

unrealized = 0.0
open_rows = []

for symbol, t in open_positions.items():
    mark = mark_prices.get(symbol) or float(t.get("entry_price", 0))
    entry = float(t.get("entry_price", 0))
    qty = float(t.get("qty", 0))
    side = t.get("side", "")

    if side == "LONG":
        upnl = (mark - entry) * qty
    else:
        upnl = (entry - mark) * qty

    margin = float(t.get("margin_used", 0))
    upnl_pct = (upnl / margin * 100) if margin else 0.0
    unrealized += upnl

    open_rows.append({
        "ID": t.get("id"),
        "Symbol": symbol,
        "Side": side,
        "Open Time": t.get("open_time"),
        "Entry": entry,
        "Mark": mark,
        "Leverage": t.get("leverage"),
        "Qty": qty,
        "Notional": t.get("notional_usdt"),
        "Margin": t.get("margin_used"),
        "SL": t.get("sl"),
        "TP1": t.get("tp1"),
        "TP2": t.get("tp2"),
        "TP3": t.get("tp3"),
        "uPnL": upnl,
        "uPnL % Margin": upnl_pct,
        "Order Mode": t.get("order_mode"),
        "Order Status": t.get("order_status"),
    })

initial = float(state.get("initial_equity", 1000))
cash = float(state.get("cash", initial))
realized = float(state.get("realized_pnl", 0.0))
equity = cash + unrealized
total_return = ((equity - initial) / initial * 100) if initial else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    metric_card("Initial Equity", fmt_money(initial))
with c2:
    metric_card("Current Equity", fmt_money(equity), f"{total_return:+.2f}%")
with c3:
    metric_card("Realized PnL", fmt_money(realized))
with c4:
    metric_card("Unrealized PnL", fmt_money(unrealized))
with c5:
    metric_card("Open Positions", str(len(open_positions)))

st.divider()

tab1, tab2, tab3 = st.tabs(["📌 Open Positions", "📜 Trade History", "📈 Equity Curve"])

with tab1:
    st.subheader("Open Paper Positions")

    if open_rows:
        df_open = pd.DataFrame(open_rows)
        st.dataframe(df_open, use_container_width=True, hide_index=True)
    else:
        st.info("Açık paper pozisyon yok.")

with tab2:
    st.subheader("Trade History")

    if trades:
        df = pd.DataFrame(trades)

        preferred = [
            "id", "symbol", "side", "status", "open_time", "close_time",
            "entry_price", "exit_price", "leverage", "qty", "notional_usdt",
            "margin_used", "sl", "tp1", "tp2", "tp3",
            "realized_pnl", "realized_pnl_pct_on_margin",
            "exit_reason", "bull_score", "bear_score", "er", "z",
            "order_mode", "order_status", "close_order_status",
        ]
        cols = [c for c in preferred if c in df.columns]
        df = df[cols].copy()

        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total_rows = len(df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * page_size
        end = start + page_size

        st.caption(f"Showing {start + 1}-{min(end, total_rows)} of {total_rows} trades")
        st.dataframe(df.iloc[start:end], use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="paper_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("Henüz paper trade kaydı yok.")

with tab3:
    st.subheader("Equity Curve")

    closed = [t for t in trades if t.get("status") == "CLOSED"]
    curve_rows = [{"time": "START", "equity": initial, "realized_pnl": 0.0}]

    running = initial
    for t in closed:
        pnl = float(t.get("realized_pnl") or 0)
        running += pnl
        curve_rows.append({
            "time": t.get("close_time"),
            "equity": running,
            "realized_pnl": pnl,
        })

    df_curve = pd.DataFrame(curve_rows)

    if len(df_curve) > 1:
        st.line_chart(df_curve.set_index("time")["equity"])
    else:
        st.info("Equity curve için kapanmış işlem yok. Açık işlemler üstte uPnL olarak görünür.")

    st.dataframe(df_curve, use_container_width=True, hide_index=True)

st.divider()
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
