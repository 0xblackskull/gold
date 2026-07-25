"""
XAUUSD Automated Trading Bot
==============================
Data    : Twelvedata (XAU/USD 15m real-time)
Exchange: Hyperliquid (GOLD-USDC perp, xyz operator)
Strategy: Var I — 3-Candle Breakout + Retest, Longs only
          50% close at 2.0R, trail rest at 1.5*ATR

Modes:
  --mode paper  : Simulates trades locally, no real orders (start here)
  --mode live   : Places real orders on Hyperliquid

Usage:
  python gold_bot.py --mode paper
  python gold_bot.py --mode live --key 0xYOUR_PRIVATE_KEY

Setup:
  pip install requests pandas pandas-ta eth-account
"""

import time
import json
import requests
import argparse
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TWELVEDATA_KEY  = "e2506d037585409aa463b00d3c9783de"
TWELVEDATA_URL  = "https://api.twelvedata.com/time_series"
HL_INFO_URL     = "https://api.hyperliquid.xyz/info"
HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"

SYMBOL          = "XAU/USD"
INTERVAL        = "15min"
LOOKBACK_BARS   = 300

RISK_PCT        = 1.0     # % equity per trade
PARTIAL_R       = 2.0     # close 50% at 2R
TRAIL_MULT      = 1.5     # trail at 1.5*ATR below highest close
PAPER_SIZE      = 0.001   # contracts for live paper (~$4.50)
PAPER_EQUITY    = 10000.0 # virtual equity for paper mode

# ─────────────────────────────────────────────
# 1.  DATA — Twelvedata
# ─────────────────────────────────────────────
def fetch_candles(n_bars: int = LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch latest XAU/USD 15m candles from Twelvedata."""
    params = {
        "symbol":     SYMBOL,
        "interval":   INTERVAL,
        "outputsize": n_bars,
        "apikey":     TWELVEDATA_KEY,
        "format":     "JSON",
        "order":      "ASC",
    }
    r = requests.get(TWELVEDATA_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Twelvedata error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError("No candle data returned from Twelvedata")

    rows = []
    for v in values:
        rows.append({
            "timestamp": pd.to_datetime(v["datetime"]),
            "Open":   float(v["open"]),
            "High":   float(v["high"]),
            "Low":    float(v["low"]),
            "Close":  float(v["close"]),
            "Volume": float(v.get("volume", 0)),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    # Drop last (still-forming) candle
    df = df.iloc[:-1]
    return df


# ─────────────────────────────────────────────
# 2.  INDICATORS
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"]  = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI"]    = ta.rsi(df["Close"], length=14)
    df["ATR"]    = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR_MA"] = df["ATR"].rolling(14).mean()
    df["ATR_OK"] = df["ATR"] > df["ATR_MA"] * 0.8
    df["BODY"]   = (df["Close"] - df["Open"]).abs()

    # Daily EMA50 resampled to 15m
    daily = df["Close"].resample("1D").last().dropna()
    d_ema = ta.ema(daily, length=50)
    if d_ema is None or d_ema.dropna().empty:
        # Not enough daily bars for EMA50 — use EMA200 as fallback
        df["DAILY_EMA50"] = df["EMA200"]
    else:
        df["DAILY_EMA50"] = d_ema.reindex(df.index, method="ffill").values

    # Session filter (UTC)
    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = ((dh >= 8.0) & (dh < 17.0)) | ((dh >= 13.0) & (dh < 22.0))

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  SIGNAL DETECTOR
# ─────────────────────────────────────────────
class SetupState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state      = "IDLE"
        self.range_high = None
        self.range_low  = None
        self.retest_ctr = 0


def detect_signal(df: pd.DataFrame, state: SetupState) -> Optional[dict]:
    if len(df) < 10:
        return None

    last = df.iloc[-1]
    if not last["IN_SESSION"] or not last["ATR_OK"]:
        state.reset()
        return None

    c   = float(last["Close"])
    atr = float(last["ATR"])

    def bull(i):  return df.iloc[i]["Close"] > df.iloc[i]["Open"]
    def body_ok(i): return df.iloc[i]["BODY"] >= 0.2 * atr

    # ── state machine ──────────────────────
    if state.state == "IDLE":
        if (bull(-4) and body_ok(-4) and
            bull(-3) and body_ok(-3) and
            bull(-2) and body_ok(-2) and
            df.iloc[-3]["Close"] > df.iloc[-4]["Close"] and
            df.iloc[-2]["Close"] > df.iloc[-3]["Close"]):
            state.range_high = max(df.iloc[-4]["High"],
                                   df.iloc[-3]["High"],
                                   df.iloc[-2]["High"])
            state.range_low  = min(df.iloc[-4]["Low"],
                                   df.iloc[-3]["Low"],
                                   df.iloc[-2]["Low"])
            state.state = "WATCHING"

    elif state.state == "WATCHING":
        if c > state.range_high + 0.2 * atr:
            state.state      = "RETEST"
            state.retest_ctr = 0
        else:
            state.reset()

    elif state.state == "RETEST":
        state.retest_ctr += 1
        rh = state.range_high
        rl = state.range_low

        touched  = last["Low"] <= rh + 0.3 * atr
        reclosed = c > rh

        if touched and reclosed:
            if (c > last["EMA50"] and
                c > last["EMA200"] and
                c > last["DAILY_EMA50"] and
                last["RSI"] < 70):
                risk = c - rl
                sig  = {
                    "time":  df.index[-1],
                    "entry": round(c, 3),
                    "sl":    round(rl, 3),
                    "tp1":   round(c + PARTIAL_R * risk, 3),
                    "risk":  round(risk, 3),
                    "atr":   round(atr, 3),
                    "rsi":   round(float(last["RSI"]), 1),
                }
                state.reset()
                return sig
            state.reset()
        elif state.retest_ctr >= 3:
            state.reset()

    return None


# ─────────────────────────────────────────────
# 4.  PAPER POSITION MANAGER
# ─────────────────────────────────────────────
class PaperPosition:
    """Simulates position management locally — no real orders."""

    def __init__(self, equity: float = PAPER_EQUITY):
        self.equity       = equity
        self.in_trade     = False
        self.entry        = None
        self.sl           = None
        self.tp1          = None
        self.risk         = None
        self.size         = None
        self.partial_done = False
        self.highest_c    = None
        self.trail_sl     = None
        self.trades       = []   # log of closed trades

    def open(self, signal: dict):
        risk_amount = self.equity * (RISK_PCT / 100)
        self.size   = round(risk_amount / signal["risk"], 4)
        self.in_trade     = True
        self.entry        = signal["entry"]
        self.sl           = signal["sl"]
        self.tp1          = signal["tp1"]
        self.risk         = signal["risk"]
        self.partial_done = False
        self.highest_c    = signal["entry"]
        self.trail_sl     = signal["sl"]

        print(f"\n  📥 PAPER TRADE OPENED")
        print(f"     Entry  : ${self.entry:,.3f}")
        print(f"     SL     : ${self.sl:,.3f}  (risk: ${self.risk:.2f}/unit)")
        print(f"     TP1    : ${self.tp1:,.3f}  (+{PARTIAL_R}R)")
        print(f"     Size   : {self.size:.4f} units")
        print(f"     $ Risk : ${self.size * self.risk:.2f}  "
              f"({RISK_PCT}% of ${self.equity:,.0f})")

    def update(self, price: float, atr: float) -> bool:
        """Returns True if position closed."""
        if not self.in_trade:
            return False

        c = price

        # Track high watermark
        if c > self.highest_c:
            self.highest_c = c

        # Partial close at TP1
        if not self.partial_done and c >= self.tp1:
            half_pnl = (c - self.entry) * (self.size / 2)
            self.equity += half_pnl
            print(f"\n  ✅ PARTIAL TP HIT @ ${c:,.3f}")
            print(f"     Closed 50% — PnL: +${half_pnl:.2f}  "
                  f"Equity: ${self.equity:,.2f}")
            self.partial_done = True
            self.trail_sl     = c - TRAIL_MULT * atr
            self.highest_c    = c

        # Update trail
        if self.partial_done:
            new_sl = self.highest_c - TRAIL_MULT * atr
            if new_sl > self.trail_sl:
                old = self.trail_sl
                self.trail_sl = new_sl
                print(f"  〰  Trail SL: ${old:,.3f} → ${self.trail_sl:,.3f}")

        # Check stop
        stop = self.trail_sl if self.partial_done else self.sl
        if c <= stop:
            remaining = self.size / 2 if self.partial_done else self.size
            pnl       = (c - self.entry) * remaining
            self.equity += pnl

            result = "WIN" if (c > self.entry) else "LOSS"
            emoji  = "🟢" if result == "WIN" else "🔴"
            print(f"\n  {emoji} TRADE CLOSED @ ${c:,.3f}  [{result}]")
            print(f"     PnL      : ${pnl:+,.2f}")
            print(f"     Equity   : ${self.equity:,.2f}")

            self.trades.append({
                "entry":  self.entry,
                "exit":   c,
                "pnl":    round(pnl, 2),
                "result": result,
                "equity": round(self.equity, 2),
            })
            self.in_trade     = False
            self.partial_done = False
            return True

        return False

    def summary(self):
        if not self.trades:
            print("\n  No closed trades yet.")
            return
        total_pnl = sum(t["pnl"] for t in self.trades)
        wins      = [t for t in self.trades if t["result"] == "WIN"]
        print(f"\n{'─'*45}")
        print(f"  Paper Trading Summary")
        print(f"{'─'*45}")
        print(f"  Trades      : {len(self.trades)}")
        print(f"  Win Rate    : {len(wins)/len(self.trades)*100:.1f}%")
        print(f"  Total PnL   : ${total_pnl:+,.2f}")
        print(f"  Final Equity: ${self.equity:,.2f}")
        print(f"  Return      : {(self.equity-PAPER_EQUITY)/PAPER_EQUITY*100:+.2f}%")
        print(f"{'─'*45}")


# ─────────────────────────────────────────────
# 5.  HYPERLIQUID LIVE EXECUTION
# ─────────────────────────────────────────────
def hl_place_order(private_key: str, is_buy: bool, size: float,
                   price: float, sl: float, reduce_only: bool = False) -> dict:
    """Sign and place order on Hyperliquid GOLD-USDC."""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        raise RuntimeError("pip install eth-account")

    account   = Account.from_key(private_key)
    timestamp = int(time.time() * 1000)

    # Get GOLD asset index from xyz operator markets
    # GOLD-USDC on Hyperliquid xyz uses asset index from metaAndAssetCtxs
    asset_idx = _get_gold_asset_index()

    order_action = {
        "type": "order",
        "orders": [{
            "a": asset_idx,
            "b": is_buy,
            "p": str(round(price, 2)),
            "s": str(round(size, 4)),
            "r": reduce_only,
            "t": {"limit": {"tif": "Gtc"}},
            "c": None,
        }],
        "grouping": "na",
    }

    msg     = json.dumps({"action": order_action, "nonce": timestamp,
                          "vaultAddress": None}, separators=(",", ":"))
    signed  = account.sign_message(encode_defunct(text=msg))
    sig_hex = signed.signature.hex()

    payload = {
        "action":    order_action,
        "nonce":     timestamp,
        "signature": {
            "r": "0x" + sig_hex[2:66],
            "s": "0x" + sig_hex[66:130],
            "v": int(sig_hex[130:132], 16),
        },
        "vaultAddress": None,
    }

    r = requests.post(HL_EXCHANGE_URL, json=payload, timeout=15)
    return r.json()


def _get_gold_asset_index() -> int:
    """Find GOLD asset index in Hyperliquid's universe."""
    r = requests.post(HL_INFO_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
    universe = r.json()[0]["universe"]
    for i, asset in enumerate(universe):
        if "GOLD" in asset["name"].upper():
            return i
    raise ValueError("GOLD not found in Hyperliquid universe")


def hl_get_equity(address: str) -> float:
    r = requests.post(HL_INFO_URL,
                      json={"type": "clearinghouseState", "user": address},
                      timeout=10)
    return float(r.json()["marginSummary"]["accountValue"])


# ─────────────────────────────────────────────
# 6.  PRINT HELPERS
# ─────────────────────────────────────────────
def print_signal(sig: dict):
    print(f"\n{'█'*50}")
    print(f"  🟡 LONG SIGNAL — {sig['time'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'─'*50}")
    print(f"  Entry   : ${sig['entry']:,.3f}")
    print(f"  Stop    : ${sig['sl']:,.3f}  (risk: ${sig['risk']:.2f})")
    print(f"  TP1 50% : ${sig['tp1']:,.3f}  (+{PARTIAL_R}R)")
    print(f"  Trail   : {TRAIL_MULT}*ATR after TP1")
    print(f"  ATR     : {sig['atr']:.2f}  |  RSI: {sig['rsi']}")
    print(f"{'█'*50}\n")


def print_status(bar_time, close, state_name, equity, trades):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"  [{now}]  Bar:{bar_time.strftime('%m-%d %H:%M')}  "
          f"${close:,.2f}  Setup:{state_name:<10}  "
          f"Equity:${equity:,.0f}  Trades:{trades}", end="\r")


# ─────────────────────────────────────────────
# 7.  MAIN LOOP
# ─────────────────────────────────────────────
def run(mode: str = "paper", private_key: str = None):
    print(f"\n{'═'*55}")
    print(f"  XAUUSD Gold Bot  |  Mode: {mode.upper()}")
    print(f"  Data: Twelvedata (XAU/USD 15m)")
    print(f"  Exchange: Hyperliquid GOLD-USDC")
    print(f"  Strategy: Var I — Partial {PARTIAL_R}R + Trail {TRAIL_MULT}*ATR")
    print(f"{'═'*55}\n")

    state    = SetupState()
    position = PaperPosition(equity=PAPER_EQUITY)
    last_bar = None

    # Live mode setup
    address = None
    if mode == "live":
        if not private_key:
            print("[!] --key required for live mode")
            return
        from eth_account import Account
        address = Account.from_key(private_key).address
        live_equity = hl_get_equity(address)
        print(f"  Wallet  : {address[:8]}...{address[-6:]}")
        print(f"  Equity  : ${live_equity:,.2f} USDC\n")

    print(f"  Watching XAU/USD 15m... (Ctrl+C to stop)\n")
    print(f"  [time]      Bar            Price      Setup       Equity   Trades")
    print(f"  {'─'*65}")

    while True:
        try:
            df   = fetch_candles()
            df   = compute_indicators(df)

            if df.empty:
                time.sleep(30)
                continue

            current_bar   = df.index[-1]
            current_price = float(df.iloc[-1]["Close"])
            current_atr   = float(df.iloc[-1]["ATR"])

            # Status line
            print_status(current_bar, current_price, state.state,
                         position.equity, len(position.trades))

            # New bar — run logic
            if last_bar is None or current_bar > last_bar:
                last_bar = current_bar

                # Manage open position
                if position.in_trade:
                    position.update(current_price, current_atr)

                # Look for signal only when flat
                if not position.in_trade:
                    signal = detect_signal(df, state)

                    if signal:
                        print_signal(signal)

                        if mode == "paper":
                            position.open(signal)

                        elif mode == "live" and address:
                            live_equity = hl_get_equity(address)
                            risk_amt    = live_equity * (RISK_PCT / 100)
                            size        = max(PAPER_SIZE,
                                             round(risk_amt / signal["risk"], 4))
                            print(f"  Placing LIVE order: {size} contracts @ ${signal['entry']}")
                            result = hl_place_order(
                                private_key, True, size,
                                signal["entry"], signal["sl"]
                            )
                            print(f"  Order result: {result}")

            # Sleep until next 15m bar close
            now_s   = time.time()
            elapsed = now_s % 900
            wait    = max(10, int(900 - elapsed + 3))
            time.sleep(min(wait, 60))

        except KeyboardInterrupt:
            print(f"\n\n  Stopped by user.\n")
            position.summary()
            break
        except Exception as e:
            print(f"\n  [!] Error: {e}")
            time.sleep(30)


# ─────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAUUSD Gold Bot — Twelvedata + Hyperliquid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  paper  — Simulates trades locally with virtual $10,000 (no real orders)
  live   — Places real orders on Hyperliquid GOLD-USDC

Examples:
  python gold_bot.py --mode paper
  python gold_bot.py --mode live --key 0xYOUR_PRIVATE_KEY

Paper mode tracks:
  - Every signal fired
  - Virtual PnL per trade
  - Running equity curve
  - Win rate and summary on exit (Ctrl+C)

Run paper mode for 2-4 weeks before switching to live.
        """
    )
    parser.add_argument("--mode", choices=["paper", "live"],
                        default="paper",
                        help="paper=simulation, live=real orders")
    parser.add_argument("--key",  type=str, default=None,
                        help="Hyperliquid private key (live mode only)")
    args = parser.parse_args()

    run(mode=args.mode, private_key=args.key)
