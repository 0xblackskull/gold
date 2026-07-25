"""
Hyperliquid GOLD-USDC Live Signal Detector + Paper Trader
==========================================================
Strategy: Var I — 3-Candle Breakout + Retest, Longs only
          50% close at 2.0R, trail rest at 1.5*ATR

Modes:
  --mode signal    : Print alerts when setup fires (no trading)
  --mode paper     : Auto-execute with tiny size (0.001 contracts)
  --mode live      : Auto-execute with full 1% risk sizing

Setup:
  pip install requests pandas pandas-ta eth-account

Usage:
  python hl_gold_trader.py --mode signal
  python hl_gold_trader.py --mode paper  --key YOUR_PRIVATE_KEY
  python hl_gold_trader.py --mode live   --key YOUR_PRIVATE_KEY
"""

import time
import json
import hmac
import hashlib
import argparse
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HL_INFO_URL   = "https://api.hyperliquid.xyz/info"
HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
COIN          = "GOLD"
INTERVAL      = "15m"
LOOKBACK_BARS = 300          # bars to fetch for indicator warmup
PAPER_SIZE    = 0.001        # contracts for paper trading (~$4.5)
RISK_PCT      = 1.0          # % equity per trade (live mode)
PARTIAL_R     = 2.0          # close 50% at 2R
TRAIL_MULT    = 1.5          # trail rest at 1.5*ATR
POLL_SECONDS  = 60           # check every 60s (fires on new 15m close)

# ─────────────────────────────────────────────
# 1.  DATA — Hyperliquid candle fetch
# ─────────────────────────────────────────────
def _hl_meta_coins() -> list:
    """Fetch all coin names from Hyperliquid meta endpoint."""
    try:
        r = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=10)
        r.raise_for_status()
        universe = r.json().get("universe", [])
        return [a["name"] for a in universe]
    except Exception:
        return []


def fetch_candles(coin: str = COIN, interval: str = INTERVAL,
                  n_bars: int = LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch last n_bars of OHLCV from Hyperliquid."""
    now_ms     = int(time.time() * 1000)
    ms_per_bar = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                  "1h": 3_600_000, "4h": 14_400_000}
    start_ms   = now_ms - n_bars * ms_per_bar[interval]

    # Hyperliquid uses the coin name from its universe list.
    # GOLD perps may be listed as "GOLD", "XAU", or with a prefix.
    # Try each candidate until one returns data.
    candidates = [coin, "XAU", "GOLD-USDC", "GOLD/USDC"]
    # Also pull from meta to find exact name
    meta_coins = _hl_meta_coins()
    for mc in meta_coins:
        if "GOLD" in mc.upper() or "XAU" in mc.upper():
            if mc not in candidates:
                candidates.insert(0, mc)

    last_error = None
    for candidate in candidates:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin":      candidate,
                "interval":  interval,
                "startTime": start_ms,
                "endTime":   now_ms,
            }
        }
        try:
            r = requests.post(HL_INFO_URL, json=payload, timeout=15)
            if r.status_code == 200:
                raw = r.json()
                if raw and len(raw) > 0:
                    print(f"    [*] Using coin name: '{candidate}'")
                    # Patch global COIN so future calls use the right name
                    global COIN
                    COIN = candidate

                    rows = []
                    for c in raw:
                        ts = pd.to_datetime(c["t"], unit="ms", utc=True).tz_localize(None)
                        rows.append({
                            "timestamp": ts,
                            "Open":   float(c["o"]),
                            "High":   float(c["h"]),
                            "Low":    float(c["l"]),
                            "Close":  float(c["c"]),
                            "Volume": float(c["v"]),
                        })
                    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
                    return df.iloc[:-1]   # drop still-forming bar
            else:
                last_error = f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            last_error = str(e)

    # Nothing worked — print available coins to help debug
    print(f"\n  [!] Could not fetch candles. Last error: {last_error}")
    if meta_coins:
        gold_coins = [c for c in meta_coins if "GOLD" in c.upper() or "XAU" in c.upper()]
        print(f"  [*] Gold-related coins on Hyperliquid: {gold_coins}")
        print(f"  [*] All available coins: {meta_coins[:30]} ...")
    raise RuntimeError("Failed to fetch candles from Hyperliquid")


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
    df["DAILY_EMA50"] = d_ema.reindex(df.index, method="ffill").values

    # Session (UTC)
    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = ((dh >= 8.0) & (dh < 17.0)) | ((dh >= 13.0) & (dh < 22.0))

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  SIGNAL DETECTOR  (stateless — runs on each new bar)
# ─────────────────────────────────────────────
class SignalState:
    """Tracks setup state across bar-by-bar calls."""
    def __init__(self):
        self.state      = "IDLE"   # IDLE / WATCHING / RETEST
        self.range_high = None
        self.range_low  = None
        self.retest_ctr = 0

    def reset(self):
        self.state = "IDLE"
        self.range_high = None
        self.range_low  = None
        self.retest_ctr = 0


def detect_signal(df: pd.DataFrame, state: SignalState) -> Optional[dict]:
    """
    Feed the full updated dataframe each call.
    Returns signal dict if entry triggered on latest closed bar, else None.
    """
    if len(df) < 10:
        return None

    # Latest closed bar is df.iloc[-1]
    last = df.iloc[-1]

    if not last["IN_SESSION"] or not last["ATR_OK"]:
        state.reset()
        return None

    c   = last["Close"]
    atr = last["ATR"]

    def bull(i): return df.iloc[i]["Close"] > df.iloc[i]["Open"]
    def body_ok(i): return df.iloc[i]["BODY"] >= 0.2 * atr
    def progressive(): return (df.iloc[-3]["Close"] > df.iloc[-4]["Close"] and
                               df.iloc[-2]["Close"] > df.iloc[-3]["Close"])

    # ── State machine ──────────────────────
    if state.state == "IDLE":
        if (bull(-4) and body_ok(-4) and
            bull(-3) and body_ok(-3) and
            bull(-2) and body_ok(-2) and
            progressive()):
            state.range_high = max(df.iloc[-4]["High"], df.iloc[-3]["High"], df.iloc[-2]["High"])
            state.range_low  = min(df.iloc[-4]["Low"],  df.iloc[-3]["Low"],  df.iloc[-2]["Low"])
            state.state      = "WATCHING"

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
            # All filters
            if (c > last["EMA50"] and
                c > last["EMA200"] and
                c > last["DAILY_EMA50"] and
                last["RSI"] < 70):

                sl   = rl
                risk = c - sl
                tp1  = c + PARTIAL_R * risk        # 50% exit at 2R
                tp2  = c + 10 * risk               # trail manages final exit

                signal = {
                    "time":       df.index[-1],
                    "entry":      round(c, 3),
                    "sl":         round(sl, 3),
                    "tp1":        round(tp1, 3),    # partial close at 2R
                    "risk":       round(risk, 3),
                    "atr":        round(atr, 3),
                    "rsi":        round(last["RSI"], 1),
                    "ema50":      round(last["EMA50"], 2),
                }
                state.reset()
                return signal
            state.reset()

        elif state.retest_ctr >= 3:
            state.reset()

    return None


# ─────────────────────────────────────────────
# 4.  HYPERLIQUID ORDER EXECUTION
# ─────────────────────────────────────────────
def get_account_equity(address: str) -> float:
    """Fetch USDC equity from Hyperliquid."""
    r = requests.post(HL_INFO_URL,
                      json={"type": "clearinghouseState", "user": address},
                      timeout=10)
    data = r.json()
    return float(data["marginSummary"]["accountValue"])


def place_order(private_key: str, coin: str, is_buy: bool,
                size: float, price: float, sl: float,
                reduce_only: bool = False) -> dict:
    """
    Place a limit order on Hyperliquid with stop loss.
    Uses eth_account to sign the L1 action.
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        print("[!] eth-account not installed: pip install eth-account")
        return {}

    account = Account.from_key(private_key)
    address = account.address

    timestamp = int(time.time() * 1000)

    # Order action
    order_action = {
        "type": "order",
        "orders": [{
            "a":   get_asset_index(coin),
            "b":   is_buy,
            "p":   str(round(price, 2)),
            "s":   str(round(size, 4)),
            "r":   reduce_only,
            "t":   {"limit": {"tif": "Gtc"}},
            "c":   None,
        }],
        "grouping": "na",
    }

    # Sign
    msg    = json.dumps({"action": order_action, "nonce": timestamp,
                         "vaultAddress": None}, separators=(",", ":"))
    sig    = account.sign_message(encode_defunct(text=msg))
    sig_hex = sig.signature.hex()

    payload = {
        "action":       order_action,
        "nonce":        timestamp,
        "signature":    {"r": "0x"+sig_hex[2:66],
                         "s": "0x"+sig_hex[66:130],
                         "v": int(sig_hex[130:132], 16)},
        "vaultAddress": None,
    }

    r = requests.post(HL_EXCHANGE_URL, json=payload, timeout=15)
    return r.json()


def get_asset_index(coin: str) -> int:
    """Get Hyperliquid asset index for a coin."""
    r = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=10)
    meta = r.json()
    for i, asset in enumerate(meta["universe"]):
        if asset["name"] == coin:
            return i
    raise ValueError(f"Coin {coin} not found in Hyperliquid universe")


def calc_position_size(equity: float, entry: float, sl: float,
                       risk_pct: float = RISK_PCT) -> float:
    """Calculate position size for 1% risk."""
    risk_amount   = equity * (risk_pct / 100)
    risk_per_unit = abs(entry - sl)
    if risk_per_unit < 0.001:
        return PAPER_SIZE
    size = risk_amount / risk_per_unit
    return max(0.001, round(size, 3))


# ─────────────────────────────────────────────
# 5.  ALERT FORMATTING
# ─────────────────────────────────────────────
def print_signal(signal: dict, mode: str, size: float = None):
    rr2_return = signal["risk"] * PARTIAL_R
    print(f"\n{'█'*55}")
    print(f"  🟡 GOLD LONG SIGNAL — {signal['time'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'─'*55}")
    print(f"  Entry       : ${signal['entry']:,.3f}")
    print(f"  Stop Loss   : ${signal['sl']:,.3f}  (risk: ${signal['risk']:.2f}/unit)")
    print(f"  TP1 (50%)   : ${signal['tp1']:,.3f}  (+{PARTIAL_R}R — close half here)")
    print(f"  Trail rest  : {TRAIL_MULT}*ATR below highest close after TP1")
    print(f"{'─'*55}")
    print(f"  ATR         : {signal['atr']:.2f}")
    print(f"  RSI         : {signal['rsi']}")
    print(f"  EMA50       : {signal['ema50']:.2f}")
    if size:
        print(f"  Size        : {size} contracts  ({'PAPER' if mode=='paper' else 'LIVE'})")
        print(f"  $ at risk   : ${size * signal['risk']:.2f}")
    print(f"{'█'*55}\n")


def print_status(bar_time, close, state_name):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}]  "
          f"Bar: {bar_time.strftime('%m-%d %H:%M')}  "
          f"Close: ${close:,.2f}  "
          f"Setup: {state_name}", end="\r")


# ─────────────────────────────────────────────
# 6.  POSITION MANAGER (paper + live)
# ─────────────────────────────────────────────
class PositionManager:
    """Tracks open position and manages partial TP + trailing stop."""

    def __init__(self):
        self.in_trade     = False
        self.entry        = None
        self.sl           = None
        self.tp1          = None
        self.risk         = None
        self.size         = None
        self.partial_done = False
        self.highest_c    = None
        self.trail_sl     = None

    def open(self, signal: dict, size: float):
        self.in_trade     = True
        self.entry        = signal["entry"]
        self.sl           = signal["sl"]
        self.tp1          = signal["tp1"]
        self.risk         = signal["risk"]
        self.size         = size
        self.partial_done = False
        self.highest_c    = signal["entry"]
        self.trail_sl     = signal["sl"]
        print(f"\n  [+] Position opened: {size} contracts @ ${self.entry:,.3f}")
        print(f"      SL: ${self.sl:,.3f}  |  TP1: ${self.tp1:,.3f}")

    def update(self, current_price: float, atr: float,
               mode: str, private_key: str = None) -> bool:
        """Returns True if position closed."""
        if not self.in_trade:
            return False

        c = current_price

        # Track highest close
        if c > self.highest_c:
            self.highest_c = c

        # Step 1: partial close at TP1
        if not self.partial_done and c >= self.tp1:
            half = round(self.size / 2, 3)
            print(f"\n  [✓] TP1 HIT — closing 50% ({half} contracts) @ ${c:,.3f}  (+{PARTIAL_R}R)")
            if mode == "live" and private_key:
                place_order(private_key, COIN, False, half, c, self.sl, reduce_only=True)
            self.partial_done = True
            self.trail_sl     = c - TRAIL_MULT * atr
            self.highest_c    = c

        # Step 2: update trailing stop
        if self.partial_done:
            new_sl = self.highest_c - TRAIL_MULT * atr
            if new_sl > self.trail_sl:
                self.trail_sl = new_sl
                print(f"\n  [~] Trail SL moved to ${self.trail_sl:,.3f}")

        # Check stop hit
        stop = self.trail_sl if self.partial_done else self.sl
        if c <= stop:
            pnl = (c - self.entry) * self.size
            print(f"\n  [✗] STOP HIT @ ${c:,.3f}  |  PnL: ${pnl:+,.2f}")
            if mode == "live" and private_key:
                remaining = round(self.size / 2 if self.partial_done else self.size, 3)
                place_order(private_key, COIN, False, remaining, c, 0, reduce_only=True)
            self.in_trade = False
            return True

        return False


# ─────────────────────────────────────────────
# 7.  MAIN LOOP
# ─────────────────────────────────────────────
def run(mode: str = "signal", private_key: str = None):
    print(f"\n{'═'*55}")
    print(f"  Hyperliquid GOLD-USDC  |  15m Signal Bot")
    print(f"  Mode: {mode.upper()}")
    print(f"  Strategy: Var I — 3-Candle Breakout + Retest")
    print(f"  Partial TP: 50% @ {PARTIAL_R}R  |  Trail: {TRAIL_MULT}*ATR")
    print(f"{'═'*55}\n")

    state    = SignalState()
    position = PositionManager()
    last_bar = None

    # Get wallet address if needed
    address = None
    if mode in ("paper", "live") and private_key:
        try:
            from eth_account import Account
            address = Account.from_key(private_key).address
            equity  = get_account_equity(address)
            print(f"  Wallet : {address[:8]}...{address[-6:]}")
            print(f"  Equity : ${equity:,.2f} USDC\n")
        except Exception as e:
            print(f"  [!] Wallet error: {e}")

    print("  Monitoring GOLD-USDC 15m bars... (Ctrl+C to stop)\n")

    while True:
        try:
            # Fetch latest candles
            df = fetch_candles()
            df = compute_indicators(df)

            if df.empty:
                time.sleep(30)
                continue

            current_bar  = df.index[-1]
            current_price = df.iloc[-1]["Close"]
            current_atr   = df.iloc[-1]["ATR"]

            # Print status line
            print_status(current_bar, current_price, state.state)

            # New bar closed — run signal detection
            if last_bar is None or current_bar > last_bar:
                last_bar = current_bar

                # Manage open position first
                if position.in_trade:
                    closed = position.update(current_price, current_atr,
                                             mode, private_key)
                    if closed:
                        state.reset()

                # Look for new signal only if flat
                if not position.in_trade:
                    signal = detect_signal(df, state)

                    if signal:
                        # Determine size
                        if mode == "paper":
                            size = PAPER_SIZE
                        elif mode == "live" and address:
                            equity = get_account_equity(address)
                            size   = calc_position_size(equity,
                                                        signal["entry"],
                                                        signal["sl"])
                        else:
                            size = None

                        print_signal(signal, mode, size)

                        # Execute if paper or live
                        if mode in ("paper", "live") and size and private_key:
                            result = place_order(
                                private_key, COIN, True,
                                size, signal["entry"], signal["sl"]
                            )
                            print(f"  Order result: {result}")
                            position.open(signal, size)

            # Sleep until next bar
            # 15m bars close on :00, :15, :30, :45
            now     = time.time()
            seconds = now % 900           # seconds into current 15m bar
            wait    = max(5, 900 - seconds + 2)   # wait until next bar + 2s buffer
            time.sleep(min(wait, POLL_SECONDS))

        except KeyboardInterrupt:
            print("\n\n  [*] Stopped by user.")
            break
        except Exception as e:
            print(f"\n  [!] Error: {e}")
            time.sleep(30)


# ─────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperliquid GOLD-USDC Live Signal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Signal only (no trading, no key needed)
  python hl_gold_trader.py --mode signal

  # Paper trade (tiny 0.001 contract size, real orders)
  python hl_gold_trader.py --mode paper --key 0xYOUR_PRIVATE_KEY

  # Live trade (full 1% risk sizing)
  python hl_gold_trader.py --mode live --key 0xYOUR_PRIVATE_KEY

Notes:
  - Signal mode works with no wallet/key
  - Paper mode sends real orders at 0.001 contract size (~$4.50 at risk)
  - Live mode uses 1% equity risk per trade
  - Private key is your Hyperliquid wallet private key (never shared)
  - Keep your key in an env variable: --key $HL_KEY
        """
    )
    parser.add_argument("--mode", choices=["signal","paper","live"],
                        default="signal",
                        help="signal=alerts only, paper=tiny orders, live=full size")
    parser.add_argument("--key",  type=str, default=None,
                        help="Hyperliquid wallet private key (for paper/live)")
    args = parser.parse_args()

    if args.mode in ("paper", "live") and not args.key:
        print("[!] --key required for paper/live mode")
        print("    Run signal mode first: python hl_gold_trader.py --mode signal")
        exit(1)

    run(mode=args.mode, private_key=args.key)
