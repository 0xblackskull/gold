"""
Dukascopy XAUUSD 15m Downloader
================================
Downloads tick data directly from Dukascopy's public API,
resamples to 15m OHLCV, and saves as xauusd_15m.csv

Usage:
    python download_xauusd.py
    python download_xauusd.py --start 2022-01-01 --end 2024-12-31
    python download_xauusd.py --start 2023-01-01 --end 2024-12-31  # faster, 2yr
"""

import struct
import lzma
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Dukascopy constants ──────────────────────────────────────────────────────
BASE_URL   = "https://datafeed.dukascopy.com/datafeed"
INSTRUMENT = "XAUUSD"
POINT      = 0.001      # XAU pip size (prices stored as integer * POINT)
TICK_SIZE  = 20         # bytes per tick record

# ── Download one hour of raw tick data ──────────────────────────────────────
def fetch_hour(instrument: str, dt: datetime) -> bytes | None:
    url = (f"{BASE_URL}/{instrument}/"
           f"{dt.year}/{dt.month-1:02d}/{dt.day:02d}/"
           f"{dt.hour:02d}h_ticks.bi5")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 0:
            return r.content
        return None
    except Exception:
        return None


# ── Decompress & parse one hour of ticks ────────────────────────────────────
def parse_ticks(raw: bytes, hour_dt: datetime) -> pd.DataFrame:
    try:
        data = lzma.decompress(raw)
    except Exception:
        return pd.DataFrame()

    n = len(data) // TICK_SIZE
    if n == 0:
        return pd.DataFrame()

    rows = []
    for i in range(n):
        chunk = data[i*TICK_SIZE:(i+1)*TICK_SIZE]
        ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)
        ts  = hour_dt + timedelta(milliseconds=ms_offset)
        mid = (ask_raw + bid_raw) / 2.0 * POINT
        rows.append({"timestamp": ts, "price": mid,
                     "volume": (ask_vol + bid_vol) / 2.0})

    return pd.DataFrame(rows).set_index("timestamp")


# ── Resample ticks → 15m OHLCV ──────────────────────────────────────────────
def resample_15m(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()
    ohlcv = ticks["price"].resample("15min").ohlc()
    ohlcv["Volume"] = ticks["volume"].resample("15min").sum()
    ohlcv.columns = ["Open","High","Low","Close","Volume"]
    return ohlcv.dropna()


# ── Main download loop ───────────────────────────────────────────────────────
def download(start: str, end: str, out_path: str = "xauusd_15m.csv"):
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    current  = start_dt
    all_ticks: list[pd.DataFrame] = []
    total_hours = int((end_dt - start_dt).total_seconds() // 3600)
    done = 0

    print(f"[*] Downloading XAUUSD 15m  {start} → {end}")
    print(f"    Total hours to fetch: {total_hours:,}  (this may take 5-15 min)")

    while current < end_dt:
        raw = fetch_hour(INSTRUMENT, current)
        if raw:
            ticks = parse_ticks(raw, current)
            if not ticks.empty:
                all_ticks.append(ticks)

        done += 1
        if done % 100 == 0:
            pct = done / total_hours * 100
            print(f"    {pct:.1f}%  ({current.strftime('%Y-%m-%d')})", end="\r")

        current += timedelta(hours=1)

    print(f"\n[*] Downloaded {done:,} hours, assembling OHLCV ...")

    if not all_ticks:
        print("[!] No data received. Check your internet connection.")
        return

    ticks_df = pd.concat(all_ticks)
    ticks_df.sort_index(inplace=True)

    ohlcv = resample_15m(ticks_df)
    # Strip timezone for backtesting.py compatibility
    ohlcv.index = ohlcv.index.tz_localize(None)

    ohlcv.to_csv(out_path)
    print(f"[+] Saved {len(ohlcv):,} bars → {out_path}")
    print(f"    Date range: {ohlcv.index[0]}  →  {ohlcv.index[-1]}")
    print(f"\n    Run backtest with:")
    print(f"    python xauusd_strategy_v3.py --csv {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download XAUUSD 15m from Dukascopy")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out",   default="xauusd_15m.csv", help="Output CSV filename")
    args = parser.parse_args()

    download(args.start, args.end, args.out)
