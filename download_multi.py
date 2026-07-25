"""
Multi-instrument Dukascopy downloader
Downloads XAGUSD, BTCUSD, EURUSD 15m data (2023-2025)
Usage: python download_multi.py
"""

import struct, lzma, time, requests, pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL  = "https://datafeed.dukascopy.com/datafeed"
TICK_SIZE = 20

INSTRUMENTS = {
    "XAGUSD": {"point": 0.0001, "file": "xagusd_15m.csv"},
    "EURUSD": {"point": 0.00001,"file": "eurusd_15m.csv"},
    "BTCUSD": {"point": 0.01,   "file": "btcusd_15m.csv"},
}

def fetch_hour(instrument, dt):
    url = (f"{BASE_URL}/{instrument}/"
           f"{dt.year}/{dt.month-1:02d}/{dt.day:02d}/"
           f"{dt.hour:02d}h_ticks.bi5")
    try:
        r = requests.get(url, timeout=20)
        return r.content if r.status_code == 200 and r.content else None
    except Exception:
        return None

def parse_ticks(raw, hour_dt, point):
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
        mid = (ask_raw + bid_raw) / 2.0 * point
        rows.append({"timestamp": ts, "price": mid,
                     "volume": (ask_vol + bid_vol) / 2.0})
    return pd.DataFrame(rows).set_index("timestamp")

def resample_15m(ticks):
    if ticks.empty:
        return pd.DataFrame()
    ohlcv = ticks["price"].resample("15min").ohlc()
    ohlcv["Volume"] = ticks["volume"].resample("15min").sum()
    ohlcv.columns = ["Open","High","Low","Close","Volume"]
    return ohlcv.dropna()

def download(instrument, point, out_file,
             start="2023-01-01", end="2025-12-31"):
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total    = int((end_dt - start_dt).total_seconds() // 3600)
    current  = start_dt
    all_ticks = []
    done = 0

    print(f"\n[*] {instrument}  {start} → {end}  ({total:,} hours)")
    while current < end_dt:
        raw = fetch_hour(instrument, current)
        if raw:
            ticks = parse_ticks(raw, current, point)
            if not ticks.empty:
                all_ticks.append(ticks)
        done += 1
        if done % 200 == 0:
            print(f"    {done/total*100:.1f}%  {current.strftime('%Y-%m-%d')}", end="\r")
        current += timedelta(hours=1)

    print(f"\n    Assembling {len(all_ticks)} chunks ...")
    if not all_ticks:
        print(f"    [!] No data for {instrument}")
        return
    ticks_df = pd.concat(all_ticks).sort_index()
    ohlcv    = resample_15m(ticks_df)
    ohlcv.index = ohlcv.index.tz_localize(None)
    ohlcv.to_csv(out_file)
    print(f"    [+] {len(ohlcv):,} bars → {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", nargs="+",
                        default=list(INSTRUMENTS.keys()),
                        help="Which instruments to download")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    args = parser.parse_args()

    for sym in args.instruments:
        if sym not in INSTRUMENTS:
            print(f"[!] Unknown instrument: {sym}")
            continue
        cfg = INSTRUMENTS[sym]
        download(sym, cfg["point"], cfg["file"],
                 args.start, args.end)
    print("\n[+] All done. Run backtest with:")
    print("    python multi_backtest.py --gold xauusd_combined.csv "
          "--silver xagusd_15m.csv --btc btcusd_15m.csv --eur eurusd_15m.csv")
