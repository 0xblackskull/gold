"""
XAUUSD 15m Backtesting Strategy  v5
=====================================
Built on Var C (RR=3, Longs only) — the best v4 variation.

New improvements:
  1. MACD crossover confirmation — entry bar must have MACD line crossing above signal
     (confirms momentum at the exact retest candle, not just trend direction)
  2. Trailing stop — instead of fixed TP at 3R, use a trailing stop that locks in
     profits as price moves in our favour. Still has a hard stop at range_low.

Variations tested:
  C  — baseline (RR=3, Long, fixed TP)       ← v4 winner, re-run for fair comparison
  D  — MACD filter added
  E  — Trailing stop (atr_trail_mult=1.5)
  F  — MACD + Trailing stop combined
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yfinance as yf
from backtesting import Backtest, Strategy

# ─────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────
def load_data(ticker="GC=F", period="60d", interval="15m", csv_path=None):
    if csv_path:
        print(f"[*] Loading {csv_path} ...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
        return df
    try:
        print(f"[*] Downloading {ticker} ({interval}) ...")
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.dropna(inplace=True); df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
            return df
    except Exception as e:
        print(f"    [!] Download failed: {e}")
    # synthetic fallback
    np.random.seed(42); price=2000.0; prices=[2000.0]; vol=0.0008; trend=0.0
    for _ in range(4999):
        shock=np.random.randn(); vol=np.clip(0.85*vol+0.15*abs(shock)*0.001+0.0005,0.0003,0.003)
        trend=0.98*trend+0.02*np.random.randn()*0.0002
        price=np.clip(price*(1+trend+shock*vol),1800,2400); prices.append(price)
    closes=np.array(prices); idx=pd.date_range("2024-01-02 08:00",periods=5000,freq="15min")
    rows=[]
    for i in range(5000):
        c=closes[i]; bv=abs(np.random.randn())*c*0.0008+c*0.0002
        o=closes[i-1] if i>0 else c; h=max(o,c)+abs(np.random.randn())*bv*0.6
        l=min(o,c)-abs(np.random.randn())*bv*0.6
        rows.append({"Open":round(o,2),"High":round(h,2),"Low":round(l,2),
                     "Close":round(c,2),"Volume":1000})
    df=pd.DataFrame(rows,index=idx)
    df["High"]=df[["Open","High","Close"]].max(axis=1)
    df["Low"]=df[["Open","Low","Close"]].min(axis=1)
    print("[*] Using synthetic fallback data")
    return df


# ─────────────────────────────────────────────
# 2.  INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"]  = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["RSI"]    = ta.rsi(df["Close"], length=14)
    df["ATR"]    = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR_MA"] = df["ATR"].rolling(14).mean()
    df["ATR_OK"] = df["ATR"] > df["ATR_MA"] * 0.8
    df["BODY"]   = (df["Close"] - df["Open"]).abs()

    # MACD (12, 26, 9)
    macd_df      = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"]   = macd_df["MACD_12_26_9"]
    df["MACDS"]  = macd_df["MACDs_12_26_9"]   # signal line
    df["MACDH"]  = macd_df["MACDh_12_26_9"]   # histogram

    # Daily EMA50 resampled to 15m
    daily_close      = df["Close"].resample("1D").last().dropna()
    daily_ema50      = ta.ema(daily_close, length=50)
    df["DAILY_EMA50"] = daily_ema50.reindex(df.index, method="ffill").values

    # Session
    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = ((dh >= 8.0) & (dh < 17.0)) | ((dh >= 13.0) & (dh < 22.0))

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  STRATEGY FACTORY
# ─────────────────────────────────────────────
def make_strategy(use_macd: bool, use_trail: bool, rr: float = 3.0):

    class BreakoutRetest(Strategy):
        _use_macd  = use_macd
        _use_trail = use_trail
        _rr        = rr

        risk_pct           = 1.0
        min_body_atr_ratio = 0.2
        breakout_atr_ratio = 0.2
        retest_wick_ratio  = 0.3
        max_retest_bars    = 3
        rsi_long_max       = 70
        atr_trail_mult     = 1.5   # trailing stop = 1.5 * ATR below highest close

        def init(self):
            self.ema50      = self.I(lambda x: x, self.data.EMA50)
            self.ema200     = self.I(lambda x: x, self.data.EMA200)
            self.daily_ema  = self.I(lambda x: x, self.data.DAILY_EMA50)
            self.rsi        = self.I(lambda x: x, self.data.RSI)
            self.atr        = self.I(lambda x: x, self.data.ATR)
            self.atr_ok     = self.I(lambda x: x, self.data.ATR_OK)
            self.in_session = self.I(lambda x: x, self.data.IN_SESSION)
            self.body       = self.I(lambda x: x, self.data.BODY)
            self.macd       = self.I(lambda x: x, self.data.MACD)
            self.macds      = self.I(lambda x: x, self.data.MACDS)
            self.macdh      = self.I(lambda x: x, self.data.MACDH)

            # state machine
            self._ls = "IDLE"; self._lrh = np.nan; self._lrl = np.nan; self._lrc = 0
            # trailing stop tracking
            self._trail_sl   = np.nan
            self._highest_c  = np.nan

        def _bull(self, i): return self.data.Close[i] > self.data.Open[i]
        def _body_ok(self, i):
            return self.body[i] >= self.min_body_atr_ratio * self.atr[-1]
        def _size(self, entry, sl):
            r = abs(entry - sl)
            if r < 1e-8: return 0.01
            return max(0.01, min(0.99, round(
                (self.equity * (self.risk_pct/100) / r) * entry / self.equity, 4)))

        def _macd_bullish_cross(self):
            """MACD line crossed above signal line on current or previous bar."""
            # Current bar: macd > signal AND previous bar: macd <= signal
            return (self.macd[-1] > self.macds[-1] and
                    self.macd[-2] <= self.macds[-2])

        def next(self):
            i = len(self.data.Close) - 1
            if i < 30: return   # need enough history for MACD warmup

            if not bool(self.in_session[-1]) or not bool(self.atr_ok[-1]):
                self._ls = "IDLE"
                return

            c   = self.data.Close[-1]
            atr = self.atr[-1]

            # ── Trailing stop management ──────────
            if self._use_trail and self.position.is_long:
                # Update highest close seen since entry
                if c > self._highest_c:
                    self._highest_c = c
                    # Move SL up to highest_close - atr_trail_mult * ATR
                    new_sl = self._highest_c - self.atr_trail_mult * atr
                    if new_sl > self._trail_sl:
                        self._trail_sl = new_sl
                        self.position.close() if c <= self._trail_sl else None
                        # Update the order's SL
                        try:
                            for trade in self.trades:
                                if trade.is_long:
                                    trade.sl = max(trade.sl or 0, new_sl)
                        except Exception:
                            pass
                return  # don't look for new entries while in trade

            # ── Long state machine ────────────────
            if not self.position.is_long:

                if self._ls == "IDLE":
                    if (self._bull(-4) and self._body_ok(-4) and
                        self._bull(-3) and self._body_ok(-3) and
                        self._bull(-2) and self._body_ok(-2) and
                        self.data.Close[-3] > self.data.Close[-4] and
                        self.data.Close[-2] > self.data.Close[-3]):
                        self._lrh = max(self.data.High[-4], self.data.High[-3], self.data.High[-2])
                        self._lrl = min(self.data.Low[-4],  self.data.Low[-3],  self.data.Low[-2])
                        self._ls  = "WATCHING"

                elif self._ls == "WATCHING":
                    if c > self._lrh + self.breakout_atr_ratio * atr:
                        self._ls = "RETEST"; self._lrc = 0
                    else:
                        self._ls = "IDLE"

                elif self._ls == "RETEST":
                    self._lrc += 1
                    rh = self._lrh; rl = self._lrl

                    touched  = self.data.Low[-1] <= rh + self.retest_wick_ratio * atr
                    reclosed = c > rh

                    if touched and reclosed:
                        # Base filters
                        filters_ok = (c > self.ema50[-1] and
                                      c > self.ema200[-1] and
                                      c > self.daily_ema[-1] and
                                      self.rsi[-1] < self.rsi_long_max)
                        # MACD filter (optional)
                        macd_ok = self._macd_bullish_cross() if self._use_macd else True

                        if filters_ok and macd_ok:
                            sl = rl
                            if self._use_trail:
                                # With trailing: set initial TP far away, rely on trail
                                tp = c + 10 * (c - sl)   # very wide TP, trail will exit
                                self._trail_sl  = sl
                                self._highest_c = c
                            else:
                                tp = c + self._rr * (c - sl)

                            self.buy(size=self._size(c, sl), sl=sl, tp=tp)
                        self._ls = "IDLE"

                    elif self._lrc >= self.max_retest_bars:
                        self._ls = "IDLE"

    label = ("MACD+" if use_macd else "") + ("Trail" if use_trail else f"RR{rr}")
    BreakoutRetest.__name__ = label
    return BreakoutRetest


# ─────────────────────────────────────────────
# 4.  RUN VARIATIONS
# ─────────────────────────────────────────────
VARIATIONS = [
    {"label": "Var C  (RR=3, No MACD, Fixed TP)",    "macd": False, "trail": False, "rr": 3.0},
    {"label": "Var D  (RR=3, MACD,    Fixed TP)",    "macd": True,  "trail": False, "rr": 3.0},
    {"label": "Var E  (RR=3, No MACD, Trail Stop)",  "macd": False, "trail": True,  "rr": 3.0},
    {"label": "Var F  (RR=3, MACD,    Trail Stop)",  "macd": True,  "trail": True,  "rr": 3.0},
]

def run_all(df):
    results = []
    for v in VARIATIONS:
        strat = make_strategy(v["macd"], v["trail"], v["rr"])
        bt    = Backtest(df, strat, cash=100_000, commission=0.0002,
                         exclusive_orders=True, trade_on_close=False)
        stats = bt.run()
        results.append({"label": v["label"], "stats": stats})
        pf = stats["Profit Factor"]
        pf_str = f"{pf:.2f}" if not np.isnan(pf) and not np.isinf(pf) else " inf"
        print(f"  {v['label']:<42}  "
              f"Trades:{stats['# Trades']:>4}  "
              f"WR:{stats['Win Rate [%]']:>5.1f}%  "
              f"PF:{pf_str:>5}  "
              f"Sharpe:{stats['Sharpe Ratio']:>7.3f}  "
              f"DD:{stats['Max. Drawdown [%]']:>6.2f}%  "
              f"Ret:{stats['Return [%]']:>6.2f}%")
    return results


# ─────────────────────────────────────────────
# 5.  CHARTS
# ─────────────────────────────────────────────
def plot_comparison(results, save_path="equity_comparison_v5.png"):
    DARK="#0D1117"; GRID="#21262D"; TEXT="#E6EDF3"
    COLORS=["#58A6FF","#3FB950","#D29922","#F85149"]

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle("XAUUSD 15m  |  v5: MACD Filter  +  Trailing Stop Comparison",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK); ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_color(GRID)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        ax.grid(color=GRID, linewidth=0.5)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    for i, r in enumerate(results):
        eq = r["stats"]["_equity_curve"]["Equity"]
        ax1.plot(eq.index, eq.values, color=COLORS[i], linewidth=1.5,
                 label=r["label"].strip())
    ax1.axhline(100_000, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.4)
    style_ax(ax1, "Equity Curves — C vs D vs E vs F")
    ax1.set_ylabel("Equity ($)", color=TEXT, fontsize=8)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT)

    # 2. Return %
    ax2 = fig.add_subplot(gs[1, 0])
    labels = [r["label"].split("(")[1].rstrip(")") for r in results]
    rets   = [r["stats"]["Return [%]"] for r in results]
    bars   = ax2.bar(labels, rets,
                     color=["#3FB950" if v >= 0 else "#F85149" for v in rets], alpha=0.85)
    ax2.axhline(0, color=TEXT, linewidth=0.6)
    for bar, val in zip(bars, rets):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.02 if val >= 0 else -0.15),
                 f"{val:.2f}%", ha="center", color=TEXT, fontsize=8)
    style_ax(ax2, "Total Return (%)")

    # 3. Sharpe
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [r["stats"]["Sharpe Ratio"] for r in results]
    bars2   = ax3.bar(labels, sharpes,
                      color=["#3FB950" if v >= 0 else "#F85149" for v in sharpes], alpha=0.85)
    ax3.axhline(0, color=TEXT, linewidth=0.6)
    for bar, val in zip(bars2, sharpes):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.005 if val >= 0 else -0.04),
                 f"{val:.3f}", ha="center", color=TEXT, fontsize=8)
    style_ax(ax3, "Sharpe Ratio")

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(results):
        eq = r["stats"]["_equity_curve"]["Equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax4.plot(dd.index, dd.values, color=COLORS[i], linewidth=1.2,
                 label=r["label"].strip(), alpha=0.8)
    ax4.axhline(0, color=TEXT, linewidth=0.4)
    style_ax(ax4, "Drawdown (%)")
    ax4.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)

    # 5. Summary table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK); ax5.axis("off")
    ax5.set_title("Summary Table", color=TEXT, fontsize=9, pad=6)
    headers = ["Variation", "Trades", "WR%", "PF", "Sharpe", "DD%", "Ret%"]
    col_x   = [0.01, 0.22, 0.36, 0.48, 0.60, 0.74, 0.87]
    y_start = 0.88
    for j, h in enumerate(headers):
        ax5.text(col_x[j], y_start+0.06, h, transform=ax5.transAxes,
                 color="#8B949E", fontsize=7.5, fontweight="bold")
    for i, r in enumerate(results):
        s   = r["stats"]
        y   = y_start - i * 0.18
        lbl = r["label"].split("(")[1].rstrip(")")
        pf  = s["Profit Factor"]
        pf_str = f"{pf:.2f}" if not np.isnan(pf) and not np.isinf(pf) else "∞"
        row = [lbl, f"{s['# Trades']:.0f}", f"{s['Win Rate [%]']:.1f}",
               pf_str, f"{s['Sharpe Ratio']:.3f}",
               f"{s['Max. Drawdown [%]']:.2f}", f"{s['Return [%]']:.2f}"]
        color = "#3FB950" if s["Return [%]"] >= 0 else "#F85149"
        for j, val in enumerate(row):
            ax5.text(col_x[j], y, val, transform=ax5.transAxes,
                     color=color if j >= 1 else "#D29922", fontsize=7.5)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"[+] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 6.  BEST VARIATION REPORT + TRADE LOG
# ─────────────────────────────────────────────
def print_best(results):
    best = max(results, key=lambda r: r["stats"]["Sharpe Ratio"])
    s    = best["stats"]
    SEP  = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  BEST: {best['label'].strip()}")
    print(f"{'═'*55}")
    rows = [
        ("Start",         str(s["Start"])),
        ("End",           str(s["End"])),
        ("Duration",      str(s["Duration"])),
        ("Final Equity",  f"${s['Equity Final [$]']:,.2f}"),
        ("Return",        f"{s['Return [%]']:.2f}%"),
        ("Buy & Hold",    f"{s['Buy & Hold Return [%]']:.2f}%"),
        (SEP, ""),
        ("Total Trades",  f"{s['# Trades']:.0f}"),
        ("Win Rate",      f"{s['Win Rate [%]']:.1f}%"),
        ("Profit Factor", f"{s['Profit Factor']:.2f}" if not np.isinf(s['Profit Factor']) else "∞"),
        ("Avg Trade",     f"{s['Avg. Trade [%]']:.2f}%"),
        ("Best Trade",    f"{s['Best Trade [%]']:.2f}%"),
        ("Worst Trade",   f"{s['Worst Trade [%]']:.2f}%"),
        (SEP, ""),
        ("Max Drawdown",  f"{s['Max. Drawdown [%]']:.2f}%"),
        ("Sharpe Ratio",  f"{s['Sharpe Ratio']:.3f}"),
        ("Sortino Ratio", f"{s['Sortino Ratio']:.3f}"),
        ("SQN",           f"{s['SQN']:.3f}"),
    ]
    for k, v in rows:
        print(f"  {k}" if v == "" else f"  {k:<28} {v}")

    trades = s["_trades"]
    if not trades.empty:
        out = trades[["EntryTime","ExitTime","EntryPrice","ExitPrice",
                       "PnL","ReturnPct","Size","Duration"]].copy()
        out["Direction"] = out["Size"].apply(lambda x: "LONG" if x > 0 else "SHORT")
        out.to_csv("trade_log_v5_best.csv", index=False)
        print(f"\n[+] Trade log → trade_log_v5_best.csv  ({len(out)} trades)")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAUUSD 15m Backtest v5")
    parser.add_argument("--csv",    type=str, default=None)
    parser.add_argument("--period", type=str, default="60d")
    args = parser.parse_args()

    raw = load_data(ticker="GC=F", period=args.period,
                    interval="15m", csv_path=args.csv)
    df  = add_indicators(raw)

    print(f"\n[*] Bars: {len(df):,}  |  "
          f"Session: {df['IN_SESSION'].sum():,}  |  "
          f"ATR-OK: {df['ATR_OK'].sum():,}\n")
    print(f"{'─'*92}")
    print(f"  {'Variation':<44} {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
          f"{'Sharpe':>8}  {'DD':>8}  {'Return':>8}")
    print(f"{'─'*92}")

    results = run_all(df)

    print(f"{'─'*92}")
    print_best(results)
    plot_comparison(results)
