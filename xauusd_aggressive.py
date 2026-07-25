"""
XAUUSD 15m Aggressive Strategy
================================
Target: 30-50% annual return
Changes from Var I (conservative baseline):
  1. Leverage 2x and 3x variants
  2. Shorts re-enabled (both directions)
  3. Trailing stop on ALL trades (not just after partial)
  4. Risk per trade: 2% (was 1%)
  5. Partial TP at 1.5R (earlier lock-in to support higher frequency)

Variations:
  BASE — Var I baseline (1% risk, longs only, partial+trail)
  AGG1 — 2% risk, both directions, 2x leverage, trail all
  AGG2 — 2% risk, both directions, 3x leverage, trail all
  AGG3 — 3% risk, both directions, 2x leverage, trail all
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
def load_data(csv_path=None):
    if csv_path:
        print(f"[*] Loading {csv_path} ...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
        return df
    raise RuntimeError("Pass --csv path")


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
    daily         = df["Close"].resample("1D").last().dropna()
    d_ema         = ta.ema(daily, length=50)
    df["DAILY_EMA50"] = d_ema.reindex(df.index, method="ffill").values
    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = ((dh >= 8.0) & (dh < 17.0)) | ((dh >= 13.0) & (dh < 22.0))
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  STRATEGY FACTORY
# ─────────────────────────────────────────────
def make_strategy(risk_pct: float, leverage: float,
                  both_directions: bool, partial_r: float = 1.5,
                  trail_mult: float = 1.5):

    class AggressiveBreakout(Strategy):
        _risk_pct        = risk_pct
        _leverage        = leverage
        _both_dir        = both_directions
        _partial_r       = partial_r
        _trail_mult      = trail_mult

        min_body_ratio   = 0.2
        breakout_ratio   = 0.2
        retest_wick      = 0.3
        max_retest_bars  = 3
        rsi_long_max     = 70
        rsi_short_min    = 30

        def init(self):
            self.ema50      = self.I(lambda x: x, self.data.EMA50)
            self.ema200     = self.I(lambda x: x, self.data.EMA200)
            self.daily_ema  = self.I(lambda x: x, self.data.DAILY_EMA50)
            self.rsi        = self.I(lambda x: x, self.data.RSI)
            self.atr        = self.I(lambda x: x, self.data.ATR)
            self.atr_ok     = self.I(lambda x: x, self.data.ATR_OK)
            self.in_session = self.I(lambda x: x, self.data.IN_SESSION)
            self.body       = self.I(lambda x: x, self.data.BODY)

            # Long state
            self._ls = "IDLE"; self._lrh = np.nan
            self._lrl = np.nan; self._lrc = 0
            # Short state
            self._ss = "IDLE"; self._srh = np.nan
            self._srl = np.nan; self._src = 0
            # Position tracking
            self._entry = np.nan; self._risk = np.nan
            self._partial_done = False
            self._highest = np.nan; self._lowest = np.nan
            self._trail_sl = np.nan
            self._direction = None   # 'long' or 'short'

        def _bull(self, i): return self.data.Close[i] > self.data.Open[i]
        def _bear(self, i): return self.data.Close[i] < self.data.Open[i]
        def _body_ok(self, i):
            return self.body[i] >= self.min_body_ratio * self.atr[-1]

        def _size(self, entry, sl):
            risk_amount   = self.equity * (self._risk_pct / 100)
            risk_per_unit = abs(entry - sl)
            if risk_per_unit < 1e-8: return 0.01
            # Apply leverage to position size
            raw  = (risk_amount / risk_per_unit) * entry / self.equity
            size = raw * self._leverage
            return max(0.01, min(0.99, round(size, 4)))

        def next(self):
            i = len(self.data.Close) - 1
            if i < 10: return

            if not bool(self.in_session[-1]) or not bool(self.atr_ok[-1]):
                self._ls = "IDLE"; self._ss = "IDLE"
                return

            c   = self.data.Close[-1]
            atr = self.atr[-1]

            # ── Manage open position ──────────────
            if self.position:
                is_long = self.position.is_long
                partial_target = (self._entry + self._partial_r * self._risk
                                  if is_long else
                                  self._entry - self._partial_r * self._risk)

                # Track extremes
                if is_long and c > self._highest:
                    self._highest = c
                if not is_long and c < self._lowest:
                    self._lowest = c

                # Partial close
                if not self._partial_done:
                    if (is_long and c >= partial_target) or \
                       (not is_long and c <= partial_target):
                        self.position.close(portion=0.5)
                        self._partial_done = True
                        if is_long:
                            self._trail_sl = c - self._trail_mult * atr
                        else:
                            self._trail_sl = c + self._trail_mult * atr

                # Update trail
                if self._partial_done:
                    if is_long:
                        new_sl = self._highest - self._trail_mult * atr
                        if new_sl > self._trail_sl:
                            self._trail_sl = new_sl
                        if c <= self._trail_sl:
                            self.position.close()
                            self._partial_done = False
                    else:
                        new_sl = self._lowest + self._trail_mult * atr
                        if new_sl < self._trail_sl:
                            self._trail_sl = new_sl
                        if c >= self._trail_sl:
                            self.position.close()
                            self._partial_done = False
                return

            # ── LONG state machine ────────────────
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
                if c > self._lrh + self.breakout_ratio * atr:
                    self._ls = "RETEST"; self._lrc = 0
                else:
                    self._ls = "IDLE"

            elif self._ls == "RETEST":
                self._lrc += 1
                rh = self._lrh; rl = self._lrl
                if self.data.Low[-1] <= rh + self.retest_wick * atr and c > rh:
                    if (c > self.ema50[-1] and c > self.ema200[-1] and
                        c > self.daily_ema[-1] and self.rsi[-1] < self.rsi_long_max):
                        sl   = rl; risk = c - sl
                        tp   = c + 10 * risk   # trail manages exit
                        size = self._size(c, sl)
                        self.buy(size=size, sl=sl, tp=tp)
                        self._entry = c; self._risk = risk
                        self._partial_done = False
                        self._highest = c; self._trail_sl = sl
                        self._direction = "long"
                    self._ls = "IDLE"
                elif self._lrc >= self.max_retest_bars:
                    self._ls = "IDLE"

            # ── SHORT state machine ───────────────
            if not self._both_dir:
                return

            if self._ss == "IDLE":
                if (self._bear(-4) and self._body_ok(-4) and
                    self._bear(-3) and self._body_ok(-3) and
                    self._bear(-2) and self._body_ok(-2) and
                    self.data.Close[-3] < self.data.Close[-4] and
                    self.data.Close[-2] < self.data.Close[-3]):
                    self._srh = max(self.data.High[-4], self.data.High[-3], self.data.High[-2])
                    self._srl = min(self.data.Low[-4],  self.data.Low[-3],  self.data.Low[-2])
                    self._ss  = "WATCHING"

            elif self._ss == "WATCHING":
                if c < self._srl - self.breakout_ratio * atr:
                    self._ss = "RETEST"; self._src = 0
                else:
                    self._ss = "IDLE"

            elif self._ss == "RETEST":
                self._src += 1
                rh = self._srh; rl = self._srl
                if self.data.High[-1] >= rl - self.retest_wick * atr and c < rl:
                    if (c < self.ema50[-1] and c < self.ema200[-1] and
                        c < self.daily_ema[-1] and self.rsi[-1] > self.rsi_short_min):
                        sl   = rh; risk = sl - c
                        tp   = c - 10 * risk
                        size = self._size(c, sl)
                        self.sell(size=size, sl=sl, tp=tp)
                        self._entry = c; self._risk = risk
                        self._partial_done = False
                        self._lowest = c; self._trail_sl = sl
                        self._direction = "short"
                    self._ss = "IDLE"
                elif self._src >= self.max_retest_bars:
                    self._ss = "IDLE"

    label = f"R{risk_pct}%_L{leverage}x_{'Both' if both_directions else 'Long'}"
    AggressiveBreakout.__name__ = label
    return AggressiveBreakout


# ─────────────────────────────────────────────
# 4.  VARIATIONS
# ─────────────────────────────────────────────
VARIATIONS = [
    {"label": "BASE  (1% risk, 1x, Long only)",   "risk": 1.0, "lev": 1.0, "both": False},
    {"label": "AGG1  (2% risk, 2x, Both dirs)",   "risk": 2.0, "lev": 2.0, "both": True},
    {"label": "AGG2  (2% risk, 3x, Both dirs)",   "risk": 2.0, "lev": 3.0, "both": True},
    {"label": "AGG3  (3% risk, 2x, Both dirs)",   "risk": 3.0, "lev": 2.0, "both": True},
]

def run_all(df):
    results = []
    for v in VARIATIONS:
        strat = make_strategy(v["risk"], v["lev"], v["both"])
        bt    = Backtest(df, strat, cash=100_000, commission=0.0002,
                         exclusive_orders=True, trade_on_close=False)
        stats = bt.run()
        results.append({"label": v["label"], "stats": stats})
        pf = stats["Profit Factor"]
        pf_str = f"{pf:.2f}" if not (np.isnan(pf) or np.isinf(pf)) else "  ∞"
        print(f"  {v['label']:<42}  "
              f"Trades:{stats['# Trades']:>4}  "
              f"WR:{stats['Win Rate [%]']:>5.1f}%  "
              f"PF:{pf_str:>5}  "
              f"Sharpe:{stats['Sharpe Ratio']:>7.3f}  "
              f"DD:{stats['Max. Drawdown [%]']:>7.2f}%  "
              f"Ret:{stats['Return [%]']:>7.2f}%")
    return results


# ─────────────────────────────────────────────
# 5.  CHARTS
# ─────────────────────────────────────────────
def plot_results(results, save_path="equity_aggressive.png"):
    DARK="#0D1117"; GRID="#21262D"; TEXT="#E6EDF3"
    COLORS=["#58A6FF","#3FB950","#D29922","#F85149"]

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle("XAUUSD 15m  |  Aggressive Variations — Leverage + Both Directions",
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
        ax1.plot(eq.index, eq.values, color=COLORS[i],
                 linewidth=1.5, label=r["label"].strip())
    ax1.axhline(100_000, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.4)
    style_ax(ax1, "Equity Curves")
    ax1.set_ylabel("Equity ($)", color=TEXT, fontsize=8)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT)

    # 2. Annual returns
    ax2 = fig.add_subplot(gs[1, 0])
    labels = [r["label"].split("(")[0].strip() for r in results]
    rets   = [r["stats"]["Return [%]"] for r in results]
    bars   = ax2.bar(labels, rets,
                     color=["#3FB950" if v >= 0 else "#F85149" for v in rets], alpha=0.85)
    ax2.axhline(0, color=TEXT, linewidth=0.6)
    ax2.axhline(30, color="#D29922", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(50, color="#F85149", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.text(3.4, 31, "30% target", color="#D29922", fontsize=7)
    ax2.text(3.4, 51, "50% target", color="#F85149", fontsize=7)
    for bar, val in zip(bars, rets):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", color=TEXT, fontsize=8)
    style_ax(ax2, "Total Return (3 years)")

    # 3. Risk-adjusted (Sharpe)
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [r["stats"]["Sharpe Ratio"] for r in results]
    bars2   = ax3.bar(labels, sharpes,
                      color=["#3FB950" if v >= 0 else "#F85149" for v in sharpes], alpha=0.85)
    ax3.axhline(0, color=TEXT, linewidth=0.6)
    ax3.axhline(1.0, color="#D29922", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, val in zip(bars2, sharpes):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", color=TEXT, fontsize=8)
    style_ax(ax3, "Sharpe Ratio  (>1.0 = good)")

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(results):
        eq = r["stats"]["_equity_curve"]["Equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax4.plot(dd.index, dd.values, color=COLORS[i],
                 linewidth=1.2, label=r["label"].strip(), alpha=0.8)
    ax4.axhline(0, color=TEXT, linewidth=0.4)
    ax4.axhline(-20, color="#F85149", linewidth=0.8, linestyle="--", alpha=0.5)
    style_ax(ax4, "Drawdown %  (red line = -20% danger zone)")
    ax4.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)

    # 5. Summary table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK); ax5.axis("off")
    ax5.set_title("Full Comparison", color=TEXT, fontsize=9, pad=6)
    headers = ["Variation", "Trades", "WR%", "Sharpe", "Max DD%", "3yr Ret%", "~Ann%"]
    col_x   = [0.01, 0.18, 0.30, 0.42, 0.55, 0.70, 0.85]
    y_start = 0.88
    for j, h in enumerate(headers):
        ax5.text(col_x[j], y_start+0.06, h, transform=ax5.transAxes,
                 color="#8B949E", fontsize=7.5, fontweight="bold")
    for i, r in enumerate(results):
        s   = r["stats"]; y = y_start - i * 0.18
        lbl = r["label"].split("(")[0].strip()
        ann = s["Return [%]"] / 3   # rough annual
        row = [lbl,
               f"{s['# Trades']:.0f}",
               f"{s['Win Rate [%]']:.1f}",
               f"{s['Sharpe Ratio']:.3f}",
               f"{s['Max. Drawdown [%]']:.1f}",
               f"{s['Return [%]']:.1f}",
               f"{ann:.1f}"]
        color = "#3FB950" if s["Return [%]"] >= 0 else "#F85149"
        for j, val in enumerate(row):
            ax5.text(col_x[j], y, val, transform=ax5.transAxes,
                     color=color if j >= 1 else "#D29922", fontsize=7.5)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"[+] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 6.  REPORT
# ─────────────────────────────────────────────
def print_best(results):
    best = max(results, key=lambda r: r["stats"]["Sharpe Ratio"])
    s    = best["stats"]
    SEP  = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  BEST: {best['label'].strip()}")
    print(f"{'═'*55}")
    rows = [
        ("Period",        "2023–2025 (3 years)"),
        ("Final Equity",  f"${s['Equity Final [$]']:,.2f}"),
        ("Total Return",  f"{s['Return [%]']:.2f}%"),
        ("Annual Return", f"~{s['Return [%]']/3:.1f}% per year"),
        ("Buy & Hold",    f"{s['Buy & Hold Return [%]']:.2f}%"),
        (SEP, ""),
        ("Total Trades",  f"{s['# Trades']:.0f}  (~{s['# Trades']/3:.0f}/year)"),
        ("Win Rate",      f"{s['Win Rate [%]']:.1f}%"),
        ("Profit Factor", f"{s['Profit Factor']:.2f}"),
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
        out.to_csv("trade_log_aggressive.csv", index=False)
        print(f"\n[+] Trade log → trade_log_aggressive.csv")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAUUSD Aggressive Backtest")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    raw = load_data(csv_path=args.csv)
    df  = add_indicators(raw)

    print(f"\n[*] Bars: {len(df):,}  |  "
          f"Session: {df['IN_SESSION'].sum():,}  |  "
          f"ATR-OK: {df['ATR_OK'].sum():,}\n")
    print(f"{'─'*100}")
    print(f"  {'Variation':<44} {'Trades':>6}  {'WR':>6}  {'PF':>6}  "
          f"{'Sharpe':>8}  {'MaxDD':>8}  {'Return':>8}")
    print(f"{'─'*100}")

    results = run_all(df)
    print(f"{'─'*100}")
    print_best(results)
    plot_results(results)
