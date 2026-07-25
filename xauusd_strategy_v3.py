"""
XAUUSD 15m Backtesting Strategy  v2
=====================================
Strategy: 3-Candle Consolidation Breakout + Retest
Improvements over v1:
  - Minimum candle body size filter (body > 0.3 * ATR per candle)
  - 3-candle range must be directional (no overlap with prior candle)
  - ATR threshold raised to 0.8x ATR_MA (kills choppy/ranging markets)
  - Breakout candle must close > 0.5 ATR above range_high (strong breakout)
  - EMA200 added as macro trend filter (long above, short below)
  - RSI filter: avoid overbought longs / oversold shorts at entry
  - Retest tolerance band: low can dip up to 0.3 ATR below range_high (wicks ok)
  - Max retest bars tightened to 3 (stale setups killed faster)
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
# 1.  DATA LOADING
# ─────────────────────────────────────────────
def _generate_synthetic_xauusd(n_bars=5000, seed=42):
    np.random.seed(seed)
    price = 2000.0
    prices = [price]
    vol = 0.0008
    trend = 0.0
    for _ in range(n_bars - 1):
        shock = np.random.randn()
        vol = 0.85 * vol + 0.15 * abs(shock) * 0.0010 + 0.0005
        vol = np.clip(vol, 0.0003, 0.003)
        trend = 0.98 * trend + 0.02 * np.random.randn() * 0.0002
        price = price * (1 + trend + shock * vol)
        price = max(1800, min(2400, price))
        prices.append(price)

    closes = np.array(prices)
    start  = pd.Timestamp("2024-01-02 08:00:00")
    idx    = pd.date_range(start, periods=n_bars, freq="15min")
    rows   = []
    for i in range(n_bars):
        c  = closes[i]
        bv = abs(np.random.randn()) * c * 0.0008 + c * 0.0002
        o  = closes[i-1] if i > 0 else c * (1 + np.random.randn() * 0.0003)
        h  = max(o, c) + abs(np.random.randn()) * bv * 0.6
        l  = min(o, c) - abs(np.random.randn()) * bv * 0.6
        rows.append({"Open": round(o,2), "High": round(h,2),
                     "Low":  round(l,2), "Close": round(c,2),
                     "Volume": int(np.random.randint(100, 5000))})
    df = pd.DataFrame(rows, index=idx)
    df["High"] = df[["Open","High","Close"]].max(axis=1)
    df["Low"]  = df[["Open","Low","Close"]].min(axis=1)
    return df


def load_data(ticker="GC=F", period="60d", interval="15m", csv_path=None):
    if csv_path:
        print(f"[*] Loading data from {csv_path} ...")
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
            df.dropna(inplace=True)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
            return df
    except Exception as e:
        print(f"    [!] Live download failed: {e}")

    print("[*] Using synthetic XAUUSD data (5,000 bars) ...")
    df = _generate_synthetic_xauusd(n_bars=5000)
    print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# 2.  INDICATORS  (all pre-computed, zero lookahead)
# ─────────────────────────────────────────────
def add_indicators(df: pd.DataFrame,
                   atr_period=14, ema_fast=50, ema_slow=200,
                   rsi_period=14, atr_multiplier=0.8) -> pd.DataFrame:
    df = df.copy()

    # Trend
    df["EMA50"]  = ta.ema(df["Close"], length=ema_fast)
    df["EMA200"] = ta.ema(df["Close"], length=ema_slow)

    # Momentum
    df["RSI"] = ta.rsi(df["Close"], length=rsi_period)

    # Volatility
    df["ATR"]    = ta.atr(df["High"], df["Low"], df["Close"], length=atr_period)
    df["ATR_MA"] = df["ATR"].rolling(atr_period).mean()
    # Raised threshold: 0.8x instead of 0.5x → only trade genuinely volatile bars
    df["ATR_OK"] = df["ATR"] > df["ATR_MA"] * atr_multiplier

    # Session (UTC)
    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = (
        ((dh >= 8.0)  & (dh < 17.0)) |   # London
        ((dh >= 13.0) & (dh < 22.0))      # New York
    )

    # Candle body size (for body filter)
    df["BODY"] = (df["Close"] - df["Open"]).abs()

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  STRATEGY
# ─────────────────────────────────────────────
class ThreeCandleBreakoutRetest(Strategy):
    """
    v2 — Tighter filters to reduce false setups in choppy markets.

    New filters vs v1:
      1. Each of the 3 setup candles must have body > min_body_atr_ratio * ATR
      2. ATR_OK threshold raised to 0.8x ATR_MA
      3. Breakout candle close must exceed range_high by > breakout_atr_ratio * ATR
      4. EMA200 macro trend filter (long only above EMA200, short only below)
      5. RSI filter: no longs above rsi_long_max, no shorts below rsi_short_min
      6. Retest wick tolerance: low can dip retest_wick_ratio * ATR below range_high
      7. max_retest_bars reduced to 3
    """

    # ── risk ────────────────────────────────
    risk_pct        = 1.0    # % equity per trade
    rr_ratio        = 2.0    # reward:risk

    # ── setup quality filters ────────────────
    min_body_atr_ratio   = 0.2   # each setup candle body must be > 30% of ATR
    breakout_atr_ratio   = 0.2   # breakout close must exceed level by > 30% ATR
    retest_wick_ratio    = 0.3   # retest low can dip up to 30% ATR below range_high
    max_retest_bars      = 3     # timeout faster (was 5)

    # ── momentum filters ─────────────────────
    rsi_long_max    = 70    # don't go long if RSI overbought
    rsi_short_min   = 30    # don't go short if RSI oversold

    def init(self):
        self.ema50      = self.I(lambda x: x, self.data.EMA50,       name="EMA50")
        self.ema200     = self.I(lambda x: x, self.data.EMA200,      name="EMA200")
        self.rsi        = self.I(lambda x: x, self.data.RSI,         name="RSI")
        self.atr        = self.I(lambda x: x, self.data.ATR,         name="ATR")
        self.atr_ok     = self.I(lambda x: x, self.data.ATR_OK,      name="ATR_OK")
        self.in_session = self.I(lambda x: x, self.data.IN_SESSION,  name="Session")
        self.body       = self.I(lambda x: x, self.data.BODY,        name="Body")

        # Long state machine
        self._long_state      = "IDLE"
        self._long_range_high = np.nan
        self._long_range_low  = np.nan
        self._long_retest_ctr = 0

        # Short state machine
        self._short_state      = "IDLE"
        self._short_range_high = np.nan
        self._short_range_low  = np.nan
        self._short_retest_ctr = 0

    # ── helpers ─────────────────────────────
    def _bullish(self, i):
        return self.data.Close[i] > self.data.Open[i]

    def _bearish(self, i):
        return self.data.Close[i] < self.data.Open[i]

    def _body_ok(self, i, atr):
        """Candle body must be meaningfully sized relative to ATR."""
        return self.body[i] >= self.min_body_atr_ratio * atr

    def _size_from_risk(self, entry, sl):
        risk_amount   = self.equity * (self.risk_pct / 100.0)
        risk_per_unit = abs(entry - sl)
        if risk_per_unit < 1e-8:
            return 0.01
        fraction = (risk_amount / risk_per_unit) * entry / self.equity
        return max(0.01, min(0.99, round(fraction, 4)))

    # ── main loop ───────────────────────────
    def next(self):
        i = len(self.data.Close) - 1
        if i < 10:
            return

        session_ok = bool(self.in_session[-1])
        atr_ok     = bool(self.atr_ok[-1])

        if not session_ok or not atr_ok:
            self._long_state  = "IDLE"
            self._short_state = "IDLE"
            return

        close  = self.data.Close[-1]
        ema50  = self.ema50[-1]
        ema200 = self.ema200[-1]
        rsi    = self.rsi[-1]
        atr    = self.atr[-1]

        # ── LONG STATE MACHINE ───────────────
        if not self.position.is_long:

            if self._long_state == "IDLE":
                # 3 consecutive bullish candles with meaningful bodies
                c1_bull = self._bullish(-4) and self._body_ok(-4, atr)
                c2_bull = self._bullish(-3) and self._body_ok(-3, atr)
                c3_bull = self._bullish(-2) and self._body_ok(-2, atr)
                # Each candle must close higher than the previous (directional momentum)
                c_progressive = (self.data.Close[-3] > self.data.Close[-4] and
                                 self.data.Close[-2] > self.data.Close[-3])

                if c1_bull and c2_bull and c3_bull and c_progressive:
                    self._long_range_high = max(self.data.High[-4],
                                                self.data.High[-3],
                                                self.data.High[-2])
                    self._long_range_low  = min(self.data.Low[-4],
                                                self.data.Low[-3],
                                                self.data.Low[-2])
                    self._long_state = "WATCHING_BREAKOUT"

            elif self._long_state == "WATCHING_BREAKOUT":
                rh = self._long_range_high
                # Breakout: close must exceed range_high by at least breakout_atr_ratio * ATR
                if close > rh + self.breakout_atr_ratio * atr:
                    self._long_state      = "WAITING_RETEST"
                    self._long_retest_ctr = 0
                else:
                    self._long_state = "IDLE"

            elif self._long_state == "WAITING_RETEST":
                self._long_retest_ctr += 1
                rh = self._long_range_high
                rl = self._long_range_low

                # Retest: low touches range_high (with wick tolerance) and closes above
                wick_tolerance = self.retest_wick_ratio * atr
                touched  = self.data.Low[-1] <= rh + wick_tolerance   # touched the level
                reclosed = close > rh                                   # closed back above

                if touched and reclosed:
                    # All filters must pass at entry
                    above_ema50  = close > ema50
                    above_ema200 = close > ema200
                    rsi_ok       = rsi < self.rsi_long_max

                    if above_ema50 and above_ema200 and rsi_ok:
                        entry = close
                        sl    = rl
                        tp    = entry + self.rr_ratio * (entry - sl)
                        size  = self._size_from_risk(entry, sl)
                        self.buy(size=size, sl=sl, tp=tp)
                    self._long_state = "IDLE"

                elif self._long_retest_ctr >= self.max_retest_bars:
                    self._long_state = "IDLE"

        # ── SHORT STATE MACHINE ──────────────
        if not self.position.is_short:

            if self._short_state == "IDLE":
                c1_bear = self._bearish(-4) and self._body_ok(-4, atr)
                c2_bear = self._bearish(-3) and self._body_ok(-3, atr)
                c3_bear = self._bearish(-2) and self._body_ok(-2, atr)
                c_progressive = (self.data.Close[-3] < self.data.Close[-4] and
                                 self.data.Close[-2] < self.data.Close[-3])

                if c1_bear and c2_bear and c3_bear and c_progressive:
                    self._short_range_high = max(self.data.High[-4],
                                                 self.data.High[-3],
                                                 self.data.High[-2])
                    self._short_range_low  = min(self.data.Low[-4],
                                                 self.data.Low[-3],
                                                 self.data.Low[-2])
                    self._short_state = "WATCHING_BREAKOUT"

            elif self._short_state == "WATCHING_BREAKOUT":
                rl = self._short_range_low
                if close < rl - self.breakout_atr_ratio * atr:
                    self._short_state      = "WAITING_RETEST"
                    self._short_retest_ctr = 0
                else:
                    self._short_state = "IDLE"

            elif self._short_state == "WAITING_RETEST":
                self._short_retest_ctr += 1
                rh = self._short_range_high
                rl = self._short_range_low

                wick_tolerance = self.retest_wick_ratio * atr
                touched  = self.data.High[-1] >= rl - wick_tolerance
                reclosed = close < rl

                if touched and reclosed:
                    below_ema50  = close < ema50
                    below_ema200 = close < ema200
                    rsi_ok       = rsi > self.rsi_short_min

                    if below_ema50 and below_ema200 and rsi_ok:
                        entry = close
                        sl    = rh
                        tp    = entry - self.rr_ratio * (sl - entry)
                        size  = self._size_from_risk(entry, sl)
                        self.sell(size=size, sl=sl, tp=tp)
                    self._short_state = "IDLE"

                elif self._short_retest_ctr >= self.max_retest_bars:
                    self._short_state = "IDLE"


# ─────────────────────────────────────────────
# 4.  RUN BACKTEST
# ─────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, cash=100_000, commission=0.0002):
    bt = Backtest(
        df, ThreeCandleBreakoutRetest,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=False,
    )
    stats = bt.run()
    return bt, stats


# ─────────────────────────────────────────────
# 5.  REPORTING
# ─────────────────────────────────────────────
def _avg_trade(stats, win: bool) -> str:
    trades = stats.get("_trades")
    if trades is None or trades.empty:
        return "N/A"
    pnl    = trades["ReturnPct"]
    subset = pnl[pnl >= 0] if win else pnl[pnl < 0]
    return "N/A" if subset.empty else f"{subset.mean():.2f}%"


def print_report(stats):
    SEP = "─" * 55
    print(f"\n{'═'*55}")
    print("  XAUUSD 15m  |  3-Candle Breakout + Retest  v3")
    print(f"{'═'*55}")

    def _f(key, fmt):
        try:
            v = stats[key]
            return "N/A" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        except Exception:
            return "N/A"

    rows = [
        ("Start",           str(stats["Start"])),
        ("End",             str(stats["End"])),
        ("Duration",        str(stats["Duration"])),
        ("Initial Equity",  "$100,000"),
        ("Final Equity",    _f("Equity Final [$]",    "${:,.2f}")),
        ("Return",          _f("Return [%]",          "{:.2f}%")),
        ("Buy & Hold Ret.", _f("Buy & Hold Return [%]","{:.2f}%")),
        (SEP, ""),
        ("Total Trades",    _f("# Trades",            "{:.0f}")),
        ("Win Rate",        _f("Win Rate [%]",        "{:.1f}%")),
        ("Profit Factor",   _f("Profit Factor",       "{:.2f}")),
        ("Avg Trade",       _f("Avg. Trade [%]",      "{:.2f}%")),
        ("Best Trade",      _f("Best Trade [%]",      "{:.2f}%")),
        ("Worst Trade",     _f("Worst Trade [%]",     "{:.2f}%")),
        ("Avg Win",         _avg_trade(stats, win=True)),
        ("Avg Loss",        _avg_trade(stats, win=False)),
        (SEP, ""),
        ("Max Drawdown",    _f("Max. Drawdown [%]",   "{:.2f}%")),
        ("Avg Drawdown",    _f("Avg. Drawdown [%]",   "{:.2f}%")),
        ("Sharpe Ratio",    _f("Sharpe Ratio",        "{:.3f}")),
        ("Sortino Ratio",   _f("Sortino Ratio",       "{:.3f}")),
        ("Calmar Ratio",    _f("Calmar Ratio",        "{:.3f}")),
        ("SQN",             _f("SQN",                 "{:.3f}")),
        (SEP, ""),
        ("Max Trade Dur.",  str(stats["Max. Trade Duration"])),
        ("Avg Trade Dur.",  str(stats["Avg. Trade Duration"])),
    ]

    for k, v in rows:
        print(f"  {k}" if v == "" else f"  {k:<28} {v}")
    print(f"{'═'*55}\n")


def save_trade_log(stats, path="trade_log.csv"):
    trades = stats["_trades"]
    if trades.empty:
        print("[!] No trades to log.")
        return None
    out = trades[["EntryTime","ExitTime","EntryPrice","ExitPrice",
                  "PnL","ReturnPct","Size","Duration"]].copy()
    out["Direction"] = out["Size"].apply(lambda s: "LONG" if s > 0 else "SHORT")
    out.to_csv(path, index=False)
    print(f"[+] Trade log saved → {path}  ({len(out)} trades)")
    return out


# ─────────────────────────────────────────────
# 6.  CHARTS
# ─────────────────────────────────────────────
def plot_results(stats, save_path="equity_curve.png"):
    trades       = stats["_trades"]
    equity_series = stats["_equity_curve"]["Equity"]

    fig = plt.figure(figsize=(16, 12), facecolor="#0D1117")
    fig.suptitle("XAUUSD 15m  |  3-Candle Breakout + Retest  v3",
                 color="#E6EDF3", fontsize=14, fontweight="bold", y=0.98)

    gs    = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    DARK  = "#0D1117"; GRID = "#21262D"; TEXT = "#E6EDF3"
    GREEN = "#3FB950"; RED  = "#F85149"; GOLD = "#D29922"; BLUE = "#58A6FF"

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_color(GRID)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        ax.grid(color=GRID, linewidth=0.5)

    # 1. Equity + Drawdown
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(equity_series.index, equity_series.values,
                     equity_series.values[0], alpha=0.15, color=GOLD)
    ax1.plot(equity_series.index, equity_series.values,
             color=GOLD, linewidth=1.5, label="Equity")
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max * 100
    ax1_dd = ax1.twinx()
    ax1_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color=RED)
    ax1_dd.set_ylabel("Drawdown %", color=RED, fontsize=8)
    ax1_dd.tick_params(colors=RED, labelsize=7)
    ax1_dd.spines[["top","right","left","bottom"]].set_color(GRID)
    ax1_dd.set_facecolor(DARK)
    style_ax(ax1, "Equity Curve  +  Drawdown")
    ax1.set_ylabel("Equity ($)", color=GOLD, fontsize=8)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT)

    # 2. Trade Return Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    if not trades.empty:
        pnl    = trades["ReturnPct"].values
        wins   = pnl[pnl >= 0]; losses = pnl[pnl < 0]
        bins   = np.linspace(pnl.min()-0.5, pnl.max()+0.5, 30)
        ax2.hist(wins,   bins=bins, color=GREEN, alpha=0.8, label=f"Wins ({len(wins)})")
        ax2.hist(losses, bins=bins, color=RED,   alpha=0.8, label=f"Losses ({len(losses)})")
        ax2.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
    style_ax(ax2, "Trade Return Distribution (%)")
    ax2.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    ax2.set_xlabel("Return %", color=TEXT, fontsize=8)

    # 3. Monthly Returns
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = equity_series.resample("ME").last().pct_change().dropna() * 100
    colors  = [GREEN if v >= 0 else RED for v in monthly.values]
    ax3.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.85)
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([d.strftime("%b %y") for d in monthly.index],
                        rotation=45, ha="right", fontsize=7)
    ax3.axhline(0, color=TEXT, linewidth=0.6)
    style_ax(ax3, "Monthly Returns (%)")

    # 4. Cumulative PnL per trade
    ax4 = fig.add_subplot(gs[2, 0])
    if not trades.empty:
        tr = trades["ReturnPct"].reset_index(drop=True)
        cs = tr.cumsum()
        ax4.plot(cs.values, color=BLUE, linewidth=1.5)
        ax4.axhline(0, color=TEXT, linewidth=0.6, linestyle="--")
        ax4.fill_between(range(len(tr)), cs.values, 0,
                         where=cs.values >= 0, alpha=0.15, color=GREEN)
        ax4.fill_between(range(len(tr)), cs.values, 0,
                         where=cs.values < 0,  alpha=0.15, color=RED)
    style_ax(ax4, "Cumulative Return % (per trade)")
    ax4.set_xlabel("Trade #", color=TEXT, fontsize=8)

    # 5. Key metrics box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK); ax5.axis("off")

    def _safe(key, fmt):
        try:
            v = stats[key]
            return "N/A" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        except Exception:
            return "N/A"

    summary = [
        ("Total Trades",  _safe("# Trades",         "{:.0f}")),
        ("Win Rate",      _safe("Win Rate [%]",      "{:.1f}%")),
        ("Profit Factor", _safe("Profit Factor",     "{:.2f}")),
        ("Sharpe Ratio",  _safe("Sharpe Ratio",      "{:.3f}")),
        ("Sortino Ratio", _safe("Sortino Ratio",     "{:.3f}")),
        ("Max Drawdown",  _safe("Max. Drawdown [%]", "{:.2f}%")),
        ("Total Return",  _safe("Return [%]",        "{:.2f}%")),
        ("SQN",           _safe("SQN",               "{:.3f}")),
    ]
    ax5.set_title("Key Metrics", color=TEXT, fontsize=9, pad=6)
    for idx, (label, val) in enumerate(summary):
        y = 0.88 - idx * 0.115
        ax5.text(0.05, y, label, transform=ax5.transAxes, color="#8B949E", fontsize=9)
        ax5.text(0.95, y, val,   transform=ax5.transAxes, color=GOLD,
                 fontsize=9, ha="right", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"[+] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAUUSD 15m Backtest v3")
    parser.add_argument("--csv",    type=str, default=None,
                        help="Path to CSV file")
    parser.add_argument("--period", type=str, default="60d",
                        help="yfinance period if no CSV (default: 60d)")
    args = parser.parse_args()

    raw = load_data(ticker="GC=F", period=args.period,
                    interval="15m", csv_path=args.csv)
    df  = add_indicators(raw)

    print(f"[*] Bars after indicator warmup : {len(df):,}")
    print(f"[*] Session bars : {df['IN_SESSION'].sum():,}  |  "
          f"ATR-OK bars : {df['ATR_OK'].sum():,}")

    print("[*] Running backtest ...")
    bt, stats = run_backtest(df)

    print_report(stats)
    trades_df = save_trade_log(stats, "trade_log.csv")
    plot_results(stats, "equity_curve.png")

    if trades_df is not None and not trades_df.empty:
        print("\nLast 10 trades:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(trades_df.tail(10).to_string(index=False))
