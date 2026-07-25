"""
XAUUSD 15m Backtesting Strategy
================================
Strategy: 3-Candle Consolidation Breakout + Retest
- Long: 3 consecutive bullish candles → breakout above range_high → retest → enter
- Short: 3 consecutive bearish candles → breakout below range_low → retest → enter
- Filters: EMA50, London/New York sessions, ATR volatility
- Risk: 1% per trade, SL at range_low/high, TP at 2R
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
from datetime import datetime

# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────
def _generate_synthetic_xauusd(n_bars=5000, seed=42):
    """Generate realistic synthetic XAUUSD 15-min data (used when live feed unavailable)."""
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
    start = pd.Timestamp("2024-01-02 08:00:00")
    idx = pd.date_range(start, periods=n_bars, freq="15min")
    rows = []
    for i in range(n_bars):
        c = closes[i]
        bv = abs(np.random.randn()) * c * 0.0008 + c * 0.0002
        o = closes[i-1] if i > 0 else c * (1 + np.random.randn() * 0.0003)
        h = max(o, c) + abs(np.random.randn()) * bv * 0.6
        l = min(o, c) - abs(np.random.randn()) * bv * 0.6
        rows.append({"Open": round(o,2), "High": round(h,2),
                     "Low": round(l,2), "Close": round(c,2),
                     "Volume": int(np.random.randint(100, 5000))})
    df = pd.DataFrame(rows, index=idx)
    df["High"] = df[["Open","High","Close"]].max(axis=1)
    df["Low"]  = df[["Open","Low","Close"]].min(axis=1)
    return df


def load_data(ticker="GC=F", period="60d", interval="15m",
              csv_path: str = None):
    """
    Load XAUUSD 15-minute OHLCV data.
    Priority: (1) csv_path if provided, (2) yfinance live download,
              (3) synthetic fallback so the script always runs.
    """
    # ── 1. CSV override ──────────────────────
    if csv_path:
        print(f"[*] Loading data from {csv_path} ...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
        return df

    # ── 2. Live download ─────────────────────
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

    # ── 3. Synthetic fallback ────────────────
    print("[*] Using synthetic XAUUSD data (5 000 bars) ...")
    df = _generate_synthetic_xauusd(n_bars=5000)
    print(f"    {len(df):,} bars  |  {df.index[0]}  →  {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# 2.  PRE-COMPUTE INDICATORS (no lookahead)
# ─────────────────────────────────────────────
def add_indicators(df: pd.DataFrame, atr_period=14, ema_period=50,
                   atr_multiplier=0.5) -> pd.DataFrame:
    df = df.copy()

    # EMA-50
    df["EMA50"] = ta.ema(df["Close"], length=ema_period)

    # ATR
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_period)
    df["ATR_MA"] = df["ATR"].rolling(atr_period).mean()
    df["ATR_OK"] = df["ATR"] > df["ATR_MA"] * atr_multiplier   # volatility filter

    # Session flags (UTC times)
    hour = df.index.hour
    minute = df.index.minute
    decimal_hour = hour + minute / 60.0
    # London: 08:00–17:00 UTC | New York: 13:00–22:00 UTC
    df["IN_SESSION"] = (
        ((decimal_hour >= 8.0) & (decimal_hour < 17.0)) |   # London
        ((decimal_hour >= 13.0) & (decimal_hour < 22.0))    # New York
    )

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  STRATEGY
# ─────────────────────────────────────────────
class ThreeCandleBreakoutRetest(Strategy):
    """
    3-Candle Consolidation Breakout + Retest Strategy for XAUUSD 15m.

    State machine per direction:
      IDLE → WATCHING_BREAKOUT → WAITING_RETEST → (trade entry)

    All index references use already-closed candles (i-1 style) to avoid
    lookahead bias. The strategy only acts on fully closed bars.
    """

    # ── tuneable parameters ──────────────────
    risk_pct        = 1.0    # % of equity risked per trade
    rr_ratio        = 2.0    # reward : risk
    max_retest_bars = 5      # max candles to wait for retest after breakout
    atr_period      = 14
    ema_period      = 50

    def init(self):
        # Pre-built indicator arrays (already in df, exposed via self.data)
        self.ema50      = self.I(lambda x: x, self.data.EMA50,  name="EMA50")
        self.atr        = self.I(lambda x: x, self.data.ATR,    name="ATR")
        self.atr_ok     = self.I(lambda x: x, self.data.ATR_OK, name="ATR_OK")
        self.in_session = self.I(lambda x: x, self.data.IN_SESSION, name="Session")

        # Internal state (long side)
        self._long_state      = "IDLE"   # IDLE / WATCHING_BREAKOUT / WAITING_RETEST
        self._long_range_high = np.nan
        self._long_range_low  = np.nan
        self._long_retest_ctr = 0

        # Internal state (short side)
        self._short_state      = "IDLE"
        self._short_range_high = np.nan
        self._short_range_low  = np.nan
        self._short_retest_ctr = 0

    # ── helpers ─────────────────────────────
    def _bullish(self, i):
        return self.data.Close[i] > self.data.Open[i]

    def _bearish(self, i):
        return self.data.Close[i] < self.data.Open[i]

    def _size_from_risk(self, entry, sl):
        """Position size so that a full SL hit = risk_pct% of equity."""
        risk_amount = self.equity * (self.risk_pct / 100.0)
        risk_per_unit = abs(entry - sl)
        if risk_per_unit < 1e-8:
            return 0.01
        raw = risk_amount / risk_per_unit
        # backtesting.py wants fraction of equity (0–1)
        fraction = raw * entry / self.equity
        return max(0.01, min(0.99, round(fraction, 4)))

    # ── main loop ───────────────────────────
    def next(self):
        # Current bar index (0-based, already closed)
        i = len(self.data.Close) - 1
        if i < 5:
            return  # not enough history

        # Filters on current bar
        session_ok  = bool(self.in_session[-1])
        atr_ok      = bool(self.atr_ok[-1])
        close       = self.data.Close[-1]
        ema         = self.ema50[-1]

        if not session_ok or not atr_ok:
            # Reset states if out of session/volatility (avoid stale setups)
            self._long_state  = "IDLE"
            self._short_state = "IDLE"
            return

        # ── LONG STATE MACHINE ───────────────
        if not self.position.is_long:
            if self._long_state == "IDLE":
                # Look for 3 consecutive bullish candles (bars i-3, i-2, i-1)
                if (self._bullish(-4) and self._bullish(-3) and self._bullish(-2)):
                    rh = max(self.data.High[-4], self.data.High[-3], self.data.High[-2])
                    rl = min(self.data.Low[-4],  self.data.Low[-3],  self.data.Low[-2])
                    self._long_range_high = rh
                    self._long_range_low  = rl
                    self._long_state      = "WATCHING_BREAKOUT"

            elif self._long_state == "WATCHING_BREAKOUT":
                # Current candle closes ABOVE range_high → breakout confirmed
                if close > self._long_range_high:
                    self._long_state      = "WAITING_RETEST"
                    self._long_retest_ctr = 0
                else:
                    # Breakout failed or new setup forming; reset
                    self._long_state = "IDLE"

            elif self._long_state == "WAITING_RETEST":
                self._long_retest_ctr += 1
                rh = self._long_range_high
                rl = self._long_range_low

                # Valid retest: candle low dips to/below range_high AND closes back above it
                if (self.data.Low[-1] <= rh) and (close > rh):
                    if close > ema:      # EMA filter
                        entry = close
                        sl    = rl
                        tp    = entry + self.rr_ratio * (entry - sl)
                        size  = self._size_from_risk(entry, sl)
                        self.buy(size=size, sl=sl, tp=tp)
                    self._long_state = "IDLE"

                elif self._long_retest_ctr >= self.max_retest_bars:
                    self._long_state = "IDLE"   # timeout

        # ── SHORT STATE MACHINE ──────────────
        if not self.position.is_short:
            if self._short_state == "IDLE":
                # 3 consecutive bearish candles
                if (self._bearish(-4) and self._bearish(-3) and self._bearish(-2)):
                    rh = max(self.data.High[-4], self.data.High[-3], self.data.High[-2])
                    rl = min(self.data.Low[-4],  self.data.Low[-3],  self.data.Low[-2])
                    self._short_range_high = rh
                    self._short_range_low  = rl
                    self._short_state      = "WATCHING_BREAKOUT"

            elif self._short_state == "WATCHING_BREAKOUT":
                if close < self._short_range_low:
                    self._short_state      = "WAITING_RETEST"
                    self._short_retest_ctr = 0
                else:
                    self._short_state = "IDLE"

            elif self._short_state == "WAITING_RETEST":
                self._short_retest_ctr += 1
                rh = self._short_range_high
                rl = self._short_range_low

                if (self.data.High[-1] >= rl) and (close < rl):
                    if close < ema:      # EMA filter
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
    """Compute avg win / avg loss from trade list since backtesting.py omits them."""
    trades = stats.get("_trades")
    if trades is None or trades.empty:
        return "N/A"
    pnl = trades["ReturnPct"]
    subset = pnl[pnl >= 0] if win else pnl[pnl < 0]
    if subset.empty:
        return "N/A"
    return f"{subset.mean():.2f}%"


def print_report(stats):
    SEP = "─" * 55
    print(f"\n{'═'*55}")
    print("  XAUUSD 15m  |  3-Candle Breakout + Retest Strategy")
    print(f"{'═'*55}")

    metrics = [
        ("Start",           stats["Start"]),
        ("End",             stats["End"]),
        ("Duration",        stats["Duration"]),
        ("Initial Equity",  "$100,000"),
        ("Final Equity",    f"${stats['Equity Final [$]']:,.2f}"),
        ("Return",          f"{stats['Return [%]']:.2f}%"),
        ("Buy & Hold Ret.", f"{stats['Buy & Hold Return [%]']:.2f}%"),
        (SEP, ""),
        ("Total Trades",    stats["# Trades"]),
        ("Win Rate",        f"{stats['Win Rate [%]']:.1f}%"),
        ("Profit Factor",   f"{stats['Profit Factor']:.2f}"),
        ("Avg Trade",       f"{stats['Avg. Trade [%]']:.2f}%"),
        ("Best Trade",      f"{stats['Best Trade [%]']:.2f}%"),
        ("Worst Trade",     f"{stats['Worst Trade [%]']:.2f}%"),
        ("Avg Win",         _avg_trade(stats, win=True)),
        ("Avg Loss",        _avg_trade(stats, win=False)),
        (SEP, ""),
        ("Max Drawdown",    f"{stats['Max. Drawdown [%]']:.2f}%"),
        ("Avg Drawdown",    f"{stats['Avg. Drawdown [%]']:.2f}%"),
        ("Sharpe Ratio",    f"{stats['Sharpe Ratio']:.3f}"),
        ("Sortino Ratio",   f"{stats['Sortino Ratio']:.3f}"),
        ("Calmar Ratio",    f"{stats['Calmar Ratio']:.3f}"),
        ("SQN",             f"{stats['SQN']:.3f}"),
        (SEP, ""),
        ("Max Trade Dur.",  str(stats["Max. Trade Duration"])),
        ("Avg Trade Dur.",  str(stats["Avg. Trade Duration"])),
    ]

    for k, v in metrics:
        if v == "":
            print(f"  {k}")
        else:
            print(f"  {k:<28} {v}")

    print(f"{'═'*55}\n")


def save_trade_log(stats, path="trade_log.csv"):
    trades = stats["_trades"]
    if trades.empty:
        print("[!] No trades to log.")
        return
    trades_out = trades[[
        "EntryTime", "ExitTime", "EntryPrice", "ExitPrice",
        "PnL", "ReturnPct", "Size", "Duration"
    ]].copy()
    trades_out["Direction"] = trades_out["Size"].apply(
        lambda s: "LONG" if s > 0 else "SHORT"
    )
    trades_out.to_csv(path, index=False)
    print(f"[+] Trade log saved → {path}  ({len(trades_out)} trades)")
    return trades_out


# ─────────────────────────────────────────────
# 6.  CUSTOM CHARTS
# ─────────────────────────────────────────────
def plot_results(stats, df: pd.DataFrame, save_path="equity_curve.png"):
    trades = stats["_trades"]

    # ── equity curve reconstruction ─────────
    equity_series = stats["_equity_curve"]["Equity"]

    fig = plt.figure(figsize=(16, 12), facecolor="#0D1117")
    fig.suptitle(
        "XAUUSD 15m  |  3-Candle Breakout + Retest",
        color="#E6EDF3", fontsize=14, fontweight="bold", y=0.98
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    DARK  = "#0D1117"
    GRID  = "#21262D"
    TEXT  = "#E6EDF3"
    GREEN = "#3FB950"
    RED   = "#F85149"
    GOLD  = "#D29922"
    BLUE  = "#58A6FF"

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[["top", "right", "left", "bottom"]].set_color(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if title:
            ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        ax.grid(color=GRID, linewidth=0.5)

    # ── 1. Equity Curve ──────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(equity_series.index, equity_series.values,
                     equity_series.values[0], alpha=0.15, color=GOLD)
    ax1.plot(equity_series.index, equity_series.values,
             color=GOLD, linewidth=1.5, label="Equity")

    # drawdown shading
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max * 100
    ax1_dd = ax1.twinx()
    ax1_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color=RED)
    ax1_dd.set_ylabel("Drawdown %", color=RED, fontsize=8)
    ax1_dd.tick_params(colors=RED, labelsize=7)
    ax1_dd.spines[["top", "right", "left", "bottom"]].set_color(GRID)
    ax1_dd.set_facecolor(DARK)

    style_ax(ax1, "Equity Curve  +  Drawdown")
    ax1.set_ylabel("Equity ($)", color=GOLD, fontsize=8)
    ax1.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT)

    # ── 2. Trade PnL Distribution ────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if not trades.empty:
        pnl = trades["ReturnPct"].values
        wins  = pnl[pnl >= 0]
        losses = pnl[pnl < 0]
        bins = np.linspace(pnl.min() - 0.5, pnl.max() + 0.5, 30)
        ax2.hist(wins,   bins=bins, color=GREEN, alpha=0.8, label=f"Wins ({len(wins)})")
        ax2.hist(losses, bins=bins, color=RED,   alpha=0.8, label=f"Losses ({len(losses)})")
        ax2.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
    style_ax(ax2, "Trade Return Distribution (%)")
    ax2.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    ax2.set_xlabel("Return %", color=TEXT, fontsize=8)

    # ── 3. Monthly Returns ───────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = equity_series.resample("ME").last().pct_change().dropna() * 100
    colors  = [GREEN if v >= 0 else RED for v in monthly.values]
    ax3.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.85)
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels(
        [d.strftime("%b %y") for d in monthly.index],
        rotation=45, ha="right", fontsize=7
    )
    ax3.axhline(0, color=TEXT, linewidth=0.6)
    style_ax(ax3, "Monthly Returns (%)")

    # ── 4. Cumulative Win/Loss ───────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if not trades.empty:
        trade_ret = trades["ReturnPct"].reset_index(drop=True)
        ax4.plot(trade_ret.cumsum().values, color=BLUE, linewidth=1.5)
        ax4.axhline(0, color=TEXT, linewidth=0.6, linestyle="--")
        ax4.fill_between(range(len(trade_ret)),
                         trade_ret.cumsum().values, 0,
                         where=trade_ret.cumsum().values >= 0,
                         alpha=0.15, color=GREEN)
        ax4.fill_between(range(len(trade_ret)),
                         trade_ret.cumsum().values, 0,
                         where=trade_ret.cumsum().values < 0,
                         alpha=0.15, color=RED)
    style_ax(ax4, "Cumulative Return % (per trade)")
    ax4.set_xlabel("Trade #", color=TEXT, fontsize=8)

    # ── 5. Summary Stats Box ─────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK)
    ax5.axis("off")

    def _safe(key, fmt="{:.2f}"):
        try:
            v = stats[key]
            if isinstance(v, float) and np.isnan(v):
                return "N/A"
            return fmt.format(v) if fmt else str(v)
        except Exception:
            return "N/A"

    summary_lines = [
        ("Total Trades",   _safe("# Trades",        "{:.0f}")),
        ("Win Rate",       _safe("Win Rate [%]",     "{:.1f}%")),
        ("Profit Factor",  _safe("Profit Factor",    "{:.2f}")),
        ("Sharpe Ratio",   _safe("Sharpe Ratio",     "{:.3f}")),
        ("Sortino Ratio",  _safe("Sortino Ratio",    "{:.3f}")),
        ("Max Drawdown",   _safe("Max. Drawdown [%]","{:.2f}%")),
        ("Total Return",   _safe("Return [%]",       "{:.2f}%")),
        ("SQN",            _safe("SQN",              "{:.3f}")),
    ]

    ax5.set_title("Key Metrics", color=TEXT, fontsize=9, pad=6)
    for idx, (label, val) in enumerate(summary_lines):
        y_pos = 0.88 - idx * 0.115
        ax5.text(0.05, y_pos, label, transform=ax5.transAxes,
                 color="#8B949E", fontsize=9)
        ax5.text(0.95, y_pos, val, transform=ax5.transAxes,
                 color=GOLD, fontsize=9, ha="right", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"[+] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Load & prepare data ──────────────────
    raw = load_data(ticker="GC=F", period="60d", interval="15m")
    # raw = load_data(ticker="GC=F", period="60d", interval="15m", csv_path="/home/claude/xauusd_15m.csv")
    df  = add_indicators(raw)

    print(f"[*] Bars after indicator warmup: {len(df):,}")
    print(f"[*] Session bars: {df['IN_SESSION'].sum():,}  |  "
          f"ATR-OK bars: {df['ATR_OK'].sum():,}")

    # ── Run backtest ─────────────────────────
    print("[*] Running backtest ...")
    bt, stats = run_backtest(df)

    # ── Console report ───────────────────────
    print_report(stats)

    # ── Save outputs ─────────────────────────
    trades_df = save_trade_log(stats, "trade_log.csv")
    plot_results(stats, df, "equity_curve.png")

    # ── Quick trade preview ──────────────────
    if trades_df is not None and not trades_df.empty:
        print("\nLast 10 trades:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(trades_df.tail(10).to_string(index=False))
