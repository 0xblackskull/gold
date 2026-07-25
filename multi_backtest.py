"""
Multi-Instrument Backtest
==========================
Runs Var I strategy on XAUUSD, XAGUSD, BTCUSD, EURUSD simultaneously.
Simulates a single $10k account trading all 4 instruments with 0.25% risk each
(total risk = 1% of account per bar, same as single-instrument version).

Output:
  - Per-instrument stats
  - Combined equity curve (all instruments on same account)
  - Total portfolio metrics
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
from pathlib import Path

# ─────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────
# Instrument config: yfinance ticker + display name
INSTRUMENTS = {
    "GOLD":   {"ticker": "GC=F",   "name": "XAUUSD (Gold)"},
    "SILVER": {"ticker": "SI=F",   "name": "XAGUSD (Silver)"},
    "BTC":    {"ticker": "BTC-USD","name": "BTCUSD"},
    "EUR":    {"ticker": "EURUSD=X","name": "EURUSD"},
}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.dropna(inplace=True)
    return df

def load_yf(ticker: str) -> pd.DataFrame:
    print(f"    Downloading {ticker} ...")
    df = yf.download(ticker, period="60d", interval="15m",
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df

def load_instrument(key: str, csv_path: str = None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        print(f"    Loading {key} from {csv_path} ...")
        df = load_csv(csv_path)
        print(f"      {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")
        return df
    cfg = INSTRUMENTS[key]
    df  = load_yf(cfg["ticker"])
    if not df.empty:
        print(f"      {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")
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

    daily = df["Close"].resample("1D").last().dropna()
    d_ema = ta.ema(daily, length=50)
    if d_ema is not None and not d_ema.dropna().empty:
        df["DAILY_EMA50"] = d_ema.reindex(df.index, method="ffill").values
    else:
        df["DAILY_EMA50"] = df["EMA200"]

    dh = df.index.hour + df.index.minute / 60.0
    df["IN_SESSION"] = ((dh >= 8.0) & (dh < 17.0)) | ((dh >= 13.0) & (dh < 22.0))
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3.  STRATEGY (Var I — partial 2R + trail)
# ─────────────────────────────────────────────
def make_strategy(risk_pct: float = 0.25):
    """
    risk_pct: 0.25% per instrument × 4 instruments = 1% total portfolio risk
    """
    class VarI(Strategy):
        _risk_pct  = risk_pct
        _partial_r = 2.0
        _trail_m   = 1.5

        def init(self):
            self.ema50  = self.I(lambda x: x, self.data.EMA50)
            self.ema200 = self.I(lambda x: x, self.data.EMA200)
            self.d_ema  = self.I(lambda x: x, self.data.DAILY_EMA50)
            self.rsi    = self.I(lambda x: x, self.data.RSI)
            self.atr    = self.I(lambda x: x, self.data.ATR)
            self.atr_ok = self.I(lambda x: x, self.data.ATR_OK)
            self.sess   = self.I(lambda x: x, self.data.IN_SESSION)
            self.body   = self.I(lambda x: x, self.data.BODY)

            self._ls="IDLE"; self._lrh=np.nan; self._lrl=np.nan; self._lrc=0
            self._entry=np.nan; self._risk=np.nan
            self._partial_done=False; self._highest=np.nan; self._trail_sl=np.nan

        def _bull(self,i): return self.data.Close[i]>self.data.Open[i]
        def _body_ok(self,i): return self.body[i]>=0.2*self.atr[-1]
        def _size(self,entry,sl):
            r=abs(entry-sl)
            if r<1e-8: return 0.01
            return max(0.01,min(0.99,round((self.equity*(self._risk_pct/100)/r)*entry/self.equity,4)))

        def next(self):
            i=len(self.data.Close)-1
            if i<10: return
            if not bool(self.sess[-1]) or not bool(self.atr_ok[-1]):
                self._ls="IDLE"; return

            c=self.data.Close[-1]; atr=self.atr[-1]

            # Manage open position
            if self.position.is_long:
                if c>self._highest: self._highest=c
                pt=self._entry+self._partial_r*self._risk
                if not self._partial_done and c>=pt:
                    self.position.close(portion=0.5)
                    self._partial_done=True
                    self._trail_sl=c-self._trail_m*atr
                    self._highest=c
                if self._partial_done:
                    new_sl=self._highest-self._trail_m*atr
                    if new_sl>self._trail_sl: self._trail_sl=new_sl
                    if c<=self._trail_sl:
                        self.position.close(); self._partial_done=False
                return

            # Setup detection
            if self._ls=="IDLE":
                if (self._bull(-4) and self._body_ok(-4) and
                    self._bull(-3) and self._body_ok(-3) and
                    self._bull(-2) and self._body_ok(-2) and
                    self.data.Close[-3]>self.data.Close[-4] and
                    self.data.Close[-2]>self.data.Close[-3]):
                    self._lrh=max(self.data.High[-4],self.data.High[-3],self.data.High[-2])
                    self._lrl=min(self.data.Low[-4],self.data.Low[-3],self.data.Low[-2])
                    self._ls="WATCHING"
            elif self._ls=="WATCHING":
                if c>self._lrh+0.2*atr: self._ls="RETEST"; self._lrc=0
                else: self._ls="IDLE"
            elif self._ls=="RETEST":
                self._lrc+=1
                rh=self._lrh; rl=self._lrl
                if self.data.Low[-1]<=rh+0.3*atr and c>rh:
                    if (c>self.ema50[-1] and c>self.ema200[-1] and
                        c>self.d_ema[-1] and self.rsi[-1]<70):
                        sl=rl; risk=c-sl
                        self.buy(size=self._size(c,sl),sl=sl,tp=c+10*risk)
                        self._entry=c; self._risk=risk
                        self._partial_done=False; self._highest=c; self._trail_sl=sl
                    self._ls="IDLE"
                elif self._lrc>=3: self._ls="IDLE"

    return VarI


# ─────────────────────────────────────────────
# 4.  RUN PER-INSTRUMENT BACKTESTS
# ─────────────────────────────────────────────
def run_instrument(key: str, df: pd.DataFrame,
                   risk_pct: float = 0.25) -> dict:
    strat  = make_strategy(risk_pct)
    bt     = Backtest(df, strat, cash=100_000, commission=0.0002,
                      exclusive_orders=True, trade_on_close=False)
    stats  = bt.run()
    return {"key": key, "name": INSTRUMENTS[key]["name"],
            "stats": stats, "equity": stats["_equity_curve"]["Equity"]}


# ─────────────────────────────────────────────
# 5.  COMBINE EQUITY CURVES
# ─────────────────────────────────────────────
def combine_equity(results: list, initial: float = 100_000) -> pd.Series:
    """
    Combine per-instrument equity curves into a single portfolio curve.
    Each instrument starts at $25k (25% of $100k portfolio).
    Portfolio equity = sum of all 4 instrument accounts.
    """
    curves = []
    for r in results:
        eq = r["equity"].copy()
        # Normalise to 25% of portfolio
        eq = eq / 100_000 * (initial / len(results))
        curves.append(eq)

    # Align on common index
    combined = pd.concat(curves, axis=1).ffill().sum(axis=1)
    return combined


# ─────────────────────────────────────────────
# 6.  PORTFOLIO METRICS
# ─────────────────────────────────────────────
def portfolio_metrics(combined: pd.Series, initial: float = 100_000) -> dict:
    total_ret  = (combined.iloc[-1] - initial) / initial * 100
    roll_max   = combined.cummax()
    dd         = (combined - roll_max) / roll_max * 100
    max_dd     = dd.min()
    daily_ret  = combined.resample("1D").last().pct_change().dropna()
    sharpe     = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    neg        = daily_ret[daily_ret < 0]
    sortino    = daily_ret.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 else 0
    n_years    = (combined.index[-1] - combined.index[0]).days / 365
    annual_ret = ((combined.iloc[-1] / initial) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    return {"total_ret": total_ret, "annual_ret": annual_ret,
            "max_dd": max_dd, "sharpe": sharpe, "sortino": sortino,
            "final_equity": combined.iloc[-1]}


# ─────────────────────────────────────────────
# 7.  CHARTS
# ─────────────────────────────────────────────
def plot_portfolio(results: list, combined: pd.Series,
                   save_path: str = "portfolio_equity.png"):
    DARK="#0D1117"; GRID="#21262D"; TEXT="#E6EDF3"
    COLORS=["#D29922","#C0C0C0","#F7931A","#3FB950"]  # Gold, Silver, BTC, EUR

    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle("Multi-Instrument Portfolio  |  XAUUSD + XAGUSD + BTCUSD + EURUSD",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK); ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_color(GRID)
        if title: ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        ax.grid(color=GRID, linewidth=0.5)

    # 1. Combined portfolio equity
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(combined.index, combined.values,
             color="#58A6FF", linewidth=2, label="Portfolio (all 4)")
    ax1.fill_between(combined.index, combined.values,
                     combined.values[0], alpha=0.1, color="#58A6FF")
    # Individual curves faded
    for i, r in enumerate(results):
        eq = r["equity"] / 100_000 * 25_000
        ax1.plot(eq.index, eq.values, color=COLORS[i],
                 linewidth=0.8, alpha=0.5, label=r["name"])
    ax1.axhline(100_000, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.4)
    style_ax(ax1, "Portfolio Equity Curve  (blue = combined, faded = per instrument)")
    ax1.set_ylabel("Equity ($)", color=TEXT, fontsize=8)
    ax1.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)

    # 2. Per-instrument returns
    ax2 = fig.add_subplot(gs[1, 0])
    names = [r["name"].split(" ")[0] for r in results]
    rets  = [r["stats"]["Return [%]"] for r in results]
    bars  = ax2.bar(names, rets,
                    color=[COLORS[i] for i in range(len(results))], alpha=0.85)
    ax2.axhline(0, color=TEXT, linewidth=0.6)
    for bar, val in zip(bars, rets):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.3 if val>=0 else -1.5),
                 f"{val:.1f}%", ha="center", color=TEXT, fontsize=8)
    style_ax(ax2, "Per-Instrument Return (%)")

    # 3. Per-instrument Sharpe
    ax3 = fig.add_subplot(gs[1, 1])
    sharpes = [r["stats"]["Sharpe Ratio"] for r in results]
    bars2   = ax3.bar(names, sharpes,
                      color=[COLORS[i] for i in range(len(results))], alpha=0.85)
    ax3.axhline(0, color=TEXT, linewidth=0.6)
    ax3.axhline(1.0, color="#3FB950", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, val in zip(bars2, sharpes):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", color=TEXT, fontsize=8)
    style_ax(ax3, "Per-Instrument Sharpe Ratio")

    # 4. Portfolio drawdown
    ax4 = fig.add_subplot(gs[2, 0])
    roll_max = combined.cummax()
    dd       = (combined - roll_max) / roll_max * 100
    ax4.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#F85149")
    ax4.plot(dd.index, dd.values, color="#F85149", linewidth=1)
    ax4.axhline(0, color=TEXT, linewidth=0.4)
    style_ax(ax4, "Portfolio Drawdown (%)")

    # 5. Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK); ax5.axis("off")
    pm = portfolio_metrics(combined)
    ax5.set_title("Portfolio Summary", color=TEXT, fontsize=9, pad=6)

    summary = [
        ("Instruments",   "4 (Gold, Silver, BTC, EUR)"),
        ("Risk/instrument","0.25% per trade"),
        ("Total Return",  f"{pm['total_ret']:.2f}%"),
        ("Annual Return", f"~{pm['annual_ret']:.1f}% / year"),
        ("Final Equity",  f"${pm['final_equity']:,.0f}"),
        ("Max Drawdown",  f"{pm['max_dd']:.2f}%"),
        ("Sharpe Ratio",  f"{pm['sharpe']:.3f}"),
        ("Sortino Ratio", f"{pm['sortino']:.3f}"),
    ]
    for idx, (label, val) in enumerate(summary):
        y = 0.88 - idx * 0.105
        ax5.text(0.05, y, label, transform=ax5.transAxes,
                 color="#8B949E", fontsize=9)
        ax5.text(0.95, y, val, transform=ax5.transAxes,
                 color="#D29922", fontsize=9, ha="right", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"[+] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 8.  PRINT REPORT
# ─────────────────────────────────────────────
def print_report(results: list, combined: pd.Series):
    pm  = portfolio_metrics(combined)
    SEP = "─" * 60

    print(f"\n{'═'*60}")
    print(f"  MULTI-INSTRUMENT PORTFOLIO RESULTS")
    print(f"{'═'*60}")
    print(f"\n  Per-instrument (0.25% risk each, $25k allocation):\n")
    print(f"  {'Instrument':<20} {'Trades':>6}  {'WR':>6}  {'Sharpe':>8}  {'DD':>8}  {'Ret':>8}")
    print(f"  {SEP}")

    for r in results:
        s = r["stats"]
        print(f"  {r['name']:<20} "
              f"{s['# Trades']:>6}  "
              f"{s['Win Rate [%]']:>5.1f}%  "
              f"{s['Sharpe Ratio']:>8.3f}  "
              f"{s['Max. Drawdown [%]']:>7.2f}%  "
              f"{s['Return [%]']:>7.2f}%")

    print(f"\n  {SEP}")
    print(f"  COMBINED PORTFOLIO")
    print(f"  {SEP}")
    print(f"  Total Return   : {pm['total_ret']:.2f}%")
    print(f"  Annual Return  : ~{pm['annual_ret']:.1f}% per year")
    print(f"  Final Equity   : ${pm['final_equity']:,.2f}")
    print(f"  Max Drawdown   : {pm['max_dd']:.2f}%")
    print(f"  Sharpe Ratio   : {pm['sharpe']:.3f}")
    print(f"  Sortino Ratio  : {pm['sortino']:.3f}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-instrument portfolio backtest")
    parser.add_argument("--gold",   type=str, default=None)
    parser.add_argument("--silver", type=str, default=None)
    parser.add_argument("--btc",    type=str, default=None)
    parser.add_argument("--eur",    type=str, default=None)
    args = parser.parse_args()

    csv_map = {
        "GOLD":   args.gold,
        "SILVER": args.silver,
        "BTC":    args.btc,
        "EUR":    args.eur,
    }

    print("\n[*] Loading instruments ...")
    datasets = {}
    for key, csv_path in csv_map.items():
        df = load_instrument(key, csv_path)
        if not df.empty:
            datasets[key] = add_indicators(df)
        else:
            print(f"    [!] Skipping {key} — no data")

    if not datasets:
        print("[!] No data loaded. Pass CSV paths or check internet connection.")
        exit(1)

    print(f"\n[*] Running backtests on {len(datasets)} instruments ...\n")
    print(f"{'─'*75}")

    results = []
    for key, df in datasets.items():
        print(f"  {INSTRUMENTS[key]['name']} ...")
        r = run_instrument(key, df, risk_pct=0.25)
        s = r["stats"]
        print(f"    Trades:{s['# Trades']}  "
              f"WR:{s['Win Rate [%]']:.1f}%  "
              f"Sharpe:{s['Sharpe Ratio']:.3f}  "
              f"DD:{s['Max. Drawdown [%]']:.2f}%  "
              f"Ret:{s['Return [%]']:.2f}%")
        results.append(r)

    print(f"{'─'*75}")

    combined = combine_equity(results)
    print_report(results, combined)
    plot_portfolio(results, combined)
