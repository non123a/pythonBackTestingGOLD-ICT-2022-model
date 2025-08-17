#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_TZ = "America/New_York"
LONDON_START, LONDON_END = "03:00", "07:00"
NY_START, NY_END = "07:00", "12:00"
PIVOT_K = 7
OTE_LEVEL = 0.77
RISK_PER_TRADE = 100.0
START_EQUITY = 10_000.0

@dataclass
class Trade:
    date: pd.Timestamp
    direction: str
    entry: float
    stop: float
    target: float
    risk: float
    reward: float
    result: str = ""         # defaults go last
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None


# ---------------------------
# CLI args
# ---------------------------

def plot_candles(df, outpath="candles.png"):
    """
    df must have Open/High/Low/Close columns, indexed by datetime
    """
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('white')  # white background
    ax.set_facecolor('white')
    ax.grid(False)  # remove grid

    width = 0.0005  # bar width in days (adjust for 1-min data)
    width2 = 0.0001

    for idx, row in df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'black'
        # draw the candle body
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
                [row['Low'], row['High']],
                color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((mdates.date2num(idx)-width/2, min(row['Open'], row['Close'])),
                                   width,
                                   abs(row['Close']-row['Open']),
                                   facecolor=color))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Candlestick Chart")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser("ICT-style London range -> NY OTE 0.77 backtest (1m)")
    p.add_argument("--excel", required=True, help="Path to data file (CSV). MetaTrader format supported.")
    p.add_argument("--tz", default=DEFAULT_TZ, help="Timezone of the timestamps, e.g., America/New_York")
    p.add_argument("--outdir", default="./out", help="Where to write results")
    p.add_argument("--london", default=f"{LONDON_START}-{LONDON_END}", help="London window HH:MM-HH:MM")
    p.add_argument("--ny", default=f"{NY_START}-{NY_END}", help="NY window HH:MM-HH:MM")
    p.add_argument("--pivot", type=int, default=PIVOT_K)
    p.add_argument("--ote", type=float, default=OTE_LEVEL)
    p.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    p.add_argument("--capital", type=float, default=START_EQUITY)
    return p.parse_args()

# ---------------------------
# Helpers
# ---------------------------
def parse_window(s: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
    """Support hh, hh:mm, hh:mm:ss for session windows"""
    def fix_time(t: str):
        parts = t.split(":")
        if len(parts) == 1:    # hh
            t = f"{parts[0]}:00:00"
        elif len(parts) == 2:  # hh:mm
            t = f"{parts[0]}:{parts[1]}:00"
        return pd.to_timedelta(t)
    a, b = s.split("-")
    return fix_time(a), fix_time(b)

def load_mt_csv(path: str, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["Date","Time","Open","High","Low","Close","Volume"])
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('DateTime', inplace=True)
    df.index = df.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    df = df[["Open","High","Low","Close"]].astype(float).sort_index()
    return df

def pivot_highs_lows(df: pd.DataFrame, k: int) -> Tuple[pd.Series, pd.Series]:
    if k % 2 == 0: raise ValueError("pivot window must be odd")
    roll_max = df["High"].rolling(k, center=True).max()
    roll_min = df["Low"].rolling(k, center=True).min()
    return (df["High"]==roll_max).fillna(False), (df["Low"]==roll_min).fillna(False)

def session_slice(day_df: pd.DataFrame, start: pd.Timedelta, end: pd.Timedelta) -> pd.DataFrame:
    d0 = day_df.index[0].normalize()
    s, e = d0 + start, d0 + end
    return day_df.loc[(day_df.index>=s) & (day_df.index<=e)]

def first_touch_after(window_df: pd.DataFrame, series_df: pd.DataFrame, level: float, side: str) -> Optional[pd.Timestamp]:
    if series_df.empty: return None
    hit = series_df[series_df["High"]>=level] if side=="high" else series_df[series_df["Low"]<=level]
    return hit.index[0] if not hit.empty else None

# ---------------------------
# Strategy logic (unchanged)
# ---------------------------

# ---------------------------
# Output
# ---------------------------
def performance_summary(trades: List[Trade], eq: pd.Series) -> dict:
    ret = eq.diff().fillna(0.0)
    sharpe = (ret.mean()/(ret.std(ddof=0)+1e-12))*np.sqrt(252*24*60)
    dd = (eq-eq.cummax()).min()
    wins = sum(1 for t in trades if t.pnl_dollars>0)
    losses = sum(1 for t in trades if t.pnl_dollars<=0)
    pf = (sum(t.pnl_dollars for t in trades if t.pnl_dollars>0)/abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars<=0))) if losses>0 else float("inf")
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": 100.0*wins/max(1,wins+losses),
        "net_pnl_dollars": float(eq.iloc[-1]-eq.iloc[0]),
        "final_equity": float(eq.iloc[-1]),
        "profit_factor": float(pf),
        "sharpe_proxy": float(sharpe),
        "max_drawdown_dollars": float(dd),
    }

# def save_outputs(outdir: str, trades: List[Trade], eq_df: pd.DataFrame, price_df: pd.DataFrame):
#     os.makedirs(outdir, exist_ok=True)
#     pd.DataFrame([asdict(t) for t in trades]).to_csv(os.path.join(outdir,"trades.csv"),index=False)
#     with open(os.path.join(outdir,"summary.json"),"w",encoding="utf-8") as f:
#         json.dump(performance_summary(trades, eq_df["equity"]),f,indent=2)
#     eq_df.to_csv(os.path.join(outdir,"equity_curve.csv"))
#     plt.figure(); price_df["Close"].plot(); plt.title("Price (Close)"); plt.tight_layout(); plt.savefig(os.path.join(outdir,"price.png"),dpi=150); plt.close()
#     plt.figure(); eq_df["equity"].plot(); plt.title("Equity ($)"); plt.tight_layout(); plt.savefig(os.path.join(outdir,"equity_curve.png"),dpi=150); plt.close()
import plotly.graph_objects as go

def plot_candles_plotly(df, outpath="candles.html"):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='black'
    )])

    # Layout settings for interactivity
    fig.update_layout(
        title="Gold 1-min Candlesticks",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True),  # allows zoom horizontally
            type="date"
        ),
        yaxis=dict(fixedrange=False),  # allows vertical zoom
        dragmode='pan'  # allows panning with mouse
    )

    fig.write_html(outpath)
    fig.show()  # optional: open in browser immediately
    print(f"Interactive chart saved to {outpath}")

def save_outputs(outdir: str, trades: List[Trade], eq_df: pd.DataFrame, price_df: pd.DataFrame):
    import os
    os.makedirs(outdir, exist_ok=True)

    # trades.csv
    # tdf = pd.DataFrame([asdict(t) for t in trades])
    tdf = pd.DataFrame(trades)

    tdf.to_csv(os.path.join(outdir, "trades.csv"), index=False)

    # summary.json
    summary = performance_summary(trades, eq_df["equity"])
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # equity_curve.csv
    eq_df.to_csv(os.path.join(outdir, "equity_curve.csv"))

    # Equity plot
    plt.figure()
    eq_df["equity"].plot()
    plt.title("Equity ($)")
    plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
    plt.close()

    # Price candlestick chart
    # plot_candles(price_df, outpath=os.path.join(outdir, "candles.png"))
    plot_candles_plotly(price_df, outpath=os.path.join(outdir, "candles.html"))
    
def calculate_trends(df_1m: pd.DataFrame, pivot_k_1h=7, pivot_k_4h=7):
    """
    Calculate 1H and 4H trend based on pivot highs/lows.
    Returns a dict with keys '1h_trend' and '4h_trend' for each datetime in df_1m.
    """
    import pandas as pd

    # Resample to 1H and 4H
    df_1h = df_1m.resample("1H").agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    df_4h = df_1m.resample("4H").agg({'Open':'first','High':'max','Low':'min','Close':'last'})

    # Pivot detection helper
    def detect_trend(df, pivot_k):
        is_ph, is_pl = pivot_highs_lows(df, pivot_k)
        trend_dict = {}
        last_trend = None
        for t in df.index:
            # Look for last pivot before this bar
            ph_idx = is_ph.loc[:t]
            pl_idx = is_pl.loc[:t]
            if len(ph_idx[ph_idx]) and len(pl_idx[pl_idx]):
                last_ph = df.loc[ph_idx[ph_idx].index[-1]]['High']
                last_pl = df.loc[pl_idx[pl_idx].index[-1]]['Low']
                if df.loc[t,'Close'] > last_ph:
                    last_trend = "UP"
                elif df.loc[t,'Close'] < last_pl:
                    last_trend = "DOWN"
            trend_dict[t] = last_trend
        return trend_dict

    trend_1h = detect_trend(df_1h, pivot_k_1h)
    trend_4h = detect_trend(df_4h, pivot_k_4h)

    # Map back to 1-min index
    trend_1h_full = df_1m.index.to_series().apply(lambda x: trend_1h[max([k for k in trend_1h if k <= x])])
    trend_4h_full = df_1m.index.to_series().apply(lambda x: trend_4h[max([k for k in trend_4h if k <= x])])

    return pd.DataFrame({
        '1h_trend': trend_1h_full,
        '4h_trend': trend_4h_full
    })

def backtest_ict_london_bos(
    df: pd.DataFrame,
    london_window,
    ny_window,
    pivot_k: int,
    trend_df: pd.DataFrame,
    ote_ratio: float = 0.236,
):
    """
    Backtest ICT London BOS strategy with fixed PnL:
      - Loss = -100
      - Win  = 324
    """
    is_ph, is_pl = pivot_highs_lows(df, pivot_k)

    def last_pl_before(t):
        idx = is_pl.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None

    def last_ph_before(t):
        idx = is_ph.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None

    trades = []
    total_days_triggered = 0

    # Merge trend_df into main df once at the start
    if "1h_trend" in df.columns:
        df = df.drop(columns=["1h_trend"])
    if "4h_trend" in df.columns:
        df = df.drop(columns=["4h_trend"])
    df = df.join(trend_df)

    for day, day_df in df.groupby(df.index.date):
        day_df = df.loc[str(day)]
        if day_df.empty:
            continue

        lon = session_slice(day_df, *london_window)
        if lon.empty:
            continue
        london_high = float(lon["High"].max())
        london_low  = float(lon["Low"].min())
        london_high_time = lon["High"].idxmax()
        london_low_time  = lon["Low"].idxmin()

        ny = session_slice(day_df, *ny_window)
        if ny.empty:
            continue

        # Bearish variables
        london_high_taken_time = None
        bearish_BOS_time = None
        pivot_high_BOS = None
        pivot_low_BOS  = None
        bearish_retrace_time = None
        bearish_low_between = None
        bearish_ote_high = None
        bearish_ote_low  = None
        bearish_ote_level = None
        bearish_entry_made = False

        # Bullish variables
        london_low_taken_time = None
        bullish_BOS_time = None
        pivot_low_BOS  = None
        pivot_high_BOS = None
        bullish_retrace_time = None
        bullish_high_between = None
        bullish_ote_low  = None
        bullish_ote_high = None
        bullish_ote_level = None
        bullish_entry_made = False

        day_had_trade = False

        for t, row in ny.iterrows():
            # London sweep
            if london_high_taken_time is None and row["High"] > london_high:
                london_high_taken_time = t
            if london_low_taken_time is None and row["Low"] < london_low:
                london_low_taken_time = t

            # ---------------- Bearish ----------------
            if london_high_taken_time is not None and bearish_BOS_time is None:
                pl_t = last_pl_before(t)
                ph_t = last_ph_before(t)
                if pl_t and ph_t:
                    b_lvl = float(df.loc[pl_t, "Low"])
                    h_lvl = float(df.loc[ph_t, "High"])
                    if row["Close"] < b_lvl:
                        bearish_BOS_time = t
                        pivot_low_BOS = b_lvl
                        pivot_high_BOS = h_lvl

            if bearish_BOS_time is not None and bearish_retrace_time is None and t > bearish_BOS_time:
                if row["High"] >= pivot_low_BOS:
                    bearish_retrace_time = t
                    sub = df.loc[bearish_BOS_time:bearish_retrace_time]
                    bearish_low_between = float(sub["Low"].min())
                    bearish_ote_high = float(pivot_high_BOS)
                    bearish_ote_low  = float(bearish_low_between)
                    bearish_ote_level = bearish_ote_high - (bearish_ote_high - bearish_ote_low) * ote_ratio

            if bearish_ote_level is not None and not bearish_entry_made and t > bearish_retrace_time:
                if row["High"] >= bearish_ote_level:
                    entry_time = t
                    entry_price = float(max(bearish_ote_level, min(row["High"], max(row["Open"], row["Close"]))))
                    sl = float(bearish_ote_high)
                    tp = float(bearish_ote_low)
                    post = ny.loc[ny.index >= entry_time]
                    exit_time, exit_price, exit_reason = None, None, None
                    pnl, rr_val = 0.0, 0.0
                    for tt, rr in post.iterrows():
                        if rr["High"] >= sl:
                            exit_time, exit_price, exit_reason = tt, sl, "sl"
                            pnl = -100.0
                            rr_val = -1.0
                            break
                        if rr["Low"] <= tp:
                            exit_time, exit_price, exit_reason = tt, tp, "tp"
                            pnl = 324.0
                            rr_val = 3.24
                            break
                    if exit_time is None:
                        exit_time = ny.index[-1]
                        exit_price = float(ny.iloc[-1]["Close"])
                        exit_reason = "close"
                        pnl = 0.0
                        rr_val = 0.0

                    trades.append({
                        "day": pd.Timestamp(day),
                        "side": "short",
                        "entry_time": entry_time, "entry_price": entry_price,
                        "sl": sl, "tp": tp,
                        "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
                        "pnl_dollars": pnl, "rr": rr_val,
                        "london_high": london_high, "london_high_time": london_high_time,
                        "london_high_taken_time": london_high_taken_time,
                        "bos_time": bearish_BOS_time,
                        "pivot_high_BOS": float(pivot_high_BOS),
                        "pivot_low_BOS": float(pivot_low_BOS),
                        "retrace_time": bearish_retrace_time,
                        "ote_level": float(bearish_ote_level),
                        "ote_high_used": float(bearish_ote_high),
                        "ote_low_used": float(bearish_ote_low),
                        "4h_trend": row.get("4h_trend", "NA"),
                        "1h_trend": row.get("1h_trend", "NA"),
                    })
                    day_had_trade = True
                    bearish_entry_made = True

            # ---------------- Bullish ----------------
            if london_low_taken_time is not None and bullish_BOS_time is None:
                ph_t = last_ph_before(t)
                pl_t = last_pl_before(t)
                if ph_t and pl_t:
                    b_lvl = float(df.loc[ph_t, "High"])
                    l_lvl = float(df.loc[pl_t, "Low"])
                    if row["Close"] > b_lvl:
                        bullish_BOS_time = t
                        pivot_high_BOS = b_lvl
                        pivot_low_BOS  = l_lvl

            if bullish_BOS_time is not None and bullish_retrace_time is None and t > bullish_BOS_time:
                if row["Low"] <= pivot_high_BOS:
                    bullish_retrace_time = t
                    sub = df.loc[bullish_BOS_time:bullish_retrace_time]
                    bullish_high_between = float(sub["High"].max())
                    bullish_ote_low  = float(pivot_low_BOS)
                    bullish_ote_high = float(bullish_high_between)
                    bullish_ote_level = bullish_ote_low + (bullish_ote_high - bullish_ote_low) * ote_ratio

            if bullish_ote_level is not None and not bullish_entry_made and t > bullish_retrace_time:
                if row["Low"] <= bullish_ote_level:
                    entry_time = t
                    entry_price = float(min(bullish_ote_level, max(row["Low"], min(row["Open"], row["Close"]))))
                    sl = float(bullish_ote_low)
                    tp = float(bullish_ote_high)
                    post = ny.loc[ny.index >= entry_time]
                    exit_time, exit_price, exit_reason = None, None, None
                    pnl, rr_val = 0.0, 0.0
                    for tt, rr in post.iterrows():
                        if rr["Low"] <= sl:
                            exit_time, exit_price, exit_reason = tt, sl, "sl"
                            pnl = -100.0
                            rr_val = -1.0
                            break
                        if rr["High"] >= tp:
                            exit_time, exit_price, exit_reason = tt, tp, "tp"
                            pnl = 324.0
                            rr_val = 3.24
                            break
                    if exit_time is None:
                        exit_time = ny.index[-1]
                        exit_price = float(ny.iloc[-1]["Close"])
                        exit_reason = "close"
                        pnl = 0.0
                        rr_val = 0.0

                    trades.append({
                        "day": pd.Timestamp(day),
                        "side": "long",
                        "entry_time": entry_time, "entry_price": entry_price,
                        "sl": sl, "tp": tp,
                        "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
                        "pnl_dollars": pnl, "rr": rr_val,
                        "london_low": london_low, "london_low_time": london_low_time,
                        "london_low_taken_time": london_low_taken_time,
                        "bos_time": bullish_BOS_time,
                        "pivot_low_BOS": float(pivot_low_BOS),
                        "pivot_high_BOS": float(pivot_high_BOS),
                        "retrace_time": bullish_retrace_time,
                        "ote_level": float(bullish_ote_level),
                        "ote_low_used": float(bullish_ote_low),
                        "ote_high_used": float(bullish_ote_high),
                        "4h_trend": row.get("4h_trend", "NA"),
                        "1h_trend": row.get("1h_trend", "NA"),
                    })
                    day_had_trade = True
                    bullish_entry_made = True

        if day_had_trade:
            total_days_triggered += 1

    if trades:
        tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
        wins = int((tdf["exit_reason"] == "tp").sum())
        losses = int((tdf["exit_reason"] == "sl").sum())
        closes = int((tdf["exit_reason"] == "close").sum())
        wr = 100.0 * wins / max(1, wins + losses)
        summary = {
            "trades": len(tdf),
            "wins": wins,
            "losses": losses,
            "closed_no_hit": closes,
            "win_rate_pct": wr,
            "net_pnl_dollars": float(tdf["pnl_dollars"].sum()),
            "days_triggered": total_days_triggered,
        }
    else:
        tdf = pd.DataFrame(columns=[
            "day","side","entry_time","entry_price","sl","tp",
            "exit_time","exit_price","exit_reason","pnl_dollars","rr",
            "bos_time","retrace_time","ote_level","ote_low_used","ote_high_used",
            "pivot_low_BOS","pivot_high_BOS",
            "london_high","london_high_time","london_high_taken_time",
            "london_low","london_low_time","london_low_taken_time",
            "1h_trend","4h_trend"
        ])
        summary = {
            "trades": 0, "wins": 0, "losses": 0, "closed_no_hit": 0,
            "win_rate_pct": 0.0, "net_pnl_dollars": 0.0, "days_triggered": 0
        }

    # Print trades with 1H and 4H trend
    for _, r in tdf.iterrows():
        print(f"\n{r['day'].date()} [{r['side'].upper()}]")
        print(f"  Entry @ {r['entry_time']}  price={r['entry_price']:.3f}  SL={r['sl']:.3f}  TP={r['tp']:.3f}")
        print(f"  BOS @ {r['bos_time']}   Retrace @ {r['retrace_time']}")
        print(f"  OTE {ote_ratio:.3f} level={r['ote_level']:.3f}  (low used={r.get('ote_low_used', float('nan')):.3f}, high used={r.get('ote_high_used', float('nan')):.3f})")
        print(f"  Exit @ {r['exit_time']}  price={r['exit_price']:.3f}  reason={r['exit_reason']}")
        print(f"  PnL=${r['pnl_dollars']:.2f}  RR={r['rr']:.2f}")
        print(f"  4h: {r.get('4h_trend', 'NA')}, 1h: {r.get('1h_trend', 'NA')}")

    print("\nSummary:", summary)
    return tdf, summary

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)
    df = load_mt_csv(args.excel, args.tz)

    # Resample 1H and 4H from 1-minute df
    df_1h = df['Close'].resample("1h").ohlc()
    df_4h = df['Close'].resample("4h").ohlc()

    # Define trend: 1 = bull, -1 = bear
    df_1h['trend'] = df_1h['close'] - df_1h['open']
    df_1h['trend'] = df_1h['trend'].apply(lambda x: 1 if x > 0 else -1)
    df_4h['trend'] = df_4h['close'] - df_4h['open']
    df_4h['trend'] = df_4h['trend'].apply(lambda x: 1 if x > 0 else -1)

    # Merge into main df
    trend_df = pd.DataFrame(index=df.index)
    trend_df['1h_trend'] = df_1h['trend'].reindex(df.index, method='ffill')
    trend_df['4h_trend'] = df_4h['trend'].reindex(df.index, method='ffill')

    # Call backtest with trend_df
    trades, summary = backtest_ict_london_bos(df, london_window, ny_window, args.pivot, trend_df)

    print("Backtest complete.")

if __name__=="__main__":
    main()
