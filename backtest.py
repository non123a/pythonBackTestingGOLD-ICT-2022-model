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
PIVOT_K = 9
OTE_LEVEL = 0.77
SL_MULT_DEFAULT = 1.0
TP_MULT_DEFAULT = 1.0
TF_FILTER_DEFAULT = "none"
RISK_PER_TRADE = 100.0

def parse_args():
    p = argparse.ArgumentParser("ICT-style London range -> NY OTE 0.77 backtest (1m)")
    p.add_argument("--excel", required=True, help="Path to data file (CSV). MetaTrader format supported.")
    p.add_argument("--tz", default=DEFAULT_TZ, help="Timezone of the timestamps, e.g., America/New_York")
    p.add_argument("--outdir", default="./out", help="Where to write results")
    p.add_argument("--london", default=f"{LONDON_START}-{LONDON_END}", help="London window HH:MM-HH:MM")
    p.add_argument("--ny", default=f"{NY_START}-{NY_END}", help="NY window HH:MM-HH:MM")
    p.add_argument("--pivot", type=int, default=PIVOT_K)
    p.add_argument("--ote", type=float, default=OTE_LEVEL)
    p.add_argument("--sl_mult", type=float, default=SL_MULT_DEFAULT)
    p.add_argument("--tp_mult", type=float, default=TP_MULT_DEFAULT)
    p.add_argument("--tf_filter", type=str, default=TF_FILTER_DEFAULT, choices=["none", "1h", "4h", "1h+4h"])
    p.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    return p.parse_args()

def parse_window(s: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
    def fix_time(t: str):
        parts = t.split(":")
        if len(parts) == 1:
            t = f"{parts[0]}:00:00"
        elif len(parts) == 2:
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

def calculate_trends(df_1m: pd.DataFrame, pivot_k_1h=7, pivot_k_4h=7):
    # Fix the FutureWarning in pandas
    df_1h = df_1m.resample("1h").agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    df_4h = df_1m.resample("4h").agg({'Open':'first','High':'max','Low':'min','Close':'last'})

    def detect_trend(df, pivot_k):
        is_ph, is_pl = pivot_highs_lows(df, pivot_k)
        trend_dict = {}
        last_trend = None
        for t in df.index:
            ph_idx = is_ph.loc[:t]
            pl_idx = is_pl.loc[:t]
            if len(ph_idx[ph_idx]) and len(pl_idx[pl_idx]):
                last_ph_t = ph_idx[ph_idx].index[-1]
                last_pl_t = pl_idx[pl_idx].index[-1]
                if last_ph_t > last_pl_t:
                    last_trend = "Downtrend"
                elif last_pl_t > last_ph_t:
                    last_trend = "Uptrend"
            trend_dict[t] = last_trend
        return trend_dict

    trend_1h = detect_trend(df_1h.dropna(), pivot_k_1h)
    trend_4h = detect_trend(df_4h.dropna(), pivot_k_4h)

    trend_1h_full = pd.Series(trend_1h).reindex(df_1m.index, method='ffill')
    trend_4h_full = pd.Series(trend_4h).reindex(df_1m.index, method='ffill')

    return pd.DataFrame({
        '1h_trend': trend_1h_full,
        '4h_trend': trend_4h_full
    }).ffill()

def is_valid_trend(row, tf_filter):
    trend_4h = row.get("4h_trend")
    trend_1h = row.get("1h_trend")
    
    if tf_filter == "none":
        return True, True
    elif tf_filter == "1h":
        return trend_1h == "Uptrend", trend_1h == "Downtrend"
    elif tf_filter == "4h":
        return trend_4h == "Uptrend", trend_4h == "Downtrend"
    elif tf_filter == "1h+4h":
        return (trend_1h == "Uptrend" and trend_4h == "Uptrend"), \
               (trend_1h == "Downtrend" and trend_4h == "Downtrend")
    return False, False

def backtest_ict_london_bos(
    df: pd.DataFrame,
    london_window: Tuple[pd.Timedelta, pd.Timedelta],
    ny_window: Tuple[pd.Timedelta, pd.Timedelta],
    pivot_k: int,
    ote_ratio: float,
    sl_mult: float,
    tp_mult: float,
    tf_filter: str,
    risk_per_trade: float
):
    is_ph, is_pl = pivot_highs_lows(df, pivot_k)

    def last_pl_before(t):
        idx = is_pl.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None

    def last_ph_before(t):
        idx = is_ph.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None

    trend_df = calculate_trends(df)
    df = df.join(trend_df)
    
    trades = []
    total_days_triggered = 0

    for day, day_df in df.groupby(df.index.date):
        day_df = df.loc[str(day)]
        if day_df.empty:
            continue

        lon = session_slice(day_df, *london_window)
        if lon.empty:
            continue
        london_high = float(lon["High"].max())
        london_low  = float(lon["Low"].min())

        ny = session_slice(day_df, *ny_window)
        if ny.empty:
            continue
        
        # --- NEW: Always use 1-minute TF for entry logic ---
        entry_ny = ny.copy()
        
        london_high_taken_time = None
        bearish_BOS_time = None
        bearish_entry_made = False

        london_low_taken_time = None
        bullish_BOS_time = None
        bullish_entry_made = False

        day_had_trade = False

        for t, row in entry_ny.iterrows():
            if london_high_taken_time is None and row["High"] > london_high:
                london_high_taken_time = t
            if london_low_taken_time is None and row["Low"] < london_low:
                london_low_taken_time = t

            # ---------------- Bearish ----------------
            if london_high_taken_time is not None and bearish_BOS_time is None:
                pl_t = last_pl_before(t)
                ph_t = last_ph_before(t)
                if pl_t and ph_t:
                    pivot_low_BOS = float(df.loc[pl_t, "Low"])
                    if row["Close"] < pivot_low_BOS:
                        bearish_BOS_time = t
                        pivot_high_BOS = float(df.loc[ph_t, "High"])

            if bearish_BOS_time is not None and not bearish_entry_made and t > bearish_BOS_time:
                retrace_candles = entry_ny.loc[bearish_BOS_time:t]
                lowest_retrace_low = float(retrace_candles["Low"].min())
                
                # Check OTE entry condition
                bearish_ote_level = pivot_high_BOS - (pivot_high_BOS - lowest_retrace_low) * ote_ratio
                
                if row["High"] >= bearish_ote_level:
                    is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                    if is_bearish:
                        entry_time = t
                        entry_price = float(max(bearish_ote_level, min(row["High"], max(row["Open"], row["Close"]))))

                        raw_sl_price = pivot_high_BOS
                        risk_in_price = abs(entry_price - raw_sl_price)
                        sl = entry_price + (risk_in_price * sl_mult)
                        tp = entry_price - (risk_in_price * tp_mult)
                        
                        if abs(sl - entry_price) > 0:
                            trade_size = risk_per_trade / abs(sl - entry_price)
                        else:
                            trade_size = 0
                        
                        exit_time, exit_price, exit_reason = None, None, None
                        pnl_dollars, rr_val = 0.0, 0.0
                        post = entry_ny.loc[entry_ny.index >= entry_time]
                        
                        for tt, rr in post.iterrows():
                            if rr["High"] >= sl:
                                exit_time, exit_price, exit_reason = tt, sl, "sl"
                                pnl_dollars = (entry_price - sl) * trade_size
                                rr_val = pnl_dollars / risk_per_trade if risk_per_trade > 0 else 0
                                break
                            if rr["Low"] <= tp:
                                exit_time, exit_price, exit_reason = tt, tp, "tp"
                                pnl_dollars = (entry_price - tp) * trade_size
                                rr_val = pnl_dollars / risk_per_trade if risk_per_trade > 0 else 0
                                break
                        
                        if exit_time is None:
                            exit_time = entry_ny.index[-1]
                            exit_price = float(entry_ny.iloc[-1]["Close"])
                            exit_reason = "close"
                            pnl_dollars = (entry_price - exit_price) * trade_size
                            rr_val = 0.0
                            
                        trades.append({
                            "day": pd.Timestamp(day),
                            "side": "short",
                            "entry_time": entry_time, "entry_price": entry_price,
                            "sl": sl, "tp": tp,
                            "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
                            "pnl_dollars": pnl_dollars, "rr": rr_val,
                            "ote_high_used": float(pivot_high_BOS),
                            "ote_low_used": lowest_retrace_low,
                            "4h_trend": row.get("4h_trend", "NA"),
                            "1h_trend": row.get("1h_trend", "NA"),
                            "pivot_high_BOS": pivot_high_BOS,
                            "pivot_low_BOS": pivot_low_BOS,
                        })
                        day_had_trade = True
                        bearish_entry_made = True
            
            # ---------------- Bullish ----------------
            if london_low_taken_time is not None and bullish_BOS_time is None:
                ph_t = last_ph_before(t)
                pl_t = last_pl_before(t)
                if ph_t and pl_t:
                    pivot_high_BOS = float(df.loc[ph_t, "High"])
                    if row["Close"] > pivot_high_BOS:
                        bullish_BOS_time = t
                        pivot_low_BOS = float(df.loc[pl_t, "Low"])
            
            if bullish_BOS_time is not None and not bullish_entry_made and t > bullish_BOS_time:
                retrace_candles = entry_ny.loc[bullish_BOS_time:t]
                highest_retrace_high = float(retrace_candles["High"].max())

                # Check OTE entry condition
                bullish_ote_level = pivot_low_BOS + (highest_retrace_high - pivot_low_BOS) * ote_ratio
                
                if row["Low"] <= bullish_ote_level:
                    is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                    if is_bullish:
                        entry_time = t
                        entry_price = float(min(bullish_ote_level, max(row["Low"], min(row["Open"], row["Close"]))))

                        raw_sl_price = pivot_low_BOS
                        risk_in_price = abs(entry_price - raw_sl_price)
                        sl = entry_price - (risk_in_price * sl_mult)
                        tp = entry_price + (risk_in_price * tp_mult)
                        
                        if abs(sl - entry_price) > 0:
                            trade_size = risk_per_trade / abs(sl - entry_price)
                        else:
                            trade_size = 0
                        
                        exit_time, exit_price, exit_reason = None, None, None
                        pnl_dollars, rr_val = 0.0, 0.0
                        post = entry_ny.loc[entry_ny.index >= entry_time]

                        for tt, rr in post.iterrows():
                            if rr["Low"] <= sl:
                                exit_time, exit_price, exit_reason = tt, sl, "sl"
                                pnl_dollars = (sl - entry_price) * trade_size
                                rr_val = pnl_dollars / risk_per_trade if risk_per_trade > 0 else 0
                                break
                            if rr["High"] >= tp:
                                exit_time, exit_price, exit_reason = tt, tp, "tp"
                                pnl_dollars = (tp - entry_price) * trade_size
                                rr_val = pnl_dollars / risk_per_trade if risk_per_trade > 0 else 0
                                break
                        
                        if exit_time is None:
                            exit_time = entry_ny.index[-1]
                            exit_price = float(entry_ny.iloc[-1]["Close"])
                            exit_reason = "close"
                            pnl_dollars = (exit_price - entry_price) * trade_size
                            rr_val = 0.0

                        trades.append({
                            "day": pd.Timestamp(day),
                            "side": "long",
                            "entry_time": entry_time, "entry_price": entry_price,
                            "sl": sl, "tp": tp,
                            "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
                            "pnl_dollars": pnl_dollars, "rr": rr_val,
                            "ote_low_used": float(pivot_low_BOS),
                            "ote_high_used": highest_retrace_high,
                            "4h_trend": row.get("4h_trend", "NA"),
                            "1h_trend": row.get("1h_trend", "NA"),
                            "pivot_low_BOS": pivot_low_BOS,
                            "pivot_high_BOS": pivot_high_BOS,
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
        tdf = pd.DataFrame()
        summary = {
            "trades": 0, "wins": 0, "losses": 0, "closed_no_hit": 0,
            "win_rate_pct": 0.0, "net_pnl_dollars": 0.0, "days_triggered": 0
        }
    
    # Printing the results
    print("\n----------------------")
    print("      TRADES LOG      ")
    print("----------------------")
    for _, r in tdf.iterrows():
        print(f"\n[{r['side'].upper()}] {r['day'].date()} @ {r['entry_time'].strftime('%H:%M')}")
        print(f"  Entry: {r['entry_price']:.3f} | SL: {r['sl']:.3f} | TP: {r['tp']:.3f}")
        print(f"  PnL: ${r['pnl_dollars']:.2f} | RR: {r['rr']:.2f} | Exit: {r['exit_reason']}")
        print(f"  TF Filter: {tf_filter} (4h trend: {r.get('4h_trend', 'NA')}, 1h trend: {r.get('1h_trend', 'NA')})")
    
    print("\n----------------------")
    print("    SUMMARY RESULTS   ")
    print("----------------------")
    print(f"Total Trades: {summary['trades']}")
    print(f"Win Rate: {summary['win_rate_pct']:.2f}%")
    print(f"Net PnL: ${summary['net_pnl_dollars']:.2f}")
    print("----------------------")
    
    return tdf, summary

def main():
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)
    df = load_mt_csv(args.excel, args.tz)

    trades, summary = backtest_ict_london_bos(
        df,
        london_window=london_window,
        ny_window=ny_window,
        pivot_k=args.pivot,
        ote_ratio=args.ote,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        tf_filter=args.tf_filter,
        risk_per_trade=args.risk
    )

if __name__=="__main__":
    main()