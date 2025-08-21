#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import optuna

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
    result: str = ""
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

def parse_args():
    p = argparse.ArgumentParser("ICT-style London range -> NY OTE backtest (1m)")
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

def performance_summary(trades: List[dict]) -> dict:
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "closed_no_hit": 0,
            "win_rate_pct": 0.0, "net_pnl_dollars": 0.0, "days_triggered": 0
        }
    
    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    
    wins = (tdf["pnl_dollars"] > 0).sum()
    losses = (tdf["pnl_dollars"] <= 0).sum()
    
    summary = {
        "trades": len(tdf),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": 100.0 * wins / max(1, wins + losses),
        "net_pnl_dollars": float(tdf["pnl_dollars"].sum())
    }
    return summary

def calculate_trends(df_1m: pd.DataFrame, pivot_k_1h=7, pivot_k_4h=7):
    # Resample to 1H and 4H
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
        return True, True  # Bullish and Bearish are both allowed
    elif tf_filter == "1h":
        return trend_1h == "Uptrend", trend_1h == "Downtrend"
    elif tf_filter == "4h":
        return trend_4h == "Uptrend", trend_4h == "Downtrend"
    elif tf_filter == "1h+4h":
        return (trend_1h == "Uptrend" and trend_4h == "Uptrend"), \
               (trend_1h == "Downtrend" and trend_4h == "Downtrend")
    return False, False

def backtest_ict_london_bos(df, london_window, ny_window, pivot, sl_mult, tp_mult, ote, tf_filter, risk_per_trade):
    is_ph, is_pl = pivot_highs_lows(df, pivot)

    def last_pl_before(t):
        idx = is_pl.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None
    def last_ph_before(t):
        idx = is_ph.loc[:t]
        return idx[idx].index[-1] if len(idx[idx]) else None

    trend_df = calculate_trends(df)
    df = df.join(trend_df)
    
    trades = []

    for day, day_df in df.groupby(df.index.date):
        day_df = df.loc[str(day)]
        if day_df.empty: continue

        lon = session_slice(day_df, *london_window)
        if lon.empty: continue
        london_high = float(lon["High"].max())
        london_low  = float(lon["Low"].min())
        london_high_time = lon["High"].idxmax()
        london_low_time  = lon["Low"].idxmin()

        ny = session_slice(day_df, *ny_window)
        if ny.empty: continue

        london_high_taken_time = None
        bearish_BOS_time = None
        bearish_retrace_time = None
        bearish_ote_level = None
        bearish_entry_made = False

        london_low_taken_time = None
        bullish_BOS_time = None
        bullish_retrace_time = None
        bullish_ote_level = None
        bullish_entry_made = False

        for t, row in ny.iterrows():
            if london_high_taken_time is None and row["High"] > london_high:
                london_high_taken_time = t
            if london_low_taken_time is None and row["Low"] < london_low:
                london_low_taken_time = t

            # --- Bearish ---
            if london_high_taken_time is not None and bearish_BOS_time is None:
                pl_t = last_pl_before(t)
                ph_t = last_ph_before(t)
                if pl_t and ph_t:
                    b_lvl = float(df.loc[pl_t, "Low"])
                    if row["Close"] < b_lvl:
                        bearish_BOS_time = t
            
            if bearish_BOS_time is not None and bearish_retrace_time is None and t > bearish_BOS_time:
                ph_t = last_ph_before(t)
                if ph_t:
                    h_lvl = float(df.loc[ph_t, "High"])
                    if row["High"] >= h_lvl:
                        bearish_retrace_time = t
                        sub = ny.loc[bearish_BOS_time:bearish_retrace_time]
                        bearish_low_between = float(sub["Low"].min())
                        bearish_ote_high = float(h_lvl)
                        bearish_ote_low  = float(bearish_low_between)
                        bearish_ote_level = bearish_ote_high - (bearish_ote_high - bearish_ote_low) * ote

            if bearish_ote_level is not None and not bearish_entry_made and t > bearish_retrace_time:
                is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                if row["High"] >= bearish_ote_level and is_bearish:
                    entry_time = t
                    entry_price = float(max(bearish_ote_level, min(row["High"], max(row["Open"], row["Close"]))))

                    # --- NEW LOGIC for SL/TP ---
                    risk_in_price = abs(entry_price - bearish_ote_high)
                    sl = entry_price + (risk_in_price * sl_mult)
                    tp = entry_price - (risk_in_price * tp_mult)
                    
                    post = ny.loc[ny.index >= entry_time]
                    pnl_dollars = 0.0
                    for tt, rr in post.iterrows():
                        if rr["High"] >= sl:
                            pnl_dollars = -risk_per_trade
                            break
                        if rr["Low"] <= tp:
                            pnl_dollars = tp_mult * risk_per_trade
                            break
                    
                    trades.append({
                        "entry_time": entry_time,
                        "pnl_dollars": pnl_dollars
                    })
                    bearish_entry_made = True

            # --- Bullish ---
            if london_low_taken_time is not None and bullish_BOS_time is None:
                ph_t = last_ph_before(t)
                pl_t = last_pl_before(t)
                if ph_t and pl_t:
                    b_lvl = float(df.loc[ph_t, "High"])
                    if row["Close"] > b_lvl:
                        bullish_BOS_time = t

            if bullish_BOS_time is not None and bullish_retrace_time is None and t > bullish_BOS_time:
                pl_t = last_pl_before(t)
                if pl_t:
                    l_lvl = float(df.loc[pl_t, "Low"])
                    if row["Low"] <= l_lvl:
                        bullish_retrace_time = t
                        sub = ny.loc[bullish_BOS_time:bullish_retrace_time]
                        bullish_high_between = float(sub["High"].max())
                        bullish_ote_low  = float(l_lvl)
                        bullish_ote_high = float(bullish_high_between)
                        bullish_ote_level = bullish_ote_low + (bullish_ote_high - bullish_ote_low) * ote
            
            if bullish_ote_level is not None and not bullish_entry_made and t > bullish_retrace_time:
                is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                if row["Low"] <= bullish_ote_level and is_bullish:
                    entry_time = t
                    entry_price = float(min(bullish_ote_level, max(row["Low"], min(row["Open"], row["Close"]))))

                    # --- NEW LOGIC for SL/TP ---
                    risk_in_price = abs(entry_price - bullish_ote_low)
                    sl = entry_price - (risk_in_price * sl_mult)
                    tp = entry_price + (risk_in_price * tp_mult)

                    post = ny.loc[ny.index >= entry_time]
                    pnl_dollars = 0.0
                    for tt, rr in post.iterrows():
                        if rr["Low"] <= sl:
                            pnl_dollars = -risk_per_trade
                            break
                        if rr["High"] >= tp:
                            pnl_dollars = tp_mult * risk_per_trade
                            break

                    trades.append({
                        "entry_time": entry_time,
                        "pnl_dollars": pnl_dollars
                    })
                    bullish_entry_made = True
    
    return trades, performance_summary(trades)


def main():
    print("Starting...")
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)
    df = load_mt_csv(args.excel, args.tz)

    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    def objective(trial):
        params = {
            "pivot": trial.suggest_int("pivot", 3, 20, step=2),
            "sl_mult": trial.suggest_float("sl_mult", 0.5, 3.0),
            "tp_mult": trial.suggest_float("tp_mult", 0.5, 5.0),
            "ote": trial.suggest_float("ote", 0.6, 0.8),
            "tf_filter": trial.suggest_categorical("tf_filter", ["none", "1h", "4h", "1h+4h"]),
        }
        trades, summary = backtest_ict_london_bos(
            df_train,
            london_window=london_window,
            ny_window=ny_window,
            **params,
            risk_per_trade=RISK_PER_TRADE
        )
        return summary["net_pnl_dollars"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print("\n✅ Best parameters (train):", study.best_params)
    print("📈 Best train score:", study.best_value)

    # --- Get and re-run top 3 trials for detailed results ---
    print("\nRe-running top 3 trials to get detailed backtest results...")
    
    trials_df = study.trials_dataframe().sort_values("value", ascending=False).head(3)
    
    top_results = []
    
    for index, row in trials_df.iterrows():
        trial_params = {
            "pivot": row["params_pivot"],
            "sl_mult": row["params_sl_mult"],
            "tp_mult": row["params_tp_mult"],
            "ote": row["params_ote"],
            "tf_filter": row["params_tf_filter"],
        }
        
        trades, summary = backtest_ict_london_bos(
            df_train,
            london_window=london_window,
            ny_window=ny_window,
            **trial_params,
            risk_per_trade=RISK_PER_TRADE
        )
        
        top_results.append({
            "rank": len(top_results) + 1,
            "net_profit": summary["net_pnl_dollars"],
            "win_rate_percent": summary["win_rate_pct"],
            "total_trades": summary["trades"],
            "pivot": trial_params["pivot"],
            "sl_mult": trial_params["sl_mult"],
            "tp_mult": trial_params["tp_mult"],
            "ote": trial_params["ote"],
            "tf_filter": trial_params["tf_filter"]
        })
        
    top_results_df = pd.DataFrame(top_results)
    
    output_path = "top_3_detailed_results.csv"
    top_results_df.to_csv(output_path, index=False)
    
    print(f"✅ Top 3 detailed results saved to {output_path}")
    print("\nTop 3 results:")
    print(top_results_df)

    # --- Final evaluation on unseen test data ---
    print("\nEvaluating on unseen test data with best parameters...")
    trades_test, summary_test = backtest_ict_london_bos(
        df_test,
        london_window=london_window,
        ny_window=ny_window,
        **study.best_params,
        risk_per_trade=RISK_PER_TRADE
    )

    print("\n📊 Test performance:", summary_test["net_pnl_dollars"])
    print("Backtest complete.")

if __name__=="__main__":
    main()