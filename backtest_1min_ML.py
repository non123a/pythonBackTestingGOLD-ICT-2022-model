#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import optuna

# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_TZ = "America/New_York"
LONDON_START, LONDON_END = "03:00", "07:00"
NY_START, NY_END = "07:00", "12:00"
RISK_PER_TRADE = 100.0
# The penalty for a trade that enters and exits on the same candle.
FLASH_TRADE_PENALTY = 1000

def parse_args():
    p = argparse.ArgumentParser("ICT-style London range -> NY OTE backtest (1m)")
    p.add_argument("--excel", required=True, help="Path to data file (CSV). MetaTrader format supported.")
    p.add_argument("--tz", default=DEFAULT_TZ, help="Timezone of the timestamps, e.g., America/New_York")
    p.add_argument("--outdir", default="./out", help="Where to write results")
    p.add_argument("--london", default=f"{LONDON_START}-{LONDON_END}", help="London window HH:MM-HH:MM")
    p.add_argument("--ny", default=f"{NY_START}-{NY_END}", help="NY window HH:MM-HH:MM")
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

def backtest_ict_london_bos(df, london_window, ny_window, pivot, sl_mult, tp_mult, ote, tf_filter, risk_per_trade):
    ote_ratio = 1.0 - ote
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
    flash_trades = 0
    total_days_triggered = 0

    for day, day_df in df.groupby(df.index.date):
        day_df = df.loc[str(day)]
        if day_df.empty: continue

        lon = session_slice(day_df, *london_window)
        if lon.empty: continue
        london_high = float(lon["High"].max())
        london_low  = float(lon["Low"].min())

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

        day_had_trade = False
        entry_ny = ny.copy()

        for t, row in entry_ny.iterrows():
            if london_high_taken_time is None and row["High"] > london_high:
                london_high_taken_time = t
            if london_low_taken_time is None and row["Low"] < london_low:
                london_low_taken_time = t

            # --- Bearish ---
            if london_high_taken_time is not None and bearish_BOS_time is None:
                pl_t = last_pl_before(t)
                ph_t = last_ph_before(t)
                if pl_t and ph_t:
                    pivot_low_BOS = float(df.loc[pl_t, "Low"])
                    if row["Close"] < pivot_low_BOS:
                        bearish_BOS_time = t
                        pivot_high_BOS = float(df.loc[ph_t, "High"])

            if bearish_BOS_time is not None and bearish_retrace_time is None and t > bearish_BOS_time:
                if pivot_high_BOS is not None:
                    retrace_candles = entry_ny.loc[bearish_BOS_time:t]
                    lowest_retrace_low = float(retrace_candles["Low"].min())
                    bearish_ote_level = pivot_high_BOS - (pivot_high_BOS - lowest_retrace_low) * ote_ratio
                    if row["High"] >= bearish_ote_level:
                        bearish_retrace_time = t

            if bearish_ote_level is not None and not bearish_entry_made and bearish_retrace_time is not None and t > bearish_retrace_time:
                is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                if row["High"] >= bearish_ote_level and is_bearish:
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
                    
                    if exit_time == entry_time:
                        flash_trades += 1
                        
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
                    })
                    day_had_trade = True
                    bearish_entry_made = True

            # --- Bullish ---
            if london_low_taken_time is not None and bullish_BOS_time is None:
                ph_t = last_ph_before(t)
                pl_t = last_pl_before(t)
                if ph_t and pl_t:
                    pivot_high_BOS = float(df.loc[ph_t, "High"])
                    if row["Close"] > pivot_high_BOS:
                        bullish_BOS_time = t
                        pivot_low_BOS = float(df.loc[pl_t, "Low"])
            
            if bullish_BOS_time is not None and bullish_retrace_time is None and t > bullish_BOS_time:
                if pivot_low_BOS is not None:
                    retrace_candles = entry_ny.loc[bullish_BOS_time:t]
                    highest_retrace_high = float(retrace_candles["High"].max())
                    bullish_ote_level = pivot_low_BOS + (highest_retrace_high - pivot_low_BOS) * ote_ratio
                    if row["Low"] <= bullish_ote_level:
                        bullish_retrace_time = t
            
            if bullish_ote_level is not None and not bullish_entry_made and bullish_retrace_time is not None and t > bullish_retrace_time:
                is_bullish, is_bearish = is_valid_trend(row, tf_filter)
                if row["Low"] <= bullish_ote_level and is_bullish:
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
                    
                    if exit_time == entry_time:
                        flash_trades += 1

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
            "flash_trades": flash_trades,
        }
    else:
        tdf = pd.DataFrame()
        summary = {
            "trades": 0, "wins": 0, "losses": 0, "closed_no_hit": 0,
            "win_rate_pct": 0.0, "net_pnl_dollars": 0.0, "days_triggered": 0,
            "flash_trades": flash_trades
        }
    
    return tdf, summary

def objective(trial, df_train, london_window, ny_window, risk_per_trade):
    params = {
        "pivot": trial.suggest_int("pivot", 3, 19, step=2),
        "sl_mult": trial.suggest_float("sl_mult", 0.5, 3.0),
        "tp_mult": trial.suggest_float("tp_mult", 0.5, 5.0),
        "ote": trial.suggest_float("ote", 0.6, 0.8),
        "tf_filter": trial.suggest_categorical("tf_filter", ["none", "1h", "4h", "1h+4h"]),
    }
    
    trades_df, summary = backtest_ict_london_bos(
        df_train,
        london_window=london_window,
        ny_window=ny_window,
        **params,
        risk_per_trade=risk_per_trade
    )
    
    penalty = summary["flash_trades"] * FLASH_TRADE_PENALTY
    return summary["net_pnl_dollars"] - penalty

def main():
    print("Starting...")
    args = parse_args()
    london_window = parse_window(args.london)
    ny_window = parse_window(args.ny)

    # --- Create output directory if it doesn't exist ---
    os.makedirs(args.outdir, exist_ok=True)
    study_db_path = f"sqlite:///{os.path.join(args.outdir, 'study.db')}"

    print("Loading data...")
    df = load_mt_csv(args.excel, args.tz)

    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    print(f"Running optimization on {len(df_train)} candles (70% of data)...")
    study = optuna.create_study(
        direction="maximize",
        storage=study_db_path,  # Save study to a file
        study_name="london-bos-backtest",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, df_train, london_window, ny_window, args.risk), n_trials=200)

    print("\n✅ Optimization complete.")
    print("📈 Best parameters found:", study.best_params)
    print("📊 Best profit (adjusted for penalties):", study.best_value)

    # --- Get final results for saving ---
    trades_train_df, summary_train = backtest_ict_london_bos(
        df_train,
        london_window=london_window,
        ny_window=ny_window,
        **study.best_params,
        risk_per_trade=args.risk
    )
    trades_test_df, summary_test = backtest_ict_london_bos(
        df_test,
        london_window=london_window,
        ny_window=ny_window,
        **study.best_params,
        risk_per_trade=args.risk
    )

    # --- Prepare and save results to files ---
    print("\nSaving detailed results...")
    
    results = {
        "best_parameters": study.best_params,
        "summary_train": summary_train,
        "summary_test": summary_test,
    }
    
    # Clean up results dictionary for JSON saving
    for k, v in results["best_parameters"].items():
        if isinstance(v, np.float64):
            results["best_parameters"][k] = float(v)

    # Save to JSON
    json_path = os.path.join(args.outdir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save trade logs to CSV
    if not trades_train_df.empty:
        trades_train_df.to_csv(os.path.join(args.outdir, "trades_train.csv"))
    if not trades_test_df.empty:
        trades_test_df.to_csv(os.path.join(args.outdir, "trades_test.csv"))
        
    print("\n🚀 All results saved!")
    print(f"Results summary: {json_path}")
    print(f"Training trade log: {os.path.join(args.outdir, 'trades_train.csv')}")
    print(f"Test trade log: {os.path.join(args.outdir, 'trades_test.csv')}")
    print("\nBacktest complete.")

if __name__=="__main__":
    main()