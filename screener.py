"""
Single-pass NSE + BSE screener, designed to be triggered repeatedly by a
GitHub Actions cron schedule (see .github/workflows/screener.yml).

Reads Telegram credentials from environment variables (set as GitHub Secrets):
  BOT_TOKEN
  CHAT_ID

Optional env vars:
  INCLUDE_BSE       "true"/"false" (default "true")
  MAX_TICKERS       integer, limits universe size (default: no limit)
  SCAN_TIMEFRAMES   comma-separated subset of: 15m,1h,4h,1d (default: all)
"""

import os
import sys
import time
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------------------------------------------------
# CONFIG (from environment — set via GitHub Secrets / workflow env)
# ----------------------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
INCLUDE_BSE = os.environ.get("INCLUDE_BSE", "true").lower() == "true"
MAX_TICKERS = int(os.environ["MAX_TICKERS"]) if os.environ.get("MAX_TICKERS") else None
SCAN_TIMEFRAMES = os.environ.get("SCAN_TIMEFRAMES", "15m,1h,4h,1d").split(",")

TIMEFRAMES = {
    "15m": "15m",
    "1h": "60m",
    "4h": "60m",   # resampled from 60m below
    "1d": "1d",
}
TIMEFRAMES = {k: v for k, v in TIMEFRAMES.items() if k in SCAN_TIMEFRAMES}

PERIOD_FOR_INTERVAL = {
    "15m": "5d",
    "60m": "1mo",
    "1d": "1y",
}

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_EQUITY_LIST_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripCodes/w"


# ----------------------------------------------------------------------
# TICKER UNIVERSE
# ----------------------------------------------------------------------
def get_nse_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["SYMBOL"].astype(str).str.strip().tolist()
        return [f"{s}.NS" for s in symbols]
    except Exception as e:
        print(f"[tickers] NSE fetch failed: {e}")
        return []


def get_bse_tickers(nse_names: set) -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(BSE_EQUITY_LIST_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        codes = df["Scrip_Cd"].astype(str).str.strip().tolist()
        names = df.get("Scrip_Name", pd.Series(dtype=str)).astype(str).str.upper().tolist()
        bse_tickers = []
        for code, name in zip(codes, names):
            if name in nse_names:
                continue
            bse_tickers.append(f"{code}.BO")
        return bse_tickers
    except Exception as e:
        print(f"[tickers] BSE fetch failed: {e}")
        return []


def get_universe(include_bse: bool = True) -> list:
    nse = get_nse_tickers()
    universe = list(nse)
    if include_bse:
        nse_roots = {t.replace(".NS", "") for t in nse}
        universe.extend(get_bse_tickers(nse_names=nse_roots))
    return universe


# ----------------------------------------------------------------------
# INDICATOR ENGINE (TTM Squeeze Pro + Synapse Trail Pro replicas)
# ----------------------------------------------------------------------
def ttm_squeeze(df, length=20, bb_mult=2.0, kc_mult_1=1.0, kc_mult_2=1.5, kc_mult_3=2.0):
    out = df.copy()
    basis = out["Close"].rolling(length).mean()
    dev = bb_mult * out["Close"].rolling(length).std(ddof=0)
    out["bb_upper"] = basis + dev
    out["bb_lower"] = basis - dev

    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - out["Close"].shift()).abs(),
        (out["Low"] - out["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    kc_basis = out["Close"].rolling(length).mean()

    for label, mult in [("1", kc_mult_1), ("2", kc_mult_2), ("3", kc_mult_3)]:
        out[f"kc_upper_{label}"] = kc_basis + mult * atr
        out[f"kc_lower_{label}"] = kc_basis - mult * atr
        out[f"squeeze_on_{label}"] = (
            (out["bb_upper"] < out[f"kc_upper_{label}"]) &
            (out["bb_lower"] > out[f"kc_lower_{label}"])
        )

    out["squeeze_fired"] = (out["squeeze_on_2"].shift(1) == True) & (out["squeeze_on_2"] == False)

    highest_high = out["High"].rolling(length).max()
    lowest_low = out["Low"].rolling(length).min()
    sma_close = out["Close"].rolling(length).mean()
    avg_val = (highest_high + lowest_low) / 2
    avg_val = (avg_val + sma_close) / 2
    diff = out["Close"] - avg_val

    def _linreg_last(series):
        y = series.values
        x = np.arange(len(y))
        if np.isnan(y).any():
            return np.nan
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1) + intercept

    out["momentum"] = diff.rolling(length).apply(_linreg_last, raw=False)
    out["momentum_rising"] = out["momentum"] > out["momentum"].shift(1)
    return out


def synapse_trail(df, atr_length=14, atr_mult=1.5, rsi_length=14, regime_length=50, htf_trend=None):
    out = df.copy()
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - out["Close"].shift()).abs(),
        (out["Low"] - out["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(atr_length).mean()

    upper = out["Close"].rolling(atr_length).max() - atr_mult * out["atr"]
    lower = out["Close"].rolling(atr_length).min() + atr_mult * out["atr"]

    trail = np.zeros(len(out))
    trend = np.zeros(len(out))
    close = out["Close"].values
    up = upper.values
    lo = lower.values

    for i in range(1, len(out)):
        if np.isnan(up[i]) or np.isnan(lo[i]):
            trail[i] = trail[i - 1]
            trend[i] = trend[i - 1]
            continue
        prev_trail = trail[i - 1]
        prev_trend = trend[i - 1]

        if prev_trend >= 0:
            new_trail = max(prev_trail, up[i]) if prev_trail != 0 else up[i]
            if close[i] < new_trail:
                trend[i] = -1
                trail[i] = lo[i]
            else:
                trend[i] = 1
                trail[i] = new_trail
        else:
            new_trail = min(prev_trail, lo[i]) if prev_trail != 0 else lo[i]
            if close[i] > new_trail:
                trend[i] = 1
                trail[i] = up[i]
            else:
                trend[i] = -1
                trail[i] = new_trail

    out["trail_stop"] = trail
    out["trend"] = trend

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_length).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_length).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))

    rolling_range = out["Close"].rolling(regime_length).max() - out["Close"].rolling(regime_length).min()
    out["regime"] = np.where(rolling_range > (out["atr"] * regime_length * 0.5), "trending", "choppy")

    if htf_trend is not None:
        out["htf_bullish"] = htf_trend.reindex(out.index, method="ffill")
    else:
        out["htf_bullish"] = True

    trend_series = pd.Series(trend, index=out.index)
    flipped_up = (trend_series == 1) & (trend_series.shift(1) == -1)
    flipped_down = (trend_series == -1) & (trend_series.shift(1) == 1)

    out["long_signal"] = flipped_up & (out["htf_bullish"] == True)
    out["short_signal"] = flipped_down & (out["htf_bullish"] == False)
    return out


def screen_signal(df) -> str:
    row = df.iloc[-1]
    long_ok = (
        bool(row.get("long_signal", False)) and
        bool(row.get("squeeze_fired", False)) and
        bool(row.get("momentum_rising", False)) and
        row.get("rsi", 0) > 50
    )
    short_ok = (
        bool(row.get("short_signal", False)) and
        bool(row.get("squeeze_fired", False)) and
        not bool(row.get("momentum_rising", True)) and
        row.get("rsi", 100) < 50
    )
    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return "NONE"


# ----------------------------------------------------------------------
# DATA FETCH
# ----------------------------------------------------------------------
def fetch_ohlc(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["High", "Low", "Close"]].dropna()


def resample_4h(df_60m):
    return df_60m.resample("4h").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()


def scan_one(ticker, tf_label):
    try:
        if tf_label == "4h":
            raw = fetch_ohlc(ticker, "60m", PERIOD_FOR_INTERVAL["60m"])
            if raw.empty or len(raw) < 60:
                return "NONE"
            df = resample_4h(raw)
        else:
            interval = TIMEFRAMES[tf_label]
            raw = fetch_ohlc(ticker, interval, PERIOD_FOR_INTERVAL.get(interval, "1mo"))
            df = raw

        if df.empty or len(df) < 60:
            return "NONE"

        df = ttm_squeeze(df)
        df = synapse_trail(df)
        return screen_signal(df)
    except Exception as e:
        print(f"[scan] {ticker} {tf_label} error: {e}")
        return "NONE"


# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------
def send_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram] Skipped — BOT_TOKEN/CHAT_ID not set. Message was:")
        print(message)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        if resp.status_code != 200:
            print(f"[telegram] Failed: {resp.text}")
    except Exception as e:
        print(f"[telegram] Error: {e}")


# ----------------------------------------------------------------------
# MAIN — one full pass, then exit (cron handles repetition)
# ----------------------------------------------------------------------
def main():
    start = time.time()
    universe = get_universe(include_bse=INCLUDE_BSE)
    if MAX_TICKERS:
        universe = universe[:MAX_TICKERS]

    print(f"Scanning {len(universe)} tickers across {list(TIMEFRAMES.keys())} timeframes...")

    signal_count = 0
    for i, ticker in enumerate(universe):
        for tf_label in TIMEFRAMES.keys():
            signal = scan_one(ticker, tf_label)
            if signal != "NONE":
                signal_count += 1
                send_telegram(f"*{signal}* signal\nTicker: `{ticker}`\nTimeframe: {tf_label}")
        if i % 100 == 0:
            print(f"  ...{i}/{len(universe)} scanned")

    elapsed = time.time() - start
    print(f"Scan complete in {elapsed:.1f}s. {signal_count} signals found.")


if __name__ == "__main__":
    main()
