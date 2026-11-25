#!/usr/bin/env python3
"""
NSE stock screener alerts - TTM Squeeze Pro + Synapse Trail Pro (replica)
Timeframes : 10m, 15m, 1h, 1d      Signals : LONG and SHORT (mirror logic)
Data       : yfinance (free)         Alerts  : Telegram bot (free)

Signal types
  SYNAPSE     trail flips direction on the last closed bar (+ RSI / ADX / HTF filters)
  SQUEEZE     TTM Squeeze Pro "fires" (compression ends) with momentum direction
  CONFLUENCE  squeeze fired within the last few bars AND trail agrees AND filters agree (best quality)

Env vars (all optional except BOT_TOKEN / CHAT_ID for Telegram)
  BOT_TOKEN, CHAT_ID, TIMEFRAMES="10m,15m,1h,1d", TICKERS="RELIANCE.NS,TCS.NS",
  UNIVERSE_LIMIT=500, FRESH_BARS=1, CONFLUENCE_BARS=3, TRAIL_ATR_LEN=10, TRAIL_ATR_MULT=3.0,
  ADX_MIN=20, USE_HTF_FILTER=1, SIDES="LONG,SHORT", DRY_RUN=0
"""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view as swv

# ------------------------------------------------------------------ config
TZ = os.getenv("MARKET_TZ", "Asia/Kolkata")
OPEN_OFFSET = os.getenv("MARKET_OPEN_OFFSET", "15min")  # NSE opens 09:15 -> bars align to :15
CLOSE_HHMM = os.getenv("MARKET_CLOSE", "15:30")
TIMEFRAMES = [t.strip() for t in os.getenv("TIMEFRAMES", "10m,15m,1h,1d").split(",") if t.strip()]
SIDES = [s.strip().upper() for s in os.getenv("SIDES", "LONG,SHORT").split(",")]
FRESH = int(os.getenv("FRESH_BARS", "1"))            # signal must be on the last N closed bars
CONF_WINDOW = int(os.getenv("CONFLUENCE_BARS", "3"))  # squeeze fire lookback for confluence
ATR_LEN = int(os.getenv("TRAIL_ATR_LEN", "10"))
ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "3.0"))
ADX_MIN = float(os.getenv("ADX_MIN", "20"))
USE_HTF = os.getenv("USE_HTF_FILTER", "1") == "1"
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "500"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
STATE_FILE = Path(os.getenv("STATE_FILE", ".state/sent.json"))
TP_R = (1.0, 2.0, 3.0)  # TP1/TP2/TP3 as multiples of initial risk

# 10m is not offered by yfinance -> built from 5m bars.
TF_CFG = {
    "10m": dict(src="5m", period="30d", rule="10min", mins=10, htf="60min"),
    "15m": dict(src="15m", period="55d", rule=None, mins=15, htf="60min"),
    "1h": dict(src="60m", period="120d", rule=None, mins=60, htf="240min"),
    "1d": dict(src="1d", period="2y", rule=None, mins=1440, htf="W-FRI"),
}

NIFTY50 = """RELIANCE TCS HDFCBANK ICICIBANK INFY HINDUNILVR ITC SBIN BHARTIARTL KOTAKBANK LT AXISBANK
ASIANPAINT MARUTI SUNPHARMA TITAN ULTRACEMCO BAJFINANCE NESTLEIND WIPRO HCLTECH M&M NTPC POWERGRID
ONGC TATASTEEL JSWSTEEL ADANIENT ADANIPORTS COALINDIA BAJAJFINSV TECHM INDUSINDBK HINDALCO GRASIM
CIPLA DRREDDY EICHERMOT BRITANNIA APOLLOHOSP DIVISLAB HEROMOTOCO BPCL SBILIFE HDFCLIFE TATACONSUM
BAJAJ-AUTO SHRIRAMFIN""".split()


# ------------------------------------------------------------------ universe
def universe():
    env = os.getenv("TICKERS")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].dropna().astype(str).tolist()
        return [s + ".NS" for s in syms][:UNIVERSE_LIMIT]
    except Exception as e:
        print(f"[universe] NSE list failed ({e}); using Nifty 50 fallback")
        return [s + ".NS" for s in NIFTY50]


# ------------------------------------------------------------------ data
def fetch(tickers, cfg, batch=100):
    out = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        try:
            raw = yf.download(chunk, period=cfg["period"], interval=cfg["src"], group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print(f"[fetch] batch failed: {e}")
            continue
        if raw is None or raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([[chunk[0]], raw.columns])
        lvl0 = set(raw.columns.get_level_values(0))
        for t in chunk:
            if t in lvl0:
                d = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(d):
                    out[t] = d
    return out


def resample(df, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    kw = {} if rule.startswith("W") else {"offset": OPEN_OFFSET}
    return df.resample(rule, **kw).agg(agg).dropna(subset=["Close"])


def drop_incomplete(df, mins):
    """Remove the still-forming last bar (bar end clipped to session close)."""
    now = pd.Timestamp.now(tz=TZ)
    h, m = map(int, CLOSE_HHMM.split(":"))
    ends = pd.Series(df.index + pd.Timedelta(minutes=mins), index=df.index)
    closes = pd.Series(df.index.normalize() + pd.Timedelta(hours=h, minutes=m), index=df.index)
    ends = ends.where(ends <= closes, closes)
    return df[(ends <= now).values]


def prepare(df, cfg, tf):
    df = df.copy()
    if tf != "1d":
        idx = df.index
        df.index = idx.tz_localize("UTC").tz_convert(TZ) if idx.tz is None else idx.tz_convert(TZ)
        if cfg["rule"]:
            df = resample(df, cfg["rule"])
        df = drop_incomplete(df, cfg["mins"])
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
        now = pd.Timestamp.now(tz=TZ)
        if len(df) and df.index[-1].date() == now.date() and now.strftime("%H:%M") < CLOSE_HHMM:
            df = df.iloc[:-1]
    return df


# ------------------------------------------------------------------ indicators
def rma(s, n):  # Wilder smoothing
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df):
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)


def linreg(s, n):
    """Rolling linear-regression value at the last point of each window (Pine's linreg(x, n, 0))."""
    a = s.values.astype(float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = swv(a, n)
        x = np.arange(n) - (n - 1) / 2
        slope = (w * x).sum(1) / (x ** 2).sum()
        out[n - 1:] = w.mean(1) + slope * (n - 1) / 2
    return pd.Series(out, index=s.index)


def rsi(c, n=14):
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)


def adx(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = rma(true_range(df), n)
    pdi, mdi = 100 * rma(pdm, n) / atr, 100 * rma(mdm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n)


def ttm_squeeze_pro(df, n=20, bb_mult=2.0, kc=(1.0, 1.5, 2.0)):
    """
    lvl: 0 = no squeeze (green dot), 1 = low (BB inside KC 2.0), 2 = mid (inside 1.5), 3 = high (inside 1.0)
    mom: linreg momentum histogram (positive = bullish, negative = bearish)
    """
    c, h, l = df["Close"], df["High"], df["Low"]
    basis = c.rolling(n).mean()
    dev = bb_mult * c.rolling(n).std(ddof=0)
    bbu, bbl = basis + dev, basis - dev
    rng = true_range(df).rolling(n).mean()
    lvl = pd.Series(0, index=df.index)
    for level, m in ((1, kc[2]), (2, kc[1]), (3, kc[0])):
        inside = (bbu < basis + m * rng) & (bbl > basis - m * rng)
        lvl = lvl.mask(inside, level)
    mid = ((h.rolling(n).max() + l.rolling(n).min()) / 2 + basis) / 2
    return pd.DataFrame({"lvl": lvl, "mom": linreg(c - mid, n)})


def synapse_trail(df, n=ATR_LEN, mult=ATR_MULT):
    """ATR trailing-stop trend engine (supertrend-style). dir: +1 uptrend, -1 downtrend."""
    atr = rma(true_range(df), n)
    hl2 = ((df["High"] + df["Low"]) / 2).values
    ub, lb = hl2 + mult * atr.values, hl2 - mult * atr.values
    c = df["Close"].values
    N = len(c)
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(N, dtype=int)
    line = np.full(N, np.nan)
    line[0] = lb[0]
    for i in range(1, N):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            d[i] = 1
        elif c[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return pd.DataFrame({"dir": d, "line": line, "atr": atr.values}, index=df.index)


def htf_bias(df, cfg):
    """+1 / -1 = higher-timeframe close above / below its EMA21, 0 = unknown."""
    try:
        h = resample(df, cfg["htf"])
        if len(h) < 8:
            return 0
        ema = h["Close"].ewm(span=21, adjust=False).mean()
        return 1 if h["Close"].iloc[-1] > ema.iloc[-1] else -1
    except Exception:
        return 0


# ------------------------------------------------------------------ signal logic
def analyze(sym, tf, df, cfg):
    if len(df) < max(70, ATR_LEN + 40):
        return []
    sq, tl = ttm_squeeze_pro(df), synapse_trail(df)
    rsi_v, adx_v = rsi(df["Close"]).iloc[-1], adx(df).iloc[-1]
    bias = htf_bias(df, cfg) if USE_HTF else 0

    fire = ((sq["lvl"].shift(1) > 0) & (sq["lvl"] == 0)).values
    flip = (tl["dir"].diff().fillna(0) != 0).values
    mom, dirn = sq["mom"].values, tl["dir"].values
    close, line, atr = df["Close"].iloc[-1], tl["line"].iloc[-1], tl["atr"].iloc[-1]
    mom_rising = bool(mom[-1] > mom[-2]) if len(mom) > 1 else False
    pre_lvl = int(sq["lvl"].shift(1).iloc[-CONF_WINDOW:].max()) if len(sq) > CONF_WINDOW else 0

    def last_true(arr, k):
        idx = np.where(arr[-k:])[0]
        return None if len(idx) == 0 else len(arr) - k + idx[-1]

    out = []
    for side, sgn in (("LONG", 1), ("SHORT", -1)):
        if side not in SIDES:
            continue
        ok_rsi = rsi_v > 50 if sgn == 1 else rsi_v < 50
        ok_htf = (not USE_HTF) or bias in (0, sgn)
        ok_adx = adx_v >= ADX_MIN
        sl = line if dirn[-1] == sgn else close - sgn * ATR_MULT * atr
        risk = abs(close - sl)
        if not np.isfinite(risk) or risk <= 0:
            continue
        tps = [close + sgn * risk * r for r in TP_R]

        def emit(kind, bar_idx):
            out.append(dict(
                symbol=sym.replace(".NS", "").replace(".BO", ""), tf=tf, side=side, type=kind,
                bar=df.index[bar_idx].strftime("%Y-%m-%d %H:%M"), price=close, sl=sl,
                tp1=tps[0], tp2=tps[1], tp3=tps[2], rsi=rsi_v, adx=adx_v,
                htf={1: "UP", -1: "DOWN", 0: "n/a"}[bias], sqz_lvl=pre_lvl, mom_up=mom_rising))

        # 1) Synapse Trail flip
        j = last_true(flip, FRESH)
        if j is not None and dirn[-1] == sgn and ok_rsi and ok_htf and ok_adx:
            emit("SYNAPSE", j)
        # 2) TTM Squeeze Pro fire
        j = last_true(fire, FRESH)
        if j is not None and np.sign(mom[-1]) == sgn:
            emit("SQUEEZE", j)
        # 3) Confluence: recent squeeze fire + trail agrees + filters agree
        j = last_true(fire, CONF_WINDOW)
        if j is not None and np.sign(mom[-1]) == sgn and dirn[-1] == sgn and ok_rsi and ok_htf and ok_adx:
            emit("CONFLUENCE", j)
    return out


# ------------------------------------------------------------------ alerts
def load_state():
    try:
        s = json.loads(STATE_FILE.read_text())
    except Exception:
        s = {}
    cutoff = time.time() - 3 * 86400
    return {k: v for k, v in s.items() if v > cutoff}


def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s))


def fmt(x):
    return f"{x:,.2f}"


def format_signal(s):
    icon = "🟢 LONG" if s["side"] == "LONG" else "🔴 SHORT"
    star = " ⭐" if s["type"] == "CONFLUENCE" else ""
    return (f"{icon} {s['symbol']} | {s['tf']} | {s['type']}{star}\n"
            f"Px {fmt(s['price'])} | SL {fmt(s['sl'])}\n"
            f"TP1 {fmt(s['tp1'])} | TP2 {fmt(s['tp2'])} | TP3 {fmt(s['tp3'])}\n"
            f"RSI {s['rsi']:.0f} | ADX {s['adx']:.0f} | HTF {s['htf']} | bar {s['bar']}")


def telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[telegram] BOT_TOKEN/CHAT_ID missing - printing only")
        return
    r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    if not r.ok:
        print(f"[telegram] error {r.status_code}: {r.text[:200]}")


def send_all(signals):
    order = {"CONFLUENCE": 0, "SYNAPSE": 1, "SQUEEZE": 2}
    tf_order = {t: i for i, t in enumerate(TIMEFRAMES)}
    signals.sort(key=lambda s: (s["side"], order[s["type"]], tf_order[s["tf"]], s["symbol"]))
    for side in ("LONG", "SHORT"):
        rows = [s for s in signals if s["side"] == side]
        if not rows:
            continue
        msg = f"{'🟢 LONG' if side == 'LONG' else '🔴 SHORT'} signals ({len(rows)})\n"
        for s in rows:
            block = "\n" + format_signal(s) + "\n"
            if len(msg) + len(block) > 3800:
                telegram(msg)
                msg = ""
            msg += block
        if msg.strip():
            telegram(msg)


# ------------------------------------------------------------------ main
def main():
    tickers = universe()
    print(f"Universe: {len(tickers)} tickers | timeframes: {TIMEFRAMES}")
    found = []
    for tf in TIMEFRAMES:
        cfg = TF_CFG.get(tf)
        if not cfg:
            print(f"[skip] unknown timeframe {tf}")
            continue
        data = fetch(tickers, cfg)
        n_ok = 0
        for sym, raw in data.items():
            try:
                df = prepare(raw, cfg, tf)
                found += analyze(sym, tf, df, cfg)
                n_ok += 1
            except Exception as e:
                print(f"[{tf}] {sym}: {e}")
        print(f"[{tf}] analysed {n_ok}/{len(tickers)}")

    state = load_state()
    fresh = []
    for s in found:
        key = f"{s['symbol']}|{s['tf']}|{s['side']}|{s['type']}|{s['bar']}"
        if key not in state:
            state[key] = time.time()
            fresh.append(s)

    if fresh:
        df = pd.DataFrame(fresh)[["side", "type", "tf", "symbol", "price", "sl", "tp1", "tp2", "tp3", "rsi", "adx", "htf", "bar"]]
        print(df.round(2).to_string(index=False))
        if not DRY_RUN:
            send_all(fresh)
    else:
        print("No new signals this run.")
    save_state(state)


if __name__ == "__main__":
    main()
