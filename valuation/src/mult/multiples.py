# src/mult/multiples.py
from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# import the module so TAG_ALTS is always available as an attribute
from src.sec_helpers import helpers as sec_helpers

# pull the functions you call directly (optional, for brevity)
from src.sec_helpers.helpers import (
    make_session,
    get_cik_for_ticker,
    fetch_companyfacts,
    tagdf_any,
    annual_end_date,
    TAG_ALTS,
)

__all__ = [
    "build_multiples_for_ticker",
    "multiples_with_peers",
    "plot_trends",
    "price_near",
]

# --- Market data helper kept local to multiples (easy to swap provider later) ---
def price_near(ticker: str, d) -> float:
    today = pd.Timestamp.today().normalize()
    anchor = min(pd.Timestamp(d), today - pd.Timedelta(days=2))
    start = anchor - pd.Timedelta(days=30)
    end   = anchor + pd.Timedelta(days=3)

    data = yf.download(ticker, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        return float("nan")

    if isinstance(data.columns, pd.MultiIndex):
        s = None
        for col in [("Adj Close", ticker), ("Close", ticker)]:
            if col in data.columns:
                s = data[col]; break
        if s is None:
            for name in ["Adj Close", "Close"]:
                try:
                    s = data.xs(name, axis=1, level=0).iloc[:, 0]
                    break
                except Exception:
                    pass
            if s is None:
                return float("nan")
    else:
        if "Adj Close" in data.columns: s = data["Adj Close"]
        elif "Close" in data.columns:   s = data["Close"]
        else:                           return float("nan")

    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")

def _safe_div(a, b):
    return np.where((b > 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan)

def build_multiples_for_ticker(ticker: str, years_back: int = 5,
                               session=None) -> pd.DataFrame:
    if session is None:
        raise ValueError("Pass a requests.Session (use make_session).")

    cik   = get_cik_for_ticker(ticker, session)
    facts = fetch_companyfacts(cik, session)

    def _series(key: str) -> pd.Series:
        df = tagdf_any(facts, key)
        return pd.Series(dtype=float) if df.empty else df.set_index("fy")["val"].astype(float)

    ebit   = _series("ebit")
    da     = _series("da")
    ebitda = _series("ebitda")
    rev    = _series("revenue")
    cash   = _series("cash")
    debt_st= _series("debt_current")
    debt_lt= _series("debt_long")
    shares = _series("shares_diluted")
    capex  = _series("capex")
    cfo    = _series("cfo")

    years = pd.Index(sorted(set().union(ebit.index, rev.index, shares.index)))
    if len(years) == 0:
        return pd.DataFrame()
    years = years[years >= (years.max() - years_back + 1)]

    df = pd.DataFrame(index=years); df.index.name = "fy"
    df["Revenue"]    = rev.reindex(years)
    df["EBIT"]       = ebit.reindex(years)
    df["D&A"]        = da.reindex(years)
    df["EBITDA"]     = ebitda.reindex(years)
    df["EBITDA"]     = df["EBITDA"].where(np.isfinite(df["EBITDA"]), df["EBIT"] + df["D&A"])

    df["Cash"]       = cash.reindex(years).fillna(0.0)
    df["Debt_ST"]    = debt_st.reindex(years).fillna(0.0)
    df["Debt_LT"]    = debt_lt.reindex(years).fillna(0.0)
    df["TotalDebt"]  = df["Debt_ST"] + df["Debt_LT"]
    df["CapEx"]      = capex.reindex(years).abs()
    df["CFO"]        = cfo.reindex(years)

    # FCF fallback: NOPAT (21% tax) + D&A - CapEx; if EBIT missing, use CFO - CapEx
    df["FCF"] = (df["EBIT"]*(1-0.21) + df["D&A"] - df["CapEx"]).where(
        np.isfinite(df["EBIT"]), df["CFO"] - df["CapEx"]
    )

    fy_ends = annual_end_date(facts, sec_helpers.TAG_ALTS["revenue"] + sec_helpers.TAG_ALTS["ebit"])
    df["fy_end"] = [pd.to_datetime(fy_ends.get(int(y), f"{int(y)}-12-31")).date() for y in df.index]

    px_map = {int(y): price_near(ticker, d) for y, d in df["fy_end"].items()}
    df["Price"]     = pd.Series(px_map)
    df["SharesDil"] = shares.reindex(years)
    df["MktCap"]    = df["Price"] * df["SharesDil"]
    df["EV"]        = df["MktCap"] + df["TotalDebt"] - df["Cash"]

    df["EV_EBITDA"] = _safe_div(df["EV"], df["EBITDA"])
    df["P_S"]       = _safe_div(df["MktCap"], df["Revenue"])
    df["P_FCF"]     = _safe_div(df["MktCap"], df["FCF"])

    cols_out = ["Revenue","EBITDA","FCF","MktCap","EV","EV_EBITDA","P_S","P_FCF","fy_end","Price"]
    return df[cols_out].sort_index()

def multiples_with_peers(ticker: str, peers: Optional[List[str]] = None, years_back: int = 5,
                         session=None):
    comp = build_multiples_for_ticker(ticker, years_back, session=session)
    if not peers:
        return comp, pd.DataFrame()

    frames = []
    for p in peers:
        try:
            f = build_multiples_for_ticker(p, years_back, session=session)
            if not f.empty:
                f["ticker"] = p
                frames.append(f)
        except Exception as e:
            print(f"Peer {p}: {e}")

    if not frames:
        return comp, pd.DataFrame()

    peers_df = pd.concat(frames, axis=0, ignore_index=False)

    cols = ["EV_EBITDA", "P_S", "P_FCF"]
    # clean bad values so means don’t blow up
    peers_clean = peers_df.copy()
    for c in cols:
        peers_clean[c] = peers_clean[c].replace([np.inf, -np.inf], np.nan)

    stats = (peers_clean.groupby(level=0)[cols]
             .agg(["median", "mean"]))  # << both
    # flatten MultiIndex columns -> EV_EBITDA_median, EV_EBITDA_mean, ...
    stats.columns = [f"{metric}_{stat}" for (metric, stat) in stats.columns]

    return comp, stats

def plot_trends(comp, stats=None, label_ticker="TICKER"):
    for col, label in [("EV_EBITDA","EV / EBITDA"), ("P_S","P / Sales"), ("P_FCF","P / FCF")]:
        plt.figure()
        plt.plot(comp.index, comp[col], marker="o", label=label_ticker)
        if stats is not None and not stats.empty:
            med_col  = f"{col}_median"
            mean_col = f"{col}_mean"
            if med_col in stats.columns:
                plt.plot(stats.index, stats[med_col], linestyle="--", marker="o", label="Peers (median)")
            if mean_col in stats.columns:
                plt.plot(stats.index, stats[mean_col], linestyle=":", marker="o", label="Peers (mean)")
        plt.title(f"{label_ticker} — {label} Trend")
        plt.xlabel("Fiscal Year"); plt.ylabel(label)
        plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
        plt.show()
