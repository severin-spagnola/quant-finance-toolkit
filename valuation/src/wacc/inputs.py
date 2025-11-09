from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import math
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

DEFAULT_RISK_FREE = 0.045
DEFAULT_MRP       = 0.055

# Sector spread estimates (from NYU Stern)
SECTOR_SPREADS = {
    "Technology": 0.016, "Healthcare": 0.017, "Consumer Defensive": 0.017,
    "Industrials": 0.018, "Financial Services": 0.018, "Communication Services": 0.018,
    "Consumer Cyclical": 0.019, "Energy": 0.020, "Materials": 0.020,
    "Real Estate": 0.021, "Utilities": 0.020, "_default": 0.018,
}

def sector_spread_lookup(sector: Optional[str]) -> float:
    if not sector: return SECTOR_SPREADS["_default"]
    return SECTOR_SPREADS.get(sector, SECTOR_SPREADS["_default"])

def fetch_peer_rows_yf(tickers: Iterable[str]) -> List[Dict]:
    if yf is None:
        raise ImportError("yfinance not installed. pip install yfinance")
    peers = []
    for tkr in tickers:
        try:
            tk = yf.Ticker(tkr)
            info = tk.info
            beta = info.get("beta")
            mcap = info.get("marketCap")
            total_debt = info.get("totalDebt")
            tax = 0.21
            try:
                fin = tk.financials
                if fin is not None and not fin.empty:
                    tax_exp = float(fin.loc["Income Tax Expense"].fillna(0).iloc[0]) if "Income Tax Expense" in fin.index else np.nan
                    pre_tax = float(fin.loc["Income Before Tax"].fillna(0).iloc[0]) if "Income Before Tax" in fin.index else np.nan
                    if math.isfinite(tax_exp) and math.isfinite(pre_tax) and pre_tax != 0:
                        est = tax_exp / pre_tax
                        if 0.0 <= est <= 0.5: tax = est
            except Exception:
                pass
            peers.append({
                "ticker": tkr,
                "beta_eq": float(beta) if beta is not None else np.nan,
                "D": float(total_debt) if total_debt is not None else 0.0,
                "E": float(mcap) if mcap is not None else 0.0,
                "tax": float(tax),
                "sector": info.get("sector"),
            })
        except Exception:
            continue
    return peers

def clean_peers(peers: List[Dict], drop_nan_beta: bool = True, de_cap: float = 2.0) -> List[Dict]:
    out = []
    for p in peers:
        beta = p.get("beta_eq", np.nan)
        D = float(p.get("D", 0.0) or 0.0)
        E = float(p.get("E", 0.0) or 0.0)
        tax = float(p.get("tax", 0.21) or 0.21)
        if drop_nan_beta and (beta is None or not np.isfinite(beta)): continue
        if E <= 0: continue
        de = D / E
        if not np.isfinite(de) or de < 0: continue
        if de_cap is not None:
            de = min(de, de_cap); D = de * E
        out.append({
            "ticker": p.get("ticker"),
            "beta_eq": float(beta),
            "D": float(D), "E": float(E),
            "tax": float(np.clip(tax, 0.0, 0.5)),
            "sector": p.get("sector"),
        })
    return out

def smooth_tax_rate(peers: List[Dict], fallback: float = 0.21) -> float:
    vals = [p["tax"] for p in peers if p.get("tax") is not None and np.isfinite(p["tax"])]
    if not vals: return fallback
    return float(np.clip(float(np.median(vals)), 0.0, 0.5))

def infer_sector(peers: List[Dict]) -> Optional[str]:
    sectors = [p.get("sector") for p in peers if p.get("sector")]
    if not sectors: return None
    vals, counts = np.unique(sectors, return_counts=True)
    return str(vals[np.argmax(counts)])

def build_baselines_for_wacc(
    *, ticker: str, peer_rows: List[Dict], overrides: Optional[Dict] = None,
) -> Dict:
    """
    Returns kwargs you can pass straight into compute_wacc (no company D/E needed):
      rf, mrp, tax, target_de_ratio, debt_pricing
    """
    peers_clean = clean_peers(peer_rows)
    de_list = [p["D"]/p["E"] for p in peers_clean if p["E"] > 0]
    target_de = float(np.median(de_list)) if de_list else 0.20
    target_de = float(np.clip(target_de, 0.0, 1.0))
    tax = smooth_tax_rate(peers_clean, fallback=0.21)
    sector = infer_sector(peers_clean)
    spread = sector_spread_lookup(sector)

    params = {
        "rf": DEFAULT_RISK_FREE,
        "mrp": DEFAULT_MRP,
        "tax": tax,
        "target_de_ratio": target_de,
        "debt_pricing": {"spread": spread},
    }
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params
