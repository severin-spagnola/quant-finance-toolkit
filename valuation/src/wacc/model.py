from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Union, Tuple
import math
import numpy as np

def _safe_ratio(numer: float, denom: float, default: float = 0.0) -> float:
    try:
        if denom == 0 or math.isclose(denom, 0.0):
            return default
        return numer / denom
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _require(cond: bool, msg: str, strict: bool):
    if strict and not cond:
        raise ValueError(msg)

# Beta relevering / unlevering functions
def unlever_beta(beta_eq: float, D: float, E: float, tax: float) -> float:
    """β_a = β_e / (1 + (1 - tax) * D/E)"""
    tax = clamp(tax, 0.0, 0.5)
    de = _safe_ratio(D, E, default=0.0)
    denom = 1.0 + (1.0 - tax) * de
    if math.isclose(denom, 0.0):
        return beta_eq
    return beta_eq / denom

def relever_beta(beta_asset: float, D: float, E: float, tax: float) -> float:
    """β_e = β_a * (1 + (1 - tax) * D/E)"""
    tax = clamp(tax, 0.0, 0.5)
    de = _safe_ratio(D, E, default=0.0)
    return beta_asset * (1.0 + (1.0 - tax) * de)

def peer_unlevered_beta(
    peers: Iterable[Dict[str, float]],
    tax_cap: float = 0.5,
    agg: str = "median",
) -> float:
    """
    peers rows need: beta_eq, D, E, tax
    Returns aggregated unlevered (asset) beta for the peer set.
    """
    unlevs: List[float] = []
    for p in peers:
        beta_eq = float(p.get("beta_eq", np.nan))
        D = float(p.get("D", 0.0)); E = float(p.get("E", 0.0))
        tax = clamp(float(p.get("tax", 0.21)), 0.0, tax_cap)
        if not np.isfinite(beta_eq) or not np.isfinite(D) or not np.isfinite(E):
            continue
        beta_a = unlever_beta(beta_eq, D=max(D, 0.0), E=max(E, 1e-9), tax=tax)
        if np.isfinite(beta_a):
            unlevs.append(beta_a)

    if len(unlevs) == 0:
        raise ValueError("peer_unlevered_beta: no valid peer inputs provided")

    unlevs = np.array(unlevs, dtype=float)
    if agg == "mean":
        return float(np.mean(unlevs))
    return float(np.median(unlevs))

def cost_of_equity(rf: float, beta: float, mrp: float) -> float:
    """CAPM: Re = Rf + beta * MRP (inputs are decimals)."""
    return float(rf) + float(beta) * float(mrp)

# Estimate cost of debt from various input formats
def estimate_cost_of_debt(
    rf: float,
    debt_input: Union[float, Dict[str, float], None],
    tax: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    """
    debt_input:
      - float: pre-tax Rd
      - {"ytm": x} or {"spread": s} (spread added to rf)
      - None: fallback rf + 1.5% (only if strict=False upstream)
    """
    if isinstance(debt_input, (int, float)) and np.isfinite(debt_input):
        pre_tax = float(debt_input)
    elif isinstance(debt_input, dict):
        if "ytm" in debt_input and np.isfinite(debt_input["ytm"]):
            pre_tax = float(debt_input["ytm"])
        elif "spread" in debt_input and np.isfinite(debt_input["spread"]):
            pre_tax = float(rf) + float(debt_input["spread"])
        else:
            pre_tax = float(rf) + 0.015
    else:
        pre_tax = float(rf) + 0.015

    pre_tax = max(0.0, float(pre_tax))
    after_tax = None if tax is None else pre_tax * (1.0 - clamp(float(tax), 0.0, 0.5))
    return pre_tax, after_tax

@dataclass
class WACCResult:
    rf: float
    mrp: float
    beta: float
    coeq: float
    codebt: float
    codebt_after_tax: float
    tax: float
    de_ratio: float
    wacc: float
    wacc_dist: Optional[np.ndarray] = None
    wacc_pct: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Union[float, Dict[str, float], List[float]]]:
        d = asdict(self)
        out = {
            "rf": d["rf"], "mrp": d["mrp"], "beta": d["beta"],
            "coeq": d["coeq"], "codebt": d["codebt"], "tax": d["tax"],
            "de_ratio": d["de_ratio"], "wacc": d["wacc"],
        }
        if d.get("wacc_dist") is not None:
            out["wacc_dist"] = d["wacc_dist"].tolist()
        if d.get("wacc_pct") is not None:
            out["wacc_pct"] = d["wacc_pct"]
        return out

# Find weighted average cost of capital mathematically
def _weighted_average_cost_of_capital(
    Re: float, Rd: float, D: float, E: float, tax: float
) -> float:
    """WACC = (E/(D+E))*Re + (D/(D+E))*Rd*(1 - tax)"""
    tax = clamp(tax, 0.0, 0.5)
    D = max(0.0, float(D)); E = max(0.0, float(E))
    V = D + E
    if math.isclose(V, 0.0):
        return Re
    w_e = E / V
    w_d = D / V
    return float(w_e * Re + w_d * Rd * (1.0 - tax))

# Find weighted average cost of capital vectorized for monte carlo
def _weighted_wacc_vectorized(
    Re: np.ndarray, Rd: np.ndarray, D: float, E: float, tax: float,
) -> np.ndarray:
    tax = clamp(tax, 0.0, 0.5)
    D = max(0.0, float(D)); E = max(0.0, float(E))
    V = D + E
    if math.isclose(V, 0.0):
        return Re.copy()
    w_e = E / V
    w_d = D / V
    return (w_e * Re) + (w_d * Rd * (1.0 - tax))

# Compute the WACC given inputs, using target_de_ratio for weights and beta relevering. Utilizes a monte carlo simulation if n_draws > 0
def compute_wacc(
    *,
    # Core inputs
    rf: float,                 # risk-free (decimal)
    mrp: float,                # market risk premium (decimal)
    tax: float,                # effective tax rate (decimal)
    target_de_ratio: float,    # REQUIRED D/E to apply in weights & beta relever
    # Beta inputs
    beta_eq: Optional[float] = None,
    beta_asset: Optional[float] = None,
    peers: Optional[Iterable[Dict[str, float]]] = None,
    # Debt inputs
    debt_pricing: Union[float, Dict[str, float], None] = None,
    # Monte Carlo
    n_draws: int = 0,
    sd_beta: Optional[float] = None,
    sd_rf: Optional[float] = None,
    sd_mrp: Optional[float] = None,
    sd_codebt: Optional[float] = None,
    # RNG control
    random_state: Optional[int] = None,
    # Behavior
    strict: bool = True,
) -> Dict[str, Union[float, Dict[str, float], List[float]]]:
    """
    Returns dict with:
      rf, mrp, beta, coeq, codebt, tax, de_ratio, wacc, (optional) wacc_dist, wacc_pct
    """
    _require(np.isfinite(rf) and rf >= 0, "rf missing/invalid", strict)
    _require(np.isfinite(mrp) and mrp >= 0, "mrp missing/invalid", strict)
    tax = clamp(float(tax), 0.0, 0.5)
    _require(np.isfinite(target_de_ratio) and float(target_de_ratio) >= 0.0,
             "target_de_ratio required and must be non-negative", strict)
    de = float(target_de_ratio)

    # Resolve beta
    if beta_eq is None:
        if beta_asset is not None:
            beta_eq = relever_beta(beta_asset, D=de, E=1.0, tax=tax)
        elif peers is not None:
            beta_a = peer_unlevered_beta(peers)
            beta_eq = relever_beta(beta_a, D=de, E=1.0, tax=tax)
        else:
            raise ValueError("Need beta_eq OR beta_asset OR peers to infer beta.")
    beta_eq = float(beta_eq)

    coeq = cost_of_equity(rf=float(rf), beta=beta_eq, mrp=float(mrp))
    _require(debt_pricing is not None, "debt_pricing required (ytm/spread/float)", strict)
    pre_tax_rd, _ = estimate_cost_of_debt(rf=float(rf), debt_input=debt_pricing, tax=tax)

    wacc_point = _weighted_average_cost_of_capital(Re=coeq, Rd=pre_tax_rd, D=de, E=1.0, tax=tax)

    wacc_draws = None; pct = None
    if n_draws and n_draws > 0:
        rng = np.random.default_rng(random_state)
        beta_draws = (rng.normal(beta_eq, sd_beta, n_draws)
                      if (sd_beta and sd_beta > 0) else np.full(n_draws, beta_eq))
        rf_draws   = (rng.normal(rf, sd_rf, n_draws)
                      if (sd_rf and sd_rf > 0) else np.full(n_draws, rf))
        mrp_draws  = (rng.normal(mrp, sd_mrp, n_draws)
                      if (sd_mrp and sd_mrp > 0) else np.full(n_draws, mrp))
        rd_draws   = (rng.normal(pre_tax_rd, sd_codebt, n_draws)
                      if (sd_codebt and sd_codebt > 0) else np.full(n_draws, pre_tax_rd))

        rf_draws  = np.clip(rf_draws, 0.0, 0.15)
        mrp_draws = np.clip(mrp_draws, 0.0, 0.20)
        rd_draws  = np.clip(rd_draws, 0.0, 0.20)

        coeq_draws = np.clip(rf_draws + beta_draws * mrp_draws, 0.0, 0.40)
        wacc_draws = _weighted_wacc_vectorized(Re=coeq_draws, Rd=rd_draws, D=de, E=1.0, tax=tax)

        pct = {
            "p5":  float(np.percentile(wacc_draws, 5)),
            "p25": float(np.percentile(wacc_draws, 25)),
            "p50": float(np.percentile(wacc_draws, 50)),
            "p75": float(np.percentile(wacc_draws, 75)),
            "p95": float(np.percentile(wacc_draws, 95)),
        }

    res = WACCResult(
        rf=float(rf),
        mrp=float(mrp),
        beta=float(beta_eq),
        coeq=float(coeq),
        codebt=float(pre_tax_rd),
        codebt_after_tax=float(pre_tax_rd * (1.0 - tax)),
        tax=float(tax),
        de_ratio=float(de),
        wacc=float(wacc_point),
        wacc_dist=wacc_draws,
        wacc_pct=pct,
    )
    return res.to_dict()
