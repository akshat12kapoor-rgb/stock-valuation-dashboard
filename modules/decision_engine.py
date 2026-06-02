"""
decision_engine.py
------------------
Blends DCF and Multiples valuations into a single intrinsic value
and produces a BUY / HOLD / SELL investment recommendation.

Blending weights
----------------
  DCF weight      : 60%  (forward-looking, company-specific)
  Multiples weight: 40%  (market-anchored, relative)

These weights can be adjusted via the blend_weight parameter.

Decision thresholds
-------------------
  BUY  → intrinsic value > current price by > MARGIN_OF_SAFETY
  SELL → current price   > intrinsic value by > MARGIN_OF_SAFETY
  HOLD → within ± MARGIN_OF_SAFETY band
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MARGIN_OF_SAFETY: float = 0.15   # 15% buffer on either side of fair value

DECISION_CONFIG = {
    "BUY":  {"emoji": "🟢", "color": "#00C853", "label": "BUY"},
    "HOLD": {"emoji": "🟡", "color": "#FFD600", "label": "HOLD"},
    "SELL": {"emoji": "🔴", "color": "#D50000", "label": "SELL"},
}


# ---------------------------------------------------------------------------
# Data class for structured output
# ---------------------------------------------------------------------------

@dataclass
class ValuationResult:
    """Holds the complete output of the valuation engine."""

    # Inputs
    ticker:          str
    current_price:   float

    # Individual model outputs
    dcf_value:       float
    multiples_value: float | None

    # Blended result
    blended_value:   float
    upside_pct:      float           # positive = upside, negative = downside

    # Decision
    decision:        str             # "BUY" | "HOLD" | "SELL"
    emoji:           str
    color:           str
    margin_of_safety: float

    # Breakdown details
    dcf_weight:      float
    multiples_weight: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_decision(
    ticker: str,
    current_price: float,
    dcf_value: float,
    multiples_value: float | None,
    dcf_weight: float = 0.60,
    margin_of_safety: float = MARGIN_OF_SAFETY,
) -> ValuationResult:
    """
    Blend DCF and Multiples valuations, then issue a BUY/HOLD/SELL call.

    Parameters
    ----------
    ticker            : Stock ticker symbol.
    current_price     : Live market price per share.
    dcf_value         : Per-share intrinsic value from the DCF model.
    multiples_value   : Per-share implied value from comparables (can be None).
    dcf_weight        : Weight given to DCF in the blend (0.0–1.0).
    margin_of_safety  : Percentage band around fair value before a signal fires.

    Returns
    -------
    ValuationResult dataclass.
    """
    multiples_weight = 1.0 - dcf_weight

    # ---- Blend ----------------------------------------------------------
    if multiples_value is not None and multiples_value > 0:
        blended = dcf_value * dcf_weight + multiples_value * multiples_weight
    else:
        # No multiples data → rely entirely on DCF
        blended          = dcf_value
        dcf_weight       = 1.0
        multiples_weight = 0.0

    blended = max(blended, 0.0)

    # ---- Upside / Downside ---------------------------------------------
    if current_price > 0:
        upside_pct = (blended - current_price) / current_price
    else:
        upside_pct = 0.0

    # ---- Decision -------------------------------------------------------
    decision = _classify(upside_pct, margin_of_safety)
    cfg      = DECISION_CONFIG[decision]

    return ValuationResult(
        ticker=ticker,
        current_price=current_price,
        dcf_value=dcf_value,
        multiples_value=multiples_value,
        blended_value=blended,
        upside_pct=upside_pct,
        decision=decision,
        emoji=cfg["emoji"],
        color=cfg["color"],
        margin_of_safety=margin_of_safety,
        dcf_weight=dcf_weight,
        multiples_weight=multiples_weight,
    )


def describe_decision(result: ValuationResult) -> str:
    """
    Return a human-readable one-liner for the investment recommendation.

    Examples
    --------
    "🟢 BUY — AAPL appears undervalued by 23.4% ($148.20 vs $120.00 fair value)"
    "🔴 SELL — AAPL appears overvalued by 31.2% ($148.20 vs $112.80 fair value)"
    """
    sign  = "+" if result.upside_pct >= 0 else ""
    label = result.decision
    emo   = result.emoji
    pct   = abs(result.upside_pct) * 100

    direction = "undervalued" if result.upside_pct > 0 else "overvalued"

    if label == "HOLD":
        return (
            f"{emo} HOLD — {result.ticker} is fairly valued "
            f"({sign}{result.upside_pct*100:.1f}% vs intrinsic "
            f"${result.blended_value:.2f})"
        )

    return (
        f"{emo} {label} — {result.ticker} appears {direction} by {pct:.1f}% "
        f"(market ${result.current_price:.2f} | fair value ${result.blended_value:.2f})"
    )


def get_valuation_breakdown(result: ValuationResult) -> dict:
    """
    Return a dict suitable for display / charting that summarises
    each model's contribution alongside the current price.
    """
    breakdown = {
        "Current Price": result.current_price,
        "DCF Value":     result.dcf_value,
        "Blended Value": result.blended_value,
    }
    if result.multiples_value is not None:
        breakdown["Multiples Value"] = result.multiples_value
    return breakdown


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(upside_pct: float, margin: float) -> str:
    """Map the upside percentage to a BUY / HOLD / SELL label."""
    if upside_pct > margin:
        return "BUY"
    if upside_pct < -margin:
        return "SELL"
    return "HOLD"
