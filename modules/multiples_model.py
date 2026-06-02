"""
multiples_model.py
------------------
Comparable-company (Comps) valuation using two market multiples:

  1. Price / Earnings  (P/E)
       Implied price = EPS × industry_avg_PE

  2. Enterprise Value / EBITDA  (EV/EBITDA)
       EV_implied   = EBITDA × industry_avg_EV_EBITDA
       Equity value = EV_implied − Total Debt + Cash
       Implied price = Equity value / shares_outstanding

Industry averages are based on long-run market medians and are used
as proxies when a full comparable-company table is unavailable.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Industry average multiples (median, as of 2024)
# Source: Damodaran, Bloomberg medians
# ---------------------------------------------------------------------------

SECTOR_MULTIPLES: dict[str, dict[str, float]] = {
    "Technology": {
        "pe":         28.0,
        "ev_ebitda":  20.0,
    },
    "Communication Services": {
        "pe":         22.0,
        "ev_ebitda":  14.0,
    },
    "Consumer Discretionary": {
        "pe":         25.0,
        "ev_ebitda":  15.0,
    },
    "Consumer Staples": {
        "pe":         22.0,
        "ev_ebitda":  14.0,
    },
    "Energy": {
        "pe":         12.0,
        "ev_ebitda":   7.0,
    },
    "Financials": {
        "pe":         13.0,
        "ev_ebitda":   9.0,    # banks use P/B more, but we include for completeness
    },
    "Health Care": {
        "pe":         22.0,
        "ev_ebitda":  14.0,
    },
    "Industrials": {
        "pe":         20.0,
        "ev_ebitda":  13.0,
    },
    "Materials": {
        "pe":         16.0,
        "ev_ebitda":  10.0,
    },
    "Real Estate": {
        "pe":         40.0,    # REITs have high P/E; EV/EBITDA more meaningful
        "ev_ebitda":  18.0,
    },
    "Utilities": {
        "pe":         18.0,
        "ev_ebitda":  11.0,
    },
    # Fallback
    "N/A": {
        "pe":         20.0,
        "ev_ebitda":  12.0,
    },
}

# Default multiples when sector not recognised
_DEFAULT_MULTIPLES: dict[str, float] = SECTOR_MULTIPLES["N/A"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sector_multiples(sector: str) -> dict[str, float]:
    """
    Return the industry-average multiples for a given sector string.

    Falls back to a broad-market default if the sector is unknown.
    """
    return SECTOR_MULTIPLES.get(sector, _DEFAULT_MULTIPLES)


def run_multiples_valuation(
    eps: float | None,
    ebitda: float | None,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
    sector: str,
    pe_override: float | None = None,
    ev_ebitda_override: float | None = None,
) -> dict:
    """
    Compute implied stock prices from P/E and EV/EBITDA multiples,
    then return a blended estimate.

    Parameters
    ----------
    eps               : Trailing twelve-month earnings per share.
    ebitda            : Trailing twelve-month EBITDA (absolute, in dollars).
    total_debt        : Total financial debt (from balance sheet).
    cash              : Cash & equivalents (from balance sheet).
    shares_outstanding: Share count used for per-share conversion.
    sector            : Company sector string from yfinance info.
    pe_override       : User-supplied P/E multiple (overrides sector default).
    ev_ebitda_override: User-supplied EV/EBITDA multiple (overrides default).

    Returns
    -------
    dict with keys:
        pe_multiple        : multiple used
        ev_ebitda_multiple : multiple used
        pe_implied_price   : price implied by P/E  (None if EPS unavailable)
        ev_implied_price   : price implied by EV/EBITDA (None if unavailable)
        blended_price      : average of available implied prices
        available_methods  : list of methods that produced a valid estimate
    """
    multiples = get_sector_multiples(sector)

    pe_mult        = pe_override        if pe_override        is not None else multiples["pe"]
    ev_ebitda_mult = ev_ebitda_override if ev_ebitda_override is not None else multiples["ev_ebitda"]

    # ---- Method 1: P/E -------------------------------------------------
    pe_implied = _pe_valuation(eps, pe_mult)

    # ---- Method 2: EV/EBITDA -------------------------------------------
    ev_implied = _ev_ebitda_valuation(
        ebitda, total_debt, cash, shares_outstanding, ev_ebitda_mult
    )

    # ---- Blended estimate (equal weight, skip None) -------------------
    available = [v for v in [pe_implied, ev_implied] if v is not None and v > 0]
    blended   = (sum(available) / len(available)) if available else None

    methods = []
    if pe_implied is not None and pe_implied > 0:
        methods.append("P/E")
    if ev_implied is not None and ev_implied > 0:
        methods.append("EV/EBITDA")

    return {
        "pe_multiple":        pe_mult,
        "ev_ebitda_multiple": ev_ebitda_mult,
        "pe_implied_price":   pe_implied,
        "ev_implied_price":   ev_implied,
        "blended_price":      blended,
        "available_methods":  methods,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pe_valuation(eps: float | None, pe_multiple: float) -> float | None:
    """
    P/E implied price = EPS × P/E multiple.
    Returns None when EPS is missing or negative (growth stocks).
    """
    if eps is None or eps <= 0:
        return None
    return eps * pe_multiple


def _ev_ebitda_valuation(
    ebitda: float | None,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
    ev_ebitda_multiple: float,
) -> float | None:
    """
    EV/EBITDA implied price.

    Steps:
      1. EV_implied = EBITDA × EV/EBITDA multiple
      2. Equity_value = EV_implied − Debt + Cash   (bridge: EV = Equity + Debt − Cash)
      3. Implied price = Equity_value / shares_outstanding
    """
    if ebitda is None or ebitda <= 0:
        return None
    if shares_outstanding is None or shares_outstanding <= 0:
        return None

    ev_implied     = ebitda * ev_ebitda_multiple
    equity_value   = ev_implied - total_debt + cash
    implied_price  = equity_value / shares_outstanding

    return implied_price if implied_price > 0 else None
