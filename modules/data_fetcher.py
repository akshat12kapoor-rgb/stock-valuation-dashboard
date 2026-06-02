"""
data_fetcher.py
---------------
Fetches and normalises financial data for a given stock ticker
using the yfinance library.

Key data retrieved:
  - Current stock price
  - Revenue (TTM and historical)
  - Free Cash Flow (Operating CF - CapEx)
  - EBITDA
  - Net Income & EPS
  - Shares outstanding
  - Debt and Cash (for EV calculation)
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str) -> dict:
    """
    Main entry point. Returns a flat dict of all financial metrics
    needed by the valuation models.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL", "MSFT".

    Returns
    -------
    dict  with keys:
        ticker, company_name, current_price,
        revenue, fcf, ebitda, net_income, eps,
        shares_outstanding, total_debt, cash,
        revenue_history, fcf_history,
        sector, industry

    Raises
    ------
    ValueError  if the ticker is invalid or data cannot be fetched.
    """
    stock = _load_ticker(ticker)
    info  = _safe_info(stock)

    # ---- Price & shares ------------------------------------------------
    current_price      = _get_price(info, stock)
    shares_outstanding = _safe_get(info, ["sharesOutstanding"], default=None)

    # ---- Financial statements ------------------------------------------
    income_stmt = _load_statement(stock, "income_stmt")
    cashflow    = _load_statement(stock, "cashflow")
    balance     = _load_statement(stock, "balance_sheet")

    # ---- Extract series ------------------------------------------------
    revenue_series = _extract_series(income_stmt, ["Total Revenue"])
    fcf_series     = _compute_fcf(cashflow)
    ebitda_series  = _extract_series(income_stmt, ["EBITDA"])
    ni_series      = _extract_series(income_stmt, ["Net Income"])

    # ---- Most-recent values (use TTM if available, else latest annual) -
    revenue    = _latest(revenue_series)
    fcf        = _latest(fcf_series)
    ebitda     = _latest(ebitda_series)
    net_income = _latest(ni_series)

    # ---- EPS -----------------------------------------------------------
    eps = _safe_get(info, ["trailingEps", "forwardEps"], default=None)
    if eps is None and net_income and shares_outstanding:
        eps = net_income / shares_outstanding

    # ---- Balance-sheet items for EV -----------------------------------
    total_debt = _latest(_extract_series(balance, ["Total Debt", "Long Term Debt"]))
    cash       = _latest(_extract_series(balance, ["Cash And Cash Equivalents",
                                                    "Cash Cash Equivalents And Short Term Investments"]))
    total_debt = total_debt or 0.0
    cash       = cash       or 0.0

    # ---- Validation ----------------------------------------------------
    if current_price is None:
        raise ValueError(f"Could not retrieve price for '{ticker}'. "
                         "Check the ticker symbol and try again.")

    if fcf is None or fcf == 0:
        # Attempt fallback: use net income as a proxy for FCF
        fcf = net_income
    if fcf is None:
        raise ValueError(f"Insufficient cash-flow data for '{ticker}'.")

    return {
        "ticker":            ticker.upper(),
        "company_name":      info.get("longName", ticker.upper()),
        "current_price":     current_price,
        "shares_outstanding": shares_outstanding,
        "revenue":           revenue,
        "fcf":               fcf,
        "ebitda":            ebitda,
        "net_income":        net_income,
        "eps":               eps,
        "total_debt":        total_debt,
        "cash":              cash,
        "revenue_history":   _to_clean_series(revenue_series),
        "fcf_history":       _to_clean_series(fcf_series),
        "sector":            info.get("sector",   "N/A"),
        "industry":          info.get("industry", "N/A"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ticker(ticker: str) -> yf.Ticker:
    """Create a yfinance Ticker object and do a basic validation check."""
    t = yf.Ticker(ticker.strip().upper())
    return t


def _safe_info(stock: yf.Ticker) -> dict:
    """Safely retrieve the .info dict; return empty dict on failure."""
    try:
        info = stock.info
        if not info or info.get("quoteType") == "MUTUALFUND":
            return {}
        return info
    except Exception:
        return {}


def _get_price(info: dict, stock: yf.Ticker) -> float | None:
    """Try multiple fields to get the current market price."""
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        val = info.get(key)
        if val and val > 0:
            return float(val)

    # Last resort: pull 1-day history
    try:
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _load_statement(stock: yf.Ticker, attr: str) -> pd.DataFrame:
    """Load a financial statement; return empty DataFrame on failure."""
    try:
        df = getattr(stock, attr)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _extract_series(df: pd.DataFrame, row_names: list[str]) -> pd.Series:
    """
    Extract the first matching row from a financial statement DataFrame.
    yfinance stores rows as the index, columns as dates (newest first).
    Returns a Series indexed by date (oldest→newest), or empty Series.
    """
    if df.empty:
        return pd.Series(dtype=float)

    for name in row_names:
        for idx in df.index:
            if name.lower() in str(idx).lower():
                row = df.loc[idx].dropna().astype(float)
                return row.sort_index()          # oldest → newest

    return pd.Series(dtype=float)


def _compute_fcf(cashflow: pd.DataFrame) -> pd.Series:
    """
    Free Cash Flow = Operating Cash Flow - Capital Expenditure.
    CapEx is usually reported as a negative number in yfinance; we
    handle both conventions.
    """
    ocf  = _extract_series(cashflow, ["Operating Cash Flow",
                                       "Cash Flow From Operations"])
    capex = _extract_series(cashflow, ["Capital Expenditure",
                                        "Capital Expenditures"])

    if ocf.empty:
        return pd.Series(dtype=float)

    if capex.empty:
        return ocf  # best we can do

    # Align on common dates
    common = ocf.index.intersection(capex.index)
    if common.empty:
        return ocf

    ocf_a   = ocf[common]
    capex_a = capex[common].abs()   # ensure positive so we subtract
    return (ocf_a - capex_a).sort_index()


def _latest(series: pd.Series) -> float | None:
    """Return the most recent (last) non-NaN value, or None."""
    if series.empty:
        return None
    val = series.dropna().iloc[-1] if not series.dropna().empty else None
    return float(val) if val is not None else None


def _to_clean_series(series: pd.Series) -> pd.Series:
    """Return a clean Series with year strings as index and float values."""
    if series.empty:
        return pd.Series(dtype=float)
    s = series.dropna()
    s.index = [str(idx)[:4] for idx in s.index]   # keep only year
    return s


def _safe_get(info: dict, keys: list[str], default=None):
    """Try multiple keys in the info dict and return the first found value."""
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return default
