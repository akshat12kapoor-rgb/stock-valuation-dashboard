"""
dcf_model.py
------------
Two-Stage Discounted Cash Flow (DCF) valuation engine.

Two-Stage Model
---------------
Real companies don't grow at a constant rate forever. The two-stage model
reflects the lifecycle of a business more accurately:

  Stage 1 (Years 1 – stage1_years):
    High-growth phase. FCF grows at `growth_rate` each year.
    e.g. a fast-growing tech company at 20% for 5 years.

  Stage 2 (Years stage1_years+1 – total_years):
    Fading / transition phase. Growth rate linearly interpolates from
    `growth_rate` down to `terminal_growth` over this period.
    This models the company maturing and losing its competitive edge.

  Terminal Value (after total_years):
    Gordon Growth Model assuming perpetual growth at `terminal_growth`.
    TV = FCF_final × (1 + g) / (WACC – g)

Formula recap
-------------
  Stage-1 FCF(t) = Base FCF × (1 + g1)^t
  Stage-2 FCF(t) = FCF(t-1) × (1 + g_fade(t))
    where g_fade linearly declines from g1 → g_terminal
  PV(t)   = FCF(t) / (1 + WACC)^t
  TV      = FCF(N) × (1 + g_terminal) / (WACC – g_terminal)
  PV(TV)  = TV / (1 + WACC)^N
  Intrinsic = (Σ PV(t) + PV(TV)) / shares_outstanding
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dcf(
    base_fcf: float,
    shares_outstanding: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
) -> dict:
    """
    Run a two-stage DCF model.

    Parameters
    ----------
    base_fcf           : Most-recent Free Cash Flow (absolute dollars).
    shares_outstanding : Total shares outstanding.
    growth_rate        : Stage-1 annual FCF growth rate (decimal, e.g. 0.15).
    wacc               : Discount rate / WACC (decimal).
    terminal_growth    : Perpetuity growth rate after Stage 2 (decimal).
    stage1_years       : Number of high-growth years (default 5).
    stage2_years       : Number of fade/transition years (default 5).

    Returns
    -------
    dict with keys:
        projected_fcfs      : list of all projected FCF values (stage1 + stage2)
        pv_fcfs             : list of their present values
        growth_rates_used   : list of growth rates applied each year
        terminal_value      : undiscounted terminal value
        pv_terminal_value   : discounted terminal value
        total_pv            : sum of all PV cash flows
        equity_value        : total equity value
        intrinsic_value     : per-share intrinsic value
        year_labels         : ["Year 1", …]
        stage1_years        : int
        stage2_years        : int
        stage1_pv           : sum of PV from stage 1
        stage2_pv           : sum of PV from stage 2
    """
    total_years = stage1_years + stage2_years
    _validate_inputs(wacc, terminal_growth)

    projected_fcfs:    list[float] = []
    pv_fcfs:           list[float] = []
    growth_rates_used: list[float] = []

    prev_fcf = base_fcf

    for t in range(1, total_years + 1):
        # Determine growth rate for this year
        if t <= stage1_years:
            # Stage 1: constant high growth
            g = growth_rate
        else:
            # Stage 2: linearly interpolate from growth_rate → terminal_growth
            # At t = stage1_years+1 → g ≈ growth_rate
            # At t = total_years    → g ≈ terminal_growth
            fade_steps  = stage2_years
            step_index  = t - stage1_years           # 1 … stage2_years
            g = growth_rate + (terminal_growth - growth_rate) * (step_index / fade_steps)

        fcf_t  = prev_fcf * (1 + g)
        pv_t   = fcf_t / (1 + wacc) ** t

        projected_fcfs.append(fcf_t)
        pv_fcfs.append(pv_t)
        growth_rates_used.append(g)
        prev_fcf = fcf_t

    # Terminal Value (Gordon Growth Model off the last projected FCF)
    fcf_final      = projected_fcfs[-1]
    terminal_value = fcf_final * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv          = terminal_value / (1 + wacc) ** total_years

    # Stage breakdown for display
    stage1_pv = sum(pv_fcfs[:stage1_years])
    stage2_pv = sum(pv_fcfs[stage1_years:])

    total_pv        = sum(pv_fcfs) + pv_tv
    equity_value    = total_pv
    intrinsic_value = (equity_value / shares_outstanding) if shares_outstanding else 0.0

    return {
        "projected_fcfs":    projected_fcfs,
        "pv_fcfs":           pv_fcfs,
        "growth_rates_used": growth_rates_used,
        "terminal_value":    terminal_value,
        "pv_terminal_value": pv_tv,
        "total_pv":          total_pv,
        "equity_value":      equity_value,
        "intrinsic_value":   max(intrinsic_value, 0.0),
        "year_labels":       [f"Year {t}" for t in range(1, total_years + 1)],
        "stage1_years":      stage1_years,
        "stage2_years":      stage2_years,
        "stage1_pv":         stage1_pv,
        "stage2_pv":         stage2_pv,
    }


def sensitivity_matrix(
    base_fcf: float,
    shares_outstanding: float,
    terminal_growth: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
    wacc_range: list[float] | None = None,
    growth_range: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a sensitivity table: intrinsic value for a grid of
    (WACC) × (Stage-1 growth rate) combinations.

    Returns a DataFrame where:
      - rows    = WACC values
      - columns = Stage-1 growth rates
      - values  = intrinsic value per share
    """
    if wacc_range is None:
        wacc_range = [round(w, 3) for w in np.arange(0.06, 0.161, 0.02)]
    if growth_range is None:
        growth_range = [round(g, 3) for g in np.arange(0.02, 0.201, 0.03)]

    records = {}
    for w in wacc_range:
        row = {}
        for g in growth_range:
            try:
                result = run_dcf(
                    base_fcf           = base_fcf,
                    shares_outstanding = shares_outstanding,
                    growth_rate        = g,
                    wacc               = w,
                    terminal_growth    = terminal_growth,
                    stage1_years       = stage1_years,
                    stage2_years       = stage2_years,
                )
                row[f"{g:.0%}"] = round(result["intrinsic_value"], 2)
            except ValueError:
                row[f"{g:.0%}"] = np.nan
        records[f"{w:.0%}"] = row

    df = pd.DataFrame(records).T
    df.index.name = "WACC \\ Growth"
    return df


def project_revenue(
    base_revenue: float,
    growth_rate: float,
    terminal_growth: float,
    stage1_years: int = 5,
    stage2_years: int = 5,
) -> list[float]:
    """
    Two-stage revenue projection (mirrors FCF logic) — used for charting.
    Stage 1: constant growth; Stage 2: linear fade to terminal growth.
    """
    revenues = []
    prev = base_revenue
    total = stage1_years + stage2_years

    for t in range(1, total + 1):
        if t <= stage1_years:
            g = growth_rate
        else:
            step = t - stage1_years
            g    = growth_rate + (terminal_growth - growth_rate) * (step / stage2_years)
        prev = prev * (1 + g)
        revenues.append(prev)

    return revenues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(wacc: float, terminal_growth: float) -> None:
    """Guard against model-breaking inputs."""
    if wacc <= 0:
        raise ValueError("WACC must be positive.")
    if terminal_growth < 0:
        raise ValueError("Terminal growth rate cannot be negative.")
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc:.1%}) must be greater than terminal growth "
            f"({terminal_growth:.1%}) — otherwise the Gordon Growth Model "
            "produces an infinite terminal value."
        )
