"""
app.py
------
Should I Invest? — Stock Valuation Dashboard
============================================

Entry point for the Streamlit application.

Run with:
    streamlit run app.py

Architecture
------------
  1. Sidebar   → user inputs (ticker, model parameters)
  2. Header    → company info + live price
  3. KPI row   → key financial metrics
  4. Charts    → FCF & revenue projections
  5. Results   → valuation table + recommendation gauge
  6. Extras    → sensitivity heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np

# Internal modules
from modules.data_fetcher    import fetch_stock_data
from modules.dcf_model       import run_dcf, sensitivity_matrix, project_revenue
from modules.multiples_model import run_multiples_valuation
from modules.decision_engine import make_decision, describe_decision, get_valuation_breakdown
from modules import charts


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Should I Invest? | Stock Valuation",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark finance theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #8B949E !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #C9D1D9 !important; font-size: 1.3rem; font-weight: 600; }
    [data-testid="stMetricDelta"]  { font-size: 0.85rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0D1117; }

    /* Decision banner */
    .decision-banner {
        border-radius: 10px;
        padding: 20px 28px;
        text-align: center;
        margin: 16px 0;
    }
    .decision-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .decision-sub {
        font-size: 0.95rem;
        color: #8B949E;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
        margin-bottom: 8px;
        margin-top: 24px;
        border-bottom: 1px solid #21262D;
        padding-bottom: 6px;
    }

    /* Valuation table */
    .val-table { width: 100%; border-collapse: collapse; }
    .val-table td, .val-table th {
        padding: 10px 14px;
        border-bottom: 1px solid #21262D;
        font-size: 0.9rem;
    }
    .val-table th { color: #8B949E; font-weight: 500; text-align: left; }
    .val-table td { color: #C9D1D9; }
    .val-table .highlight { color: #58A6FF; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — User Inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 Stock Valuation")
    st.caption("DCF + Comparable Company Analysis")

    st.markdown("---")
    st.markdown("### 🔍 Stock Ticker")

    # Popular tickers grouped for the combobox
    POPULAR_TICKERS = [
        # Mega-cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        # Consumer
        "WMT", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP",
        # Energy / Industrials
        "XOM", "CVX", "COP", "GE", "CAT", "BA", "HON", "RTX",
        # Other notable
        "BRK-B", "NFLX", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN",
        # ETFs (for testing)
        "SPY", "QQQ", "DIA",
    ]

    ticker_input = st.selectbox(
        "Select or search ticker",
        options          = POPULAR_TICKERS,
        index            = 0,
        help             = "Type to search. For unlisted tickers, select 'Custom…' below.",
        label_visibility = "collapsed",
    )

    use_custom = st.checkbox("Enter custom ticker", value=False)
    if use_custom:
        custom_ticker = st.text_input(
            "Custom ticker",
            value            = "",
            max_chars        = 10,
            placeholder      = "e.g. TSM, BABA, SHOP",
            label_visibility = "collapsed",
        ).strip().upper()
        if custom_ticker:
            ticker_input = custom_ticker

    st.markdown("---")
    st.markdown("### ⚙️ DCF Assumptions")

    # --- Model type selector ---
    model_type = st.radio(
        "DCF Model",
        options    = ["Single-Stage", "Two-Stage", "Three-Stage"],
        index      = 1,
        horizontal = True,
        help       = (
            "**Single-Stage**: constant growth for 10 years → Terminal Value.  \n"
            "**Two-Stage**: high growth (N years) → Terminal Value.  \n"
            "**Three-Stage**: high growth → linear fade → Terminal Value."
        ),
    )

    _MODEL_CAPTIONS = {
        "Single-Stage": "One constant growth rate, 10-year horizon → Terminal Value",
        "Two-Stage":    "**Stage 1** — high growth  ·  then Terminal Value",
        "Three-Stage":  "**Stage 1** — high growth  ·  **Stage 2** — fade to terminal",
    }
    st.caption(_MODEL_CAPTIONS[model_type])

    growth_rate_pct = st.slider(
        "FCF Growth Rate" if model_type == "Single-Stage" else "Stage 1 FCF Growth Rate",
        min_value  = 0,
        max_value  = 40,
        value      = 10,
        step       = 1,
        format     = "%d%%",
        help       = "Annual FCF growth rate during the high-growth phase.",
    )
    growth_rate = growth_rate_pct / 100.0

    # Stage 1 duration — shown for Two-Stage and Three-Stage only
    if model_type == "Single-Stage":
        stage1_years = 10   # fixed, not user-configurable
    else:
        stage1_years = st.slider(
            "Stage 1 Duration (years)",
            min_value = 1,
            max_value = 10,
            value     = 5,
            step      = 1,
            help      = "How many years the high-growth phase lasts.",
        )

    # Stage 2 duration — shown for Three-Stage only
    if model_type == "Three-Stage":
        stage2_years = st.slider(
            "Stage 2 Duration (years)",
            min_value = 1,
            max_value = 10,
            value     = 5,
            step      = 1,
            help      = "Fade/transition years where growth linearly declines "
                        "from Stage 1 rate down to the terminal growth rate.",
        )
    else:
        stage2_years = 0    # no fade period for Single- or Two-Stage

    wacc_pct = st.slider(
        "Discount Rate (WACC)",
        min_value  = 5,
        max_value  = 25,
        value      = 10,
        step       = 1,
        format     = "%d%%",
        help       = "Weighted Average Cost of Capital — your required return. "
                     "Typically 8–12% for large-cap equities.",
    )
    wacc = wacc_pct / 100.0

    terminal_growth_pct = st.slider(
        "Terminal Growth Rate",
        min_value  = 1,
        max_value  = 5,
        value      = 3,
        step       = 1,
        format     = "%d%%",
        help       = "Long-run perpetuity growth rate after Stage 2. "
                     "Usually close to GDP growth (~2–3%).",
    )
    terminal_growth = terminal_growth_pct / 100.0

    st.markdown("---")
    st.markdown("### 📊 Comparable Multiples")

    use_custom_multiples = st.toggle("Override sector multiples", value=False)
    if use_custom_multiples:
        custom_pe = st.number_input("P/E Multiple",
                                    min_value=1.0, max_value=100.0, value=20.0, step=0.5)
        custom_ev = st.number_input("EV/EBITDA Multiple",
                                    min_value=1.0, max_value=50.0, value=12.0, step=0.5)
    else:
        custom_pe = None
        custom_ev = None

    st.markdown("---")
    st.markdown("### ⚖️ Blending Weight")
    dcf_weight_pct = st.slider(
        "DCF weight in final value",
        min_value  = 0,
        max_value  = 100,
        value      = 60,
        step       = 5,
        format     = "%d%%",
        help       = "How much weight to give DCF vs Multiples in the blended fair value.",
    )
    dcf_weight = dcf_weight_pct / 100.0

    run_btn = st.button("🚀 Analyse", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Data via Yahoo Finance · For educational purposes only · Not financial advice")


# ---------------------------------------------------------------------------
# Main — run analysis
# ---------------------------------------------------------------------------

# Guard: only run when ticker provided and button pressed (or first load of AAPL)
if not ticker_input:
    st.info("Enter a ticker symbol in the sidebar and press **Analyse**.")
    st.stop()

# Use session state to cache the last successful fetch
if "last_ticker" not in st.session_state:
    st.session_state["last_ticker"] = None
if "stock_data" not in st.session_state:
    st.session_state["stock_data"] = None

should_run = run_btn or (st.session_state["stock_data"] is None)

# ---------------------------------------------------------------------------
# Data fetch (with spinner + error handling)
# ---------------------------------------------------------------------------

if should_run or st.session_state["last_ticker"] != ticker_input:
    with st.spinner(f"Fetching data for **{ticker_input}**…"):
        try:
            data = fetch_stock_data(ticker_input)
            st.session_state["stock_data"]  = data
            st.session_state["last_ticker"] = ticker_input
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected error fetching data: {exc}")
            st.stop()

data = st.session_state["stock_data"]

# ---------------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------------

# ---- DCF ---------------------------------------------------------------
try:
    dcf_result = run_dcf(
        base_fcf           = data["fcf"],
        shares_outstanding = data["shares_outstanding"],
        growth_rate        = growth_rate,
        wacc               = wacc,
        terminal_growth    = terminal_growth,
        stage1_years       = stage1_years,
        stage2_years       = stage2_years,
    )
except ValueError as exc:
    st.error(f"❌ DCF Error: {exc}")
    st.stop()

# ---- Multiples ---------------------------------------------------------
mult_result = run_multiples_valuation(
    eps               = data["eps"],
    ebitda            = data["ebitda"],
    total_debt        = data["total_debt"],
    cash              = data["cash"],
    shares_outstanding= data["shares_outstanding"],
    sector            = data["sector"],
    pe_override       = custom_pe,
    ev_ebitda_override= custom_ev,
)

# ---- Decision ----------------------------------------------------------
decision = make_decision(
    ticker          = data["ticker"],
    current_price   = data["current_price"],
    dcf_value       = dcf_result["intrinsic_value"],
    multiples_value = mult_result["blended_price"],
    dcf_weight      = dcf_weight,
)

# ---- Revenue projection (for chart) -----------------------------------
rev_projection = project_revenue(
    base_revenue    = data["revenue"] or 0,
    growth_rate     = growth_rate,
    terminal_growth = terminal_growth,
    stage1_years    = stage1_years,
    stage2_years    = stage2_years,
)

# ---- Sensitivity -------------------------------------------------------
sens_df = sensitivity_matrix(
    base_fcf           = data["fcf"],
    shares_outstanding = data["shares_outstanding"],
    terminal_growth    = terminal_growth,
    stage1_years       = stage1_years,
    stage2_years       = stage2_years,
)


# ===========================================================================
# LAYOUT
# ===========================================================================

# ---------------------------------------------------------------------------
# Header — company name + live price
# ---------------------------------------------------------------------------

col_title, col_price, col_upside = st.columns([3, 1, 1])

with col_title:
    st.markdown(f"## {data['company_name']}  `{data['ticker']}`")
    st.caption(f"**Sector:** {data['sector']}  ·  **Industry:** {data['industry']}")

with col_price:
    st.metric(
        label = "Live Price",
        value = f"${data['current_price']:,.2f}",
    )

with col_upside:
    upside_sign = "+" if decision.upside_pct >= 0 else ""
    st.metric(
        label = "Signal",
        value = f"{decision.emoji} {decision.decision}",
        delta = f"{upside_sign}{decision.upside_pct:.1%} vs fair value",
        delta_color = "normal" if decision.upside_pct >= 0 else "inverse",
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# KPI Row — 5 quick stats
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">Key Financial Metrics</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

def _fmt_b(val):
    """Format a large dollar value in billions."""
    if val is None:
        return "N/A"
    return f"${val/1e9:.2f}B"

with k1:
    st.metric("Revenue (TTM)", _fmt_b(data["revenue"]))
with k2:
    st.metric("Free Cash Flow", _fmt_b(data["fcf"]))
with k3:
    st.metric("EBITDA", _fmt_b(data["ebitda"]))
with k4:
    st.metric("Net Income", _fmt_b(data["net_income"]))
with k5:
    eps_str = f"${data['eps']:.2f}" if data["eps"] is not None else "N/A"
    st.metric("EPS (TTM)", eps_str)


# ---------------------------------------------------------------------------
# Decision Banner
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">Investment Recommendation</p>',
            unsafe_allow_html=True)

col_gauge, col_decision = st.columns([1, 3])

with col_gauge:
    st.plotly_chart(
        charts.upside_gauge(decision.upside_pct, data["ticker"], decision.decision),
        use_container_width = True,
    )

with col_decision:
    color_map = {"BUY": "#0D2D0D", "HOLD": "#2D2D0A", "SELL": "#3D1515"}
    border_map= {"BUY": "#3FB950", "HOLD": "#D29922", "SELL": "#F85149"}
    bg    = color_map[decision.decision]
    bdr   = border_map[decision.decision]

    # Strip leading emoji from describe_decision since banner already shows it
    desc_text = describe_decision(decision)
    # Remove the leading "🟢 BUY —" / "🔴 SELL —" / "🟡 HOLD —" prefix
    for prefix in ["🟢 BUY —", "🔴 SELL —", "🟡 HOLD —",
                   "🟢 BUY", "🔴 SELL", "🟡 HOLD"]:
        if desc_text.startswith(prefix):
            desc_text = desc_text[len(prefix):].lstrip(" —").strip()
            break

    st.markdown(f"""
    <div class="decision-banner" style="background:{bg}; border:2px solid {bdr};">
        <p class="decision-text" style="color:{decision.color}">
            {decision.emoji} {decision.decision}
        </p>
        <p class="decision-sub">{desc_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # Valuation breakdown table
    breakdown = get_valuation_breakdown(decision)
    rows_html = ""
    for label, val in breakdown.items():
        is_current = "Current Price" in label
        is_blend   = "Blended" in label
        cls = "highlight" if is_blend else ""
        if is_current:
            vs_market = "—"
        else:
            raw_pct   = (val - data["current_price"]) / data["current_price"]
            sign      = "+" if raw_pct > 0 else ""
            color_str = "#3FB950" if raw_pct > 0 else "#F85149"
            vs_market = (
                f"<span style='color:{color_str};font-weight:600'>"
                f"{sign}{raw_pct:.1%}</span>"
            )
        rows_html += (
            f"<tr><td>{label}</td>"
            f"<td class='{cls}'>${val:,.2f}</td>"
            f"<td class='{cls}'>{vs_market}</td></tr>"
        )

    st.markdown(f"""
    <table class="val-table">
      <thead><tr>
        <th>Metric</th><th>Value / Share</th><th>vs Market Price</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Multiples used
    st.caption(
        f"**Multiples used:** P/E {mult_result['pe_multiple']:.1f}×  |  "
        f"EV/EBITDA {mult_result['ev_ebitda_multiple']:.1f}×  |  "
        f"Methods: {', '.join(mult_result['available_methods']) or 'N/A'}"
    )


# ---------------------------------------------------------------------------
# Cash-flow & Revenue Charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown('<p class="section-header">Projections</p>', unsafe_allow_html=True)

c_fcf, c_rev = st.columns(2)

_bar_config = {"displayModeBar": False}

with c_fcf:
    st.plotly_chart(
        charts.fcf_projection_chart(
            historical   = data["fcf_history"],
            projected    = dcf_result["projected_fcfs"],
            year_labels  = dcf_result["year_labels"],
            ticker       = data["ticker"],
            stage1_years = stage1_years,
        ),
        use_container_width = True,
        config = _bar_config,
    )

with c_rev:
    st.plotly_chart(
        charts.revenue_projection_chart(
            historical  = data["revenue_history"],
            projected   = rev_projection,
            year_labels = dcf_result["year_labels"],
            ticker      = data["ticker"],
        ),
        use_container_width = True,
        config = _bar_config,
    )


# ---------------------------------------------------------------------------
# Valuation Comparison Chart
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">Valuation Comparison</p>', unsafe_allow_html=True)

st.plotly_chart(
    charts.valuation_comparison_chart(
        current_price   = data["current_price"],
        dcf_value       = dcf_result["intrinsic_value"],
        multiples_value = mult_result["blended_price"],
        blended_value   = decision.blended_value,
        ticker          = data["ticker"],
    ),
    config = _bar_config,
    use_container_width = True,
)


# ---------------------------------------------------------------------------
# DCF Detail Expander
# ---------------------------------------------------------------------------

with st.expander(f"📋 {model_type} DCF Model Detail", expanded=False):
    # Year-by-year table
    stages = (
        ["Stage 1"] * dcf_result["stage1_years"] +
        ["Stage 2"] * dcf_result["stage2_years"]
    )
    dcf_table = pd.DataFrame({
        "Year"              : dcf_result["year_labels"],
        "Stage"             : stages,
        "Growth Rate"       : [f"{g:.1%}" for g in dcf_result["growth_rates_used"]],
        "Projected FCF ($B)": [f"{v/1e9:.3f}" for v in dcf_result["projected_fcfs"]],
        "PV of FCF ($B)"    : [f"{v/1e9:.3f}" for v in dcf_result["pv_fcfs"]],
    })
    st.dataframe(dcf_table, hide_index=True, use_container_width=True)

    # Value breakdown
    total_pv = dcf_result["total_pv"] or 1
    s1_pct   = dcf_result["stage1_pv"]         / total_pv * 100
    s2_pct   = dcf_result["stage2_pv"]         / total_pv * 100
    tv_pct   = dcf_result["pv_terminal_value"] / total_pv * 100

    c1, c2, c3, c4 = st.columns(4)
    _s1_label = "Forecast PV" if model_type == "Single-Stage" else "Stage 1 PV"
    c1.metric(_s1_label,           f"${dcf_result['stage1_pv']/1e9:.2f}B",
              delta=f"{s1_pct:.0f}% of total", delta_color="off")
    if model_type == "Three-Stage":
        c2.metric("Stage 2 PV",    f"${dcf_result['stage2_pv']/1e9:.2f}B",
                  delta=f"{s2_pct:.0f}% of total", delta_color="off")
    else:
        c2.metric("Stage 2 PV",    "—", delta="N/A (not modelled)", delta_color="off")
    c3.metric("PV of Terminal Val",f"${dcf_result['pv_terminal_value']/1e9:.2f}B",
              delta=f"{tv_pct:.0f}% of total", delta_color="off")
    c4.metric("DCF Intrinsic Value",f"${dcf_result['intrinsic_value']:.2f}")

    if model_type == "Single-Stage":
        _model_note = (
            f"Single constant growth rate of **{growth_rate:.0%}** applied for "
            f"**{stage1_years} years**, then a terminal value using Gordon Growth Model."
        )
    elif model_type == "Two-Stage":
        _model_note = (
            f"High-growth phase at **{growth_rate:.0%}** for **{stage1_years} years**, "
            f"followed directly by a terminal value growing at **{terminal_growth:.0%}** in perpetuity."
        )
    else:
        _model_note = (
            f"Stage 2 growth linearly declines from **{growth_rate:.0%}** → **{terminal_growth:.0%}** "
            f"over {stage2_years} year{'s' if stage2_years != 1 else ''}, "
            f"reducing dependence on the terminal value assumption."
        )
    st.caption(
        f"Terminal value is **{tv_pct:.0f}%** of total equity value. " + _model_note
    )


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown('<p class="section-header">Sensitivity Analysis — Intrinsic Value per Share</p>',
            unsafe_allow_html=True)

st.caption(
    "Each cell shows the DCF intrinsic value for a given combination of "
    "**WACC** (rows) and **FCF Growth Rate** (columns). "
    "Green = above current market price  ·  Red = below."
)

st.plotly_chart(
    charts.sensitivity_heatmap(
        sensitivity_df = sens_df,
        current_price  = data["current_price"],
        ticker         = data["ticker"],
    ),
    use_container_width = True,
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "**Disclaimer:** This tool is for educational and informational purposes only. "
    "It does not constitute financial advice. Always conduct your own research "
    "before making investment decisions. Data sourced from Yahoo Finance via yfinance."
)
