"""
charts.py
---------
All Plotly visualisations for the Stock Valuation Dashboard.

Charts
------
  1. fcf_projection_chart      — historical + projected FCF bars
  2. revenue_projection_chart  — historical + projected revenue bars
  3. valuation_comparison_chart— market price vs each model vs blended
  4. sensitivity_heatmap       — intrinsic value grid (WACC × growth)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

THEME = dict(
    bg_color     = "#0E1117",
    paper_color  = "#161B22",
    grid_color   = "#21262D",
    text_color   = "#C9D1D9",
    accent_blue  = "#58A6FF",
    accent_green = "#3FB950",
    accent_red   = "#F85149",
    accent_amber = "#D29922",
    font_family  = "Inter, Arial, sans-serif",
)

_layout_defaults = dict(
    plot_bgcolor  = THEME["bg_color"],
    paper_bgcolor = THEME["paper_color"],
    font          = dict(color=THEME["text_color"], family=THEME["font_family"]),
    margin        = dict(l=40, r=20, t=50, b=40),
    legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis         = dict(gridcolor=THEME["grid_color"], zeroline=False),
    yaxis         = dict(gridcolor=THEME["grid_color"], zeroline=False),
)


# ---------------------------------------------------------------------------
# 1. FCF Projection Chart
# ---------------------------------------------------------------------------

def fcf_projection_chart(
    historical: pd.Series,
    projected: list[float],
    year_labels: list[str],
    ticker: str,
    stage1_years: int = 5,
) -> go.Figure:
    """
    Bar chart: historical FCF + Stage 1 (green) + Stage 2 fade (teal).

    Uses a single Bar trace with per-bar colors so every bar is full-width
    (avoids the thin-bar artifact from barmode="group" with non-overlapping traces).
    """
    fig = go.Figure()

    if not historical.empty:
        try:
            last_hist_year = int(list(historical.index)[-1])
        except (ValueError, IndexError):
            last_hist_year = 2024
    else:
        last_hist_year = 2024

    proj_years = [str(last_hist_year + i) for i in range(1, len(projected) + 1)]

    # Build one unified bar trace with per-bar colors
    all_x      = list(historical.index) + proj_years           if not historical.empty else proj_years
    all_y      = [v / 1e9 for v in historical.values] + [v / 1e9 for v in projected] \
                 if not historical.empty else [v / 1e9 for v in projected]

    hist_len   = len(historical) if not historical.empty else 0
    s1_len     = stage1_years
    s2_len     = len(projected) - stage1_years

    colors = (
        [THEME["accent_blue"]] * hist_len +
        [THEME["accent_green"]] * s1_len +
        ["#4EC9B0"] * s2_len
    )
    opacities = (
        [0.55] * hist_len +
        [1.0]  * s1_len  +
        [0.85] * s2_len
    )

    fig.add_trace(go.Bar(
        x              = all_x,
        y              = all_y,
        marker_color   = colors,
        marker_opacity = opacities,
        showlegend     = False,
        hovertemplate  = "%{x}: $%{y:.2f}B<extra></extra>",
    ))

    # Invisible legend proxies
    for name, color, opacity in [
        ("Historical FCF",        THEME["accent_blue"],  0.55),
        ("Stage 1 (high growth)", THEME["accent_green"], 1.0),
        ("Stage 2 (fade)",        "#4EC9B0",             0.85),
    ]:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name         = name,
            marker_color = color,
            opacity      = opacity,
            showlegend   = True,
        ))

    # Divider: historical → forecast
    if not historical.empty:
        fig.add_vline(
            x=last_hist_year + 0.5, line_dash="dot",
            line_color=THEME["text_color"], line_width=1, opacity=0.4,
            annotation_text="Forecast →", annotation_position="top right",
            annotation_font=dict(color=THEME["text_color"], size=10),
        )

    # Divider: stage 1 → stage 2
    if s2_len > 0:
        fig.add_vline(
            x=last_hist_year + stage1_years + 0.5, line_dash="dash",
            line_color=THEME["accent_amber"], line_width=1, opacity=0.5,
            annotation_text="Fade →", annotation_position="top right",
            annotation_font=dict(color=THEME["accent_amber"], size=9),
        )

    fig.update_layout(
        **_layout_defaults,
        title       = f"{ticker} — Free Cash Flow ($B)",
        yaxis_title = "FCF ($B)",
        bargap      = 0.25,
    )
    fig.update_xaxes(
        tickangle = -45 if (hist_len + len(projected)) > 8 else 0,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Revenue Projection Chart
# ---------------------------------------------------------------------------

def revenue_projection_chart(
    historical: pd.Series,
    projected: list[float],
    year_labels: list[str],
    ticker: str,
) -> go.Figure:
    """
    Bar chart: historical revenue (muted) + projected revenue (bright).
    Uses real calendar years for both series so bars are contiguous.
    """
    fig = go.Figure()

    if not historical.empty:
        try:
            last_hist_year = int(list(historical.index)[-1])
        except (ValueError, IndexError):
            last_hist_year = 2024
    else:
        last_hist_year = 2024

    proj_years = [str(last_hist_year + i) for i in range(1, len(projected) + 1)]

    if not historical.empty:
        fig.add_trace(go.Bar(
            x            = list(historical.index),
            y            = [v / 1e9 for v in historical.values],
            name         = "Historical Revenue",
            marker_color = THEME["accent_blue"],
            opacity      = 0.55,
        ))

    fig.add_trace(go.Bar(
        x            = proj_years,
        y            = [v / 1e9 for v in projected],
        name         = "Projected Revenue",
        marker_color = THEME["accent_amber"],
    ))

    if not historical.empty:
        fig.add_vline(
            x          = last_hist_year + 0.5,
            line_dash  = "dot",
            line_color = THEME["text_color"],
            line_width = 1,
            opacity    = 0.4,
            annotation_text     = "Forecast →",
            annotation_position = "top right",
            annotation_font     = dict(color=THEME["text_color"], size=10),
        )

    fig.update_layout(
        **_layout_defaults,
        title       = f"{ticker} — Revenue ($B)",
        barmode     = "group",
        yaxis_title = "Revenue ($B)",
        bargap      = 0.15,
        bargroupgap = 0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Valuation Comparison Chart
# ---------------------------------------------------------------------------

def valuation_comparison_chart(
    current_price: float,
    dcf_value: float,
    multiples_value: float | None,
    blended_value: float,
    ticker: str,
) -> go.Figure:
    """
    Horizontal bar chart comparing:
      - Current market price (red dashed line)
      - DCF intrinsic value
      - Multiples implied value (if available)
      - Blended fair value

    A horizontal line marks the current price so the gap is obvious.
    """
    labels = ["DCF Value", "Multiples Value", "Blended Fair Value"]
    values = [dcf_value, multiples_value, blended_value]
    colors = [THEME["accent_blue"], THEME["accent_amber"], THEME["accent_green"]]

    # Filter out None entries
    valid = [(l, v, c) for l, v, c in zip(labels, values, colors)
             if v is not None and v > 0]

    if not valid:
        valid = [("DCF Value", dcf_value, THEME["accent_blue"])]

    v_labels, v_values, v_colors = zip(*valid)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x            = list(v_values),
        y            = list(v_labels),
        orientation  = "h",
        marker_color = list(v_colors),
        text         = [f"${v:,.2f}" for v in v_values],
        textposition = "outside",
        name         = "Intrinsic Value",
    ))

    # Current price vertical line
    fig.add_vline(
        x            = current_price,
        line_dash    = "dash",
        line_color   = THEME["accent_red"],
        line_width   = 2,
        annotation_text = f"Market Price ${current_price:,.2f}",
        annotation_font_color = THEME["accent_red"],
        annotation_position   = "top right",
    )

    fig.update_layout(
        **_layout_defaults,
        title    = f"{ticker} — Valuation Comparison",
        xaxis_title = "Price per Share ($)",
        showlegend  = False,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Sensitivity Heatmap
# ---------------------------------------------------------------------------

def sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    current_price: float,
    ticker: str,
) -> go.Figure:
    """
    Annotated heatmap showing intrinsic value per share across a grid of:
      - Rows    → WACC (discount rate)
      - Columns → FCF growth rate

    Cells are colour-coded green (above market price) / red (below).
    """
    z = sensitivity_df.values.astype(float)
    x = list(sensitivity_df.columns)   # growth rate labels  e.g. "2%", "5%"
    y = list(sensitivity_df.index)     # WACC labels          e.g. "6%", "8%"

    # Anchor colorscale so that current_price maps to yellow (midpoint).
    # We spread ±70% around the price so BUY zone = green, SELL zone = red.
    spread = current_price * 0.70
    zmin   = max(0.0, current_price - spread)
    zmax   = current_price + spread

    # Midpoint fraction where current_price sits in [zmin, zmax]
    mid = (current_price - zmin) / (zmax - zmin)
    mid = max(0.05, min(0.95, mid))   # clamp

    colorscale = [
        [0.0,        "#8B0000"],   # deep red  — heavily overvalued
        [mid * 0.6,  "#F85149"],   # red       — overvalued
        [mid * 0.9,  "#D29922"],   # amber     — fair / slight overvalue
        [mid,        "#D29922"],   # anchor    — exactly at market price
        [mid * 1.1 + (1 - mid) * 0.1, "#3FB950"],  # green — undervalued
        [1.0,        "#00C853"],   # bright green — deep value
    ]

    # Plain-text cell annotations (no HTML — Plotly ignores HTML in annotations)
    annotations = []
    for i, row_label in enumerate(y):
        for j, col_label in enumerate(x):
            val = z[i][j]
            if not np.isnan(val):
                signal = "▲" if val > current_price else "▼"
                # Use small font for cells; fit "$123" + signal
                annotations.append(dict(
                    x         = col_label,
                    y         = row_label,
                    text      = f"${val:,.0f} {signal}",
                    showarrow = False,
                    font      = dict(size=9, color="white"),
                    xref      = "x",
                    yref      = "y",
                ))

    fig = go.Figure(go.Heatmap(
        z          = z,
        x          = x,
        y          = y,
        colorscale = colorscale,
        zmin       = zmin,
        zmax       = zmax,
        showscale  = True,
        hovertemplate = (
            "WACC: %{y}<br>"
            "Growth: %{x}<br>"
            "Intrinsic Value: $%{z:,.2f}<extra></extra>"
        ),
        colorbar = dict(
            title      = "Intrinsic Value",
            tickprefix = "$",
            tickfont   = dict(color=THEME["text_color"], size=10),
            title_font = dict(color=THEME["text_color"], size=11),
            thickness  = 14,
        ),
    ))

    fig.update_layout(
        plot_bgcolor  = THEME["bg_color"],
        paper_bgcolor = THEME["paper_color"],
        font          = dict(color=THEME["text_color"], family=THEME["font_family"]),
        margin        = dict(l=60, r=20, t=60, b=60),
        legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        title = (
            f"<b>{ticker}</b> — Sensitivity Analysis  "
            f"<span style='font-size:12px;color:#8B949E'>"
            f"| Market Price: ${current_price:,.2f}</span>"
        ),
        xaxis = dict(
            title      = "FCF Growth Rate",
            type       = "category",    # force string labels — prevents % stripping
            gridcolor  = THEME["grid_color"],
            zeroline   = False,
            tickfont   = dict(color=THEME["text_color"]),
            title_font = dict(color=THEME["text_color"]),
        ),
        yaxis = dict(
            title      = "WACC (Discount Rate)",
            type       = "category",    # force string labels
            gridcolor  = THEME["grid_color"],
            zeroline   = False,
            tickfont   = dict(color=THEME["text_color"]),
            title_font = dict(color=THEME["text_color"]),
            autorange  = "reversed",
        ),
        annotations = annotations,
        height      = 440,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Gauge / KPI helper (returns a Plotly indicator figure)
# ---------------------------------------------------------------------------

def upside_gauge(upside_pct: float, ticker: str, decision: str) -> go.Figure:
    """
    A half-circle gauge showing the upside / downside percentage.
    Green = upside, red = downside.
    """
    color_map = {"BUY": THEME["accent_green"],
                 "HOLD": THEME["accent_amber"],
                 "SELL": THEME["accent_red"]}
    bar_color = color_map.get(decision, THEME["accent_blue"])

    pct_display = upside_pct * 100   # keep signed

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct_display,
        number = dict(
            suffix    = "%",
            font      = dict(size=42, color=bar_color, family=THEME["font_family"]),
            valueformat = ".1f",
        ),
        gauge  = dict(
            axis   = dict(
                range     = [-60, 60],
                tickvals  = [-60, -30, -15, 0, 15, 30, 60],
                ticktext  = ["-60%", "-30%", "-15%", "0", "+15%", "+30%", "+60%"],
                tickcolor = THEME["text_color"],
                tickfont  = dict(color=THEME["text_color"], size=9),
            ),
            bar    = dict(color=bar_color, thickness=0.25),
            bgcolor= THEME["bg_color"],
            borderwidth = 0,
            steps  = [
                dict(range=[-60, -15], color="#2a1010"),
                dict(range=[-15,  15], color="#1e1e0a"),
                dict(range=[ 15,  60], color="#0a1e0a"),
            ],
            threshold = dict(
                line      = dict(color="#8B949E", width=2),
                thickness = 0.75,
                value     = 0,
            ),
        ),
        title = dict(
            text = f"<b>{ticker}</b>  Upside / Downside",
            font = dict(size=12, color=THEME["text_color"]),
        ),
        domain = dict(x=[0, 1], y=[0.05, 1]),
    ))

    fig.update_layout(
        plot_bgcolor  = THEME["bg_color"],
        paper_bgcolor = THEME["paper_color"],
        font          = dict(color=THEME["text_color"], family=THEME["font_family"]),
        margin        = dict(l=10, r=10, t=30, b=5),
        height        = 220,
    )
    return fig
