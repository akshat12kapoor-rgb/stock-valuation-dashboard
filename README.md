# 📈 Should I Invest? — Stock Valuation Dashboard

An interactive Streamlit dashboard that evaluates whether a stock is undervalued,
fairly valued, or overvalued using **Discounted Cash Flow (DCF)** analysis and
**Comparable Company Multiples** (P/E, EV/EBITDA).

---

## Features

| Feature | Details |
|---|---|
| **Live data** | Fetches real financials via Yahoo Finance (yfinance) |
| **DCF Model** | 5-year FCF forecast + Gordon Growth terminal value |
| **Comps Model** | P/E and EV/EBITDA multiples, sector-calibrated |
| **Decision Engine** | BUY / HOLD / SELL with margin-of-safety logic |
| **Sensitivity Analysis** | Heatmap of intrinsic value across WACC × growth grid |
| **Interactive sliders** | Real-time recalculation of growth, WACC, terminal growth |
| **Dark theme** | Finance-grade dark UI with Plotly charts |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/akshat12kapoor-rgb/stock-valuation-dashboard.git
cd stock-valuation-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Project Structure

```
stock-valuation-dashboard/
├── app.py                    # Streamlit UI — main entry point
├── modules/
│   ├── __init__.py
│   ├── data_fetcher.py       # yfinance data collection + normalisation
│   ├── dcf_model.py          # DCF engine + sensitivity matrix
│   ├── multiples_model.py    # P/E & EV/EBITDA comparable valuation
│   ├── decision_engine.py    # BUY / HOLD / SELL blending logic
│   └── charts.py             # Plotly chart builders
├── requirements.txt
└── README.md
```

---

## Valuation Methodology

### 1. Discounted Cash Flow (DCF)

```
Projected FCF(t) = Base FCF × (1 + growth_rate)^t
PV(t)            = FCF(t) / (1 + WACC)^t
Terminal Value   = FCF(year5) × (1 + g) / (WACC - g)
Intrinsic Value  = (Σ PV + PV of TV) / shares_outstanding
```

### 2. Comparable Multiples

```
P/E implied price      = EPS × Industry P/E
EV/EBITDA implied price = (EBITDA × EV/EBITDA − Debt + Cash) / shares
```

### 3. Blending

```
Fair Value = DCF × 60% + Multiples × 40%   (adjustable via slider)
```

### 4. Decision Logic

| Signal | Condition |
|--------|-----------|
| 🟢 BUY  | Fair Value > Market Price by > 15% |
| 🟡 HOLD | Within ±15% band |
| 🔴 SELL | Market Price > Fair Value by > 15% |

---

## Key Concepts (Interview-Ready)

**Why DCF?**
DCF values a company on its intrinsic ability to generate cash — independent of
market sentiment. The discount rate (WACC) reflects the time value of money and
business risk.

**Why Multiples?**
Multiples are market-anchored. They answer: *"what are investors paying for similar
companies right now?"* They complement DCF by grounding it in real market pricing.

**Why Blend?**
Neither method is perfect. DCF is sensitive to assumptions; multiples are sensitive
to market cycles. Blending reduces model risk.

**What is Terminal Value?**
Most of a company's value comes from cash flows beyond the 5-year forecast.
The terminal value captures this using the Gordon Growth Model — assuming the
company grows at a stable long-run rate forever.

---

## Disclaimer

This tool is for **educational and informational purposes only**.
It does not constitute financial advice. Always conduct your own due diligence.
