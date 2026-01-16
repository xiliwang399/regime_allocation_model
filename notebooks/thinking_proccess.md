## Logic Flow: Mandate → Sub-Problems → Code Components

### Step 0: Translate mandate into explicit goals
**Mandate goals**
- Primary: capital preservation
- Secondary: positive risk-adjusted returns across cycles
- Constraints: rules-based, explainable, low overfitting risk, implementable, includes policy/geopolitical awareness

↓

### Step 1: Break the mandate into 3 independent “jobs”

#### Job A — Regime Recognition (detect environment)
**Question:** What type of market regime are we in?
**Output:** regime probabilities (smoothed, no hard switching)

Code: `src/regime_model.py`  
Notebook: `01_regime_model.ipynb`

↓

#### Job B — Structural Portfolio Construction (build base portfolios)
**Question:** Given the regime, what is a sensible FI allocation?
**Output:** base portfolio weights per regime

Code: `src/allocation_model.py`  
Notebook: `02_allocation_engine.ipynb`

↓

#### Job C — Capital Preservation + Policy/Geo Overlay (controlled adjustments)
**Question:** Do we need additional defensiveness / shock awareness beyond regimes?
**Output:** final implementable weights

Code: `src/allocation_model.py`  
Notebook: `03_backtest_overlay.ipynb`

↓

### Step 2: Evaluate the system against the RFP concerns
**Backtest / stress test checks**

Code: `src/backtester.py`, `src/metrics.py`  
Notebook: `03_backtest_overlay.ipynb`
