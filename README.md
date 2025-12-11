# Regime-Based Fixed Income Allocation Model

*A systematic multi-regime framework for capital-preserving portfolio construction*

This repository contains a fully modular quantitative framework for **regime-aware fixed income allocation**.
The project was developed to explore how a rules-based investment process can adapt to today’s environment of:

* elevated macro uncertainty
* volatile inflation dynamics
* changing monetary policy regimes
* geopolitical and policy-driven shocks
* correlation instability across fixed-income assets

The model’s central objective is **capital preservation**, with a secondary aim of achieving **stable, risk-adjusted returns** across multiple market environments.

---

## 1. Concept Overview

The framework integrates three layers:

### A. **Regime Identification (HMM Layer)**

A Gaussian Hidden Markov Model is fitted on a set of macro-financial indicators, including volatility indices, credit spreads, and yield curve measures.
Key features:

* Selecting the number of regimes via **BIC**
* Computing posterior probabilities for each state
* Applying exponential smoothing to reduce noise
* Automatically labeling regimes by severity using weighted macro stress scores

Typical regime labels include:

* Carry-Friendly
* Volatility Transition
* Defensive
* Capital-Preservation Mode

These classifications evolve over time and form the foundation of the allocation engine.

---

### B. **Macro–Geopolitical Profiling (Risk Signal Layer)**

To capture the influence of broader economic and political conditions, the model incorporates:

* Inflation expectations (5-year breakeven)
* Growth momentum (ISM)
* Policy uncertainty (EPU)
* Geopolitical risk (GPR)

The indicators are transformed into **z-scored risk signals**, which drive tilts such as:

* Increasing inflation hedges during elevated inflation risk
* Reducing credit exposure during growth slowdowns
* Raising safe-asset shares when policy/geopolitical fragility increases

This layer introduces responsiveness to structural risk conditions that are not captured by returns alone.

---

### C. **Allocation Layer (Structural + Tilted Weights)**

For each regime, the model computes a long-only structural portfolio using:

* Regime-conditional means and Ledoit–Wolf shrinkage covariances
* Regime-specific risk aversion levels
* IPS-style allocation bounds
* Posterior-weighted blend of regime portfolios

A second step applies macro tilts and incorporates:

* Dynamic safe-asset targets
* Inflation and growth adjustments
* Policy/geopolitical hedging
* Tail-hedge minimum floors

The final output is a **capital-preserving, volatility-aware allocation** that responds to shifts in both macro data and regime structure.

---

## 2. Backtesting & Evaluation

The repository includes a full backtesting toolkit with:

* Monthly rebalancing
* Transaction cost modeling
* Comparison with a static barbell benchmark
* Crisis-period analytics (e.g., GFC, 2020 shock)
* Performance statistics: return, volatility, Sharpe ratio, drawdowns
* Turnover and structural allocation diagnostics

---

## 3. Repository Structure

```
regime-allocation-model/
    src/
        data_loader.py
        regime_model.py
        factor_engine.py
        allocation_model.py
        backtester.py
        metrics.py
        __init__.py
    notebooks/
        01_regime_estimation.ipynb
        02_macro_profile.ipynb
        03_allocation_backtest.ipynb
    config.py
    inputs.xlsx
    .gitignore
    README.md
```

Each module is written for clarity and extensibility, allowing users to modify the factor set, asset universe, or regime logic.

---

## 4. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks in sequence:

1. `01_regime_estimation.ipynb`
2. `02_macro_profile.ipynb`
3. `03_allocation_backtest.ipynb`

---

## 5. Possible Extensions

* Walk-forward training and out-of-sample testing
* Alternative volatility models (HAR, GARCH)
* Text-based policy signals
* Expanded fixed-income universe
* Scenario-based stress testing

---

## 6. Author Notes

This project was developed as part of a broader research initiative on systematic fixed-income allocation.
All model design and implementation in this repository were completed individually.


