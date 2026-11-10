# Which SME customers should an energy supplier fight to keep, and with what?

**Targeting the customers an offer can *change* rather than the ones most likely to leave turns
the same €10k retention budget from €3.2k into €11.1k of net value — 3.5× more.**

> **New to data science? Start with the teaching edition:** [`notebooks/00_sme_energy_retention_explained.ipynb`](notebooks/00_sme_energy_retention_explained.ipynb) (or the `.html` copy to read in a browser). Same code and numbers as the short presentation notebook, with every idea explained from zero, toy examples, and how to read every output.

A retention and pricing case for a gas & power supplier: **churn, price, risk and
uplift turned into one € decision**.

## The finding
Spending a €10k retention budget on the customers **most likely to leave** returns €0.7k (discount)
or €3.2k (call) net. Spending it on the customers whose decision an offer can **change**, with an
optimiser choosing the offer, returns **€11.1k, 3.5× more** and 83% of the planted-truth ceiling.
Across six random pilot draws: 1.8–5.7× the best rule-of-thumb list.

- **Risk ≠ persuadable.** The riskiest customers are often lost causes.
- **Averages hide effects.** The account-manager call cuts churn ~9 points for multi-product customers
  and *raises* it ~6 points for loyal customers never contacted before.
- **The model over-promises at the margin.** Estimated value keeps rising with budget; true value
  peaks around €10k (the winner's curse).
- **Observational price data is confounded.** Customers given the biggest price cuts churned most,
  because the company cut prices for customers it was already losing, so the price question went
  to a randomised pilot.
- **GenAI with guard-rails.** A local LLM writes account-manager briefs; grounding makes 95% usable
  (vs 18%), and a deterministic fact-checker catches the invented numbers that remain (1 in 20).

## Data
| | Source | Status |
|---|---|---|
| 14,606 SME customers, 12 months of tariffs, churn | BCG *PowerCo* ([Kaggle, CC0](https://www.kaggle.com/datasets/erolmasimov/powerco)) | **real** |
| Retention pilot (no offer / 5% discount / account-manager call) | `src/simulate.py` | **simulated** on the real customers, with planted effects |

## How it works
| Step | Module | Method |
|---|---|---|
| Features + leakage audit | `features.py` | snapshot-safe features; 61 future-dated rows masked |
| Churn propensity | `propensity.py` | LightGBM vs logistic, out-of-fold, isotonic calibration, SHAP |
| Margin at risk | `risk.py` | P(churn) × margin, concentration |
| Price sensitivity | `elasticity.py` | logit with controls → shows confounding |
| Pilot | `simulate.py` | randomised arms, potential outcomes, planted heterogeneity |
| Uplift | `uplift.py` | arm ATEs with CIs, segment effects, T-/X-learner, Qini |
| Decision | `optimise.py` | multiple-choice knapsack (`scipy.optimize.milp`), budget curve |
| GenAI | `genai_briefs.py`, `factcheck.py` | Qwen2.5-3B-Instruct (local) + regex fact-checker |

## Repository
```
notebooks/00_sme_energy_retention_final.ipynb       the case, end to end (15-min presentation)
notebooks/00_sme_energy_retention_explained.ipynb   teaching edition: everything explained from zero
tools/                                              rebuild either notebook (build_final.py, build_explained.py)
src/                                                all logic; the notebook only calls it
tests/                                              21 tests (pytest)
data/raw/                                           downloaded on first run
data/llm/briefs.jsonl                               cached LLM output (input dataset)
run_analysis.py                                     rebuilds headline numbers without the notebook
```

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/jupyter nbconvert --execute --to notebook --inplace notebooks/00_sme_energy_retention_final.ipynb
.venv/Scripts/python run_analysis.py --robustness
.venv/Scripts/python -m src.genai_briefs        # optional: regenerate LLM briefs (GPU, torch + transformers 4.46)
```

## Limitations
The pilot is simulated: the € results show the method recovering a known truth, not a forecast of
any real company's numbers. Value of a saved customer = 12 months of margin. One snapshot, so no
time-based validation. Uplift estimates are noisy at ~2,400 customers per arm. See §9 of the notebook.
