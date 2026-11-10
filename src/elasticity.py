"""Price sensitivity from observational data, and why it can't be trusted on its own.

Question: do customers whose price rose churn more? The naive way is to regress churn on the
2015 price change. Prices are not set at random, though: the company re-prices customers it is
worried about (retention tariffs, renegotiations), so price changes and churn risk move together
for reasons other than price. This is called *confounding*.

We fit the regression with and without controls and show the coefficient moving around. That
instability is the evidence that observational data can't answer the pricing question, which is
why the decision step relies on the randomised pilot instead.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Price change in percent of the January price, capped to avoid zero-price artefacts.
    base = d["price_energy_jan"].where(d["price_energy_jan"] > 0)
    d["price_change_pct"] = (100 * d["price_energy_change"] / base).clip(-50, 50)
    d["price_gap_pct"] = (100 * d["price_gap_vs_channel"]).clip(-50, 50)
    d["log_margin"] = np.log1p(d["net_margin"].clip(lower=0))
    # 16 customers in tiny channels never churned, which breaks the logit (perfect separation).
    d["channel"] = d["channel"].astype(str).replace({"Channel other": "Channel unknown"})
    return d.dropna(subset=["price_change_pct"])


FORMULAS = {
    "naive": "churn ~ price_change_pct",
    "+ price level vs peers": "churn ~ price_change_pct + price_gap_pct",
    "+ customer controls": ("churn ~ price_change_pct + price_gap_pct + tenure_years + multi_product"
                            " + log_cons_12m + log_margin + months_to_end + C(channel)"),
}


def fit_models(df: pd.DataFrame) -> dict:
    d = _prep(df)
    return {name: smf.logit(f, data=d).fit(disp=0, cov_type="HC1") for name, f in FORMULAS.items()}


def price_effect_table(models: dict) -> pd.DataFrame:
    """The price-change coefficient under each specification, as an odds ratio per +1% price
    change, and as the average change in churn probability (percentage points) per +1%."""
    rows = []
    for name, res in models.items():
        b = res.params["price_change_pct"]
        lo, hi = res.conf_int().loc["price_change_pct"]
        me = res.get_margeff(at="overall").summary_frame().loc["price_change_pct"]
        rows.append({"model": name, "odds_ratio_per_1pct": np.exp(b),
                     "or_ci95": f"{np.exp(lo):.3f} - {np.exp(hi):.3f}",
                     "pp_per_1pct": 100 * me["dy/dx"], "p_value": res.pvalues["price_change_pct"],
                     "n": int(res.nobs)})
    return pd.DataFrame(rows).set_index("model").round(4)


def churn_by_price_change(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """The raw pattern behind the naive model: churn rate by price-change quintile."""
    d = _prep(df)
    d["price_change_band"] = pd.qcut(d["price_change_pct"].rank(method="first"), bins,
                                     labels=[f"Q{i}" for i in range(1, bins + 1)])
    return d.groupby("price_change_band", observed=True).agg(
        median_change_pct=("price_change_pct", "median"), churn_rate=("churn", "mean"),
        customers=("churn", "size"))
