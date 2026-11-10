"""Margin at risk: the churn probability turned into euros.

expected margin lost = P(churn in next 3 months) x 12-month net margin

This is why calibration matters: if the probabilities were twice too high, this number would be
twice too high, and so would the budget someone signs off on.
"""
import numpy as np
import pandas as pd


def margin_at_risk(df: pd.DataFrame, p_col: str = "p_churn") -> pd.Series:
    return df[p_col] * df["net_margin"]


def portfolio_summary(df: pd.DataFrame, p_col: str = "p_churn") -> pd.Series:
    mar = margin_at_risk(df, p_col)
    return pd.Series({
        "customers": len(df),
        "annual_net_margin_eur": df["net_margin"].sum(),
        "expected_churners": df[p_col].sum(),
        "margin_at_risk_eur": mar.sum(),
        "margin_at_risk_share": mar.sum() / df["net_margin"].sum(),
    })


def by_segment(df: pd.DataFrame, by: str, p_col: str = "p_churn") -> pd.DataFrame:
    d = df.assign(mar=margin_at_risk(df, p_col))
    t = d.groupby(by, observed=True).agg(customers=("mar", "size"), churn_prob=(p_col, "mean"),
                                         net_margin_eur=("net_margin", "sum"),
                                         margin_at_risk_eur=("mar", "sum"))
    t["share_of_risk"] = t["margin_at_risk_eur"] / t["margin_at_risk_eur"].sum()
    return t.sort_values("margin_at_risk_eur", ascending=False)


def concentration(df: pd.DataFrame, p_col: str = "p_churn") -> pd.DataFrame:
    """Sorted from most to least at risk: what share of customers holds what share of the
    euros at risk? A steep curve means a small, targeted list covers most of the exposure."""
    mar = np.sort(margin_at_risk(df, p_col).to_numpy())[::-1]
    cum = np.cumsum(mar) / mar.sum()
    share = np.arange(1, len(mar) + 1) / len(mar)
    return pd.DataFrame({"share_of_customers": share, "share_of_margin_at_risk": cum})
