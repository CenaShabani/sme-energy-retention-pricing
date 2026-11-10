"""The decision: given a campaign budget, which customers get which offer?

For every customer i and offer a we know (estimated):
  tau      = drop in churn probability caused by the offer      (from the uplift model)
  m        = 12-month net margin of the customer
  spend    = what the offer costs us in expectation
             discount: 25% of margin, paid only if the customer stays -> 0.25 * m * P(stay | offer)
             call:     EUR 30, paid whatever happens
  value    = margin saved - spend = tau * m - spend

Then: choose at most one offer per customer to maximise total value, keeping total spend within
budget. That is a "multiple-choice knapsack", solved exactly as a small integer program.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

from .config import CALL_COST_EUR, DISCOUNT_MARGIN_SHARE, KEY_ACCOUNT_QUANTILE


def campaign_eligible(df: pd.DataFrame) -> pd.Series:
    """Everyone except key accounts, which get a named account manager instead of a campaign."""
    return df["net_margin"] < df["net_margin"].quantile(KEY_ACCOUNT_QUANTILE)


def offer_economics(df: pd.DataFrame, tau_discount, tau_call, p_col: str = "p_churn") -> pd.DataFrame:
    """Expected spend and value of each offer for each customer."""
    m = df["net_margin"].to_numpy()
    p0 = df[p_col].to_numpy()
    td, tc = np.asarray(tau_discount), np.asarray(tau_call)
    stay_with_discount = np.clip(1 - p0 + td, 0, 1)
    spend_d = DISCOUNT_MARGIN_SHARE * m * stay_with_discount
    spend_c = np.full_like(m, CALL_COST_EUR, dtype=float)
    return pd.DataFrame({
        "spend_discount": spend_d, "value_discount": td * m - spend_d,
        "spend_call": spend_c, "value_call": tc * m - spend_c,
    }, index=df.index)


def solve(econ: pd.DataFrame, budget: float) -> pd.Series:
    """Exact integer program. Returns the chosen offer per customer ('none', 'discount', 'call').
    Offers with negative expected value are never chosen, because they lose money even before
    the budget runs out."""
    n = len(econ)
    value = np.concatenate([econ["value_discount"], econ["value_call"]])
    spend = np.concatenate([econ["spend_discount"], econ["spend_call"]])
    upper = (value > 0).astype(float)  # forbid loss-making offers
    one_offer_each = hstack([identity(n), identity(n)]).tocsr()
    constraints = [
        LinearConstraint(one_offer_each, 0, 1),
        LinearConstraint(csr_matrix(spend.reshape(1, -1)), 0, budget),
    ]
    res = milp(c=-value, constraints=constraints, integrality=np.ones(2 * n),
               bounds=Bounds(0, upper), options={"time_limit": 60})
    x = np.round(res.x).astype(int)
    choice = np.where(x[:n] == 1, "discount", np.where(x[n:] == 1, "call", "none"))
    return pd.Series(choice, index=econ.index, name="offer")


def rank_and_fill(score: pd.Series, spend: pd.Series, budget: float, offer: str) -> pd.Series:
    """The common practice: sort customers by a score and give everyone the same offer until the
    budget runs out."""
    out = pd.Series("none", index=score.index, name="offer")
    remaining = budget
    for i in score.sort_values(ascending=False).index:
        if spend.at[i] <= remaining:  # skip anyone who no longer fits, keep filling
            out.at[i] = offer
            remaining -= spend.at[i]
    return out


def evaluate(policy: pd.Series, df: pd.DataFrame, po: pd.DataFrame) -> dict:
    """What actually happens under a policy, which is the one thing only a simulation can tell us.

    Uses each customer's REAL churn outcome plus the PLANTED response rates, averaged over the
    simulation's coin flips (so one lucky draw on a big customer cannot swing the answer):
      a real churner who gets offer a stays with probability r_a       -> saved
      a real stayer who gets a call leaves with probability s_call     -> sleeping dog woken
    """
    m = df["net_margin"]
    y0 = po["y_none"]
    is_d, is_c = policy == "discount", policy == "call"
    persuaded = y0 * (is_d * po["r_discount"] + is_c * po["r_call"])
    woken = (1 - y0) * is_c * po["s_call"]
    saved = persuaded - woken
    stays = 1 - y0 + saved
    spend = is_d * DISCOUNT_MARGIN_SHARE * m * stays + is_c * CALL_COST_EUR
    margin_saved = (saved * m).sum()
    return {
        "discounts": int(is_d.sum()),
        "calls": int(is_c.sum()),
        "customers_saved": float(saved.sum()),
        "sleeping_dogs_woken": float(woken.sum()),
        "spend_eur": float(spend.sum()),
        "margin_saved_eur": float(margin_saved),
        "net_value_eur": float(margin_saved - spend.sum()),
        "roi": float((margin_saved - spend.sum()) / spend.sum()) if spend.sum() else 0.0,
    }


def budget_curve(econ: pd.DataFrame, df: pd.DataFrame, po: pd.DataFrame, budgets) -> pd.DataFrame:
    """Solve at each budget. 'estimated' is what the model promises; 'true' is what the planted
    truth delivers. The gap between them is the over-optimism of acting on noisy estimates."""
    rows = []
    for b in budgets:
        policy = solve(econ, b)
        chosen_d, chosen_c = policy == "discount", policy == "call"
        est = econ.loc[chosen_d, "value_discount"].sum() + econ.loc[chosen_c, "value_call"].sum()
        ev = evaluate(policy, df, po)
        rows.append({"budget_eur": b, "estimated_net_value_eur": est,
                     "true_net_value_eur": ev["net_value_eur"], "spend_eur": ev["spend_eur"],
                     "discounts": ev["discounts"], "calls": ev["calls"]})
    return pd.DataFrame(rows)
