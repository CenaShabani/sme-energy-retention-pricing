"""SIMULATED retention pilot, layered on the real customers.

Why simulate: the real data can tell us *who* is likely to churn, but not *whether an offer
would change their mind*. The data shows customers whose price went down churning more often,
which is a confounded pattern no regression can untangle. In practice you settle it by running a
randomised pilot. We have no access to the supplier's pilot data, so we simulate one with effects we
plant ourselves. Because we know the truth, every estimate in the uplift and decision steps can
be checked against it.

How the effects are planted (in plain words):
- Every customer keeps their REAL churn outcome as the "no offer" outcome, Y(none).
- Discount: a real churner is persuaded to stay with probability r_discount(x). Price-sensitive
  customers (paying more than channel peers, short tenure) are much easier to persuade.
- Call: a real churner is persuaded with probability r_call(x). Relationship customers
  (several products, high margin) respond to a call. But a call also *wakes up* some loyal,
  long-tenured, never-touched customers, who then shop around and leave: the "sleeping dogs".
So for each customer we know all three potential outcomes Y(none), Y(discount), Y(call). In the
pilot half each customer is randomly shown only one of them, as in a real experiment.
"""
import numpy as np
import pandas as pd

from .config import PILOT_SHARE, SEED

# Every planted number lives here. Tests and the notebook read the truth from this dict.
PLANTED = {
    "discount_base": 0.10,           # persuasion rate for a churner with no price grievance
    "discount_price_gap": 0.50,      # extra persuasion at the top of the price-gap distribution
    "discount_short_tenure": 0.20,   # extra persuasion if tenure <= 4 years
    "call_base": 0.15,
    "call_relationship": 0.40,       # extra persuasion if multi-product (gas or >1 product)
    "call_high_margin": 0.20,        # extra persuasion if in the top margin quartile
    "call_sleeping_dog_rate": 0.08,  # share of loyal non-churners a call pushes to leave
    "cap": 0.75,                     # nobody is persuaded with certainty
}


def sleeping_dog_segment(df: pd.DataFrame) -> pd.Series:
    """Loyal, long-tenured, single-product customers whose contract was never modified."""
    return (df["tenure_years"] >= 6) & (df["multi_product"] == 0) & (df["product_modified"] == 0)


def persuasion_rates(df: pd.DataFrame) -> pd.DataFrame:
    """The planted response of each customer to each offer (the ground truth)."""
    k = PLANTED
    gap_rank = df["price_gap_vs_channel"].rank(pct=True)
    r_discount = (k["discount_base"] + k["discount_price_gap"] * gap_rank ** 2
                  + k["discount_short_tenure"] * (df["tenure_years"] <= 4))
    high_margin = df["net_margin"] >= df["net_margin"].quantile(0.75)
    r_call = (k["call_base"] + k["call_relationship"] * df["multi_product"]
              + k["call_high_margin"] * high_margin)
    s_call = k["call_sleeping_dog_rate"] * sleeping_dog_segment(df)
    return pd.DataFrame({
        "r_discount": r_discount.clip(0, k["cap"]),
        "r_call": r_call.clip(0, k["cap"]),
        "s_call": s_call.astype(float),
    }, index=df.index)


def potential_outcomes(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Y(none), Y(discount), Y(call) for every customer. 1 = churned."""
    rng = np.random.default_rng(seed)
    rates = persuasion_rates(df)
    y0 = df["churn"].to_numpy()
    u_disc, u_call, v_call = rng.random((3, len(df)))
    y_disc = np.where(y0 == 1, (u_disc >= rates["r_discount"]).astype(int), 0)
    y_call = np.where(y0 == 1, (u_call >= rates["r_call"]).astype(int),
                      (v_call < rates["s_call"]).astype(int))
    return pd.concat([rates, pd.DataFrame(
        {"y_none": y0, "y_discount": y_disc, "y_call": y_call}, index=df.index)], axis=1)


def run_pilot(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split the book into a pilot half and a roll-out half; randomise offers in the pilot.

    Returns the potential outcomes plus: `in_pilot`, the randomly assigned `arm` (pilot only),
    and `y_observed`, the only outcome a real company would ever see.
    """
    rng = np.random.default_rng(seed + 1)
    po = potential_outcomes(df, seed)
    in_pilot = rng.random(len(df)) < PILOT_SHARE
    arm = np.where(in_pilot, rng.choice(["none", "discount", "call"], size=len(df)), None)
    observed = np.select(
        [arm == "none", arm == "discount", arm == "call"],
        [po["y_none"], po["y_discount"], po["y_call"]], default=-1)
    po["in_pilot"] = in_pilot
    po["arm"] = arm
    po["y_observed"] = np.where(in_pilot, observed, np.nan)
    return po


def true_effects(df: pd.DataFrame, po: pd.DataFrame) -> pd.DataFrame:
    """The planted *expected* churn reduction per customer, which no model ever sees.

    Uses the calibrated churn probability as the customer's baseline risk:
    tau = p * r - (1 - p) * s  (persuaded churners minus woken sleeping dogs)."""
    p = df["p_churn"]
    return pd.DataFrame({
        "tau_discount": p * po["r_discount"],
        "tau_call": p * po["r_call"] - (1 - p) * po["s_call"],
    }, index=df.index)
