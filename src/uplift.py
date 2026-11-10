"""Uplift: for each customer, how much does each offer *reduce* their churn probability?

A churn model answers "who will leave?". Uplift answers "whose decision can we change?".
Those are different people: a customer who is certain to leave whatever we do ("lost cause")
scores high on churn but zero on uplift.

The four customer types, for any one offer:
  persuadable  - leaves without the offer, stays with it        -> target
  sure thing   - stays either way                               -> offer is wasted money
  lost cause   - leaves either way                              -> offer is wasted money
  sleeping dog - stays without the offer, leaves *because of* it -> never target

Estimators (all fitted on the randomised pilot only):
- arm-level average effect with a confidence interval (the "did it work at all?" check)
- T-learner: one churn model per arm, effect = difference of predictions
- X-learner: a refinement that is more stable when the outcome is rare (as here, ~10% churn)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import SEED
from .features import MODEL_FEATURES

# Small, heavily regularised trees: each pilot arm has ~2,400 customers and ~230 churners.
STAGE_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=80,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                    verbose=-1, random_state=SEED)

UPLIFT_FEATURES = MODEL_FEATURES + ["p_churn"]


def ate_table(po: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per pilot arm and each offer's average effect vs no offer, with a 95% CI.
    Effect = churn(no offer) - churn(offer): positive means the offer reduced churn."""
    pilot = po[po["in_pilot"]]
    rates = pilot.groupby("arm")["y_observed"].agg(["mean", "size"])
    p0, n0 = rates.loc["none"]
    rows = []
    for arm in ["discount", "call"]:
        p1, n1 = rates.loc[arm]
        effect = p0 - p1
        se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
        rows.append({"offer": arm, "customers": int(n1), "churn_with_offer": p1,
                     "churn_without": p0, "effect_pp": 100 * effect,
                     "ci95_low_pp": 100 * (effect - 1.96 * se),
                     "ci95_high_pp": 100 * (effect + 1.96 * se),
                     "z": effect / se})
    return pd.DataFrame(rows).set_index("offer").round(3)


def segment_table(df: pd.DataFrame, po: pd.DataFrame, by: str) -> pd.DataFrame:
    """Offer effect within business segments, straight from the randomised pilot.
    Transparent, model-free, and noisy in small segments, hence the confidence intervals."""
    pilot = po[po["in_pilot"]].join(df[[by]])
    rows = []
    for seg, g in pilot.groupby(by, observed=True):
        r = g.groupby("arm")["y_observed"].agg(["mean", "size"])
        if "none" not in r.index:
            continue
        p0, n0 = r.loc["none"]
        for arm in ["discount", "call"]:
            if arm not in r.index:
                continue
            p1, n1 = r.loc[arm]
            se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
            rows.append({by: seg, "offer": arm, "n_offer": int(n1), "n_control": int(n0),
                         "effect_pp": 100 * (p0 - p1), "ci95_pp": 100 * 1.96 * se})
    return pd.DataFrame(rows).round(2)


def _clf():
    return lgb.LGBMClassifier(**STAGE_PARAMS)


def _reg():
    return lgb.LGBMRegressor(**STAGE_PARAMS)


def t_learner(df: pd.DataFrame, po: pd.DataFrame, offer: str) -> np.ndarray:
    """Fit P(churn | x) separately on the no-offer arm and the offer arm; the difference is the
    estimated effect. Returns effects for every customer (churn reduction, positive = good)."""
    pilot = po["in_pilot"]
    X = df[UPLIFT_FEATURES]
    ctrl = pilot & (po["arm"] == "none")
    trt = pilot & (po["arm"] == offer)
    m0 = _clf().fit(X[ctrl], po.loc[ctrl, "y_observed"].astype(int))
    m1 = _clf().fit(X[trt], po.loc[trt, "y_observed"].astype(int))
    return m0.predict_proba(X)[:, 1] - m1.predict_proba(X)[:, 1]


def x_learner(df: pd.DataFrame, po: pd.DataFrame, offer: str) -> np.ndarray:
    """X-learner (Kunzel et al., 2019), in three steps:
    1. as the T-learner: outcome models mu0 (no offer) and mu1 (offer);
    2. impute each pilot customer's individual effect using the *other* arm's model:
         treated:  D = mu0(x) - y        (what we predicted without the offer minus what happened)
         control:  D = y - mu1(x)
       then fit a regression of D on x within each arm (tau1, tau0);
    3. blend the two by the share of customers in each arm (here roughly 50/50).
    Positive = the offer reduces churn."""
    pilot = po["in_pilot"]
    X = df[UPLIFT_FEATURES]
    ctrl = pilot & (po["arm"] == "none")
    trt = pilot & (po["arm"] == offer)
    y_c = po.loc[ctrl, "y_observed"].astype(int)
    y_t = po.loc[trt, "y_observed"].astype(int)
    mu0 = _clf().fit(X[ctrl], y_c)
    mu1 = _clf().fit(X[trt], y_t)
    d_t = mu0.predict_proba(X[trt])[:, 1] - y_t.to_numpy()
    d_c = y_c.to_numpy() - mu1.predict_proba(X[ctrl])[:, 1]
    tau1 = _reg().fit(X[trt], d_t)
    tau0 = _reg().fit(X[ctrl], d_c)
    g = trt.sum() / (trt.sum() + ctrl.sum())
    return g * tau0.predict(X) + (1 - g) * tau1.predict(X)


def qini_curve(tau_hat: np.ndarray, realised_effect: np.ndarray, points: int = 50) -> pd.DataFrame:
    """Rank customers by estimated effect; cumulative customers actually saved as we go down the
    list. `realised_effect` = y_none - y_offer per customer, which only a simulation can know.
    A random ordering gives the straight diagonal."""
    order = np.argsort(-tau_hat)
    gain = np.cumsum(np.asarray(realised_effect)[order])
    n = len(order)
    idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    return pd.DataFrame({"share_targeted": (idx + 1) / n, "customers_saved": gain[idx],
                         "random": gain[-1] * (idx + 1) / n})


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's curve and the random diagonal, normalised by the number saved
    if everyone were targeted. 0 = no better than random."""
    total = curve["random"].iloc[-1]
    gap = curve["customers_saved"] - curve["random"]
    return float(np.trapz(gap, curve["share_targeted"]) / abs(total)) if total else 0.0
