"""Churn propensity: who is likely to leave in the next 3 months, with probabilities we can
multiply by euros.

Three ideas carry this module:
1. Out-of-fold (OOF) prediction: every customer is scored by a model that never saw them,
   so the scores are an honest picture of how the model does on new customers.
2. Calibration: a gradient-boosted model ranks well but its raw scores are not reliable
   probabilities. Downstream we compute EUR = probability x margin, so we calibrate.
3. SHAP: per-customer explanations of *why* the score is high, for account managers.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED
from .features import CATEGORICAL, MODEL_FEATURES

GBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=SEED,
)


def make_gbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**GBM_PARAMS)


def make_logit():
    """The simple benchmark: if this were as good, we would ship it instead."""
    numeric = [c for c in MODEL_FEATURES if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(self.medians_, idx[1])
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def out_of_fold(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """OOF churn probabilities for three models: logistic benchmark, raw GBM, calibrated GBM."""
    X, y = df[MODEL_FEATURES], df["churn"].to_numpy()
    out = pd.DataFrame(index=df.index, columns=["p_logit", "p_gbm_raw", "p_gbm_cal"], dtype=float)
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for train, test in folds.split(X, y):
        Xtr, Xte, ytr = X.iloc[train], X.iloc[test], y[train]
        out.iloc[test, 0] = make_logit().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        out.iloc[test, 1] = make_gbm().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        # Isotonic calibration, fitted on inner folds of the training part only.
        cal = CalibratedClassifierCV(make_gbm(), method="isotonic", cv=3)
        out.iloc[test, 2] = cal.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return out


def scorecard(y: np.ndarray, oof: pd.DataFrame) -> pd.DataFrame:
    """Ranking quality (AUC, average precision) and probability quality (Brier, log loss)."""
    rows = {}
    for col in oof.columns:
        p = oof[col].clip(1e-6, 1 - 1e-6)
        rows[col] = {
            "AUC": roc_auc_score(y, p),
            "avg_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p),
            "mean_predicted": p.mean(),
            "actual_rate": y.mean(),
        }
    return pd.DataFrame(rows).T.round(4)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Customers grouped into deciles of predicted risk: predicted vs actual churn per group."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    t = d.groupby("decile", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"),
                                               customers=("y", "size"), churners=("y", "sum"))
    return t


def lift_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """If we can only contact the riskiest X%, what share of all churners do we reach?"""
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    n = len(y_sorted)
    rows = []
    for k in range(1, bins + 1):
        top = y_sorted[: int(n * k / bins)]
        rows.append({"top_share": k / bins, "churn_rate": top.mean(),
                     "share_of_churners_reached": top.sum() / y_sorted.sum(),
                     "lift": top.mean() / y_sorted.mean()})
    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """GBM on the whole book, used only for explanations (scores come from OOF)."""
    return make_gbm().fit(df[MODEL_FEATURES], df["churn"])


def shap_values(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions in log-odds units: positive pushes towards churn."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    return pd.DataFrame(values, index=X.index, columns=X.columns)


# Margin says how much a customer is worth, not why they might leave, so it is not shown to
# account managers as a "reason".
NOT_A_REASON = {"net_margin", "margin_net_pow_ele"}


def top_drivers(shap_df: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> pd.Series:
    """The k features pushing each customer's risk up the most, as (feature, value, shap)."""
    def row_drivers(i):
        s = shap_df.loc[i].drop(list(NOT_A_REASON & set(shap_df.columns)))
        top = s[s > 0].sort_values(ascending=False).head(k)
        return [(f, X.at[i, f], float(v)) for f, v in top.items()]
    return pd.Series({i: row_drivers(i) for i in shap_df.index})
