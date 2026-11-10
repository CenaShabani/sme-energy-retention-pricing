"""The whole analysis as one function, used by run_analysis.py and the brief generator.

The capstone notebook calls the same steps one by one, so the reader sees each result as it is
produced. This function exists so a script can rebuild everything without a notebook.
"""
import numpy as np
import pandas as pd

from . import acquisition, elasticity, features, genai_briefs, optimise, propensity, simulate, uplift
from .config import DEFAULT_BUDGET_EUR, SEED

BRIEFS_PER_OFFER = 20


def run(budget: float = DEFAULT_BUDGET_EUR, seed: int = SEED) -> dict:
    clients, prices = acquisition.load()
    df = features.build(clients, prices)

    oof = propensity.out_of_fold(df)
    df["p_churn"] = oof["p_gbm_cal"]

    price_models = elasticity.fit_models(df)

    po = simulate.run_pilot(df, seed)
    truth = simulate.true_effects(df, po)
    tau_d = uplift.x_learner(df, po, "discount")
    tau_c = uplift.x_learner(df, po, "call")

    rollout = (~po["in_pilot"]) & optimise.campaign_eligible(df)
    R, P = df[rollout], po[rollout]
    econ = optimise.offer_economics(R, tau_d[rollout.to_numpy()], tau_c[rollout.to_numpy()])
    econ_true = optimise.offer_economics(R, truth.loc[rollout, "tau_discount"], truth.loc[rollout, "tau_call"])

    policies = {
        "Riskiest first, discount": optimise.rank_and_fill(R["p_churn"], econ["spend_discount"], budget, "discount"),
        "Riskiest first, call": optimise.rank_and_fill(R["p_churn"], econ["spend_call"], budget, "call"),
        "Uplift + optimiser": optimise.solve(econ, budget),
        "Planted truth (ceiling)": optimise.solve(econ_true, budget),
    }
    comparison = pd.DataFrame({k: optimise.evaluate(v, R, P) for k, v in policies.items()}).T

    # Explanations and fact sheets for the customers the optimiser chose, highest value first.
    final_model = propensity.fit_final(df)
    chosen = policies["Uplift + optimiser"]
    chosen = chosen[chosen != "none"]
    value = np.where(chosen == "discount", econ.loc[chosen.index, "value_discount"],
                     econ.loc[chosen.index, "value_call"])
    ranked = pd.Series(value, index=chosen.index).sort_values(ascending=False).index
    top = [i for offer in ("discount", "call")
           for i in [j for j in ranked if chosen[j] == offer][:BRIEFS_PER_OFFER]]
    X_top = df.loc[top, features.MODEL_FEATURES]
    drivers = propensity.top_drivers(propensity.shap_values(final_model, X_top), X_top)
    sheets = []
    for i in top:
        row = df.loc[i].copy()
        row.name = df.at[i, "id"]
        sheets.append(genai_briefs.fact_sheet(row, drivers[i], chosen[i], econ.loc[i]))

    return {"df": df, "oof": oof, "price_models": price_models, "po": po, "truth": truth,
            "tau_discount": tau_d, "tau_call": tau_c, "rollout": rollout, "econ": econ,
            "econ_true": econ_true, "policies": policies, "comparison": comparison,
            "final_model": final_model, "fact_sheets": sheets}
