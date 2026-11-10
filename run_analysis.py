"""Rebuild every headline number and figure without the notebook: python run_analysis.py
Add --robustness to re-run the pilot with six random draws (about 2 minutes)."""
import sys
import warnings

import pandas as pd

from src import optimise, plots, simulate, uplift
from src.config import OUTPUTS
from src.pipeline import run

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    r = run()
    R, P = r["df"][r["rollout"]], r["po"][r["rollout"]]
    print(uplift.ate_table(r["po"]), "\n")
    print(r["comparison"].round(1), "\n")
    curve = optimise.budget_curve(r["econ"], R, P, [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000])
    print(curve.round(0))
    r["comparison"].to_csv(OUTPUTS / "policy_comparison.csv")
    curve.to_csv(OUTPUTS / "budget_curve.csv", index=False)
    plots.policy_comparison(r["comparison"]).savefig(OUTPUTS / "policy_comparison.png", bbox_inches="tight")
    plots.budget_curve(curve).savefig(OUTPUTS / "budget_curve.png", bbox_inches="tight")
    print(f"\nWrote outputs to {OUTPUTS}")

    if "--robustness" in sys.argv:
        df, rows = r["df"], []
        for seed in [1, 2, 3, 4, 5, 42]:
            po = simulate.run_pilot(df, seed)
            truth = simulate.true_effects(df, po)
            keep = (~po["in_pilot"]) & optimise.campaign_eligible(df)
            R, P = df[keep], po[keep]
            econ = optimise.offer_economics(R, uplift.x_learner(df, po, "discount")[keep.to_numpy()],
                                            uplift.x_learner(df, po, "call")[keep.to_numpy()])
            econ_true = optimise.offer_economics(R, truth.loc[keep, "tau_discount"], truth.loc[keep, "tau_call"])
            b = 10_000
            best_list = max(
                optimise.evaluate(optimise.rank_and_fill(R["p_churn"], econ[f"spend_{o}"], b, o), R, P)["net_value_eur"]
                for o in ("discount", "call"))
            up = optimise.evaluate(optimise.solve(econ, b), R, P)["net_value_eur"]
            ceiling = optimise.evaluate(optimise.solve(econ_true, b), R, P)["net_value_eur"]
            rows.append({"seed": seed, "best_list": best_list, "uplift_optimiser": up, "ceiling": ceiling,
                         "x_best_list": up / best_list, "share_of_ceiling": up / ceiling})
        robust = pd.DataFrame(rows)
        print("\nRobustness across pilot draws (EUR 10k budget):")
        print(robust.round(2))
        robust.to_csv(OUTPUTS / "robustness.csv", index=False)
