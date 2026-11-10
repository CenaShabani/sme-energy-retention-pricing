"""Builds the lean presentation notebook: python tools/build_final.py .  (run from the project root)"""
import sys
from pathlib import Path

import nbformat as nbf

PROJECT = Path(sys.argv[1])
nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Which SME customers should an energy supplier fight to keep, and with what?
### A retention & pricing case for an energy retailer: churn, price, risk and uplift turned into one € decision

**The finding.** Spending a €10k retention budget on the customers *most likely to leave*, which is
what most retention teams do, returns almost nothing: **€0.7k net with a discount, €3.2k with an
account-manager call.** Spending the same €10k on the customers whose decision the offer can
actually *change*, and letting an optimiser pick which offer each one gets, returns
**€11.1k net, 3.5× more**, which is 83% of what perfect knowledge would achieve.

Three things make the difference, and each one is a lesson for the client:
1. **High churn risk is not the same as persuadable.** The riskiest customers are often lost causes.
2. **The average effect hides the real one.** The account-manager call cuts churn by ~9 points for
   multi-product customers and *raises* it by ~6 points for loyal customers we have never
   contacted. A plain A/B readout (+2.5 points on average) would miss both.
3. **The model is over-optimistic at the margin.** Its promised value keeps rising with budget,
   while the true value peaks around €10k. The analysis says how much to spend, not just on whom.

**Data:** 14,606 real SME energy customers (BCG PowerCo, public, CC0) with real 2015 tariffs and
real churn. The retention pilot is **simulated** on top of them, with effects I planted so every
estimate can be checked against a known truth. Every simulated element is labelled.
""")

md(r"""
## 1. The question

An energy retailer sells gas and power to small businesses. About **1 in 10 leave every
quarter**. The commercial team has a retention budget and two levers:

| Offer | What it is | What it costs |
|---|---|---|
| **Discount** | 5% off the energy rate for 12 months | ~25% of that customer's annual margin, *paid by everyone who stays* |
| **Call** | a proactive account-manager call, no price change | ~€30 per call |

**The decision:** *for a given budget, which customers get which offer, and how much should we spend?*

It touches all four analytics domains in one chain, and each one feeds the next:

| Lens | Question | Method | Section |
|---|---|---|---|
| CLM / CRM | Who is likely to leave? | gradient boosting, calibrated, explained with SHAP | 3 |
| Risk | How many € are at risk, and where? | probability × margin | 4 |
| Revenue mgmt | Does price drive churn? | logistic regression, plus why it can't be trusted | 5 |
| Marketing | Whose decision does an offer change? | randomised pilot + uplift modelling | 6 |
| **Decision** | Who gets what, within budget? | integer optimisation | 7 |
| GenAI | Turn the list into briefs people act on | local LLM + deterministic fact-checker | 8 |
""")

code(r"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

from src import (acquisition, features, propensity, risk, elasticity, simulate, uplift,
                 optimise, genai_briefs, factcheck, plots)
from src.config import DEFAULT_BUDGET_EUR, DISCOUNT_MARGIN_SHARE, CALL_COST_EUR
""")

md(r"""
## 2. Data

**Real:** the BCG *PowerCo* dataset ([Kaggle, CC0](https://www.kaggle.com/datasets/erolmasimov/powerco)),
a European gas & power supplier's SME book: one row per customer (consumption, margins, tenure,
products, channel, contract dates, **churn in the next 3 months**) plus 12 months of 2015 tariffs
per customer. Channel and campaign codes are hashed in the source; I relabel them *Channel A, B…*.

`acquisition.load()` downloads it if needed and fails loudly if the schema has changed.
""")

code(r"""
clients, prices = acquisition.load()
df = features.build(clients, prices)
df = df.join(features.business_segments(df))
print(f"{len(df):,} customers | churn rate {df['churn'].mean():.1%} | "
      f"annual net margin €{df['net_margin'].sum():,.0f} | {len(prices):,} monthly price rows")
""")

md(r"""
**Leakage audit.** A model that sees the future looks great in testing and fails in production.
The snapshot is 1 Jan 2016, so every feature must be knowable on that date. The check below found
**61 customers with a product change dated after the snapshot**, and they churn more than average.
Using that field would leak the answer, so `features.build()` masks it for them.
""")

code(r"""
features.leakage_audit(df)
""")

md(r"""
## 3. CLM lens: who is likely to leave?

Three models, all scored **out-of-fold** (each customer is scored by a model that never saw
them):
- a **logistic regression**, the simple benchmark. If it were as good, I would use it.
- a **gradient-boosted tree model** (LightGBM), which handles non-linear effects and interactions.
- the same GBM, **calibrated**. Later steps multiply probabilities by euros, so a "30%" must
  really mean 30%.
""")

code(r"""
oof = propensity.out_of_fold(df)
df["p_churn"] = oof["p_gbm_cal"]
propensity.scorecard(df["churn"].to_numpy(), oof)
""")

md(r"""
**Reading it.** AUC measures ranking (0.5 = coin flip, 1 = perfect). The GBM ranks clearly better
than the logistic benchmark (0.72 vs 0.65), so the extra complexity earns its place. Calibration
barely changes ranking but fixes the **level**: the raw GBM predicts 8.9% churn on average against
9.7% actual. Under-predicting risk would understate the euros at risk by the same proportion.
""")

code(r"""
y = df["churn"].to_numpy()
plots.calibration(propensity.calibration_table(y, oof["p_gbm_raw"].to_numpy()),
                  propensity.calibration_table(y, oof["p_gbm_cal"].to_numpy()));
""")

code(r"""
propensity.lift_table(y, df["p_churn"].to_numpy()).head(3)
""")

md(r"""
The riskiest 10% of customers churn at **~35%, 3.6× the base rate**. That is a useful list, but
section 6 shows it is the wrong list to spend money on. (Why not a better churn model? Doubling the
features and a ~4× bigger model add only ~0.002 AUC, and the learning curve is flat: the limit is
information missing from the data, such as complaints, competitor offers and usage trends. The teaching
edition runs these tests.)

**Why is the model worried?** SHAP splits each prediction into the contribution of each feature.
Averaged over customers, it gives the global picture below. Per customer, it gives the "reasons"
the account manager sees in section 8.
""")

code(r"""
final_model = propensity.fit_final(df)
shap_all = propensity.shap_values(final_model, df[features.MODEL_FEATURES])
plots.shap_bar(shap_all, genai_briefs.FEATURE_LABELS);
""")

md(r"""
Margin, time left on the contract, subscribed power and the acquisition campaign dominate. Price variables matter,
but less than the commercial team usually assumes, which leads to section 5. (SHAP explains the
*model*, not the world: "high margin pushes risk up" is an association, not a cause.)

## 4. Risk lens: how many euros are at risk?

`expected margin lost = P(churn) × annual net margin`. This is the number the CFO cares about,
and the reason calibration mattered.
""")

code(r"""
risk.portfolio_summary(df).to_frame("value")
""")

code(r"""
plots.concentration(risk.concentration(df));
""")

md(r"""
About **€290k of €2.8M annual margin (~11%) is at risk**, and it is highly concentrated: the top
10% of customers hold over half of it. The top 1% by margin are **key accounts**, and those get a
named account manager rather than a campaign, so they are excluded from the targeting below
(`optimise.campaign_eligible`).

## 5. Revenue-management lens: does price drive churn?

The obvious RGM question: if we raise or cut prices, how does churn respond? Every customer's
energy price *fell* during 2015 (median −3.6%). Here is churn by size of the cut:
""")

code(r"""
plots.churn_by_price_change(elasticity.churn_by_price_change(df));
""")

md(r"""
Read naively: **the bigger the price cut, the more customers leave.** Nobody believes cutting
prices causes churn. The likely story is the reverse: the company cut prices for customers it
was already worried about. Price is *set in response to* churn risk. This is **confounding**, and
regression can't fix it. Watch the price coefficient as controls are added:
""")

code(r"""
elasticity.price_effect_table(elasticity.fit_models(df))
""")

md(r"""
The effect is tiny, not significant, and changes direction with the specification. **The
observational data can't say whether a price offer keeps a customer.** The honest move is to
stop here and run an experiment, and that is what the next section does.

## 6. Marketing lens: whose decision can an offer change?

### The pilot (SIMULATED)
A random half of the book is split into three equal arms: no offer, discount, call. In a real
engagement this is a 6–8 week field test. Here it is simulated on the real customers
(`src/simulate.py`):
- each customer keeps their **real** churn outcome as the "no offer" outcome;
- a real churner who gets an offer is **persuaded to stay** with a probability I planted. Price-
  sensitive customers respond to the discount; relationship customers respond to the call;
- a call also **pushes some loyal, never-contacted customers to shop around** ("sleeping dogs").

Because the truth is planted, everything below can be **checked** against it.
""")

code(r"""
pd.Series(simulate.PLANTED).to_frame("planted value")
""")

code(r"""
po = simulate.run_pilot(df)
truth = simulate.true_effects(df, po)
ate = uplift.ate_table(po)
ate
""")

code(r"""
plots.pilot_effects(ate);
""")

md(r"""
Both offers work **on average**, with confidence intervals clear of zero. A standard A/B readout
would stop here: "discount beats call, roll out the discount". That would be expensive,
because the average hides who the offers work for:
""")

code(r"""
seg = uplift.segment_table(df, po, "relationship")
plots.segment_effects(seg, "relationship", "call",
                      "The call helps relationship customers, and backfires on loyal ones");
seg
""")

md(r"""
The call cuts churn by **~9 points** for multi-product customers and **raises** it by **~6 points**
for loyal single-product customers who have never been contacted. Phoning them reminds them they
have a choice.

### Individual-level uplift
Segments are transparent but coarse. An **X-learner** estimates each customer's own effect: two
churn models (with and without the offer) give each pilot customer an *imputed* individual
effect, and a second model learns those effects from customer features. It is fitted on the pilot
only, then applied to the other half of the book.
""")

code(r"""
tau_d = uplift.x_learner(df, po, "discount")
tau_c = uplift.x_learner(df, po, "call")
rollout = ~po["in_pilot"]
for offer, est in [("discount", tau_d), ("call", tau_c)]:
    r = np.corrcoef(est[rollout], truth.loc[rollout, f"tau_{offer}"])[0, 1]
    print(f"{offer:8s}: correlation between estimated and planted effect = {r:.2f}")
""")

md(r"""
**Does it rank the right people first?** A Qini curve sorts customers by estimated effect and
counts how many are *actually saved* as we work down the list (the simulation knows the true
outcome under every offer). The comparison is the usual practice of ranking by churn risk.
""")

code(r"""
realised_call = (po["y_none"] - po["y_call"]).to_numpy()[rollout]
plots.qini({
    "rank by uplift (X-learner)": uplift.qini_curve(tau_c[rollout], realised_call),
    "rank by churn risk": uplift.qini_curve(df.loc[rollout, "p_churn"].to_numpy(), realised_call),
    "rank by planted truth": uplift.qini_curve(truth.loc[rollout, "tau_call"].to_numpy(), realised_call),
}, "Call: who should we phone first?");
""")

md(r"""
For the **call**, ranking by uplift saves far more customers than ranking by risk, because risk
ranking phones sleeping dogs too. (For the discount, risk ranking does nearly as well, since that
effect is roughly proportional to risk. Uplift modelling isn't magic, and it matters most when
an action can backfire.)

## 7. The decision: who gets what, within budget?

For each eligible customer in the roll-out half and each offer:

`value = estimated drop in churn × annual margin − expected cost of the offer`

The discount's cost is paid by **everyone who stays**, including customers who would have
stayed anyway. That is why blanket discounts destroy margin. The optimiser then picks **at most
one offer per customer** to maximise total value within budget (a multiple-choice knapsack,
solved exactly with `scipy.optimize.milp`), and never picks an offer with negative expected value.
""")

code(r"""
eligible = rollout & optimise.campaign_eligible(df)
R, P = df[eligible], po[eligible]
econ = optimise.offer_economics(R, tau_d[eligible], tau_c[eligible])
econ_true = optimise.offer_economics(R, truth.loc[eligible, "tau_discount"], truth.loc[eligible, "tau_call"])
budget = DEFAULT_BUDGET_EUR

policies = {
    "Riskiest first, discount": optimise.rank_and_fill(R["p_churn"], econ["spend_discount"], budget, "discount"),
    "Riskiest first, call": optimise.rank_and_fill(R["p_churn"], econ["spend_call"], budget, "call"),
    "Uplift + optimiser": optimise.solve(econ, budget),
    "Planted truth (ceiling)": optimise.solve(econ_true, budget),
}
comparison = pd.DataFrame({k: optimise.evaluate(v, R, P) for k, v in policies.items()}).T
comparison
""")

code(r"""
plots.policy_comparison(comparison);
""")

md(r"""
**Same €10k, very different outcomes.** The riskiest-first lists spend the budget on lost causes
and sure things. The uplift + optimiser policy mostly phones relationship customers, gives a
discount only to a few high-risk, price-sensitive, high-margin customers, and leaves the sleeping
dogs alone.

### How much should we spend?
""")

code(r"""
curve = optimise.budget_curve(econ, R, P, [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000])
plots.budget_curve(curve, chosen_budget=budget);
curve
""")

md(r"""
The model's **promised** value keeps climbing; the **true** value peaks around €10k and then
falls. Past that point the optimiser is spending on customers whose estimated effect is mostly
noise. This is the *winner's curse*: picking the largest estimates also picks the largest
over-estimates. In practice I would recommend **€7.5–10k, plus a hold-out group to measure what
really happened**, and not the larger number the model suggests.

### ▶ Live: change the budget
""")

code(r"""
BUDGET = 5_000   # <- change me and re-run

policy = optimise.solve(econ, BUDGET)
pd.DataFrame({
    "uplift + optimiser": optimise.evaluate(policy, R, P),
    "riskiest first, call": optimise.evaluate(
        optimise.rank_and_fill(R["p_churn"], econ["spend_call"], BUDGET, "call"), R, P),
}).T[["discounts", "calls", "customers_saved", "spend_eur", "net_value_eur", "roi"]]
""")

md(r"""
## 8. GenAI: from a target list to briefs people act on

An account manager needs two sentences per customer, not a SHAP table. A small **open-weights
LLM (Qwen2.5-3B-Instruct, run locally, so no customer data leaves the machine)** drafted a brief
for 40 of the targeted customers (20 discount, 20 call), using two prompt styles:
- **open**: the facts in a friendly sentence plus "write a persuasive brief";
- **grounded**: a numbered fact sheet, explicit rules (only these numbers, only this offer), and a
  fixed output shape.

Generation ran once, offline (`python -m src.genai_briefs`). The text is loaded here as an input
dataset. **The checking below runs live**: plain Python and regular expressions with no LLM-as-judge,
so the checker can't hallucinate either.
""")

code(r"""
briefs = genai_briefs.load()
checked = factcheck.check_all(briefs)
summary = factcheck.summary(checked)
plots.factcheck_bars(summary);
summary
""")

code(r"""
FLAGS = ["unsupported_numbers", "missing_recommended_offer", "unapproved_discount", "unfilled_placeholder", "too_long"]
example = checked[(checked["condition"] == "open") & (~checked["passes"])].iloc[0]
print("OPEN-PROMPT BRIEF:\n", example["text"][:900], "...")
print("\nFLAGGED:", {k: example[k] for k in FLAGS})
same = checked[(checked["condition"] == "grounded") & (checked["customer"] == example["customer"])].iloc[0]
print("\nGROUNDED BRIEF, same customer:\n", same["text"])
print("\nFLAGGED:", {k: same[k] for k in FLAGS})
""")

md(r"""
And the hallucinations that got through grounding, which is why the checker exists:
""")

code(r"""
for _, r in checked[(checked["condition"] == "grounded") & checked["has_unsupported_number"]].iterrows():
    print(f"invented {r['unsupported_numbers']} in:\n  {r['text']}\n")
""")

md(r"""
**What grounding fixed, and what it didn't.** The grounded prompt made **95% of briefs usable
as-is vs 18%** for the open prompt, which mostly wrote long letters with "[Account Manager's Name]"
placeholders. But **both styles invented a number in 1 brief in 20**. The grounded ones did it by
doing their own arithmetic ("about 18% of the annual margin"). Grounding improves behaviour; it does
not guarantee facts. That's why the deterministic check sits between the LLM and the account
manager: a failed brief is regenerated or goes to a human. The design trusts the LLM with
language and never with numbers or commitments.

## 9. Limitations

- **The pilot is simulated.** The planted effects are plausible but invented, so the € results
  show the *method* working against a known truth and are not a forecast of any real supplier's numbers. The
  first real step is a properly powered pilot.
- **The value of a saved customer is 12 months of margin.** That understates long-tenure
  customers; a lifetime-value model would change who is worth saving.
- **One churn window (3 months), one snapshot, one year of prices.** No seasonality, no
  competitor prices, no repeat-contact effects.
- **Uplift estimates are noisy** with ~2,400 customers per arm (the budget curve shows the cost).
  A larger pilot or shrinking estimates toward segment averages would help.
- **Optimisation treats the estimates as true.** A robust version would optimise a lower
  confidence bound of value, not the point estimate.
- **The fact-checker checks numbers, offers and format, not tone or whether the reasons are
  sensible.** It handles simple negation ("no discount needed" is not a promise) but not every
  phrasing. 40 briefs per prompt style is enough to see the pattern, not to quote error rates
  precisely.

## 10. Recommendation

1. **Stop ranking retention targets by churn risk alone.** Rank by *expected € saved net of offer cost*.
2. **Default to the call for multi-product customers; never phone loyal, untouched single-product
   customers.** Reserve the discount for the few high-risk, price-sensitive, high-margin customers.
3. **Spend ~€7.5–10k per quarter on this book, not more,** and keep a random hold-out so the true
   return is measured, not assumed.
4. **Run the real pilot first.** This notebook is the analysis plan for it: the same code runs
   unchanged on real pilot outcomes.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q                      # 17 tests
.venv/Scripts/jupyter nbconvert --execute --to notebook --inplace notebooks/00_sme_energy_retention_final.ipynb
.venv/Scripts/python -m src.genai_briefs                 # optional: regenerate LLM briefs (GPU)
```
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = PROJECT / "notebooks" / "00_sme_energy_retention_final.ipynb"
nbf.write(nb, out)
print("wrote", out)
