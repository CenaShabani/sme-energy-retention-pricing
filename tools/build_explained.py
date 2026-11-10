"""Builds the TEACHING EDITION: notebooks/00_sme_energy_retention_explained.ipynb

    python tools/build_explained.py .        (run from the project root)

Same analysis and the same src/ calls as the lean notebook (tools/build_final.py), but every idea
is explained from zero: plain words, new-word tables, runnable toy examples, the src/ code shown and
walked through, how to read every output, check-yourself questions and an interview one-liner.
Toy cells only ever touch made-up toy data; every real number comes from src/.
"""
import sys
from pathlib import Path

import nbformat as nbf

PROJECT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


def check(*qa):
    """Check-yourself questions with hidden answers."""
    parts = ["#### ✅ Check yourself"]
    for q, a in qa:
        parts.append(f"<details><summary><b>{q}</b> (click for the answer)</summary>\n\n{a}\n\n</details>")
    md("\n\n".join(parts))


def say(s):
    md(f"#### 🎤 Say it in the interview\n> {s.strip()}")


def words(*rows):
    lines = ["#### 📖 New words", "| Word | What it means |", "|---|---|"]
    lines += [f"| **{w}** | {m} |" for w, m in rows]
    md("\n".join(lines))


# =================================================================================================
md(r"""
# Teaching edition: which customers should an energy supplier fight to keep, and with what?

**Who this is for:** anyone, including someone who has never done data science. Every word is
explained the first time it appears, every method is first shown on a tiny made-up example you can
follow by hand, and every table and chart comes with "how to read this".

**This is the long version.** The short version you present in an interview is
`00_sme_energy_retention_final.ipynb`. Both run exactly the same code from the `src/` folder, so their
numbers are identical.

**Time to work through it:** about 2–3 hours if you run every cell and answer the check-yourself
questions. Do it in two sittings: sections 0–4, then 5–9.
""")

md(r"""
## 0. How to use this notebook (skip if you know Jupyter)

A **notebook** is a document made of **cells**. There are two kinds:
- **Text cells** (like this one): explanations. You just read them.
- **Code cells** (grey boxes with `In [ ]:` next to them): Python code. Click one and press
  **Shift + Enter** to run it. Whatever it produces (a number, a table, a chart) appears underneath.

Run the cells **from top to bottom**, because later cells use results from earlier ones. If something
looks broken: *Kernel → Restart & Run All*.

### The Python you will see
You don't need to write Python to follow this, but these patterns come up again and again:

| You see | It means |
|---|---|
| `df` | a **DataFrame**: a table, like a spreadsheet. Rows are things (customers), columns are facts about them. |
| `df["churn"]` | one column of the table |
| `df[df["churn"] == 1]` | only the rows where churn equals 1 (a filter) |
| `df["net_margin"].mean()` / `.sum()` | the average / total of a column |
| `df.groupby("channel")["churn"].mean()` | the average churn *within each channel* separately |
| `x = something` | store a result under the name `x` to use later |
| `np.` / `pd.` | functions from **numpy** (number crunching) and **pandas** (tables) |
| `from src import propensity` then `propensity.out_of_fold(df)` | call a function this project wrote, kept in the file `src/propensity.py` |
| `show(propensity.out_of_fold)` | print that function's code so you can read it (a helper defined below) |

### The story in one paragraph
An energy supplier sells gas and electricity to small businesses. Every quarter about 1 in 10 of them
leaves for a competitor. The company has a limited budget for keeping customers: it can offer a
**discount** or have an **account manager call** them. The obvious approach is to spend the budget on
the customers *most likely to leave*. This notebook shows why that is the wrong list, and builds the
right one: customers whose decision an offer can actually **change**, with an optimiser choosing who
gets which offer. Same budget, more than three times the return.

### Map of the notebook
| § | Question | Main method |
|---|---|---|
| 1 | What is the business problem? | — |
| 2 | What data do we have, and can we trust it? | data loading, leakage check |
| 3 | Who is likely to leave? | classification, gradient boosting, calibration, SHAP |
| 4 | How many euros are at risk? | expected value |
| 5 | Does price drive leaving? | regression, confounding |
| 6 | Whose decision can an offer change? | randomised experiment, uplift modelling |
| 7 | Who gets what, within budget? | optimisation |
| 8 | Turning the list into something people act on | an LLM + a fact-checker |
| 9 | What it all means | limitations, recommendation, glossary |
""")

code(r"""
# Setup: load the libraries and this project's own code. Run this first.
import sys, warnings, inspect
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

from src import (acquisition, features, propensity, risk, elasticity, simulate, uplift,
                 optimise, genai_briefs, factcheck, plots)
from src.config import DEFAULT_BUDGET_EUR, DISCOUNT_MARGIN_SHARE, CALL_COST_EUR


def show(fn):
    # Print the source code of one of this project's functions, with the file it lives in.
    file = Path(inspect.getsourcefile(fn)).relative_to(ROOT).as_posix()
    display(Markdown(f"**`{fn.__name__}`** from `{file}`\n```python\n{inspect.getsource(fn)}```"))

print("Ready.")
""")

# =================================================================================================
md(r"""
---
## 1. The business problem

### The idea in plain words
Think of a gym. Some members will cancel next month. The gym can phone members or give them a free
month, but it has a limited budget. Who should it spend it on?

- If it phones someone who was going to stay anyway, the money is **wasted**.
- If it phones someone who will leave no matter what, the money is also **wasted**.
- The money only does something for people **on the fence**: they would leave without the offer and
  stay with it.

That's the whole case. The company here is an energy supplier, not a gym, but the logic is identical.
""")

words(
    ("SME", "Small or medium-sized enterprise: a small business (a bakery, a garage, an office)."),
    ("Churn", "A customer leaving. *Churn rate* = the share of customers who leave in a period. Here: leaving within 3 months."),
    ("Margin (net margin)", "What the company earns from a customer after costs, here per year in euros. Not revenue: profit."),
    ("Retention offer", "Something given to keep a customer: here a **discount** (5% off the energy price for a year) or a **call** from an account manager."),
    ("Budget", "The euros available for offers. Every offer costs money, so we can't give one to everybody."),
)

md(r"""
### The two offers and what they cost

| Offer | What the customer gets | What it costs us |
|---|---|---|
| **Discount** | 5% off the energy rate for 12 months | about **25% of that customer's yearly margin** (5% of the bill is ~25% of the profit on it), and it's paid by **everyone who stays**, including customers who would have stayed anyway |
| **Call** | an account manager phones them; no price change | about **€30** (half an hour of a person's time) |
""")

code(r"""
# These assumptions live in one file, src/config.py, so every part of the project uses the same ones.
print(f"Discount costs {DISCOUNT_MARGIN_SHARE:.0%} of the customer's yearly margin (if they stay).")
print(f"A call costs EUR {CALL_COST_EUR:.0f}.")
print(f"Default campaign budget: EUR {DEFAULT_BUDGET_EUR:,.0f}.")
""")

check(
    ("Why is giving a discount to a customer who would have stayed anyway a waste?",
     "Because they would have paid full price and stayed. The discount just hands away 25% of their margin for nothing."),
    ("What's the only kind of customer an offer actually helps?",
     "A customer who would leave *without* the offer and stays *because of* it. Later we call them **persuadables**."),
)

# =================================================================================================
md(r"""
---
## 2. The data, and whether we can trust it

### The idea in plain words
Data here is two tables, like two spreadsheets:
1. **Clients:** one row per customer (14,606 rows). Columns describe them: how much energy they use,
   how long they've been a customer, their margin, and whether they **left in the next 3 months** (`churn`: 1 = left, 0 = stayed).
2. **Prices:** one row per customer per month of 2015, with the prices they were charged.

It's real data from a European energy supplier, published by the consultancy BCG for training
purposes (the *PowerCo* dataset, free to use).
""")

words(
    ("Row / column", "A row is one thing (one customer). A column is one fact about every thing (e.g. everyone's margin)."),
    ("Label / target", "The column we want to predict. Here `churn`."),
    ("Feature", "A column we use to make the prediction (tenure, consumption, price...)."),
    ("Snapshot date", "The moment the data was frozen. Here 1 January 2016. Churn is measured in the 3 months *after* it."),
)

md("### Look inside the code: how the data is loaded")
code(r"show(acquisition.load)")
md(r"""
**What it does, step by step:**
1. `download()` fetches the files from Kaggle the first time (and skips this if they're already on disk).
2. `pd.read_csv` reads each file into a table. `parse_dates` tells pandas which columns are dates.
3. The `assert` lines are **safety checks**: if a column is missing, if a customer appears twice, or if
   `churn` contains anything other than 0 and 1, the program stops with a clear message instead of
   quietly producing wrong numbers later. That's a habit worth having: fail loudly and early.
""")

md("### The real run")
code(r"""
clients, prices = acquisition.load()
print(f"Clients table: {clients.shape[0]:,} rows x {clients.shape[1]} columns")
print(f"Prices table:  {prices.shape[0]:,} rows x {prices.shape[1]} columns")
clients.head(3).T
""")

md(r"""
### How to read this output
`.head(3)` shows the first 3 customers; `.T` flips the table so each **column of the data becomes a row**
(easier to read when there are many columns). So each column above (`0`, `1`, `2`) is one customer.
The most important rows:

| Column | Meaning |
|---|---|
| `id` | the customer's code (scrambled for privacy) |
| `channel_sales`, `origin_up` | how the customer was sold to / which campaign brought them in (also scrambled) |
| `cons_12m` | electricity used in the last 12 months (kWh) |
| `has_gas` | `t` if they also buy gas from us |
| `date_activ`, `date_end` | when the contract started / is due to end |
| `num_years_antig` | **tenure**: how many years they've been a customer |
| `net_margin` | yearly profit from this customer (€) |
| `nb_prod_act` | number of products they have with us |
| `churn` | **the label**: 1 if they left within 3 months of the snapshot |
""")

code(r"""
print(f"Share of customers who left (churn rate): {clients['churn'].mean():.1%}")
print(f"Total yearly margin of the whole book:    EUR {clients['net_margin'].sum():,.0f}")
""")

md(r"""
### Turning raw columns into features
Raw columns aren't always in a useful form. "Contract ends on 2016-06-15" is more useful as
"5.4 months until the contract ends". "Price in January" and "price in December" are more useful as
"how much did the price change". This step is called **feature engineering**.
""")
code(r"show(features.build)")
md(r"""
**Walkthrough of the decisions in this code:**
- `price_features(prices)` squashes 12 monthly price rows per customer into one row: the December
  price, the change since January, and how much it jumped around (`std` = standard deviation, a
  measure of spread).
- `_readable_codes` renames the scrambled channel codes to `Channel A`, `Channel B`... and pools tiny
  groups into `other`, so the model can't memorise a handful of customers.
- `months_to_end`, `months_since_modif`: dates turned into "months from the snapshot".
- `np.log1p(...)` takes the logarithm of consumption. Consumption ranges from 0 to 6 million kWh;
  the log squashes that range so a few huge customers don't dominate. (`log1p` = log of 1 + x, so 0 stays 0.)
- `price_gap_vs_channel`: how much more (or less) this customer pays than the typical customer in the
  same sales channel. +10% means they pay 10% above their peers, a plausible reason to shop around.
- The **future_modif** lines handle *leakage*, explained next.
""")

md(r"""
### Leakage: the most common way to fool yourself

**The idea in plain words.** Imagine predicting who wins a race using the finishing photo. You'd be
right every time, and completely useless, because you'd only have the photo after the race. **Leakage**
is when a feature contains information you wouldn't have at the moment you make the prediction.

**Toy example.** Our snapshot is 1 January 2016. Look at four events for one customer:
""")
code(r"""
toy = pd.DataFrame({
    "event": ["contract started", "last price change", "product changed", "customer left"],
    "date": pd.to_datetime(["2012-03-01", "2015-07-01", "2016-01-20", "2016-02-10"]),
})
snapshot = pd.Timestamp("2016-01-01")
toy["known on 1 Jan 2016?"] = np.where(toy["date"] <= snapshot, "yes - can use", "NO - that's the future")
toy
""")
md(r"""
The "product changed" event happened **after** the snapshot, typically because the customer was
already on their way out (switching plans, negotiating). A model that sees it would look brilliant in
testing and fail in real use.

**The real check:** for every date column, how many customers have a date after the snapshot, and do
those customers churn more?
""")
code(r"""
df = features.build(clients, prices)
df = df.join(features.business_segments(df))
features.leakage_audit(df)
""")
md(r"""
### How to read this output
- `rows_after_snapshot`: how many customers have a date in the future for that column.
- `churn_rate_if_after`: churn rate among just those customers (compare with ~9.7% overall).
- `handling`: what the code does about it.

`date_end` and `date_renewal` are in the future for most customers, but that's fine: a contract's end
date is **scheduled in advance**, so it's genuinely known on 1 January. `date_modif_prod` is different:
a product change can't be known before it happens.
""")
code(r"""
leak = features.leakage_audit(df).set_index("column").loc["date_modif_prod"]
print(f"{leak['rows_after_snapshot']} customers have a product change dated after the snapshot, "
      f"and {leak['churn_rate_if_after']:.1%} of them churned (vs {df['churn'].mean():.1%} overall).")
print("So features.build() sets that field to 'unknown' for them instead of using the future date.")
""")

check(
    ("A colleague adds the feature 'number of complaints in the 3 months after the snapshot'. What's wrong?",
     "It's leakage: those complaints happen after the moment we make the prediction, so we couldn't know them when deciding."),
    ("Why is it OK to use the contract end date even though it's in the future?",
     "Because it was agreed when the contract was signed. It's a scheduled fact known on the snapshot date, not something that happened later."),
)
say("The first thing I did was a leakage audit: 61 customers had a product change dated after the snapshot, and they churned more, so I masked it.")

# =================================================================================================
md(r"""
---
## 3. Who is likely to leave? (the churn model)

### The idea in plain words
We have 14,606 past customers and we know who left. A **model** looks for patterns ("customers with
short tenure and high prices leave more often") and turns them into a **probability** for each customer:
"this one has a 32% chance of leaving". Predicting a yes/no outcome like this is called
**classification**.
""")

words(
    ("Model", "A recipe, learned from data, that turns features into a prediction."),
    ("Probability", "A number from 0 to 1 (or 0–100%). 0.3 = a 30% chance."),
    ("Training", "Showing the model past examples with known answers so it learns the patterns."),
    ("Logistic regression", "The simplest classification model: adds up the features with weights and squeezes the result into 0–1."),
    ("Decision tree", "A model that asks yes/no questions ('tenure ≤ 4 years?') and ends in a prediction."),
    ("Gradient boosting (GBM)", "Hundreds of small trees built one after another, each fixing the mistakes of the ones before. LightGBM is a fast implementation."),
)

md(r"""
### Toy example 1: logistic regression on 8 customers
Eight made-up customers. The only feature is tenure (years with us). Short-tenure customers leave more.
""")
code(r"""
from sklearn.linear_model import LogisticRegression

toy = pd.DataFrame({"tenure_years": [1, 2, 2, 3, 5, 6, 8, 10],
                    "churn":        [1, 1, 0, 1, 0, 0, 0, 0]})
logit = LogisticRegression().fit(toy[["tenure_years"]], toy["churn"])
toy["predicted_probability"] = logit.predict_proba(toy[["tenure_years"]])[:, 1].round(2)
print(toy)

grid = pd.DataFrame({"tenure_years": np.linspace(0, 11, 100)})
plt.figure(figsize=(5, 3))
plt.scatter(toy["tenure_years"], toy["churn"], color="#eb6834", zorder=3, label="actual (1 = left)")
plt.plot(grid["tenure_years"], logit.predict_proba(grid)[:, 1], color="#2a78d6", label="predicted probability")
plt.xlabel("tenure (years)"); plt.ylabel("probability of leaving"); plt.legend(); plt.show()
""")
md(r"""
**How to read it.** Orange dots are the truth (1 = left, 0 = stayed). The blue S-shaped curve is the
model's probability. It's high for short tenure and falls as tenure grows. That S-shape (the
**logistic function**) is what keeps the output between 0 and 1. With many features, logistic regression
does the same thing with a weighted sum of all of them.
""")

md(r"""
### Toy example 2: a decision tree, then boosting
A **decision tree** asks yes/no questions. Here it learns from two features.
""")
code(r"""
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

toy = pd.DataFrame({"tenure_years":   [1, 2, 2, 3, 5, 6, 8, 10],
                    "pays_above_peers":[1, 1, 0, 1, 1, 0, 0, 0],
                    "churn":           [1, 1, 0, 1, 0, 0, 0, 0]})
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(toy[["tenure_years", "pays_above_peers"]], toy["churn"])
print(export_text(tree, feature_names=["tenure_years", "pays_above_peers"]))
""")
md(r"""
**How to read it.** Each `|---` line is a question; indentation shows which answer leads where. Follow it
like a flowchart: first "is `pays_above_peers` ≤ 0.5?" (i.e. *not* paying above peers). If yes → class 0
(stays). If no → "is tenure ≤ 4 years?" If yes → class 1 (leaves), if no → class 0. The tree found by
itself that over-charged *new* customers are the ones who leave.

One tree is either too simple or memorises the data. **Boosting** builds many small trees in sequence:
each new tree is trained on **what the previous trees still got wrong** (the "residuals"). Watch it
happen over three rounds:
""")
code(r"""
X = toy[["tenure_years", "pays_above_peers"]]
y = toy["churn"].astype(float)
prediction = np.full(len(y), y.mean())          # round 0: everyone gets the average (3/8 = 0.375)
learning_rate = 0.5
rounds = {"round 0": prediction.copy()}
for r in range(1, 4):
    residual = y - prediction                    # what we still get wrong
    stump = DecisionTreeRegressor(max_depth=1).fit(X, residual)   # a 1-question tree that predicts the error
    prediction = prediction + learning_rate * stump.predict(X)    # nudge the prediction towards the truth
    rounds[f"round {r}"] = prediction.copy()
out = pd.DataFrame(rounds).round(2)
out.insert(0, "actual", y)
out
""")
md(r"""
**How to read it.** Each column is the prediction after that many rounds. Round 0 gives everyone the
average. Each round, a one-question tree looks at the remaining errors and nudges the predictions
towards the truth. The `learning_rate` (0.5 here, 0.03 in the real model) controls how big each nudge is:
small nudges over many rounds learn carefully. The real model does this **400 times** with slightly
bigger trees. That's gradient boosting.

(Two customers end at −0.02, a "negative probability". That's because this toy works directly on 0/1
numbers to stay simple. The real model works on a stretched scale called *log-odds* and converts back
at the end, so its answers always stay between 0 and 1.)
""")

md(r"""
### Testing a model honestly: cross-validation and "out-of-fold"
A model always looks good on the customers it learned from (like a student who has seen the exam).
So we test it on customers it **hasn't seen**:

**5-fold cross-validation:** split the customers into 5 groups ("folds"). Train on 4, predict the 5th.
Repeat 5 times so every group gets predicted once. Each customer ends up with an **out-of-fold**
prediction made by a model that never saw them.
""")
code(r"""
n, k = 20, 5
fold_of = np.arange(n) % k
grid = np.array([[0 if fold_of[i] == f else 1 for i in range(n)] for f in range(k)])
plt.figure(figsize=(7, 2.2))
plt.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(["#eb6834", "#2a78d6"]), aspect="auto")
plt.yticks(range(k), [f"round {f+1}" for f in range(k)]); plt.xticks(range(n), range(1, n + 1), fontsize=8)
plt.grid(False); plt.xlabel("20 example customers")
plt.title("blue = used for training, orange = predicted (never seen)"); plt.show()
""")
md("""Every customer is orange exactly once: that's their honest, out-of-fold prediction.""")

md(r"""
### Measuring how good the model is
Two different questions:
1. **Ranking:** does the model put leavers above stayers? Measured by **AUC**.
2. **Probability quality:** when it says 30%, do ~30% actually leave? Measured by **Brier score** and
   **calibration** (below).

**AUC by hand.** Take every pair of (one leaver, one stayer). AUC = the share of pairs where the leaver
got the higher score. 0.5 = coin flip, 1.0 = perfect.
""")
code(r"""
from sklearn.metrics import roc_auc_score

toy = pd.DataFrame({"customer": list("ABCDEF"), "churn": [1, 1, 0, 0, 0, 1],
                    "score": [0.9, 0.4, 0.5, 0.2, 0.1, 0.7]})
leavers, stayers = toy[toy.churn == 1], toy[toy.churn == 0]
pairs = [(l.customer, s.customer, l.score > s.score) for l in leavers.itertuples() for s in stayers.itertuples()]
print(pd.DataFrame(pairs, columns=["leaver", "stayer", "leaver scored higher?"]))
print(f"\nby hand: {sum(p[2] for p in pairs)}/{len(pairs)} = {np.mean([p[2] for p in pairs]):.3f}")
print(f"sklearn: {roc_auc_score(toy.churn, toy.score):.3f}  (same number)")
""")

md(r"""
### Calibration: does "30%" really mean 30%?
**Analogy:** a weather forecaster who says "30% chance of rain" is well calibrated if it rains on about
3 of every 10 such days. A model can rank customers perfectly and still have the wrong *level*, e.g.
saying 20% when the truth is 30%.

Why we care here: later we multiply probabilities by euros. If the probabilities are 10% too low, every
euro figure is 10% too low.

**Toy:** group predictions into bins and compare the average prediction with the actual rate.
""")
code(r"""
toy = pd.DataFrame({"predicted": [0.05, 0.08, 0.10, 0.12, 0.10, 0.30, 0.35, 0.40, 0.25, 0.30],
                    "actual":    [0,    0,    0,    1,    0,    1,    0,    1,    0,    1]})
toy["bin"] = np.where(toy["predicted"] < 0.2, "low (<20%)", "high (>=20%)")
toy.groupby("bin").agg(average_prediction=("predicted", "mean"), actual_rate=("actual", "mean"), customers=("actual", "size"))
""")
md(r"""
**How to read it.** In the low bin the model said ~9% on average and 1 in 5 (20%) actually left, so it
**under-predicts**. In the high bin it said ~32% and 60% left, again too low. A **calibration** step
learns a correction from data ("when the model says 9%, the truth is about 20%") and applies it.
The project uses **isotonic calibration**, which learns a step-shaped correction that never changes the
order of customers.
""")

md("### Look inside the code: how the real models are trained")
code(r"show(propensity.out_of_fold)")
md(r"""
**Walkthrough:**
- `StratifiedKFold(n_splits, shuffle=True)` makes the 5 folds. *Stratified* means each fold keeps the
  same ~9.7% churn rate, so no fold is accidentally easy.
- For each fold, three models are trained on the other 4 folds and predict this one:
  1. `make_logit()`: logistic regression, the **simple benchmark**. If it were as good, we'd use it.
  2. `make_gbm()`: LightGBM gradient boosting.
  3. `CalibratedClassifierCV(make_gbm(), method="isotonic", cv=3)`: the same GBM plus a calibration
     correction, learned **only from the training folds** (so calibration doesn't peek either).
- `out.iloc[test, ...] = ...` stores the predictions for the held-out customers only.
""")

md("### The real run")
code(r"""
oof = propensity.out_of_fold(df)
df["p_churn"] = oof["p_gbm_cal"]          # the calibrated probability is what everything later uses
y = df["churn"].to_numpy()
scores = propensity.scorecard(y, oof)
scores
""")
md(r"""
### How to read this output
One row per model (`p_logit` = logistic benchmark, `p_gbm_raw` = boosting, `p_gbm_cal` = boosting + calibration).

| Column | Meaning | Better is |
|---|---|---|
| `AUC` | ranking quality (0.5 = random, 1 = perfect) | higher |
| `avg_precision` | ranking quality focused on the rare leavers (random would be ≈ the churn rate, 0.097) | higher |
| `brier` | average squared gap between probability and outcome | lower |
| `log_loss` | like Brier, but punishes confident mistakes harder | lower |
| `mean_predicted` | the model's average probability | close to `actual_rate` |
| `actual_rate` | the real churn rate | — |
""")
code(r"""
s = scores
print(f"Boosting ranks better than the simple benchmark: AUC {s.loc['p_gbm_raw','AUC']:.3f} vs {s.loc['p_logit','AUC']:.3f}.")
print(f"But raw boosting predicts {s.loc['p_gbm_raw','mean_predicted']:.1%} churn on average vs {s.loc['p_gbm_raw','actual_rate']:.1%} actual: too low.")
print(f"After calibration it predicts {s.loc['p_gbm_cal','mean_predicted']:.1%}: the level is now right, "
      f"and the ranking (AUC {s.loc['p_gbm_cal','AUC']:.3f}) is unchanged.")
""")

code(r"""
plots.calibration(propensity.calibration_table(y, oof["p_gbm_raw"].to_numpy()),
                  propensity.calibration_table(y, oof["p_gbm_cal"].to_numpy()));
""")
md(r"""
### How to read this chart
- Customers are sorted by predicted risk and split into 10 equal groups (**deciles**). Each dot is one group.
- **x-axis:** the group's average predicted probability. **y-axis:** the share in that group who actually left.
- The **dashed diagonal** is perfection: predicted = actual.
- Dots **above** the line: the model under-predicts for that group. Below: it over-predicts.
- Both lines climb steeply to the right, which means the model *ranks* well: its riskiest 10% really are much riskier.
""")

code(r"propensity.lift_table(y, df['p_churn'].to_numpy()).head(3)")
md(r"""
### How to read this table
If we could only contact the riskiest X% of customers (`top_share`):
- `churn_rate`: how many of them actually leave.
- `share_of_churners_reached`: what fraction of *all* leavers we'd reach.
- `lift`: how many times more leavers than a random pick (churn_rate ÷ overall rate).
""")
code(r"""
lt = propensity.lift_table(y, df["p_churn"].to_numpy()).iloc[0]
print(f"The riskiest 10% churn at {lt['churn_rate']:.0%}, {lt['lift']:.1f}x the average, "
      f"and contain {lt['share_of_churners_reached']:.0%} of everyone who leaves.")
print("Useful list. But section 6 shows it's the wrong list to spend money on.")
""")

md(r"""
### Why does the model think a customer is risky? (SHAP)
**The idea.** For one customer, SHAP splits "how far this customer's score is from the average" into a
contribution from each feature, and the contributions add up exactly.

**Toy:** a made-up model where risk = base 10% + effects. For one customer:
""")
code(r"""
toy = pd.DataFrame({
    "feature": ["(average customer)", "short tenure (2 years)", "pays 8% above peers", "has gas too"],
    "contribution_pct_points": [10.0, +6.0, +4.0, -3.0],
})
toy["running_total"] = toy["contribution_pct_points"].cumsum()
toy
""")
md(r"""
Start at the average (10%), add +6 for short tenure, +4 for the high price, −3 for having gas: this
customer's risk is 17%, and we can say *why*. The real model has 26 features and works on a log-odds
scale (a stretched version of probability), but the idea is identical.

**One warning:** SHAP explains the **model**, not the world. "High margin pushes risk up" means the
model learned that pattern, not that margin *causes* leaving. Section 5 is about exactly that difference.
""")
code(r"""
final_model = propensity.fit_final(df)
shap_all = propensity.shap_values(final_model, df[features.MODEL_FEATURES])
plots.shap_bar(shap_all, genai_briefs.FEATURE_LABELS);
""")
md(r"""
### How to read this chart
Each bar is one feature. Its length is the **average size of its push** on customers' scores (in either
direction). Longer = the model relies on it more. It doesn't say whether the feature pushes up or down;
that differs per customer.
""")
code(r"""
# one high-risk customer (at the 99th percentile of risk) and the model's top reasons
example = (df["p_churn"] - df["p_churn"].quantile(0.99)).abs().idxmin()
X1 = df.loc[[example], features.MODEL_FEATURES]
drivers = propensity.top_drivers(propensity.shap_values(final_model, X1), X1)[example]
print(f"Example customer: churn probability {df.at[example, 'p_churn']:.0%} (riskier than 99% of customers).")
print("Top reasons the model gives (push = how much each one raises the score, in log-odds):")
for feat, value, push in drivers:
    shown = f"{value:.3f}" if isinstance(value, float) else str(value)
    print(f"  {genai_briefs.FEATURE_LABELS.get(feat, feat):35s} value = {shown:>10}   push = {push:+.2f}")
""")

check(
    ("Why test a model on customers it didn't train on?",
     "Because any model looks good on data it has memorised. Out-of-fold predictions show how it does on new customers, which is the real use."),
    ("Calibration didn't change AUC. Why bother?",
     "AUC only measures *ranking*; calibration only fixes the *level* and keeps the order. We multiply probabilities by euros later, so the level must be right."),
    ("What's the difference between AUC 0.5 and 0.72?",
     "0.5 means the model ranks a leaver above a stayer only half the time, like a coin. 0.72 means it does so 72% of the time."),
)
say("Boosting beat a logistic benchmark, 0.72 AUC vs 0.65 out-of-fold, and I calibrated it because I'm about to multiply its probabilities by euros.")

md(r"""
### Why isn't the churn model better?
An AUC around 0.72 can look disappointing. Before accepting it, three honest tests: (1) is it being
judged the right way? (2) would more data help? (3) would more features or a bigger model help?

**Test 1: the "50% cut-off" trap.** A common way to judge a classifier is: predict "leaves" if the
probability is above 50%, then count hits.
""")
code(r"""
pp = oof["p_gbm_cal"].to_numpy()
flagged = pp >= 0.5
print(f"Customers with risk >= 50%: {flagged.sum()} of {len(pp):,}")
print(f"Real leavers caught: {(flagged & (y == 1)).sum()} of {y.sum():,}")
print(f"Accuracy: {(flagged == y).mean():.1%}  vs {1 - y.mean():.1%} for simply predicting 'nobody leaves'")
""")
md(r"""
### How to read this
Judged this way the model looks terrible: it catches only a small share of leavers and barely beats
"nobody leaves". But that's the wrong test. Only 9.7% of customers leave, so **almost nobody is more
likely than not to leave**: even a customer at 30% risk, three times the average, is still "probably
stays". The model is meant to **rank** customers and give **probabilities** to multiply by euros, and on
that test the riskiest 10% leave at ~35%, 3.6× the average. Accuracy at 50% is the wrong yardstick for
rare events.

**Test 2: would more data help?** Train on smaller random subsets of customers and watch the score.
If it were still climbing at 100%, more customers would help.
""")
code(r"""
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
cv = StratifiedKFold(5, shuffle=True, random_state=0)
X_base = df[features.MODEL_FEATURES]

rng = np.random.default_rng(0)
curve_rows = []
for share in [0.25, 0.5, 0.75, 1.0]:
    idx = rng.choice(len(y), int(share * len(y)), replace=False)
    p_sub = cross_val_predict(propensity.make_gbm(), X_base.iloc[idx], y[idx], cv=cv, method="predict_proba")[:, 1]
    curve_rows.append({"share_of_customers_used": share, "AUC": roc_auc_score(y[idx], p_sub)})
learning = pd.DataFrame(curve_rows)
learning
""")
md(r"""
### How to read this
Each row trains and tests the same model on a random share of the customers. The score rises from 25%
to 50%, then goes flat. More customers *of the same kind of data* won't help.

**Test 3: more features, bigger model.** Add every price column (December level, change over the year,
average and volatility of all six price series, the off-peak/peak spread), margin per kWh, tenure in
months and months to renewal. Then also try a much bigger, more finely tuned model.
""")
code(r"""
import lightgbm as lgb

pr_ = prices.sort_values(["id", "price_date"]); g = pr_.groupby("id")
price_cols = [c for c in prices.columns if c.startswith("price_") and c != "price_date"]
rich = pd.concat([g[price_cols].last().add_suffix("_dec"),
                  (g[price_cols].last() - g[price_cols].first()).add_suffix("_chg"),
                  g[price_cols].mean().add_suffix("_mean"),
                  g[price_cols].std().add_suffix("_std")], axis=1)
rich["offpeak_peak_spread"] = rich["price_off_peak_var_dec"] - rich["price_peak_var_dec"]
X_rich = X_base.join(df[["id"]]).merge(rich, left_on="id", right_index=True, how="left").drop(columns="id")
X_rich["margin_per_kwh"] = df["net_margin"] / (df["cons_12m"] + 1)
X_rich["tenure_months"] = (features.SNAPSHOT - df["date_activ"]).dt.days / 30.4
X_rich["months_to_renewal"] = (df["date_renewal"] - features.SNAPSHOT).dt.days / 30.4

bigger = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=63, min_child_samples=20,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2,
                            verbose=-1, random_state=0)
tests = {"current model, current features": (propensity.make_gbm(), X_base),
         f"current model, {X_rich.shape[1]} features": (propensity.make_gbm(), X_rich),
         f"bigger tuned model, {X_rich.shape[1]} features": (bigger, X_rich)}
rows = {}
for name, (model, X_) in tests.items():
    p_ = cross_val_predict(model, X_, y, cv=cv, method="predict_proba")[:, 1]
    rows[name] = {"AUC": roc_auc_score(y, p_), "avg_precision": average_precision_score(y, p_)}
more = pd.DataFrame(rows).T
more
""")
code(r"""
gain = more["AUC"].iloc[-1] - more["AUC"].iloc[0]
print(f"Doubling the features and using a model ~4x bigger changed AUC by {gain:+.3f}.")
print("The model isn't the bottleneck. The DATA is: what drives an SME to switch supplier (a competitor's offer,")
print("a complaint, a bad bill, a renewal conversation, falling usage) simply isn't in this dataset.")
""")
md(r"""
### How to read this
One row per test, same customers, same 5 folds. If a richer model or more features were the answer, AUC
would jump. It moves by about 0.002, which is noise. (Note the first row differs slightly from the 0.716 above:
it uses a different random split into folds. Differences of ±0.01 between splits are normal.)

**So what do you do with a churn model that can't get better?** Exactly what this case does: stop trying to
predict *who leaves* more precisely, and ask *whose decision can we change* (§6). Even a perfect churn
model would only find lost causes more accurately.
""")
check(("Why is 'accuracy at a 50% cut-off' a bad way to judge this model?",
       "With 9.7% churn, almost nobody is above 50%, so the model 'predicts' almost no leavers, and predicting 'nobody leaves' is already 90% accurate. Ranking (AUC, lift) and probability quality are what matter."),
      ("The learning curve is flat after 50% of the data. What does that tell you?",
       "Collecting more customers with the same columns won't help. Only new kinds of information (complaints, competitor offers, usage trends) would."),
      ("Richer features and a bigger model added ~0.002 AUC. What would you ask the client for?",
       "Behavioural data: contact-centre logs and complaints, billing disputes, usage month by month, renewal-quote history, and which competitor offers reach their region."))
say("The churn model is limited by the information in the data, not the modelling: doubling the features and a much bigger model added 0.002 AUC, and the learning curve is flat. That's why I didn't spend effort on a better churn model and asked a different question instead: whose decision can we change?")

# =================================================================================================
md(r"""
---
## 4. How many euros are at risk?

### The idea in plain words
**Expected value** = probability × amount. If a customer worth €200 a year has a 30% chance of leaving,
the expected loss is 0.3 × €200 = **€60**. Add it up across customers and you get the euros at risk.
""")
words(("Expected value", "Probability × amount: the average outcome if the situation repeated many times."),
      ("Margin at risk", "Sum over customers of P(churn) × yearly margin."))
code(r"""
toy = pd.DataFrame({"customer": ["A", "B", "C"], "p_churn": [0.30, 0.05, 0.60], "yearly_margin_eur": [200, 2000, 50]})
toy["expected_loss_eur"] = toy["p_churn"] * toy["yearly_margin_eur"]
print(toy)
print(f"\nTotal at risk: EUR {toy['expected_loss_eur'].sum():.0f}. Notice B: low risk but big margin, so B matters most.")
""")
code(r"show(risk.portfolio_summary)")
md(r"""
**Walkthrough:** `margin_at_risk` is literally `p_churn × net_margin` per customer. The summary adds them
up and also reports the total margin and the expected number of leavers (the sum of all probabilities:
100 customers at 10% each means about 10 expected leavers). This is why the probabilities had to be
calibrated in §3: every number here scales with them.
""")
code(r"risk.portfolio_summary(df).to_frame('value')")
code(r"""
ps = risk.portfolio_summary(df)
print(f"Expected leavers next quarter: {ps['expected_churners']:,.0f} customers.")
print(f"Margin at risk: EUR {ps['margin_at_risk_eur']:,.0f} out of EUR {ps['annual_net_margin_eur']:,.0f} "
      f"({ps['margin_at_risk_share']:.1%}).")
""")
code(r"plots.concentration(risk.concentration(df));")
md(r"""
### How to read this chart
Customers are sorted from most € at risk to least. **x-axis:** share of customers counted so far.
**y-axis:** share of all € at risk they hold. The dashed diagonal is what you'd see if risk were spread
evenly. The higher the curve bows above it, the more concentrated the risk.

**Key accounts.** The top 1% of customers by margin (about 146) get a named account manager in real
companies, not a campaign. Two of them are huge (€10k and €24k margin). They're excluded from the
targeting below (`optimise.campaign_eligible`), otherwise luck with two customers would swamp the result.
""")
check(("Customer X: 50% risk, €40 margin. Customer Y: 10% risk, €400 margin. Who has more € at risk?",
       "X: 0.5 × 40 = €20. Y: 0.1 × 400 = €40. Y, even though X is five times as likely to leave."))
say("About €294k of €2.8M yearly margin is at risk, and the top 10% of customers hold over half of it.")

# =================================================================================================
md(r"""
---
## 5. Does price drive leaving? (and why the data can't tell us)

### The idea in plain words
The obvious question: if we cut prices, do fewer customers leave? The obvious method: look at whether
customers whose price changed more left more often. The problem is **confounding**.

**Analogy:** ice-cream sales and drownings rise and fall together. Does ice cream cause drowning? No: hot
weather causes both. Hot weather is a **confounder**: it affects both the "cause" and the "outcome" and
creates a fake link between them.
""")
words(("Correlation", "Two things moving together. Says nothing by itself about cause."),
      ("Causation", "Changing one thing changes the other."),
      ("Confounder", "A third thing that drives both, creating a misleading correlation."),
      ("Regression", "A method that estimates how an outcome changes with each feature, holding the others fixed."),
      ("Controls", "Extra features added to a regression to account for confounders. Only works for confounders you *have in the data*."))

md("### Toy example: ice cream and drownings (simulated)")
code(r"""
import statsmodels.formula.api as smf
rng = np.random.default_rng(0)
days = 3650                                                    # ten summers' worth of days
temperature = rng.uniform(5, 35, days)
ice_cream = 20 * temperature + rng.normal(0, 60, days)         # depends on temperature only
drownings = 0.3 * temperature + rng.normal(0, 1, days)         # depends on temperature only
toy = pd.DataFrame({"temperature": temperature, "ice_cream": ice_cream, "drownings": drownings})

naive = smf.ols("drownings ~ ice_cream", toy).fit()
controlled = smf.ols("drownings ~ ice_cream + temperature", toy).fit()
print(f"Naive:      +1000 ice creams -> {1000*naive.params['ice_cream']:+.2f} drownings  (looks like a strong effect)")
print(f"Controlled: +1000 ice creams -> {1000*controlled.params['ice_cream']:+.2f} drownings  (about zero, once temperature is in)")
""")
md(r"""
Here it was easy because we *had* the temperature column. The real problem is when the confounder
**isn't in the data**.

### The real data
Every customer's energy price fell during 2015. Split them into 5 equal groups by how much it fell:
""")
code(r"plots.churn_by_price_change(elasticity.churn_by_price_change(df));")
md(r"""
### How to read this chart
Each bar is one fifth of the customers (a **quintile**). Q1 had the **biggest price cuts** (median shown
underneath), Q5 the smallest. The bar height is the share who left.

Read naively: *the bigger the price cut, the more customers leave.* Nobody believes cutting prices
causes leaving. The likely story is the reverse: **the company cut prices for customers it was already
worried about losing.** The retention team's judgement ("this one sounded unhappy on the phone") is the
confounder, and it's not a column in the data.
""")
code(r"show(elasticity.price_effect_table)")
code(r"""
price_table = elasticity.price_effect_table(elasticity.fit_models(df))
price_table
""")
md(r"""
### How to read this table
Three versions of the same regression, adding more controls each time (`naive` → `+ price level vs
peers` → `+ customer controls`: tenure, products, consumption, margin, contract timing, channel).

| Column | Meaning |
|---|---|
| `odds_ratio_per_1pct` | how the *odds* of leaving change for a 1% price increase. 1.00 = no effect; 1.01 = 1% higher odds |
| `or_ci95` | the 95% **confidence interval**: the range of values consistent with the data. If it contains 1.00, we can't rule out "no effect" |
| `pp_per_1pct` | the same thing in **percentage points** of churn probability per 1% price change |
| `p_value` | roughly: how surprising this result would be if there were truly no effect. Below 0.05 is the usual bar for "significant" |
| `n` | customers used |
""")
code(r"""
t = price_table
print(f"The estimated price effect goes from {t['odds_ratio_per_1pct'].iloc[0]:.3f} to {t['odds_ratio_per_1pct'].iloc[-1]:.3f} "
      f"as controls are added: it changes direction, and no version is significant "
      f"(p-values {t['p_value'].min():.2f} to {t['p_value'].max():.2f}).")
print("Conclusion: this data cannot tell us whether a price offer keeps a customer. We need an experiment.")
""")
check(("Why can't we just add more controls until the price effect is right?",
       "Controls only fix confounders we can measure. The retention team's decision of whom to re-price isn't in the data, so no regression can remove it."),
      ("What's the reverse-causality story here?",
       "The causal arrow runs from 'likely to leave' to 'got a price cut', not the other way round."))
say("The biggest price cuts went to the customers who then left most, which is reverse causality: the company cut prices for customers it was already losing. So I stopped trusting observational data and designed an experiment.")

# =================================================================================================
md(r"""
---
## 6. Whose decision can an offer change? (experiment + uplift)

### 6.1 The four kinds of customer
For any one offer, each customer is one of four types. We never see which, because each customer
only lives one version of reality, but it's the key idea of this section:
""")
code(r"""
types = pd.DataFrame({
    "type": ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"],
    "without the offer": ["leaves", "stays", "leaves", "stays"],
    "with the offer":    ["stays", "stays", "leaves", "LEAVES"],
    "should we target?": ["YES - the only ones the offer helps", "no - money wasted", "no - money wasted",
                          "NEVER - the offer makes them leave"],
})
types
""")
md(r"""
A **churn model** finds people who will leave without an offer: persuadables **and lost causes, mixed
together**. An **uplift model** tries to find only the persuadables. "Sleeping dogs" sounds odd, but it's
real: phoning a loyal customer who never thinks about their contract can remind them to shop around.
""")
words(("Experiment / A-B test / RCT", "Randomly split customers into groups and treat each group differently. RCT = randomised controlled trial."),
      ("Control group", "The group that gets no offer, for comparison."),
      ("Treatment", "What a group receives (here: discount, or call)."),
      ("Potential outcomes", "What *would* happen to a customer with and without the offer. We only ever observe one of the two."),
      ("Uplift / treatment effect", "Churn without the offer minus churn with it: how much the offer *reduces* leaving."),
      ("ATE", "Average treatment effect: uplift averaged over everyone."))

md(r"""
### 6.2 Why randomise? Toy example
If we let account managers choose whom to call, they'd pick certain customers, and any difference in
results could be due to *who* they picked. Random assignment (a coin flip per customer) makes the
groups alike in **everything**, measured or not.
""")
code(r"""
rng = np.random.default_rng(1)
toy = pd.DataFrame({"tenure": rng.integers(1, 14, 2000), "margin": rng.gamma(2, 100, 2000)})
toy["group"] = np.where(rng.random(2000) < 0.5, "offer", "no offer")   # the coin flip
toy.groupby("group")[["tenure", "margin"]].mean()
""")
md("""Both groups have almost identical average tenure and margin, without us trying. That's what makes any later difference in churn *caused* by the offer.""")

md(r"""
### 6.3 The pilot in this project (SIMULATED)
We don't have the energy company's experiment data, so the pilot is **simulated on the real customers**:
- each customer keeps their **real** churn outcome as "what happens with no offer";
- a real leaver who gets an offer is **persuaded to stay** with a probability that we **plant**
  (price-sensitive customers respond to the discount, relationship customers respond to the call);
- a call makes some loyal, never-contacted customers leave (the planted **sleeping dogs**).

Because we planted the truth, we can later **check** whether the methods find it. The methods themselves
never see these planted numbers, only the randomised outcomes, exactly like a real experiment.
""")
code(r"show(simulate.persuasion_rates)")
md(r"""
**Walkthrough:** `gap_rank` is each customer's price gap turned into a percentile (0 = cheapest, 1 = most
expensive relative to peers). The discount persuades `0.10 + 0.50 × gap_rank² + 0.20 if tenure ≤ 4`: the
most over-charged, newer customers are easiest to win back with a discount. The call persuades
`0.15 + 0.40 if multi-product + 0.20 if high margin`: relationship customers. `s_call` is the sleeping-dog
rate: 8% of loyal, single-product, never-changed customers who would have stayed are pushed out by a call.
""")
code(r"pd.Series(simulate.PLANTED).to_frame('planted value')")
code(r"""
po = simulate.run_pilot(df)
truth = simulate.true_effects(df, po)
po.loc[po["in_pilot"], ["arm", "y_none", "y_discount", "y_call", "y_observed"]].head(6)
""")
md(r"""
### How to read this table
One row per customer in the pilot. `arm` is the coin flip (`none`, `discount`, `call`). `y_none`,
`y_discount`, `y_call` are the three **potential outcomes** (1 = leaves), which only a simulation knows.
`y_observed` is the one outcome a real company would see: the one matching the arm.
""")

md(r"""
### 6.4 Did each offer work on average? ATE by hand
Formula for the difference between two percentages, with its uncertainty:
- effect = churn(no offer) − churn(offer)
- **standard error** SE = √( p₀(1−p₀)/n₀ + p₁(1−p₁)/n₁ ): how much the effect would wobble if we re-ran the experiment
- **95% confidence interval** = effect ± 1.96 × SE
- **z** = effect ÷ SE. If |z| > 1.96, the effect is "significant at 5%".
""")
code(r"""
p0, n0 = 0.11, 2400     # toy: 11% churn without the offer, 2,400 customers
p1, n1 = 0.06, 2400     # 6% with the offer
effect = p0 - p1
se = np.sqrt(p0*(1-p0)/n0 + p1*(1-p1)/n1)
print(f"effect = {effect:.3f} ({100*effect:.1f} percentage points)")
print(f"SE     = {se:.4f}")
print(f"95% CI = {100*(effect-1.96*se):.1f} to {100*(effect+1.96*se):.1f} pp")
print(f"z      = {effect/se:.1f}  (> 1.96, so significant)")
""")
code(r"show(uplift.ate_table)")
code(r"""
ate = uplift.ate_table(po)
ate
""")
md(r"""
### How to read this table
One row per offer. `churn_with_offer` / `churn_without`: churn in that arm and in the no-offer arm.
`effect_pp`: the drop in churn in percentage points. `ci95_low_pp` / `ci95_high_pp`: the confidence interval.
`z`: effect ÷ standard error.
""")
code(r"""
for offer in ate.index:
    r = ate.loc[offer]
    print(f"{offer:8s}: cuts churn by {r['effect_pp']:.1f} points (95% CI {r['ci95_low_pp']:.1f} to {r['ci95_high_pp']:.1f}), z = {r['z']:.1f}")
plots.pilot_effects(ate);
""")
md(r"""
**How to read the chart:** each dot is the estimated effect, the line through it is the 95% CI. The vertical
line is zero ("no effect"). Both intervals sit right of zero: both offers work on average. A standard A/B
readout would stop here and say "roll out the discount". That would be expensive, because the average
hides *who* the offers work for.
""")

md("### 6.5 The average hides the real story")
code(r"""
seg = uplift.segment_table(df, po, "relationship")
plots.segment_effects(seg, "relationship", "call", "The call helps relationship customers, and backfires on loyal ones");
seg
""")
md(r"""
### How to read this
The pilot split by customer type. `effect_pp` > 0 means the offer reduced churn; < 0 means it *increased*
churn. `ci95_pp` is the ± half-width of the confidence interval. In the chart, a dot left of zero is an
offer that backfires.
""")
code(r"""
c = seg[seg["offer"] == "call"].set_index("relationship")["effect_pp"]
print(f"The call: {c['multi-product']:+.1f} points for multi-product customers, "
      f"{c['loyal, single product, never changed']:+.1f} for loyal never-contacted customers "
      f"(negative = it made them leave).")
print(f"The average call effect ({ate.loc['call','effect_pp']:.1f} points) hides both.")
""")

md(r"""
### 6.6 From segments to individual customers: T-learner and X-learner
Segments are clear but coarse. An **uplift model** estimates each customer's own effect.

**T-learner ("two models"):** train one churn model on the no-offer group and one on the offer group.
For any customer, uplift = prediction from model 1 − prediction from model 2.

**X-learner:** a refinement in three steps that's more stable when leaving is rare. Toy with 12 customers
and one feature (`multi_product`), using simple group averages as the "models":
""")
code(r"""
toy = pd.DataFrame({
    "multi_product": [0, 0, 0, 1, 1, 1,   0, 0, 0, 1, 1, 1],
    "got_offer":     [0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1],
    "churned":       [1, 0, 0, 1, 1, 0,   1, 0, 0, 0, 0, 0],
})
# Step 1 (this is also the T-learner): a 'model' per group = average churn for each feature value
mu0 = toy[toy.got_offer == 0].groupby("multi_product")["churned"].mean()   # without offer
mu1 = toy[toy.got_offer == 1].groupby("multi_product")["churned"].mean()   # with offer
print("T-learner uplift by group (mu0 - mu1):"); print((mu0 - mu1).rename("uplift"))

# Step 2: impute each customer's individual effect using the OTHER group's model
toy["imputed_effect"] = np.where(
    toy.got_offer == 1,
    toy.multi_product.map(mu0) - toy.churned,   # treated: predicted without - what happened
    toy.churned - toy.multi_product.map(mu1))   # control: what happened - predicted with
# Step 3: learn the imputed effects from features (here: average them per feature value) and blend
x_uplift = toy.groupby("multi_product")["imputed_effect"].mean()
print("\nX-learner uplift by group:"); print(x_uplift.rename("uplift"))
""")
md(r"""
Both say the offer helps multi-product customers (by 0.67) and does nothing for the others. With simple
group averages the two methods give the same answer. The X-learner earns its keep with real models and
real data: it uses *both* groups to estimate each effect, which makes it steadier when leaving is rare
and the groups are small. With real data, the "models" are small boosted trees on all 27 features
instead of group averages. Here's the real code:
""")
code(r"show(uplift.x_learner)")
code(r"""
tau_d = uplift.x_learner(df, po, "discount")
tau_c = uplift.x_learner(df, po, "call")
rollout = ~po["in_pilot"]
for offer, est in [("discount", tau_d), ("call", tau_c)]:
    r = np.corrcoef(est[rollout], truth.loc[rollout, f"tau_{offer}"])[0, 1]
    print(f"{offer:8s}: correlation between estimated and planted effect = {r:.2f}  (1 = perfect, 0 = useless)")
""")
md(r"""
The model was fitted on the pilot half and applied to the other half (`rollout`). It doesn't recover the
truth perfectly (that's honest: ~2,400 customers per group is not much), but it's clearly informative.

### 6.7 Does ranking by uplift beat ranking by risk? The Qini curve
**Toy:** 10 customers sorted by some score. Walk down the list and count customers *actually saved*:
""")
code(r"""
toy = pd.DataFrame({"customer": list("ABCDEFGHIJ"),
                    "saved_if_offered": [1, 1, 0, 1, 0, 0, 1, 0, 0, -1]})   # -1 = a sleeping dog
toy["saved_so_far"] = toy["saved_if_offered"].cumsum()
toy["random_order"] = np.linspace(toy["saved_if_offered"].sum() / 10, toy["saved_if_offered"].sum(), 10)
print(toy)
plt.figure(figsize=(5, 3)); plt.plot(range(1, 11), toy["saved_so_far"], "o-", color="#2a78d6", label="this ranking")
plt.plot(range(1, 11), toy["random_order"], "--", color="#8a8984", label="random order")
plt.xlabel("customers targeted (best first)"); plt.ylabel("customers saved"); plt.legend(); plt.show()
""")
md("""A good ranking climbs fast at the start (persuadables first) and may fall at the end (sleeping dogs last). Now the real one for the call:""")
code(r"""
realised_call = (po["y_none"] - po["y_call"]).to_numpy()[rollout]
plots.qini({
    "rank by uplift (X-learner)": uplift.qini_curve(tau_c[rollout], realised_call),
    "rank by churn risk": uplift.qini_curve(df.loc[rollout, "p_churn"].to_numpy(), realised_call),
    "rank by planted truth": uplift.qini_curve(truth.loc[rollout, "tau_call"].to_numpy(), realised_call),
}, "Call: who should we phone first?");
""")
md(r"""
### How to read this chart
**x-axis:** share of customers phoned, best-ranked first. **y-axis:** customers actually saved so far.
**Grey dashed:** a random order. **Green:** perfect knowledge (the ceiling). **Blue:** our uplift model.
**Orange:** the usual practice, phoning the riskiest first. Blue climbs far higher than orange: risk
ranking wastes calls on lost causes and wakes up sleeping dogs. (For the *discount*, risk ranking does
nearly as well, because that effect grows with risk. Uplift matters most when an action can backfire.)
""")
check(("What does a churn model miss that an uplift model captures?",
       "A churn model can't tell persuadables from lost causes (both leave without an offer), and it can't see sleeping dogs at all."),
      ("Why must the pilot be randomised?",
       "So the groups are alike in everything. Then any difference in churn is caused by the offer, not by who was chosen."),
      ("The call's average effect is +2.5 points. Should we call everyone?",
       "No. It's +9 for multi-product customers and −6 for loyal untouched ones. Calling everyone throws away some of the gain and actively loses loyal customers."))
say("Both offers work on average, but the call cuts churn about 9 points for multi-product customers and raises it about 6 for loyal ones we'd never contacted. The A/B average hides both, so I modelled uplift per customer.")

# =================================================================================================
md(r"""
---
## 7. Who gets what, within the budget? (optimisation)

### The idea in plain words
For each customer and each offer we now know (estimated): how much the offer cuts their chance of
leaving, how much they're worth, and what the offer costs. Turn it into one number:

**value = (drop in churn probability × yearly margin) − expected cost of the offer**

Then choose, for every customer, *nothing*, *discount* or *call*, to get the most total value without
spending more than the budget. Choosing items under a spending limit is the classic **knapsack
problem** (what to pack in a bag that can only carry so much).
""")
words(("Optimisation", "Finding the best choice among many, under rules (constraints)."),
      ("Constraint", "A rule the solution must respect: here, total spend ≤ budget, and at most one offer per customer."),
      ("Integer program (MILP)", "An optimisation where choices are yes/no. Solved exactly by `scipy.optimize.milp`."))
code(r"""
toy = pd.DataFrame({"customer": ["A", "B", "C"], "yearly_margin": [400, 400, 100],
                    "p_churn": [0.40, 0.05, 0.40], "uplift_discount": [0.15, 0.01, 0.15], "uplift_call": [0.10, 0.01, 0.10]})
# discount cost = 25% of margin x probability they stay (paid by everyone who stays)
toy["discount_cost"] = DISCOUNT_MARGIN_SHARE * toy["yearly_margin"] * (1 - toy["p_churn"] + toy["uplift_discount"])
toy["discount_value"] = toy["uplift_discount"] * toy["yearly_margin"] - toy["discount_cost"]
toy["call_cost"] = CALL_COST_EUR
toy["call_value"] = toy["uplift_call"] * toy["yearly_margin"] - toy["call_cost"]
toy
""")
md(r"""
### How to read this
- **A** (risky, valuable, persuadable): the call is worth +€10; the discount *loses* money because it costs
  25% of the margin for everyone who stays.
- **B** (would stay anyway): both offers lose money. A sure thing.
- **C** (risky but small): too small to be worth even a €30 call.

That's why "give the riskiest customers a discount" fails: the discount's cost lands on everyone who stays.

### Toy knapsack: brute force vs the solver
Four customers, two offers each, a €70 budget. With only 3⁴ = 81 combinations we can try them all, then
check the solver finds the same best answer.
""")
code(r"""
import itertools
econ = pd.DataFrame({"spend_discount": [40, 10, 25, 30], "value_discount": [50, -3, 20, 35],
                     "spend_call":     [30, 30, 30, 30], "value_call":     [45, 12, -8, 31]})
best, best_combo = -np.inf, None
for combo in itertools.product(["none", "discount", "call"], repeat=len(econ)):
    spend = sum(econ.at[i, f"spend_{a}"] for i, a in enumerate(combo) if a != "none")
    value = sum(econ.at[i, f"value_{a}"] for i, a in enumerate(combo) if a != "none")
    if spend <= 70 and value > best:
        best, best_combo = value, combo
print(f"brute force: {best_combo}  value = {best}")
print(f"solver:      {tuple(optimise.solve(econ, 70))}")
""")
code(r"show(optimise.solve)")
md(r"""
**Walkthrough:** each customer gets two yes/no variables (discount? call?). `c=-value` because `milp`
*minimises*, so minimising minus value maximises value. Constraint 1 (`one_offer_each`): each customer's
two variables add up to at most 1. Constraint 2: total spend ≤ budget. `Bounds(0, upper)` forbids any offer
whose value is negative, because it would lose money even with budget to spare. With ~14,000 variables
the solver still finishes in well under a second.

### The real decision
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
md(r"""
### How to read this table
Four ways to spend the same budget (rows), all judged by the **planted truth** (what would really
happen, which only a simulation can know):

| Column | Meaning |
|---|---|
| `discounts`, `calls` | how many of each offer the policy gives |
| `customers_saved` | expected leavers who stay because of the offer, minus sleeping dogs pushed out |
| `sleeping_dogs_woken` | loyal customers the policy pushed to leave |
| `spend_eur` | what the offers cost |
| `margin_saved_eur` | yearly margin kept thanks to the offers |
| `net_value_eur` | margin saved − spend: **the bottom line** |
| `roi` | net value ÷ spend: euros returned per euro spent |

"Riskiest first" is common practice: sort by churn risk and give everyone the same offer until the money
runs out. "Planted truth" is the best possible with perfect knowledge, a ceiling no real method can reach.
""")
code(r"""
c = comparison["net_value_eur"]
print(f"Same EUR {budget:,.0f} budget:")
for k, v in c.items():
    print(f"  {k:28s} EUR {v:>9,.0f} net")
print(f"\nUplift + optimiser earns {c['Uplift + optimiser'] / max(c['Riskiest first, discount'], c['Riskiest first, call']):.1f}x "
      f"the best riskiest-first list, and {c['Uplift + optimiser'] / c['Planted truth (ceiling)']:.0%} of the ceiling.")
plots.policy_comparison(comparison);
""")

md(r"""
### How much should we spend? The winner's curse
**Toy first.** 1,000 customers, and the offer *truly* does nothing for any of them. Our estimates are just
noise. We pick the 10 with the highest estimates:
""")
code(r"""
rng = np.random.default_rng(3)
true_effect = np.zeros(1000)
estimate = true_effect + rng.normal(0, 1, 1000)        # pure noise
top10 = np.argsort(-estimate)[:10]
print(f"Average ESTIMATED effect of the 10 we picked: {estimate[top10].mean():+.2f}")
print(f"Average TRUE effect of the 10 we picked:      {true_effect[top10].mean():+.2f}")
""")
md(r"""
Picking the biggest estimates also picks the biggest **over**-estimates. That's the **winner's curse**.
The more customers the budget lets us pick, the deeper we go into estimates that are mostly noise:
""")
code(r"""
curve = optimise.budget_curve(econ, R, P, [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000])
plots.budget_curve(curve, chosen_budget=budget);
curve
""")
md(r"""
### How to read this
**x-axis:** campaign budget. **Orange:** the value the model *promises* (from its own estimates).
**Green:** the value the planted truth *delivers*. The gap between them is the model's over-optimism.
""")
code(r"""
peak = curve.loc[curve["true_net_value_eur"].idxmax()]
print(f"True value peaks at a budget of EUR {peak['budget_eur']:,.0f} (EUR {peak['true_net_value_eur']:,.0f} net). "
      f"At EUR 30,000 the model promises EUR {curve['estimated_net_value_eur'].iloc[-1]:,.0f} "
      f"but delivers EUR {curve['true_net_value_eur'].iloc[-1]:,.0f}.")
print("Recommendation: spend about EUR 7.5-10k, with a random hold-out group to measure what really happens.")
""")
md("### ▶ Try it: change the budget and re-run")
code(r"""
BUDGET = 5_000   # <- change me (try 2_000, 10_000, 20_000) and press Shift+Enter

policy = optimise.solve(econ, BUDGET)
pd.DataFrame({
    "uplift + optimiser": optimise.evaluate(policy, R, P),
    "riskiest first, call": optimise.evaluate(optimise.rank_and_fill(R["p_churn"], econ["spend_call"], BUDGET, "call"), R, P),
}).T[["discounts", "calls", "customers_saved", "spend_eur", "net_value_eur", "roi"]]
""")
check(("Why does the optimiser give so few discounts?",
       "Because a discount costs 25% of margin for every customer who stays, including those who'd have stayed anyway. It only pays for a few high-risk, persuadable, high-margin customers."),
      ("If the model says 'spend more and you'll earn more', why not believe it?",
       "The winner's curse: the extra customers it picks are the ones with the most over-estimated effects. The true value peaks and then falls."))
say("Same €10k: riskiest-first returns about €3k, uplift plus an exact optimiser returns €11k. And the budget curve told me when to stop: past about €10k the model's promised gains are mostly noise.")

# =================================================================================================
md(r"""
---
## 8. GenAI: turning the list into briefs people act on

### The idea in plain words
An account manager needs two readable sentences per customer, not a table of probabilities. A
**large language model (LLM)**, the technology behind ChatGPT and Claude, is good at writing such text.
It's also known to **hallucinate**: write fluent, confident sentences containing made-up facts. So the
design is: **the LLM writes the words, but every number and promise is checked by plain code** before
anyone sees it.
""")
words(("LLM", "A model trained on huge amounts of text to predict the next word. It writes fluent text."),
      ("Prompt", "The instructions and information you give the LLM."),
      ("Hallucination", "When an LLM states something false or invented as if it were fact."),
      ("Grounding", "Giving the LLM only the facts it may use, plus strict rules, to reduce hallucination."),
      ("Open-weights / local model", "A model you download and run on your own computer (here Qwen2.5-3B-Instruct). Customer data never leaves the machine."),
      ("Regular expression (regex)", "A pattern for finding text, e.g. 'any number'. Used here to pull every number out of a brief."))

md(r"""
### The two prompts that were compared
The model wrote a brief for 40 targeted customers, twice: once with an **open** prompt and once with a
**grounded** one. Here are both prompts, exactly as sent, for the first customer:
""")
code(r"""
briefs = genai_briefs.load()
first = briefs[briefs["customer"] == briefs["customer"].iloc[0]].set_index("condition")
print("=== OPEN PROMPT ===\n" + first.loc["open", "prompt"])
print("\n=== GROUNDED PROMPT ===\n" + first.loc["grounded", "prompt"])
""")
code(r"show(genai_briefs.prompt_grounded)")
md(r"""
The grounded prompt gives a numbered **fact sheet**, explicit **rules** ("every number must appear in the
fact sheet"), and a **fixed shape** for the answer. The facts come from `genai_briefs.fact_sheet`, which
takes them straight from the model outputs (churn probability, margin, the optimiser's chosen offer, the
SHAP reasons), so the LLM never has to compute anything. The generation ran once, offline, on a GPU
(`python -m src.genai_briefs`); this notebook loads the saved text and does the checking live.

### How the checker works: toy example
""")
code(r"""
toy_facts = {"customer": "cb5171895e4a15e2", "churn_probability_pct": 32.4, "annual_net_margin_eur": 865,
             "recommended_offer": "call", "offer_terms": "a proactive account-manager call (costs about EUR 30)"}
toy_brief = ("Why at risk: 32% churn probability. Action: book a proactive call. "
             "Worth it because: EUR 865 margin, and they've been loyal for 12 years.")
print("Numbers found in the brief:", factcheck.numbers_in(toy_brief))
print("Numbers allowed (from the facts):", factcheck.allowed_numbers(toy_facts))
print("Check result:", {k: v for k, v in factcheck.check(toy_brief, toy_facts).items() if k != "unsupported_numbers"})
print("Invented:", factcheck.check(toy_brief, toy_facts)["unsupported_numbers"], " <- '12 years' isn't in the facts")
""")
md(r"""
`32%` passes (it's 32.4 rounded), `865` passes, `30` isn't used, `12` is caught: the model made it up.
""")
code(r"show(factcheck.check)")
md(r"""
**Walkthrough:** (1) find every number in the text and keep those that don't match a fact (allowing
rounding); (2) check the approved offer is mentioned; (3) check no discount is promised when the
approved offer was a call ("no discount needed" is fine: negation is understood); (4) check for unfilled
template placeholders like `[Account Manager's Name]`; (5) check it isn't too long to be useful. A brief
passes only if all five are clean. No second LLM judges the first: the checker is plain, testable code.
""")
code(r"""
checked = factcheck.check_all(briefs)
summary = factcheck.summary(checked)
plots.factcheck_bars(summary);
summary
""")
md(r"""
### How to read this
One row per prompt style, 40 briefs each. `pass_rate`: share usable as-is. The other columns: the share
of briefs with each kind of error. In the chart, orange is the open prompt and blue the grounded one.
""")
code(r"""
s = summary
print(f"Usable as-is: grounded {s.loc['grounded','pass_rate']:.0%} vs open {s.loc['open','pass_rate']:.0%}.")
print(f"Open prompt left a [placeholder] in {s.loc['open','placeholder_rate']:.0%} of briefs and was too long in {s.loc['open','too_long_rate']:.0%}.")
print(f"But BOTH invented a number in {s.loc['grounded','invented_number_rate']:.0%} (grounded) and "
      f"{s.loc['open','invented_number_rate']:.0%} (open) of briefs. Grounding fixes the format, not the facts.")
for _, r in checked[(checked["condition"] == "grounded") & checked["has_unsupported_number"]].iterrows():
    print(f"\n  invented {r['unsupported_numbers']} in a grounded brief:\n  {r['text']}")
""")
check(("Why not let a second LLM check the first one?",
       "It could hallucinate too, and it doesn't give the same answer every time. For numbers and commitments we want a check that is deterministic and tested."),
      ("Grounding made 95% of briefs usable. Why keep the checker?",
       "Because grounded briefs still invented numbers about 1 time in 20 (by doing their own arithmetic). The checker catches those before an account manager reads them."))
say("The LLM writes the language, never the numbers. Grounding made 95% of briefs usable versus 18%, but both styles still invented a number one time in twenty, so a deterministic checker sits between the model and the people.")

# =================================================================================================
md(r"""
---
## 9. What it all means

### Limitations (what this does NOT show)
- **The pilot is simulated.** The euro results show the method finding a known truth; they're not a
  forecast of a real company's numbers. The next real step is a properly sized pilot.
- **A saved customer is worth 12 months of margin** here. Long-standing customers are probably worth more.
- **One snapshot**, so the churn model couldn't be tested on a *later* period, the most honest test.
- **Uplift estimates are noisy** with ~2,400 customers per group; the budget curve shows the cost of that.
- **The fact-checker** checks numbers, offers and format, not tone or whether the reasons make sense.

### Recommendation
1. Rank retention targets by **expected euros saved net of cost**, not by churn risk.
2. **Call relationship customers; never phone loyal, untouched single-product ones.** Discounts only for a
   few high-risk, price-sensitive, high-margin customers.
3. Spend **~€7.5–10k per quarter** with a random hold-out group, not the larger amount the model suggests.
4. **Run the real pilot first.** This notebook is the analysis plan for it.

### Glossary (everything in one place)
| Word | Meaning |
|---|---|
| ATE | Average treatment effect: how much an offer changes the outcome on average |
| AUC | Probability the model scores a random leaver above a random stayer (0.5 random, 1 perfect) |
| Brier score | Average squared gap between predicted probability and what happened |
| Calibration | Whether predicted probabilities match real frequencies |
| Churn | A customer leaving |
| Confidence interval | The range of values consistent with the data (95% = the usual level) |
| Confounder | Something that drives both a "cause" and an "outcome", creating a misleading link |
| Cross-validation | Repeatedly training on part of the data and testing on the rest |
| Expected value | Probability × amount |
| Feature | A column used to make a prediction |
| Gradient boosting | Many small decision trees in sequence, each fixing the last ones' errors |
| Hallucination | An LLM stating made-up facts |
| Knapsack problem | Choosing items with values and costs to maximise value within a budget |
| Leakage | Using information you wouldn't have at prediction time |
| Logistic regression | A model that turns a weighted sum of features into a probability |
| Out-of-fold | A prediction made by a model that never saw that row |
| p-value | How surprising a result would be if there were no real effect |
| Persuadable / sure thing / lost cause / sleeping dog | The four customer types for an offer (§6.1) |
| Potential outcomes | What would happen with and without an offer; only one is ever observed |
| Qini curve | Customers saved as you work down a ranked list |
| RCT | Randomised controlled trial: an experiment with random assignment |
| SHAP | Splits a prediction into contributions from each feature |
| T-learner / X-learner | Ways to estimate each customer's uplift from an experiment |
| Uplift | The drop in churn caused by an offer |
| Winner's curse | Picking the largest estimates also picks the largest over-estimates |

### Where next
- `README.md`: the headline numbers and the limitations, stated plainly.
- `src/`: every step of the chain, one module each, with the tests that pin it.
""")

# =================================================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = PROJECT / "notebooks" / "00_sme_energy_retention_explained.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
