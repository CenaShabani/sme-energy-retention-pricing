"""Paths, seeds and every business assumption in one place.

If an interviewer asks "where does that number come from?", the answer is this file.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_LLM = ROOT / "data" / "llm"
OUTPUTS = ROOT / "outputs"

KAGGLE_REF = "erolmasimov/powerco"  # BCG PowerCo SME book, CC0
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"

SEED = 42

# The client file is a snapshot taken in January 2016; `churn` = left in the next 3 months.
SNAPSHOT = pd.Timestamp("2016-01-01")

# ---- Business assumptions for the retention decision (stated, not estimated) ----
# Offer 1: 5% off the energy rate for 12 months. With net margin at roughly 20% of the bill,
# 5% of the bill is about 25% of the customer's annual net margin. Paid only by customers who stay.
DISCOUNT_MARGIN_SHARE = 0.25
# Offer 2: a proactive account-manager call: ~30 minutes at ~EUR 60/hour, fully loaded.
CALL_COST_EUR = 30.0
# Value of keeping a customer = the next 12 months of net margin (deliberately conservative:
# no lifetime value, no win-back).
ACTIONS = ("none", "discount", "call")

# The pilot: a random half of the book is used to run the (simulated) randomised test;
# the decision is then made for the other half.
PILOT_SHARE = 0.5

DEFAULT_BUDGET_EUR = 10_000.0

# Key accounts (top 1% by net margin, ~146 customers, ~9% of margin) are handled by a named
# account manager, not by a campaign. They are excluded from the targeting decision.
KEY_ACCOUNT_QUANTILE = 0.99
