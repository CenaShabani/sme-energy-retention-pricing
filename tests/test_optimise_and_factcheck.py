import itertools

import numpy as np
import pandas as pd

from src import factcheck, optimise


def _toy():
    return pd.DataFrame({
        "spend_discount": [40, 10, 25, 5, 30], "value_discount": [50, -3, 20, 4, 35],
        "spend_call": [30, 30, 30, 30, 30], "value_call": [45, 12, -8, 10, 31],
    })


def test_optimiser_matches_brute_force():
    econ, budget = _toy(), 80
    best = 0
    for combo in itertools.product(["none", "discount", "call"], repeat=len(econ)):
        spend = sum(econ.at[i, f"spend_{a}"] for i, a in enumerate(combo) if a != "none")
        value = sum(econ.at[i, f"value_{a}"] for i, a in enumerate(combo) if a != "none")
        if spend <= budget:
            best = max(best, value)
    chosen = optimise.solve(econ, budget)
    got = sum(econ.at[i, f"value_{a}"] for i, a in chosen.items() if a != "none")
    assert np.isclose(got, best)


def test_optimiser_respects_budget_and_never_loses_money():
    econ = _toy()
    chosen = optimise.solve(econ, 50)
    spend = sum(econ.at[i, f"spend_{a}"] for i, a in chosen.items() if a != "none")
    assert spend <= 50
    for i, a in chosen.items():
        if a != "none":
            assert econ.at[i, f"value_{a}"] > 0


FACTS = {
    "customer": "cb5171895e4a15e2c72cbbd7cf054cbf", "churn_probability_pct": 32.4,
    "annual_net_margin_eur": 865, "years_as_customer": 5, "recommended_offer": "call",
    "offer_terms": "a proactive account-manager call, no price change (costs about EUR 30)",
    "expected_net_value_eur": 216, "top_risk_drivers": ["months until contract end: 2.5"],
}
GOOD = ("Why at risk: 32.4% churn probability, contract ends in 2.5 months. "
        "Action: book a proactive account-manager call. Worth it because: EUR 865 margin a year.")


def test_correct_brief_passes():
    assert factcheck.check(GOOD, FACTS)["passes"]


def test_every_injected_number_is_caught():
    rng = np.random.default_rng(0)
    caught = 0
    for _ in range(200):
        fake = int(rng.integers(1000, 99999))
        caught += factcheck.check(GOOD + f" They have been with us since {fake}.", FACTS)["has_unsupported_number"]
    assert caught == 200


def test_rounding_is_not_an_error():
    assert factcheck.check(GOOD.replace("32.4%", "32%"), FACTS)["passes"]


def test_unapproved_discount_is_caught():
    bad = GOOD + " Offer them a 10% discount to stay."
    result = factcheck.check(bad, FACTS)
    assert result["unapproved_discount"] and not result["passes"]


def test_ids_are_not_numbers():
    assert factcheck.numbers_in("customer cb5171895e4a15e2") == []


def test_list_numbering_is_not_a_claim():
    text = "Key points:\n1. **Contract ends soon**\n2. Book a proactive account-manager call.\n" + GOOD
    assert factcheck.check(text, FACTS)["passes"]


def test_digits_copied_from_an_id_are_not_claims():
    facts = {**FACTS, "customer": "93831978e4a15e2c72cbbd7cf054cbf0"}
    assert factcheck.check("Customer 93831978: " + GOOD, facts)["passes"]


def test_derived_numbers_are_flagged():
    # "18% of margin" is arithmetic the model did itself; it is not on the fact sheet
    assert factcheck.check(GOOD + " That is 18% of their margin.", FACTS)["has_unsupported_number"]


def test_saying_no_discount_is_not_a_promise():
    assert factcheck.check(GOOD + " No discount is needed for this customer.", FACTS)["passes"]
