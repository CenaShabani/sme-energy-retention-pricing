import numpy as np
import pandas as pd

from src import simulate, uplift


def test_simulation_is_deterministic(book):
    a, b = simulate.run_pilot(book, seed=7), simulate.run_pilot(book, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_discount_never_causes_churn(pilot):
    assert (pilot["y_discount"] <= pilot["y_none"]).all()


def test_sleeping_dogs_only_where_planted(book, pilot):
    woken = (pilot["y_none"] == 0) & (pilot["y_call"] == 1)
    assert not woken[~simulate.sleeping_dog_segment(book)].any()
    assert woken.any()


def test_pilot_is_randomised_and_balanced(book, pilot):
    assert abs(pilot["in_pilot"].mean() - 0.5) < 0.02
    arms = pilot.loc[pilot["in_pilot"], "arm"].value_counts(normalize=True)
    assert (arms.sub(1 / 3).abs() < 0.03).all()
    # randomisation check: baseline churn risk is the same in every arm
    risk = book.loc[pilot["in_pilot"]].groupby(pilot.loc[pilot["in_pilot"], "arm"])["p_churn"].mean()
    assert risk.max() - risk.min() < 0.01


def test_pilot_detects_the_discount(pilot):
    ate = uplift.ate_table(pilot)
    assert ate.loc["discount", "ci95_low_pp"] > 0


def test_x_learner_recovers_planted_effects(book, pilot):
    truth = simulate.true_effects(book, pilot)
    rollout = ~pilot["in_pilot"]
    for offer in ["discount", "call"]:
        est = uplift.x_learner(book, pilot, offer)[rollout.to_numpy()]
        corr = np.corrcoef(est, truth.loc[rollout, f"tau_{offer}"])[0, 1]
        assert corr > 0.4, f"{offer}: correlation with planted truth only {corr:.2f}"
