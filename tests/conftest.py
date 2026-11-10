import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def book():
    """The real customer book with features and out-of-fold churn probabilities."""
    from src import acquisition, features, propensity
    clients, prices = acquisition.load()
    df = features.build(clients, prices)
    df["p_churn"] = propensity.out_of_fold(df)["p_gbm_cal"]
    return df


@pytest.fixture(scope="session")
def pilot(book):
    from src import simulate
    return simulate.run_pilot(book)
