from src import acquisition, features
from src.config import SNAPSHOT


def test_schema_and_size():
    clients, prices = acquisition.load()
    assert len(clients) == 14_606
    assert clients["churn"].between(0, 1).all()
    assert prices["price_date"].dt.year.eq(2015).all()


def test_no_future_information(book):
    # every product-change date the model can see is on or before the snapshot
    seen = book["months_since_modif"].dropna()
    assert (seen >= 0).all()
    masked = book["date_modif_prod"] > SNAPSHOT
    assert book.loc[masked, "months_since_modif"].isna().all()


def test_model_features_exist(book):
    missing = set(features.MODEL_FEATURES) - set(book.columns)
    assert not missing


def test_probabilities_are_calibrated_on_average(book):
    # calibrated probabilities should add up to roughly the observed number of churners
    assert abs(book["p_churn"].mean() - book["churn"].mean()) < 0.005
