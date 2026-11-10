"""Get the PowerCo data and load it with a schema check.

The dataset is public on Kaggle (CC0). Kaggle serves public datasets anonymously through its
API download URL, so no token is needed.
"""
import io
import zipfile

import pandas as pd
import requests

from .config import DATA_RAW, KAGGLE_URL

CLIENT_COLUMNS = {
    "id", "channel_sales", "cons_12m", "cons_gas_12m", "cons_last_month", "date_activ",
    "date_end", "date_modif_prod", "date_renewal", "forecast_cons_12m", "forecast_cons_year",
    "forecast_discount_energy", "forecast_meter_rent_12m", "forecast_price_energy_off_peak",
    "forecast_price_energy_peak", "forecast_price_pow_off_peak", "has_gas", "imp_cons",
    "margin_gross_pow_ele", "margin_net_pow_ele", "nb_prod_act", "net_margin",
    "num_years_antig", "origin_up", "pow_max", "churn",
}
PRICE_COLUMNS = {
    "id", "price_date", "price_off_peak_var", "price_peak_var", "price_mid_peak_var",
    "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix",
}
DATE_COLUMNS = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]


def download(force: bool = False) -> None:
    """Download and unzip the dataset into data/raw (skipped if already there)."""
    if (DATA_RAW / "client_data.csv").exists() and not force:
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    resp = requests.get(KAGGLE_URL, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    zipfile.ZipFile(io.BytesIO(resp.content)).extractall(DATA_RAW)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clients, prices), validated. Fails loudly if the schema has drifted."""
    download()
    clients = pd.read_csv(DATA_RAW / "client_data.csv", parse_dates=DATE_COLUMNS)
    prices = pd.read_csv(DATA_RAW / "price_data.csv", parse_dates=["price_date"])

    missing = CLIENT_COLUMNS - set(clients.columns)
    assert not missing, f"client_data is missing {missing}"
    missing = PRICE_COLUMNS - set(prices.columns)
    assert not missing, f"price_data is missing {missing}"
    assert clients["id"].is_unique, "one row per customer expected"
    assert set(clients["churn"].unique()) <= {0, 1}
    assert clients["id"].isin(prices["id"]).all(), "every customer needs a price history"
    return clients, prices
