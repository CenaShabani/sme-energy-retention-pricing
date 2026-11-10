"""Turn the raw client + price files into one row per customer.

Every feature must be knowable on the snapshot date (1 Jan 2016). Nothing here may use
information from after that date. That is the leakage rule, and `leakage_audit` checks it.
"""
import numpy as np
import pandas as pd

from .config import SNAPSHOT

# Columns the churn model is allowed to see.
MODEL_FEATURES = [
    "channel", "campaign", "has_gas", "nb_prod_act", "tenure_years",
    "months_to_end", "months_since_modif", "product_modified",
    "log_cons_12m", "log_cons_gas_12m", "log_cons_last_month", "log_forecast_cons_12m",
    "forecast_discount_energy", "forecast_meter_rent_12m",
    "forecast_price_energy_off_peak", "forecast_price_energy_peak", "forecast_price_pow_off_peak",
    "margin_net_pow_ele", "net_margin", "pow_max",
    "price_energy_dec", "price_power_dec", "price_energy_change", "price_power_change",
    "price_energy_volatility", "price_gap_vs_channel",
]
CATEGORICAL = ["channel", "campaign"]


def _readable_codes(s: pd.Series, prefix: str, min_count: int = 50) -> pd.Series:
    """The source hashes channel/campaign codes. Rename them 'Channel A', 'Channel B'... by size,
    pooling tiny groups into 'Other' so the model can't memorise a handful of customers."""
    counts = s.value_counts()
    names = {}
    letter = iter("ABCDEFGH")
    for code, n in counts.items():
        if code == "MISSING":
            names[code] = f"{prefix} unknown"
        elif n >= min_count:
            names[code] = f"{prefix} {next(letter)}"
        else:
            names[code] = f"{prefix} other"
    return s.map(names)


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per customer summarising their 2015 tariff: the December level, the change over
    the year, and how much it moved month to month."""
    p = prices.sort_values(["id", "price_date"])
    g = p.groupby("id")
    out = pd.DataFrame({
        "price_energy_dec": g["price_off_peak_var"].last(),
        "price_energy_jan": g["price_off_peak_var"].first(),
        "price_power_dec": g["price_off_peak_fix"].last(),
        "price_power_jan": g["price_off_peak_fix"].first(),
        "price_energy_volatility": g["price_off_peak_var"].std(),
    })
    out["price_energy_change"] = out["price_energy_dec"] - out["price_energy_jan"]
    out["price_power_change"] = out["price_power_dec"] - out["price_power_jan"]
    return out


def build(clients: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """The modelling table: one row per customer, readable names, snapshot-safe features."""
    df = clients.merge(price_features(prices), left_on="id", right_index=True, how="left")

    df["channel"] = _readable_codes(df["channel_sales"], "Channel")
    df["campaign"] = _readable_codes(df["origin_up"], "Campaign")
    df["has_gas"] = (df["has_gas"] == "t").astype(int)
    df["tenure_years"] = df["num_years_antig"]
    df["months_to_end"] = (df["date_end"] - SNAPSHOT).dt.days / 30.44
    # 61 customers have a product modification dated *after* the snapshot. That is future
    # information (and those customers churn more), so for them the field is set to unknown.
    future_modif = df["date_modif_prod"] > SNAPSHOT
    df["months_since_modif"] = ((SNAPSHOT - df["date_modif_prod"]).dt.days / 30.44).mask(future_modif)
    df["product_modified"] = (df["date_modif_prod"] != df["date_activ"]).astype(float).mask(future_modif)
    df["multi_product"] = ((df["nb_prod_act"] > 1) | (df["has_gas"] == 1)).astype(int)
    for col in ["cons_12m", "cons_gas_12m", "cons_last_month", "forecast_cons_12m"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # How expensive is this customer's energy rate compared with peers sold through the same
    # channel? A positive gap = paying more than similar customers = a reason to shop around.
    channel_median = df.groupby("channel")["price_energy_dec"].transform("median")
    df["price_gap_vs_channel"] = df["price_energy_dec"] / channel_median - 1

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Check every date the model sees is on or before the snapshot, except contract end dates,
    which are scheduled in advance and therefore known on the snapshot date."""
    handling = {
        "date_activ": "used (tenure)",
        "date_modif_prod": "future dates masked in build()",
        "date_renewal": "scheduled in advance; not used as a feature",
        "date_end": "scheduled in advance; used as months_to_end",
    }
    rows = []
    for col, how in handling.items():
        after = (df[col] > SNAPSHOT)
        rows.append({"column": col, "rows_after_snapshot": int(after.sum()),
                     "churn_rate_if_after": round(df.loc[after, "churn"].mean(), 3) if after.any() else None,
                     "handling": how})
    return pd.DataFrame(rows)


def business_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Two hypothesis-driven segmentations an account team would propose before any modelling:
    - relationship depth: does the customer have more than one product with us, and if not,
      are they a long-standing customer we have never touched?
    - price position: is the customer paying more than peers in the same channel?"""
    loyal_untouched = (df["tenure_years"] >= 6) & (df["multi_product"] == 0) & (df["product_modified"] == 0)
    relationship = np.select(
        [df["multi_product"] == 1, loyal_untouched],
        ["multi-product", "loyal, single product, never changed"], default="other single product")
    price = np.where(df["price_gap_vs_channel"] > 0.02, "pays above channel peers", "at or below peers")
    return pd.DataFrame({"relationship": relationship, "price_position": price}, index=df.index)
