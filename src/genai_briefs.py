"""GenAI step: an LLM drafts a short brief for the account manager of each targeted customer.

Why an LLM here at all: the optimiser produces a list of customers, offers and SHAP drivers.
An account manager needs two readable sentences per customer, not a table. That is exactly the
kind of narrow, low-risk job a small model can do, *if* it sticks to the facts.

Two prompt styles are compared on the same customers:
  "open"     - the facts in a friendly sentence plus "write a brief". The usual first attempt.
  "grounded" - the facts as a numbered fact sheet, explicit rules (only these numbers, this offer,
               no other promises), and a fixed output shape.
The fact-checker (src/factcheck.py) then measures how often each style invents numbers or
promises the wrong thing.

Generation runs ONCE, offline, on a local open-weights model, and the text is saved to
data/llm/briefs.jsonl. The notebook does not call an LLM; it loads that file as an input
dataset and runs the fact-checker live. To regenerate:
    python -m src.genai_briefs
"""
import json

import pandas as pd

from .config import CALL_COST_EUR, DATA_LLM, DISCOUNT_MARGIN_SHARE, SEED

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BRIEFS_PATH = DATA_LLM / "briefs.jsonl"

FEATURE_LABELS = {
    "channel": "sales channel", "campaign": "acquisition campaign", "has_gas": "also buys gas",
    "nb_prod_act": "number of active products", "tenure_years": "years as a customer",
    "months_to_end": "months until contract end", "months_since_modif": "months since last product change",
    "product_modified": "product was changed since sign-up", "log_cons_12m": "electricity use (12 months)",
    "log_cons_gas_12m": "gas use (12 months)", "log_cons_last_month": "electricity use last month",
    "log_forecast_cons_12m": "forecast electricity use", "forecast_discount_energy": "current discount",
    "forecast_meter_rent_12m": "meter rent", "forecast_price_energy_off_peak": "forecast off-peak energy price",
    "forecast_price_energy_peak": "forecast peak energy price", "forecast_price_pow_off_peak": "forecast power price",
    "margin_net_pow_ele": "net margin on power", "net_margin": "total net margin", "pow_max": "subscribed power",
    "price_energy_dec": "current energy price", "price_power_dec": "current power price",
    "price_energy_change": "energy price change in 2015", "price_power_change": "power price change in 2015",
    "price_energy_volatility": "energy price volatility", "price_gap_vs_channel": "price vs channel peers",
    "p_churn": "churn probability",
}

OFFER_TEXT = {
    "discount": f"5% off the energy rate for 12 months (costs about {int(DISCOUNT_MARGIN_SHARE * 100)}% of margin)",
    "call": f"a proactive account-manager call, no price change (costs about EUR {int(CALL_COST_EUR)})",
}


def _driver_value(feature: str, value) -> str:
    """Show driver values in business units (the model uses logs; people don't)."""
    if feature == "price_gap_vs_channel":
        return f"{100 * value:+.0f}% vs peers"
    if feature.startswith("log_"):
        return "high" if value > 8 else "low"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"
    return str(value)


def fact_sheet(row: pd.Series, drivers: list, offer: str, econ_row: pd.Series) -> dict:
    """Everything the LLM is allowed to say about one customer."""
    value = econ_row[f"value_{offer}"]
    return {
        "customer": row.name if isinstance(row.name, str) else str(row.name),
        "churn_probability_pct": round(100 * row["p_churn"], 1),
        "annual_net_margin_eur": round(float(row["net_margin"])),
        "years_as_customer": int(row["tenure_years"]),
        "recommended_offer": offer,
        "offer_terms": OFFER_TEXT[offer],
        "expected_net_value_eur": round(float(value)),
        "top_risk_drivers": [f"{FEATURE_LABELS.get(f, f)}: {_driver_value(f, v)}" for f, v, _ in drivers],
    }


def prompt_open(facts: dict) -> str:
    return (
        f"Our customer {facts['customer'][:8]} has a {facts['churn_probability_pct']}% chance of "
        f"leaving and brings in about EUR {facts['annual_net_margin_eur']} a year. We plan to offer "
        f"{facts['offer_terms']}. Main reasons they may leave: {'; '.join(facts['top_risk_drivers'])}. "
        "Write a short, persuasive retention brief for the account manager."
    )


def prompt_grounded(facts: dict) -> str:
    sheet = "\n".join([
        f"1. Customer: {facts['customer'][:8]}",
        f"2. Churn probability (next 3 months): {facts['churn_probability_pct']}%",
        f"3. Annual net margin: EUR {facts['annual_net_margin_eur']}",
        f"4. Years as customer: {facts['years_as_customer']}",
        f"5. Approved offer: {facts['offer_terms']}",
        f"6. Expected net value of the offer: EUR {facts['expected_net_value_eur']}",
        f"7. Top risk drivers: {'; '.join(facts['top_risk_drivers'])}",
    ])
    return (
        "You write retention briefs for account managers at an energy supplier.\n"
        f"FACT SHEET\n{sheet}\n\n"
        "RULES\n"
        "- Use only facts from the fact sheet. Do not invent numbers, dates, prices or history.\n"
        "- Every number you write must appear in the fact sheet.\n"
        "- Recommend exactly the approved offer (item 5). Do not promise any other discount or perk.\n"
        "- At most 70 words, in this shape:\n"
        "  Why at risk: <one sentence using item 2 and item 7>\n"
        "  Action: <one sentence stating item 5>\n"
        "  Worth it because: <one sentence using item 3 or item 6>"
    )


def generate(fact_sheets: list[dict], model_id: str = MODEL_ID, max_new_tokens: int = 160) -> pd.DataFrame:
    """Run both prompt styles through a local model. Needs torch + transformers (and a GPU for
    speed). Greedy decoding so a rerun gives the same text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto")
    rows = []
    for facts in fact_sheets:
        for condition, build in [("open", prompt_open), ("grounded", prompt_grounded)]:
            prompt = build(facts)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append({"customer": facts["customer"], "condition": condition, "facts": facts,
                         "prompt": prompt, "text": text, "model": model_id})
    return pd.DataFrame(rows)


def save(briefs: pd.DataFrame) -> None:
    DATA_LLM.mkdir(parents=True, exist_ok=True)
    with open(BRIEFS_PATH, "w", encoding="utf-8") as f:
        for rec in briefs.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load() -> pd.DataFrame:
    """The cached briefs: an input dataset like any other, produced by `python -m src.genai_briefs`."""
    return pd.read_json(BRIEFS_PATH, lines=True)


if __name__ == "__main__":
    from .pipeline import run

    result = run()
    sheets = result["fact_sheets"]
    print(f"Generating briefs for {len(sheets)} customers x 2 prompt styles with {MODEL_ID}")
    save(generate(sheets))
    print(f"Saved to {BRIEFS_PATH}")
