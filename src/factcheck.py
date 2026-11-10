"""A deterministic fact-checker for LLM-written briefs.

The LLM is allowed to write prose; it is not allowed to make up facts. Every number in a brief
must trace back to the fact sheet it was given, it must recommend the offer the optimiser chose,
and it must never promise a discount the optimiser did not approve. The checks are plain Python
and regular expressions, with no second LLM acting as judge, so the check itself can't
hallucinate and gives the same answer every time.
"""
import re

import pandas as pd

# A number standing on its own (not inside an ID such as "24011ae4"), with optional thousands
# separators and decimals.
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z_\d])")
OFFER_WORDS = {
    "discount": re.compile(r"\bdiscount|\b5\s?%\s?off|\brate reduction", re.I),
    "call": re.compile(r"\bcall\b|\bphone\b|\breach out\b|\bcheck-in\b|\bmeeting\b", re.I),
}


# "No rebate approved" or "without a discount" is not a promise. A concession word only counts if
# none of these appear in the few words before it.
NEGATION = re.compile(r"\b(?:no|not|without|none|never|isn't|aren't|won't)\b[^.\n]{0,25}$", re.I)


def promises(pattern: re.Pattern, text: str) -> bool:
    """True if `pattern` occurs at least once without a negation just before it."""
    return any(not NEGATION.search(text[max(0, m.start() - 40):m.start()]) for m in pattern.finditer(text))


# List numbering ("1.", "2)") at the start of a line is layout, not a claim.
LIST_MARKER = re.compile(r"(?m)^[ 	>*#-]*\d{1,2}[.)](?=\s)")
ID_FIELDS = ("customer", "account")
# Format rules: a brief must be usable as-is by the person it is for.
PLACEHOLDER = re.compile(r"\[[A-Z][^\]]{2,40}\]")       # e.g. "[Account Manager's Name]"
MAX_WORDS = 120


def numbers_in(text: str) -> list[float]:
    text = LIST_MARKER.sub("", text)
    return [float(m.replace(",", "")) for m in NUMBER.findall(text)]


def allowed_numbers(facts: dict) -> list[float]:
    """Every number that appears anywhere in the fact sheet (including nested driver values)."""
    out = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out.extend(numbers_in(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(facts)
    return out


def _inside_id(x: float, facts: dict) -> bool:
    """A long digit run copied from an ID (e.g. a hash prefix '93831978') is a reference, not a
    number. Short numbers never qualify: '18' appears inside most hashes by chance."""
    return x == int(x) and len(str(int(x))) >= 5 and any(str(int(x)) in str(facts.get(f, "")) for f in ID_FIELDS)


def _supported(x: float, allowed: list[float]) -> bool:
    """A number is supported if it matches a fact up to rounding (e.g. 23.4% written as 23%)."""
    return any(abs(x - a) <= max(0.51, 0.01 * abs(a)) for a in allowed)


def check(text: str, facts: dict) -> dict:
    allowed = allowed_numbers(facts)
    unsupported = [x for x in numbers_in(text) if not _supported(x, allowed) and not _inside_id(x, facts)]
    offer = facts["recommended_offer"]
    mentions_offer = bool(OFFER_WORDS[offer].search(text))
    # The costly error: promising a discount the optimiser did not approve gives away margin.
    # (A discount brief that also mentions phoning the customer is fine, since someone has to.)
    unapproved_discount = offer != "discount" and promises(OFFER_WORDS["discount"], text)
    placeholder = bool(PLACEHOLDER.search(text))
    too_long = len(text.split()) > MAX_WORDS
    return {
        "unsupported_numbers": unsupported,
        "has_unsupported_number": bool(unsupported),
        "missing_recommended_offer": not mentions_offer,
        "unapproved_discount": unapproved_discount,
        "unfilled_placeholder": placeholder,
        "too_long": too_long,
        "passes": (not unsupported and mentions_offer and not unapproved_discount
                   and not placeholder and not too_long),
    }


def check_all(briefs: pd.DataFrame) -> pd.DataFrame:
    """briefs needs columns `text` and `facts` (a dict per row)."""
    results = [check(t, f) for t, f in zip(briefs["text"], briefs["facts"])]
    return pd.concat([briefs.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def summary(checked: pd.DataFrame, by: str = "condition") -> pd.DataFrame:
    return checked.groupby(by).agg(
        briefs=("passes", "size"),
        pass_rate=("passes", "mean"),
        invented_number_rate=("has_unsupported_number", "mean"),
        missing_offer_rate=("missing_recommended_offer", "mean"),
        unapproved_discount_rate=("unapproved_discount", "mean"),
        placeholder_rate=("unfilled_placeholder", "mean"),
        too_long_rate=("too_long", "mean"),
    ).round(3)
