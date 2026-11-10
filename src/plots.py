"""Every figure in the notebook. One function per figure, each returning the matplotlib Figure.

Colour roles are fixed across the project so a reader learns them once:
  BLUE   = the recommended approach (uplift + optimiser, calibrated model)
  ORANGE = the common practice it is compared against (riskiest-first, raw model)
  AQUA   = the planted truth / ceiling
  GREY   = reference lines (random, perfect calibration, zero)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 10,
    "legend.frameon": False,
})


def _eur(x, _=None):
    """€2.5k, €10k, €722: never round a tick label into a different number."""
    return f"€{x/1000:g}k" if abs(x) >= 1000 else f"€{x:,.0f}"


def calibration(cal_raw: pd.DataFrame, cal_cal: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    top = max(cal_raw[["predicted", "actual"]].max().max(), cal_cal[["predicted", "actual"]].max().max())
    ax.plot([0, top], [0, top], color=GREY, lw=1, ls="--", label="perfect calibration")
    ax.plot(cal_raw["predicted"], cal_raw["actual"], "o-", color=ORANGE, lw=2, ms=6, label="raw GBM")
    ax.plot(cal_cal["predicted"], cal_cal["actual"], "o-", color=BLUE, lw=2, ms=6, label="calibrated GBM")
    ax.set(xlabel="predicted churn probability (decile average)", ylabel="actual churn rate",
           title="Calibration: do the probabilities mean what they say?")
    ax.legend(loc="upper left")
    return fig


def concentration(conc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(conc["share_of_customers"], conc["share_of_margin_at_risk"], color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    for q in (0.1, 0.2):
        y = np.interp(q, conc["share_of_customers"], conc["share_of_margin_at_risk"])
        ax.plot(q, y, "o", color=BLUE, ms=7)
        ax.annotate(f"top {q:.0%} of customers\n= {y:.0%} of € at risk", (q, y), xytext=(12, -28),
                    textcoords="offset points", color=INK)
    ax.set(xlabel="share of customers (most at risk first)", ylabel="share of margin at risk",
           title="Margin at risk is concentrated")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return fig


def churn_by_price_change(t: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(t))
    ax.bar(x, 100 * t["churn_rate"], color=ORANGE, width=0.6)
    for i, (rate, med) in enumerate(zip(t["churn_rate"], t["median_change_pct"])):
        ax.text(i, 100 * rate + 0.2, f"{100*rate:.1f}%", ha="center", color=INK)
    ax.set_xticks(x, [f"{lab}\n({m:+.1f}%)" for lab, m in zip(t.index, t["median_change_pct"])])
    ax.set_ylim(0, 100 * t["churn_rate"].max() * 1.15)
    ax.set(xlabel="2015 energy-price change, quintile (median change)", ylabel="churn rate (%)",
           title="Customers with the biggest price CUTS churned most")
    ax.grid(axis="x", visible=False)
    return fig


def pilot_effects(ate: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    y = np.arange(len(ate))
    ax.errorbar(ate["effect_pp"], y, xerr=[ate["effect_pp"] - ate["ci95_low_pp"],
                                           ate["ci95_high_pp"] - ate["effect_pp"]],
                fmt="o", color=BLUE, ms=8, capsize=5, lw=2)
    ax.axvline(0, color=GREY, lw=1)
    for yi, (e, lo, hi) in enumerate(zip(ate["effect_pp"], ate["ci95_low_pp"], ate["ci95_high_pp"])):
        ax.text(hi + 0.25, yi, f"{e:.1f} pp  [{lo:.1f}, {hi:.1f}]", va="center", color=INK)
    ax.set_yticks(y, ate.index)
    ax.set_ylim(-0.6, len(ate) - 0.4)
    ax.set_xlim(min(-1, ate["ci95_low_pp"].min() - 0.5), ate["ci95_high_pp"].max() + 4)
    ax.set(xlabel="churn reduction vs no offer (percentage points, 95% CI)",
           title="Pilot: average effect of each offer")
    return fig


def segment_effects(seg: pd.DataFrame, by: str, offer: str, title: str):
    s = seg[seg["offer"] == offer].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 0.75 * len(s) + 1.4))
    y = np.arange(len(s))
    ax.errorbar(s["effect_pp"], y, xerr=s["ci95_pp"], fmt="o", color=BLUE, ms=8, capsize=5, lw=2)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y, s[by].astype(str))
    ax.set(xlabel=f"churn reduction from the {offer} (pp, 95% CI)", title=title)
    return fig


def qini(curves: dict, title: str):
    """curves: label -> qini DataFrame. The first entry is drawn in BLUE, the second ORANGE,
    a third in AQUA."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    colours = [BLUE, ORANGE, AQUA]
    for (label, c), col in zip(curves.items(), colours):
        ax.plot(c["share_targeted"], c["customers_saved"], color=col, lw=2, label=label)
    first = next(iter(curves.values()))
    ax.plot(first["share_targeted"], first["random"], color=GREY, lw=1, ls="--", label="random order")
    ax.set(xlabel="share of customers targeted (best first)", ylabel="customers saved (cumulative)",
           title=title)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="lower right")
    return fig


def policy_comparison(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    order = comp["net_value_eur"].sort_values().index
    colours = [AQUA if "truth" in k else BLUE if "Uplift" in k else ORANGE for k in order]
    ax.barh(order, comp.loc[order, "net_value_eur"], color=colours, height=0.6)
    for i, k in enumerate(order):
        v = comp.at[k, "net_value_eur"]
        ax.text(v + 150, i, f"€{v:,.0f} net  ({comp.at[k, 'roi']:.1f}x ROI)", va="center", color=INK)
    ax.axvline(0, color=GREY, lw=1)
    ax.xaxis.set_major_formatter(_eur)
    ax.set_xlim(right=comp["net_value_eur"].max() * 1.55)
    ax.set(xlabel="net value = margin kept - offer cost (same budget)",
           title="Same budget, very different returns")
    ax.grid(axis="y", visible=False)
    return fig


def budget_curve(curve: pd.DataFrame, chosen_budget: float | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["budget_eur"], curve["estimated_net_value_eur"], "o-", color=ORANGE, lw=2, ms=6,
            label="what the model promises")
    ax.plot(curve["budget_eur"], curve["true_net_value_eur"], "o-", color=AQUA, lw=2, ms=6,
            label="what the planted truth delivers")
    best = curve.loc[curve["true_net_value_eur"].idxmax()]
    ax.annotate(f"true value peaks at {_eur(best['budget_eur'])} budget", (best["budget_eur"],
                best["true_net_value_eur"]), xytext=(-40, -60), textcoords="offset points", color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    if chosen_budget is not None:
        ax.axvline(chosen_budget, color=GREY, lw=1, ls=":")
    ax.xaxis.set_major_formatter(_eur)
    ax.yaxis.set_major_formatter(_eur)
    ax.set(xlabel="campaign budget", ylabel="net value", title="How much should we spend?")
    ax.legend(loc="upper left")
    return fig


def shap_bar(shap_df: pd.DataFrame, labels: dict, k: int = 12):
    imp = shap_df.abs().mean().sort_values().tail(k)
    fig, ax = plt.subplots(figsize=(6, 0.32 * k + 1.2))
    ax.barh([labels.get(c, c) for c in imp.index], imp.to_numpy(), color=BLUE, height=0.6)
    ax.set(xlabel="mean |SHAP| (log-odds of churn)", title="What drives the churn model")
    ax.grid(axis="y", visible=False)
    return fig


def factcheck_bars(summary: pd.DataFrame):
    cols = ["invented_number_rate", "missing_offer_rate", "unapproved_discount_rate", "placeholder_rate", "too_long_rate"]
    names = ["invented a number", "missed the approved action", "promised an unapproved discount",
             "left a [placeholder] unfilled", "too long to use (>120 words)"]
    fig, ax = plt.subplots(figsize=(6.6, 4))
    y = np.arange(len(cols))
    h = 0.36
    for j, (cond, col) in enumerate([("open", ORANGE), ("grounded", BLUE)]):
        if cond in summary.index:
            vals = 100 * summary.loc[cond, cols].to_numpy(dtype=float)
            ax.barh(y + (j - 0.5) * h, vals, height=h - 0.04, color=col, label=f"{cond} prompt")
            for yi, v in zip(y, vals):
                ax.text(v + 1, yi + (j - 0.5) * h, f"{v:.0f}%", va="center", color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set(xlabel="share of briefs with the error", title="LLM briefs: errors caught by the fact-checker")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return fig
