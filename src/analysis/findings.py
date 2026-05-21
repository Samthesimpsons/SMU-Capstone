"""Plot, table, and SVG-export renderers backing notebooks/findings.ipynb."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RISK_BAND_NAMES: dict[int, str] = {
    0: "Conservative",
    1: "Income",
    2: "Balanced",
    3: "Aggressive",
}
BANDS_ORDER: list[str] = ["Conservative", "Income", "Balanced", "Aggressive"]
BAND_COLOURS: dict[str, str] = {
    "Conservative": "#1f77b4",
    "Income": "#2ca02c",
    "Balanced": "#ff7f0e",
    "Aggressive": "#d62728",
}
MODEL_DISPLAY_MAP: dict[str, str] = {
    "light_gcn": "LightGCN",
    "random_forest": "Random Forest",
}
METRIC_COLUMNS: list[str] = [
    "ndcg_at_k",
    "roi_at_k",
    "recall_at_k",
    "profile_coherence_at_k",
    "profile_coherence_lift_at_k",
]
PRETTY_METRIC: dict[str, str] = {
    "ndcg_at_k": "nDCG@10",
    "roi_at_k": "ROI@10 (mo.)",
    "recall_at_k": "Recall@10",
    "profile_coherence_at_k": "PC@10",
    "profile_coherence_lift_at_k": "PC-lift@10",
}
PC_METRIC_ORDER: list[str] = [
    "PC-lift@10",
    "PC@10",
    "ROI@10 (mo.)",
    "Recall@10",
    "nDCG@10",
]
CONTRAST_DESCRIPTIONS: dict[str, str] = {
    "λ=0 vs baseline": "Stratification effect",
    "λ=1 vs baseline": "Stratification + PC-loss effect",
    "λ=1 vs λ=0 (PC-loss)": "PC-loss effect (matched architecture)",
}
CONTRAST_COLOURS: dict[str, str] = {
    "λ=0 vs baseline": "#4C72B0",
    "λ=1 vs baseline": "#DD8452",
    "λ=1 vs λ=0 (PC-loss)": "#55A868",
}
CONTRAST_ORDER: list[str] = list(CONTRAST_DESCRIPTIONS.keys())
SIGNIFICANCE_ALPHA: float = 0.05

PLOT_RC: dict[str, Any] = {
    "figure.figsize": (8, 4.5),
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


@dataclass(frozen=True)
class NotebookPaths:
    """Locations of every precomputed artefact rendered by findings.ipynb."""

    eda_directory: Path
    rq2_directory: Path
    rq3_decomp_directory: Path
    rq3_panel_directory: Path
    rq4_results_directory: Path
    rq4_baseline_directory: Path

    @classmethod
    def from_root(cls, root: Path) -> NotebookPaths:
        """Resolve the default thesis-run paths under the project root."""
        return cls(
            eda_directory=root / "outputs/eda",
            rq2_directory=root
            / "outputs/analysis/transaction_return_regression/20260504_004849",
            rq3_decomp_directory=root
            / "outputs/analysis/baseline_decomposition/20260427_122215",
            rq3_panel_directory=root
            / "outputs/analysis/panel_regression/20260504_003145",
            rq4_results_directory=root
            / "outputs/results/evaluation/pc_lgcn/20260505_125105",
            rq4_baseline_directory=root
            / "outputs/results/evaluation/light_gcn/20260427_122215/eb788_00006",
        )


def configure_matplotlib() -> None:
    """Apply the thesis plot defaults to matplotlib's global rcParams."""
    plt.rcParams.update(PLOT_RC)


def load_eda_summary(eda_directory: Path) -> dict[str, Any]:
    """Read outputs/eda/summary.json into a plain dict."""
    with open(eda_directory / "summary.json") as fh:
        return json.load(fh)


def _format_p_value(p: float) -> str:
    """Render a p-value with scientific notation below 1e-3, fixed otherwise."""
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _finalise(fig: plt.Figure, save_path: Path | None) -> None:
    """Either render inline or write `fig` to `save_path` as SVG and close it."""
    if save_path is None:
        plt.show()
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_rq1_discordance_distribution(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Bar chart of pairwise discordance d in {0,1,2,3} across all scoreable Buys."""
    discordance = eda_summary["transaction_discordance_summary"]
    counts = discordance["discordance_counts"]
    total = discordance["transactions_with_both_bands"]

    distribution = pd.DataFrame(
        [{"d": int(k), "count": v, "share": v / total} for k, v in counts.items()]
    ).sort_values("d")
    distribution["label"] = distribution["d"].map(
        {
            0: "d=0 (exact match)",
            1: "d=1 (within tolerance)",
            2: "d=2",
            3: "d=3",
        }
    )

    print(
        f"Total scoreable Buys:      {total:,}\n"
        f"Coherent (d<=1) share:     {discordance['fraction_coherent_default']:.1%}\n"
        f"Strict-coherent (d=0):     {discordance['fraction_coherent_strict']:.1%}\n"
        f"Mean profile discordance:  {discordance['mean_discordance']:.3f} bands"
    )

    fig, ax = plt.subplots()
    colours = ["#2ca02c", "#9bcd9b", "#ff9900", "#d62728"]
    ax.bar(distribution["label"], distribution["share"], color=colours)
    for x, value in enumerate(distribution["share"]):
        ax.text(x, value + 0.01, f"{value:.1%}", ha="center", va="bottom")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Share of scoreable Buys")
    ax.set_title(f"Discordance distribution across {total:,} Buy transactions")
    ax.grid(axis="x", visible=False)

    coherent_share = discordance["fraction_coherent_default"]
    discoherent_share = 1.0 - coherent_share
    _draw_coherence_bracket(
        ax, 0, 1, 0.56, f"Coherent (d≤1): {coherent_share:.1%}", "#1f4e1f"
    )
    _draw_coherence_bracket(
        ax, 2, 3, 0.56, f"Discoherent (d≥2): {discoherent_share:.1%}", "#7a1f1f"
    )

    ax.text(
        0.98,
        0.97,
        f"Mean profile discordance: {discordance['mean_discordance']:.3f} bands",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="#888", alpha=0.9
        ),
    )

    plt.tight_layout()
    _finalise(fig, save_path)


def _draw_coherence_bracket(
    ax: plt.Axes, x_left: float, x_right: float, y: float, label: str, colour: str
) -> None:
    ax.plot([x_left, x_right], [y, y], color=colour, linewidth=1.5)
    ax.plot([x_left, x_left], [y - 0.015, y], color=colour, linewidth=1.5)
    ax.plot([x_right, x_right], [y - 0.015, y], color=colour, linewidth=1.5)
    ax.text(
        (x_left + x_right) / 2,
        y + 0.015,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color=colour,
        fontweight="bold",
    )


def plot_rq1_self_discordance(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Histogram of per-customer discordant share; demonstrates the bimodal customer trait."""
    self_disc = eda_summary["customer_self_discordance_summary"]
    populations = eda_summary["populations"]
    transaction_summary = eda_summary["transaction_discordance_summary"]

    population_discordance_rate = 1 - transaction_summary["fraction_coherent_default"]
    mean_buys_per_scoreable_customer = (
        populations["transactions_with_both_bands"] / populations["customers_with_band"]
    )

    print(
        f"Population discordance rate (per Buy):                 {population_discordance_rate:.1%}\n"
        f"Mean Buys per customer      (with a known band):       {mean_buys_per_scoreable_customer:.2f}\n"
        f"Fully coherent customers    (every Buy in tolerance):  {self_disc['fraction_fully_coherent']:.1%}\n"
        f"Fully discordant customers  (every Buy out of toler.): {self_disc['fraction_fully_discordant']:.1%}"
    )

    edges = self_disc["discordant_share_histogram_edges"]
    counts = np.array(self_disc["discordant_share_histogram_counts"], dtype=float)
    total_customers = counts.sum()
    shares = counts / total_customers
    bin_labels = [
        f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}%" for i in range(len(counts))
    ]

    bar_colours = ["#cccccc"] * len(counts)
    bar_colours[0] = "#2ca02c"
    bar_colours[-1] = "#d62728"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(counts))
    ax.bar(positions, shares, color=bar_colours, edgecolor="white")
    for x, value in zip(positions, shares):
        ax.text(x, value + 0.01, f"{value:.1%}", ha="center", va="bottom", fontsize=9)

    ax.annotate(
        f"Fully coherent (exactly 0%): {self_disc['fraction_fully_coherent']:.1%}",
        xy=(positions[0], shares[0]),
        xytext=(positions[0] + 1.8, shares[0] + 0.10),
        arrowprops=dict(arrowstyle="-", linestyle=":", color="#1f4e1f", linewidth=1.2),
        fontsize=9,
        color="#1f4e1f",
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.annotate(
        f"Fully discordant (exactly 100%): {self_disc['fraction_fully_discordant']:.1%}",
        xy=(positions[-1], shares[-1]),
        xytext=(positions[-1] - 1.8, shares[-1] + 0.30),
        arrowprops=dict(arrowstyle="-", linestyle=":", color="#7a1f1f", linewidth=1.2),
        fontsize=9,
        color="#7a1f1f",
        fontweight="bold",
        ha="right",
        va="center",
    )

    mean_x_position = population_discordance_rate * len(counts) - 0.5
    ax.axvline(mean_x_position, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        mean_x_position + 0.1,
        max(shares) * 0.55,
        f"Population mean ({population_discordance_rate:.1%})\n",
        fontsize=9,
        va="top",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=20)
    ax.set_ylim(0, max(shares) * 1.3)
    ax.set_ylabel("Share of customers")
    ax.set_xlabel("Per-customer discordant share (fraction of Buys with d >= 2)")
    ax.set_title("Per-customer self-discordance is bimodal")
    ax.grid(axis="x", visible=False)

    ax.text(
        0.98,
        0.97,
        f"Mean Buys per customer: {mean_buys_per_scoreable_customer:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="#888", alpha=0.9
        ),
    )

    plt.tight_layout()
    _finalise(fig, save_path)


def plot_rq1_per_band_coherence(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Per-band coherent share showing the U-shape on Conservative and Aggressive extremes."""
    risk_level_counts = eda_summary["transaction_discordance_by_risk_level"]
    rows = []
    for band_label, by_d in risk_level_counts.items():
        total_band = sum(by_d.values())
        coherent_band = by_d.get("0", 0) + by_d.get("1", 0)
        rows.append(
            {
                "band_label": band_label,
                "transactions": total_band,
                "coherent_share": coherent_band / total_band if total_band else 0,
            }
        )
    band_table = pd.DataFrame(rows)
    band_table["band_label"] = pd.Categorical(
        band_table["band_label"], categories=BANDS_ORDER, ordered=True
    )
    band_table = band_table.sort_values("band_label")

    fig, ax = plt.subplots()
    colours = [BAND_COLOURS[label] for label in band_table["band_label"]]
    ax.bar(
        band_table["band_label"].astype(str),
        band_table["coherent_share"],
        color=colours,
    )
    for x, (count, share) in enumerate(
        zip(band_table["transactions"], band_table["coherent_share"])
    ):
        ax.text(
            x,
            share + 0.01,
            f"{share:.1%}\nn={count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coherent share (d <= 1)")
    ax.set_title("Per-band coherent share: U-shape on the ordinal extremes")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    _finalise(fig, save_path)


def plot_rq1_yearly_discordance(
    eda_summary: dict[str, Any], save_path: Path | None = None
) -> None:
    """Per-year mean discordance line chart with the 2018 MiFID II elevation flagged."""
    year_disc = eda_summary["mean_discordance_by_year"]
    buy_coverage = eda_summary["buy_coverage_by_year"]
    years = sorted(year_disc)
    year_values = [year_disc[y] for y in years]
    mean_across_years = float(np.mean(year_values))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        years,
        year_values,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Per-year mean",
        zorder=3,
    )
    ax.axhline(
        mean_across_years,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"Average across years = {mean_across_years:.2f}",
    )

    for x, year in enumerate(years):
        coverage = buy_coverage[year]
        partial_flag = " *" if _is_partial_year(coverage) else ""
        ax.annotate(
            f"n = {coverage['count']:,}{partial_flag}\n{year_values[x]:.2f}",
            xy=(year, year_values[x]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    partial_years = [y for y in years if _is_partial_year(buy_coverage[y])]
    caption = (
        " * partial year ("
        + ", ".join(
            f"Cutt off date: {buy_coverage[y]['last_date']}" for y in partial_years
        )
        + ")"
    )
    ax.text(0.01, -0.18, caption, transform=ax.transAxes, fontsize=8, color="#666666")

    ax.set_ylim(0.6, 1.15)
    ax.set_ylabel("Mean discordance (bands)")
    ax.set_xlabel("Calendar year")
    ax.set_title("Discordance is stable across the macro window")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    _finalise(fig, save_path)


def _is_partial_year(coverage: dict[str, Any]) -> bool:
    last = pd.Timestamp(coverage["last_date"])
    return (last.month, last.day) != (12, 31)


def load_rq2_artefacts(rq2_directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the RQ2 coefficients dataframe and the run summary dict."""
    coefficients = pd.read_csv(rq2_directory / "coefficients.csv")
    with open(rq2_directory / "summary.json") as fh:
        rq2_summary = json.load(fh)
    return coefficients, rq2_summary


_RQ2_DISPLAY_ORDER: list[str] = [
    "Intercept",
    "is_coherent",
    "asset_volatility",
    "C(customer_type)[T.Legal Entity]",
    "C(customer_type)[T.Mass]",
    "C(customer_type)[T.Premium]",
    "C(customer_type)[T.Professional]",
    "C(year)[T.2019]",
    "C(year)[T.2020]",
    "C(year)[T.2021]",
    "C(year)[T.2022]",
]
_RQ2_TERM_EXPLANATIONS: dict[str, str] = {
    "Intercept": "Baseline = Inactive customer, 2018, all numeric regressors = 0",
    "is_coherent": "Coherence premium after controlling for vol, segment, year",
    "asset_volatility": "Return penalty per 1.0 unit of annualised vol (negative = low-vol won)",
    "C(customer_type)[T.Legal Entity]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Mass]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Premium]": "Segment effect vs Inactive reference",
    "C(customer_type)[T.Professional]": "Segment effect vs Inactive reference",
    "C(year)[T.2019]": "Calendar effect vs 2018 reference (pre-COVID equity recovery)",
    "C(year)[T.2020]": "Calendar effect vs 2018 reference (COVID rebound)",
    "C(year)[T.2021]": "Calendar effect vs 2018 reference (continued post-COVID recovery)",
    "C(year)[T.2022]": "Calendar effect vs 2018 reference (bond / rate-shock drawdown)",
}


def _rq2_filtered_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    filtered = coefficients[coefficients["term"].isin(_RQ2_DISPLAY_ORDER)].copy()
    return filtered.set_index("term").reindex(_RQ2_DISPLAY_ORDER).reset_index()


def style_rq2_coefficient_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Return the human-readable display table for the RQ2 regression coefficients."""
    filtered = _rq2_filtered_coefficients(coefficients)
    display_df = filtered.assign(
        estimate=lambda d: d["estimate"].map("{:+.4f}".format),
        std_error=lambda d: d["std_error"].map("{:.4f}".format),
        ci=lambda d: d.apply(
            lambda r: f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]", axis=1
        ),
        p_value=lambda d: d["p_value"].map(_format_p_value),
        meaning=lambda d: d["term"].map(_RQ2_TERM_EXPLANATIONS),
    )
    return display_df[["term", "estimate", "std_error", "ci", "p_value", "meaning"]]


def plot_rq2_forest(
    coefficients: pd.DataFrame,
    rq2_summary: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """Forest plot of every RQ2 regression coefficient with 95% CIs in pp."""
    coherent_mean = rq2_summary["mean_realised_return_coherent"]
    discordant_mean = rq2_summary["mean_realised_return_discordant"]
    raw_slice_gap_pp = (coherent_mean - discordant_mean) * 100

    filtered = _rq2_filtered_coefficients(coefficients)
    plot_df = filtered.copy()
    plot_df["significant"] = plot_df["p_value"] < SIGNIFICANCE_ALPHA
    plot_df["estimate_pp"] = plot_df["estimate"] * 100
    plot_df["ci_lower_pp"] = plot_df["ci_lower"] * 100
    plot_df["ci_upper_pp"] = plot_df["ci_upper"] * 100
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    y_positions = np.arange(len(plot_df))
    sig_colour = "#1f4e1f"
    ns_colour = "#888888"
    point_colours = [sig_colour if s else ns_colour for s in plot_df["significant"]]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for y, row, colour in zip(y_positions, plot_df.itertuples(), point_colours):
        ax.errorbar(
            row.estimate_pp,
            y,
            xerr=[
                [row.estimate_pp - row.ci_lower_pp],
                [row.ci_upper_pp - row.estimate_pp],
            ],
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=1.6,
            capsize=4,
            markersize=7,
        )
        label = (
            f"{row.estimate_pp:+.2f} pp  "
            f"(SE {row.std_error * 100:.2f}, "
            f"p={_format_p_value(row.p_value)})"
        )
        ax.text(
            row.ci_upper_pp + 0.6,
            y,
            label,
            va="center",
            ha="left",
            fontsize=8.5,
            color=colour,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["term"], fontsize=9)
    ax.set_xlabel("Coefficient estimate (pp on 6-month realised return)")
    fig.suptitle(
        "realised_return ~ is_coherent + asset_volatility + C(customer_type) + C(year)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(
        min(plot_df["ci_lower_pp"].min(), 0) - 2,
        plot_df["ci_upper_pp"].max() + 28,
    )

    summary_rows = [
        (
            "Mean realised return (overall):",
            f"{rq2_summary['mean_realised_return_overall']:+.3%}",
        ),
        ("Mean realised return (coherent):", f"{coherent_mean:+.3%}"),
        ("Mean realised return (discordant):", f"{discordant_mean:+.3%}"),
        ("Raw slice gap (coh. - disc.):", f"{raw_slice_gap_pp:+.2f} pp"),
        ("Full regression terms:", f"{len(coefficients)}"),
        ("References absorbed in Intercept:", "2018, Inactive"),
    ]
    label_width = max(len(row_label) for row_label, _ in summary_rows) + 2
    value_width = max(len(row_value) for _, row_value in summary_rows)
    summary_text = "\n".join(
        f"{row_label:<{label_width}}{row_value:>{value_width}}"
        for row_label, row_value in summary_rows
    )
    ax.text(
        0.985,
        0.97,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45", facecolor="white", edgecolor="#888", alpha=0.95
        ),
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=sig_colour,
            markersize=8,
            label=f"Significant (p < {SIGNIFICANCE_ALPHA})",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=ns_colour,
            markersize=8,
            label=f"Not significant (p >= {SIGNIFICANCE_ALPHA})",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    _finalise(fig, save_path)


def style_rq3_main_results(rq3_decomp_directory: Path) -> Any:
    """Headline metrics table for the two FAR-Trans baselines at their primary-metric optima."""
    main_results = pd.read_csv(rq3_decomp_directory / "main_results.csv")
    display_columns = [
        "display_name",
        "best_trial_id",
        "primary_metric",
        "ndcg_at_k_mean",
        "roi_at_k_mean",
        "recall_at_k_mean",
        "profile_coherence_at_k_mean",
        "profile_coherence_lift_at_k_mean",
    ]
    renamed = main_results[display_columns].rename(
        columns={
            "display_name": "Model",
            "best_trial_id": "Best trial",
            "primary_metric": "Primary metric",
            "ndcg_at_k_mean": "nDCG@10",
            "roi_at_k_mean": "ROI@10 (mo.)",
            "recall_at_k_mean": "Recall@10",
            "profile_coherence_at_k_mean": "PC@10",
            "profile_coherence_lift_at_k_mean": "PC-lift@10",
        }
    )
    return renamed.style.format(
        {
            "nDCG@10": "{:.3f}",
            "ROI@10 (mo.)": "{:+.4f}",
            "Recall@10": "{:.3f}",
            "PC@10": "{:.3f}",
            "PC-lift@10": "{:.3f}",
        }
    )


def build_rq3_per_band_summary(
    rq3_panel_directory: Path, eda_summary: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-(band, model) mean PC@10 plus the band-conditional random baseline pi(b)."""
    panel = pd.read_csv(rq3_panel_directory / "panel.csv")
    pi_series = _compute_pi_series(eda_summary)

    summary = panel.groupby(["model", "band_label"], as_index=False)[
        "coherent_share"
    ].agg(mean_pc="mean", n_observations="count")
    summary["model_display"] = summary["model"].map(MODEL_DISPLAY_MAP)
    summary["band_label"] = pd.Categorical(
        summary["band_label"], categories=BANDS_ORDER, ordered=True
    )
    summary["pi_b"] = summary["band_label"].astype(str).map(pi_series.to_dict())
    summary["lift"] = (summary["mean_pc"] / summary["pi_b"]).round(3)
    summary = summary.sort_values(["model_display", "band_label"]).reset_index(
        drop=True
    )
    return summary, pi_series


def _compute_pi_series(eda_summary: dict[str, Any]) -> pd.Series:
    asset_band_dist = eda_summary["asset_band_distribution_by_category"]
    total_per_band: dict[str, int] = {band: 0 for band in RISK_BAND_NAMES.values()}
    for band_counts in asset_band_dist.values():
        for band_label, count in band_counts.items():
            total_per_band[band_label] += count
    total_assets = sum(total_per_band.values())
    pi: dict[str, float] = {}
    for band_idx, label in RISK_BAND_NAMES.items():
        coherent_count = sum(
            total_per_band[other_label]
            for other_idx, other_label in RISK_BAND_NAMES.items()
            if abs(other_idx - band_idx) <= 1
        )
        pi[label] = coherent_count / total_assets
    return pd.Series(pi).reindex(BANDS_ORDER)


def style_rq3_per_band_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the band/model PC@10 summary in the column order shown in the notebook."""
    return summary[
        ["band_label", "model_display", "n_observations", "mean_pc", "pi_b", "lift"]
    ]


def plot_rq3_per_band(
    summary: pd.DataFrame,
    pi_series: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart of mean PC@10 by band/model, with pi(b) reference lines."""
    models = list(summary["model_display"].drop_duplicates())
    tab10 = plt.get_cmap("tab10")
    model_colours = {model: tab10(i) for i, model in enumerate(models)}

    x_positions = np.arange(len(BANDS_ORDER))
    group_width = 0.78
    bar_width = group_width / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_handles = []
    for i, model in enumerate(models):
        rows = (
            summary[summary["model_display"] == model]
            .set_index("band_label")
            .reindex(BANDS_ORDER)
            .reset_index()
        )
        offsets = -group_width / 2 + bar_width / 2 + i * bar_width
        xs = x_positions + offsets
        bars = ax.bar(
            xs,
            rows["mean_pc"],
            width=bar_width * 0.95,
            color=model_colours[model],
            label=model,
            edgecolor="white",
        )
        bar_handles.append(bars)
        for x, pc, lift in zip(xs, rows["mean_pc"], rows["lift"]):
            ax.text(
                x,
                pc + 0.012,
                f"Mean PC: {pc:.1%}\nPC Lift: {lift:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=model_colours[model],
                fontweight="bold",
            )

    for j, band in enumerate(BANDS_ORDER):
        pi_val = pi_series[band]
        left = x_positions[j] - group_width / 2
        right = x_positions[j] + group_width / 2
        ax.hlines(pi_val, left, right, colors="black", linestyles="--", linewidth=1.4)
        ax.text(
            right,
            pi_val - 0.008,
            f"pi_b={pi_val:.1%}",
            va="top",
            ha="right",
            fontsize=8,
            color="black",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(BANDS_ORDER)
    ax.set_ylabel("Mean profile-coherence rate (PC@k)")
    ax.set_ylim(0, max(summary["mean_pc"].max(), pi_series.max()) * 1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Mean profile-coherence by customer band and model",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Averaged across 69 monthly splits; PC Lift = mean PC / random-baseline pi_b",
        fontsize=9,
        style="italic",
    )

    baseline_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Random baseline pi_b",
    )
    ax.legend(
        handles=[*bar_handles, baseline_handle],
        loc="upper left",
        fontsize=9,
        title="Model",
    )
    plt.tight_layout()
    _finalise(fig, save_path)


def load_rq4_per_split(
    rq4_baseline_directory: Path, rq4_results_directory: Path
) -> dict[str, pd.DataFrame]:
    """Per-split metric tables for the LightGCN baseline and the two stratified trials."""
    return {
        "baseline": _read_per_split(
            rq4_baseline_directory / "per_split_metrics.csv", "baseline"
        ),
        "lambda_0": _read_per_split(
            rq4_results_directory / "stratified_lambda_0.0/per_split_metrics.csv",
            "lambda_0",
        ),
        "lambda_1": _read_per_split(
            rq4_results_directory / "stratified_lambda_1.0/per_split_metrics.csv",
            "lambda_1",
        ),
    }


def _read_per_split(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trial"] = label
    return df


def style_rq4_aggregate(per_split: dict[str, pd.DataFrame]) -> Any:
    """Aggregate-mean metrics across 69 splits for baseline, λ=0, λ=1."""
    baseline = per_split["baseline"]
    lambda_0 = per_split["lambda_0"]
    lambda_1 = per_split["lambda_1"]

    aggregate = pd.DataFrame(
        {
            "Configuration": [
                "LightGCN baseline (RQ3)",
                "Stratified, λ=0 (ablation)",
                "Stratified, λ=1 (treatment)",
            ],
            "nDCG@10": [
                baseline["ndcg_at_k"].mean(),
                lambda_0["ndcg_at_k"].mean(),
                lambda_1["ndcg_at_k"].mean(),
            ],
            "ROI@10 (mo.)": [
                baseline["roi_at_k"].mean(),
                lambda_0["roi_at_k"].mean(),
                lambda_1["roi_at_k"].mean(),
            ],
            "Recall@10": [
                baseline["recall_at_k"].mean(),
                lambda_0["recall_at_k"].mean(),
                lambda_1["recall_at_k"].mean(),
            ],
            "PC@10": [
                baseline["profile_coherence_at_k"].mean(),
                lambda_0["profile_coherence_at_k"].mean(),
                lambda_1["profile_coherence_at_k"].mean(),
            ],
            "PC-lift@10": [
                baseline["profile_coherence_lift_at_k"].mean(),
                lambda_0["profile_coherence_lift_at_k"].mean(),
                lambda_1["profile_coherence_lift_at_k"].mean(),
            ],
        }
    )

    print("Aggregate metrics: mean across 69 evaluation splits, per configuration")
    return aggregate.style.format(
        {
            "nDCG@10": "{:.4f}",
            "ROI@10 (mo.)": "{:+.4f}",
            "Recall@10": "{:.4f}",
            "PC@10": "{:.4f}",
            "PC-lift@10": "{:.3f}",
        }
    )


def compute_rq4_paired_table(per_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-format mean Δ and win rate for each (contrast, metric) pair."""
    paired = pd.concat(
        [
            _paired_summary(
                per_split["lambda_0"], per_split["baseline"], "λ=0 vs baseline"
            ),
            _paired_summary(
                per_split["lambda_1"], per_split["baseline"], "λ=1 vs baseline"
            ),
            _paired_summary(
                per_split["lambda_1"], per_split["lambda_0"], "λ=1 vs λ=0 (PC-loss)"
            ),
        ],
        ignore_index=True,
    )
    paired["metric"] = paired["metric"].map(PRETTY_METRIC)
    return paired


def _paired_summary(
    treatment: pd.DataFrame, control: pd.DataFrame, label: str
) -> pd.DataFrame:
    rows = []
    for column in METRIC_COLUMNS:
        delta = treatment[column].to_numpy() - control[column].to_numpy()
        rows.append(
            {
                "contrast": label,
                "metric": column,
                "mean_delta": float(delta.mean()),
                "win_rate": float((delta > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def style_rq4_paired_table(paired_table: pd.DataFrame) -> pd.DataFrame:
    """Pivoted display of mean_delta and win_rate for the three RQ4 contrasts."""
    formatted = paired_table.assign(
        mean_delta=lambda d: d["mean_delta"].map("{:+.5f}".format),
        win_rate=lambda d: d["win_rate"].map("{:.0%}".format),
    )
    return formatted.pivot(
        index="contrast",
        columns="metric",
        values=["mean_delta", "win_rate"],
    )


def plot_rq4_paired_deltas(
    paired_table: pd.DataFrame, save_path: Path | None = None
) -> None:
    """Per-metric grouped barh of mean Δ across the three contrasts."""
    fig, axes = plt.subplots(1, len(PC_METRIC_ORDER), figsize=(22, 4.5), sharey=True)

    for axis, metric in zip(axes, PC_METRIC_ORDER):
        subset = paired_table[paired_table["metric"] == metric].set_index("contrast")
        deltas = np.array([subset.loc[c, "mean_delta"] for c in CONTRAST_ORDER])
        y_positions = np.arange(len(CONTRAST_ORDER))
        bar_colors = [CONTRAST_COLOURS[c] for c in CONTRAST_ORDER]

        axis.barh(
            y_positions, deltas, color=bar_colors, edgecolor="black", linewidth=0.6
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(CONTRAST_ORDER)
        axis.set_title(metric, fontsize=11, weight="bold")
        axis.grid(True, axis="x", linewidth=0.3, alpha=0.5)
        axis.invert_yaxis()

        max_abs = max(np.abs(deltas).max(), 1e-6)
        axis.set_xlim(-max_abs * 1.9, max_abs * 1.9)
        span = axis.get_xlim()[1] - axis.get_xlim()[0]

        for y_pos, delta in zip(y_positions, deltas):
            is_positive = delta >= 0
            offset = 0.03 * span if is_positive else -0.03 * span
            horizontal_alignment = "left" if is_positive else "right"
            axis.text(
                delta + offset,
                y_pos,
                f"{delta:+.5f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    linewidth=0.6,
                ),
            )

    axes[0].set_ylabel("Contrast")
    fig.suptitle(
        "Paired per-split contrasts: mean Δ (treatment − control) across 69 monthly splits",
        fontsize=13,
        weight="bold",
    )
    fig.legend(
        handles=_contrast_legend_handles(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _finalise(fig, save_path)


def plot_rq4_paired_win_rates(
    paired_table: pd.DataFrame, save_path: Path | None = None
) -> None:
    """Per-metric grouped barh of win rate across the three contrasts."""
    fig, axes = plt.subplots(1, len(PC_METRIC_ORDER), figsize=(22, 4.5), sharey=True)

    for axis, metric in zip(axes, PC_METRIC_ORDER):
        subset = paired_table[paired_table["metric"] == metric].set_index("contrast")
        win_rates = np.array([subset.loc[c, "win_rate"] for c in CONTRAST_ORDER])
        y_positions = np.arange(len(CONTRAST_ORDER))
        bar_colors = [CONTRAST_COLOURS[c] for c in CONTRAST_ORDER]

        axis.barh(
            y_positions,
            win_rates,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(CONTRAST_ORDER)
        axis.set_title(metric, fontsize=11, weight="bold")
        axis.set_xlim(0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.grid(True, axis="x", linewidth=0.3, alpha=0.5)
        axis.invert_yaxis()

        for y_pos, win in zip(y_positions, win_rates):
            axis.text(
                win + 0.02 if win < 0.85 else win - 0.02,
                y_pos,
                f"{win:.0%}",
                va="center",
                ha="left" if win < 0.85 else "right",
                fontsize=9.5,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    linewidth=0.6,
                ),
            )

    axes[0].set_ylabel("Contrast")
    fig.suptitle(
        "Paired per-split contrasts: win rate (share of 69 splits where treatment > control)",
        fontsize=13,
        weight="bold",
    )
    coin_flip_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Coin-flip reference (50%)",
    )
    fig.legend(
        handles=[*_contrast_legend_handles(), coin_flip_handle],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    _finalise(fig, save_path)


def _contrast_legend_handles() -> list[Any]:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=CONTRAST_COLOURS[c],
            label=f"{c}  :  {CONTRAST_DESCRIPTIONS[c]}",
        )
        for c in CONTRAST_ORDER
    ]


def export_figures_as_svg(
    output_directory: Path,
    eda_summary: dict[str, Any],
    paths: NotebookPaths,
) -> list[Path]:
    """Render every findings figure to SVG under `output_directory` and return the written paths."""
    output_directory.mkdir(parents=True, exist_ok=True)
    coefficients, rq2_summary = load_rq2_artefacts(paths.rq2_directory)
    rq3_per_band_summary, pi_series = build_rq3_per_band_summary(
        paths.rq3_panel_directory, eda_summary
    )
    rq4_per_split = load_rq4_per_split(
        paths.rq4_baseline_directory, paths.rq4_results_directory
    )
    rq4_paired = compute_rq4_paired_table(rq4_per_split)

    figure_plan: list[tuple[str, Any]] = [
        (
            "rq1_discordance_distribution.svg",
            lambda p: plot_rq1_discordance_distribution(eda_summary, save_path=p),
        ),
        (
            "rq1_self_discordance.svg",
            lambda p: plot_rq1_self_discordance(eda_summary, save_path=p),
        ),
        (
            "rq1_per_band_coherence.svg",
            lambda p: plot_rq1_per_band_coherence(eda_summary, save_path=p),
        ),
        (
            "rq1_yearly_discordance.svg",
            lambda p: plot_rq1_yearly_discordance(eda_summary, save_path=p),
        ),
        (
            "rq2_coefficient_forest.svg",
            lambda p: plot_rq2_forest(coefficients, rq2_summary, save_path=p),
        ),
        (
            "rq3_per_band_pc.svg",
            lambda p: plot_rq3_per_band(rq3_per_band_summary, pi_series, save_path=p),
        ),
        (
            "rq4_paired_deltas.svg",
            lambda p: plot_rq4_paired_deltas(rq4_paired, save_path=p),
        ),
        (
            "rq4_paired_win_rates.svg",
            lambda p: plot_rq4_paired_win_rates(rq4_paired, save_path=p),
        ),
    ]
    written: list[Path] = []
    for filename, render in figure_plan:
        target = output_directory / filename
        render(target)
        written.append(target)
    return written
