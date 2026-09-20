"""Phase 1 plots.

Raw values stay visible on every chart (spec §14): no composite scores, no
normalisation that would hide the underlying rate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def complexity_curve(out_dir: Path) -> Path | None:
    import pandas as pd

    path = out_dir / "by_complexity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    contracts = sorted(df["contract"].unique())
    fig, axes = plt.subplots(1, len(contracts), figsize=(6 * len(contracts), 4.2), squeeze=False)

    for ax, contract in zip(axes[0], contracts):
        sub = df[df["contract"] == contract]
        for (model, gran), group in sub.groupby(["model", "granularity"]):
            group = group.sort_values("level")
            ax.plot(group["level"], group["full_call_exact_mean"], marker="o",
                    linewidth=1.8, markersize=5, label=f"{model} / {gran}")
            # Bootstrap CI band: 5-case cells are wide, and the chart must say so.
            ax.fill_between(group["level"], group["ci_low"], group["ci_high"], alpha=0.13)
        _style(ax, f"Full-call exact match by complexity ({contract})",
               "Complexity level", "Full-call exact match")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    target = out_dir / "fig_complexity.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def calibration(out_dir: Path) -> Path | None:
    import pandas as pd

    path = out_dir / "needle_confidence.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888", label="perfect calibration")
    for contract, group in df.groupby("contract"):
        group = group.sort_values("bucket")
        ax.plot(group["bucket"], group["decision_correct"], marker="o",
                label=f"decision correct ({contract})")
        for _, row in group.iterrows():
            ax.annotate(f"n={int(row['n'])}", (row["bucket"], row["decision_correct"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    _style(ax, "Needle 3 confidence calibration", "Reported confidence (bucket)",
           "Observed decision accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    target = out_dir / "fig_calibration.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def accuracy_vs_latency(out_dir: Path) -> Path | None:
    import pandas as pd

    path = out_dir / "overall.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for _, row in df.iterrows():
        ax.scatter(row["latency_p50"], row["full_call_exact"], s=70)
        ax.annotate(f"{row['model']}\n{row['granularity']} / {row['contract']}",
                    (row["latency_p50"], row["full_call_exact"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=7)
    _style(ax, "Accuracy against median latency", "Median episode latency (ms)",
           "Full-call exact match")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    target = out_dir / "fig_accuracy_latency.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def generate_all(out_dir: Path) -> list[Path]:
    made = [f for f in (complexity_curve(out_dir), calibration(out_dir),
                        accuracy_vs_latency(out_dir)) if f]
    for path in made:
        print(f"  plot: {path}")
    return made
