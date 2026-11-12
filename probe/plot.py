import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats import METHODS, LABELS, REFS, REF_LABELS

COLORS = ("#4878a8", "#72b7b2", "#e39c37", "#9876b5", "#d5635b", "#bd2d43")
MARKERS = ("o", "s", "D", "v", "P", "^")


def curves(ax, values, ylabel):
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        ax.plot([0, 25, 50, 75, 100], values[method], marker=marker, label=LABELS[method], color=color)
    ax.set(xlabel="Student prefix read (%)", ylabel=ylabel, xticks=[0, 25, 50, 75, 100])
    ax.grid(alpha=0.2)


def save(fig, destination, name):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "probe"))
    a = p.parse_args()
    destination = Path(a.out)
    data = json.loads((destination / "numbers.json").read_text())
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    curves(axes[0], data["fig2a_follow_rate_pct"], "Follow rate (%)")
    curves(axes[1], data["fig2b_survival_score"], "Survival score")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[0].legend(fontsize=8)
    save(fig, destination, "fig2_capture")
    panels = 2 if "etoken" in data else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 3.8), squeeze=False, sharey=True)
    ax = axes[0, 0]
    colors = ["#bd2d43" if r == "cut" else "#888888" if r == "txt" else "#d99b97" for r in REFS]
    ax.barh(range(len(REFS)), [data["fig3a_residual_drift_pct"][r] for r in REFS], color=colors)
    ax.set(yticks=range(len(REFS)), yticklabels=[REF_LABELS[r] for r in REFS], xlabel="Residual drift (%)")
    ax.invert_yaxis()
    if panels == 2:
        ax = axes[0, 1]
        ax.barh(range(len(REFS)), [data["etoken"]["fig3b_false_belief"][r] for r in REFS], color=colors)
        real = data["etoken"]["fig3b_false_belief"]["real"]
        ax.axvline(real, linestyle="--", color="black", label=f"Real image ({real:.2f})")
        ax.set(xlabel="False-belief probability")
        ax.legend(fontsize=8)
    save(fig, destination, "fig3_references")
    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    curves(ax, data["fig5_corrective_density_pct"], "Corrective density (%)")
    ax.legend(fontsize=8)
    save(fig, destination, "fig5_corrective_density")
    print(f"Saved Fig. 2, 3 and 5 as PDF and PNG in {destination}")


if __name__ == "__main__":
    main()
