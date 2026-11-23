"""Plot paired closed-loop intervention effects from the registered receiver study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "runs" / "receiver_head_adaptation_20261123_001"


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5.5, 2.25), layout="constrained")
    for level, (access, color) in enumerate([("normal", "#215e8c"), ("source_aware", "#966422")]):
        result = summary["aggregate"][access]["adapted_minus_original"]
        values = np.array(result["seed_values"])
        ax.scatter(values, level + np.linspace(-.12, .12, len(values)),
                   s=22, facecolors="none", edgecolors=color, linewidths=1.2)
        lo, hi = result["ci95"]
        ax.plot([lo, hi], [level + .22] * 2, color=color, linewidth=2.5)
        ax.scatter(result["mean"], level + .22, marker="D", color=color, s=32, zorder=3)
    ax.axvline(0, color=".35", linestyle=":", linewidth=1)
    ax.set_yticks([0, 1], ["Normal RR", "Source-aware"])
    ax.set_xlabel("Accuracy change after head adaptation (percentage points)")
    ax.set_ylim(1.48, -.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.18)
    fig.savefig(HERE / "figures" / "receiver_adaptation.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
