"""Render Figure 1 at its 119 mm print width from frozen plotting inputs.

Only layout/typography changes; no evaluation or statistical refitting.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    source = HERE / "figure_inputs.json"
    d = json.loads(source.read_text())["diagnostics"]
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9,
                         "pdf.fonttype": 42})
    colors = {"rr_cap": "tab:blue", "dac": "tab:orange"}
    names = {"rr_cap": "RR", "dac": "DAC"}
    fig = plt.figure(figsize=(119 / 25.4, 5.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    primitives = {}
    for i, arm in enumerate(["rr_cap", "dac"]):
        for j, condition in enumerate(["slow_5ms", "medium_2ms"]):
            values = [r["range"] for r in d["fresh_seed"]
                      if r["condition"] == condition and r["checkpoint_arm"] == arm]
            assert len(values) == 10
            axes[0].scatter(np.linspace(i * 2 + j - .12, i * 2 + j + .12, 10),
                            values, color=colors[arm], marker=["o", "^"][j], s=23)
            primitives[f"range_{arm}_{condition}"] = values
    axes[0].set_xticks(range(4), ["RR\nSlow", "RR\nMedium", "DAC\nSlow", "DAC\nMedium"],
                      rotation=40, ha="right")
    axes[0].set_ylabel("Mean suffix-return range")
    axes[0].set_title("(a) Feasible sender sets")
    for arm in ["rr_cap", "dac"]:
        rows = [r for r in d["prefix_seed"]
                if r["condition"] == "slow_5ms" and r["checkpoint_arm"] == arm]
        x = [r["cold_delta"] for r in rows]
        y = [r["warm_unconditional_delta"] for r in rows]
        primitives[f"prefix_{arm}"] = [x, y]
        axes[1].scatter(x, y, label=names[arm], color=colors[arm],
                        marker="o" if arm == "rr_cap" else "^", s=25)
    axes[1].axhline(0, color="grey", lw=.6)
    axes[1].axvline(0, color="grey", lw=.6)
    axes[1].set_xlabel("Cold DAC − RR return")
    axes[1].set_ylabel("Prefix DAC − RR return")
    axes[1].set_title("(b) Slow / 5 ms")
    axes[1].legend(loc="best", frameon=False, handlelength=1)
    for arm in ["rr_cap", "dac"]:
        for mode, style in [("measured", "-"), ("fixed", "--")]:
            xs = sorted({r["deadline_ms"] for r in d["readiness_seed"]})
            ys = [float(np.mean([r["mean_k"] for r in d["readiness_seed"]
                                if r["checkpoint_arm"] == arm and r["mode"] == mode
                                and r["deadline_ms"] == x])) for x in xs]
            primitives[f"readiness_{arm}_{mode}"] = [xs, ys]
            axes[2].plot(xs, ys, style, label=f"{names[arm]} {mode}", color=colors[arm])
    axes[2].set_xlabel("Reception deadline (ms)")
    axes[2].set_ylabel("Delivered packets / step")
    axes[2].set_title("(c) Medium link; trained at 2 ms")
    axes[2].legend(loc="lower right", ncol=2, frameon=False, columnspacing=1)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(HERE / "diagnostics.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    (HERE / "figure1_validation.json").write_text(json.dumps({
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width_mm": 119, "axes_font_pt": 10, "ticks_and_legend_pt": 9,
        "numeric_primitives": primitives, "new_scientific_evaluation": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
