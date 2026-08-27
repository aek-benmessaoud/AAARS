"""Generate all publication figures for AAARS paper."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, r"F:\Project11-AAARS")

FIG_DIR = os.path.join(r"F:\Project11-AAARS", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def fig1_risk_trajectory():
    """Figure 1: Risk score trajectory under boustro vs minerich."""
    from src.experiments.runner import run_episode, DEFAULT_CFG

    cfg = {**DEFAULT_CFG, "max_steps": 3000, "trace_stride": 20,
           "aaars": {"w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
                     "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                     "blend_threshold": 0.30, "num_bins": 5}}

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for ax_idx, (alloc, color, label) in enumerate([
        ("boustro", "#2196F3", "BoustroLanes"),
        ("minerich", "#FF5722", "MineRichness"),
    ]):
        ax = axes[ax_idx]
        r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
        trace = json.loads(r["_trace"])

        ts = [e["t"] for e in trace]
        risk = [e.get("risk_score", 0) for e in trace]
        tc = [e.get("temporal_clustering", -1) for e in trace]
        sc = [e.get("spatial_concentration", -1) for e in trace]
        fi = [e.get("frequency_instability", -1) for e in trace]

        valid = [i for i, e in enumerate(trace) if tc[i] >= 0]
        ts_v = [ts[i] for i in valid]
        risk_v = [risk[i] for i in valid]
        tc_v = [tc[i] for i in valid]
        sc_v = [sc[i] for i in valid]
        fi_v = [fi[i] for i in valid]

        ax.plot(ts_v, risk_v, color=color, linewidth=2, label="Risk score")
        ax.fill_between(ts_v, 0, risk_v, alpha=0.15, color=color)
        ax.plot(ts_v, tc_v, color="#9C27B0", linewidth=1, alpha=0.6,
                linestyle="--", label="TC")
        ax.plot(ts_v, sc_v, color="#4CAF50", linewidth=1, alpha=0.6,
                linestyle=":", label="SC")
        ax.plot(ts_v, fi_v, color="#FF9800", linewidth=1, alpha=0.6,
                linestyle="-.", label="FI")

        ax.axhline(y=0.30, color="gray", linewidth=0.8, linestyle="--", alpha=0.5,
                    label="Blend threshold (0.30)")
        ax.set_ylabel("Score")
        ax.set_title(f"{label} (seed=42)")
        ax.set_ylim(-0.02, 0.65)
        ax.legend(loc="upper right", ncol=3, framealpha=0.9)

    axes[1].set_xlabel("Simulation step")
    fig.suptitle("AAARS Risk Score Trajectory by Allocation Policy", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_risk_trajectory.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig1_risk_trajectory.pdf"))
    plt.close(fig)
    print("  Saved fig1_risk_trajectory.png/pdf")


def fig2_fc_comparison():
    """Figure 2: FC comparison across methods (bar chart from confirmation)."""
    raw_path = os.path.join(r"F:\Project11-AAARS", "results", "raw",
                            "power_confirm.json")
    with open(raw_path) as f:
        results = json.load(f)

    methods = [
        ("chao1_ci", "Chao1 CI", "#78909C"),
        ("chao92_ci", "Chao92 CI", "#546E7A"),
        ("aaars", "AAARS", "#2196F3"),
        ("discrete_aaars", "Discrete AAARS", "#90CAF9"),
        ("oracle_95", "Oracle 95%", "#4CAF50"),
        ("fixed_2", "Fixed-2", "#FF9800"),
        ("diminishing", "Diminishing", "#FF5722"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax_idx, (alloc, alloc_label) in enumerate([
        ("boustro", "BoustroLanes"),
        ("minerich", "MineRichness"),
    ]):
        ax = axes[ax_idx]
        subset = [r for r in results if r["alloc"] == alloc]
        n_total = len(subset)

        names, fc_pcts, colors, stop_fracs = [], [], [], []

        for method_key, method_name, color in methods:
            key_t = f"{method_key}__t"
            key_r = f"{method_key}__recall"
            stops = [r for r in subset if r.get(key_t) is not None]
            fc = sum(1 for r in stops if r.get(key_r, 100) < 95.0)
            fc_pct = 100.0 * fc / len(stops) if stops else 0
            stop_frac = 100.0 * len(stops) / n_total

            names.append(method_name)
            fc_pcts.append(fc_pct)
            colors.append(color)
            stop_fracs.append(stop_frac)

        x = np.arange(len(names))
        bars = ax.bar(x, fc_pcts, color=colors, edgecolor="white", linewidth=0.5)

        for i, (bar, sf) in enumerate(zip(bars, stop_fracs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{sf:.0f}%\nstop", ha="center", va="bottom", fontsize=7,
                    color="#666")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("False Certification %" if ax_idx == 0 else "")
        ax.set_title(alloc_label)
        ax.set_ylim(0, 50)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    fig.suptitle("False Certification Rate by Method and Allocation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_fc_comparison.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig2_fc_comparison.pdf"))
    plt.close(fig)
    print("  Saved fig2_fc_comparison.png/pdf")


def fig3_sweep_heatmap():
    """Figure 3: FC reduction heatmap from config sweep."""
    raw_path = os.path.join(r"F:\Project11-AAARS", "results", "raw",
                            "config_sweep.json")
    with open(raw_path) as f:
        results = json.load(f)

    K_VALUES = [30, 60, 120]
    N_VALUES = [3, 6, 12]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax_idx, (metric, title) in enumerate([
        ("aaars", "AAARS FC%"),
        ("chao1", "Chao1 FC%"),
        ("delta", "FC Reduction (pp)"),
    ]):
        ax = axes[ax_idx]
        data = np.zeros((len(N_VALUES), len(K_VALUES)))

        for i, n in enumerate(N_VALUES):
            for j, k in enumerate(K_VALUES):
                subset = [r for r in results if r["alloc"] == "minerich"
                          and r["num_mines"] == k and r["num_agents"] == n]
                if metric == "delta":
                    aaars_stops = [r for r in subset if r.get("aaars__t") is not None]
                    chao1_stops = [r for r in subset if r.get("chao1_ci__t") is not None]
                    if aaars_stops and chao1_stops:
                        aaars_fc = sum(1 for r in aaars_stops
                                       if r.get("aaars__recall", 100) < 95.0)
                        chao1_fc = sum(1 for r in chao1_stops
                                       if r.get("chao1_ci__recall", 100) < 95.0)
                        data[i, j] = (100.0 * chao1_fc / len(chao1_stops) -
                                      100.0 * aaars_fc / len(aaars_stops))
                else:
                    key_t = f"{metric}__t"
                    key_r = f"{metric}__recall"
                    stops = [r for r in subset if r.get(key_t) is not None]
                    if stops:
                        fc = sum(1 for r in stops if r.get(key_r, 100) < 95.0)
                        data[i, j] = 100.0 * fc / len(stops)

        if metric == "delta":
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=50)
        else:
            im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=70)

        for i in range(len(N_VALUES)):
            for j in range(len(K_VALUES)):
                val = data[i, j]
                color = "white" if abs(val) > 30 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f"N={n}" for n in N_VALUES])
        ax.set_xlabel("Mine count (K)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Agent count (N)")
    fig.suptitle("False Certification under MineRichness Allocation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_sweep_heatmap.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_sweep_heatmap.pdf"))
    plt.close(fig)
    print("  Saved fig3_sweep_heatmap.png/pdf")


def fig4_stop_time_comparison():
    """Figure 4: Stop time comparison AAARS vs Chao1 under minerich (80 seeds)."""
    raw_path = os.path.join(r"F:\Project11-AAARS", "results", "raw",
                            "power_confirm.json")
    with open(raw_path) as f:
        results = json.load(f)

    minerich = [r for r in results if r["alloc"] == "minerich"]
    chao1_times = [r["chao1_ci__t"] for r in minerich if r.get("chao1_ci__t")]
    aaars_times = [r["aaars__t"] for r in minerich if r.get("aaars__t")]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(minerich))
    width = 0.35

    c1_t = [r.get("chao1_ci__t") or 6000 for r in minerich]
    aa_t = [r.get("aaars__t") or 6000 for r in minerich]

    ax.bar(x - width/2, c1_t, width, label="Chao1 CI", color="#78909C")
    ax.bar(x + width/2, aa_t, width, label="AAARS", color="#2196F3")

    # Mark FC episodes
    for i, r in enumerate(minerich):
        if r.get("chao1_ci__recall", 100) < 95.0:
            ax.plot(i - width/2, c1_t[i], "x", color="red", markersize=8, mew=2)
        if r.get("aaars__recall", 100) < 95.0:
            ax.plot(i + width/2, aa_t[i], "x", color="red", markersize=8, mew=2)

    ax.axhline(y=6000, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stop time (steps)")
    ax.set_title("Stop Time under MineRichness (red X = false certification)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_stop_times.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_stop_times.pdf"))
    plt.close(fig)
    print("  Saved fig4_stop_times.png/pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_risk_trajectory()
    fig2_fc_comparison()
    fig3_sweep_heatmap()
    fig4_stop_time_comparison()
    print("Done!")
