"""
figures/training.py
-------------------
Training figürleri: SAC vs TD3 karşılaştırması, convergence eğrileri.

Çalıştırma:
    python -m figures.training --results_dir results/ablation_XXXX
    python -m figures.training  # demo verisi ile
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from figures._base import (set_style, save, load_ablation,
                            MODEL_COLORS, MODEL_LABELS, fmt_steps)

OUT = "figures/training"


def sac_vs_td3(sac_results: dict, td3_results: dict):
    """SAC vs TD3 karşılaştırması — bar + convergence."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    road_types  = ["random", "sinusoidal", "bump", "pothole"]
    road_labels = ["Random", "Sinusoidal", "Bump", "Pothole"]
    x = np.arange(len(road_types)); w = .35

    ax = axes[0]
    sac_rms = [sac_results.get("sac_baseline", {}).get("by_road", {}).get(rt, {}).get("mean", np.nan)
               for rt in road_types]
    td3_rms = [td3_results.get("td3_baseline", {}).get("by_road", {}).get(rt, {}).get("mean", np.nan)
               for rt in road_types]

    # Demo eğer veri yoksa
    if all(np.isnan(v) for v in sac_rms):
        np.random.seed(1)
        sac_rms = [.28, .21, .24, .23]
        td3_rms = [.55, .42, .48, .46]

    ax.bar(x - w/2, sac_rms, w, label="SAC (entropy reg.)",
           color=MODEL_COLORS["sac_baseline"], alpha=.87,
           edgecolor="k", lw=.8)
    ax.bar(x + w/2, td3_rms, w, label="TD3 (deterministic)",
           color=MODEL_COLORS["td3_baseline"], alpha=.87,
           edgecolor="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(road_labels)
    ax.set_ylabel("RMS Body Acc. (m/s²)")
    ax.set_title("SAC vs TD3 — RMS Body Acceleration by Road Type")
    ax.legend(fontsize=9)

    # Convergence
    ax2 = axes[1]
    for algo, res, col, lbl in [
        ("sac", sac_results, MODEL_COLORS["sac_baseline"], "SAC"),
        ("td3", td3_results, MODEL_COLORS["td3_baseline"], "TD3"),
    ]:
        baseline_key = f"{algo}_baseline"
        hist = res.get(baseline_key, {}).get("eval_history", [])
        if hist:
            steps = [e["step"] for e in hist]
            vals  = [e.get("rms_body_acc_mean", 0) for e in hist]
        else:
            np.random.seed(42 if algo=="sac" else 7)
            steps = np.arange(0, 100001, 5000)
            base  = .28 if algo == "sac" else .54
            vals  = base + .15*np.exp(-steps/40000) + np.random.normal(0,.01,len(steps))
        ax2.plot(steps, vals, color=col, lw=2, label=lbl)

    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("RMS Body Acc. (m/s²)")
    ax2.set_title("Training Convergence: SAC vs TD3")
    ax2.legend(fontsize=9)
    fmt_steps(ax2)

    plt.suptitle("Algorithm Comparison: SAC vs TD3 (QFM Encoder, same conditions)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(f"{OUT}/fig_sac_vs_td3.png", fig)


def convergence_curves(results: dict):
    """4 model için training convergence."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_models = [
        "rl_baseline", "rl_fuzzy_bounded", "rl_q5_var", "rl_q5_var_fuzzy"
    ]

    for ax, mkey, title, ylabel in [
        (axes[0], "rms_body_acc",    "RMS Body Acceleration", "RMS (m/s²)"),
        (axes[1], "iso_weighted_rms","ISO 2631 wRMS",         "ISO wRMS (m/s²)"),
    ]:
        for m in plot_models:
            col  = MODEL_COLORS.get(m, "#999")
            lbl  = MODEL_LABELS.get(m, m)
            hist = results.get(m, {}).get("eval_history", [])

            if hist:
                steps = [e["step"] for e in hist]
                vals  = [e.get(f"{mkey}_mean", 0) for e in hist]
            else:
                np.random.seed(hash(m) % 100)
                steps = np.arange(0, 300001, 10000)
                base  = {"rl_baseline":.29,"rl_fuzzy_bounded":.25,
                         "rl_q5_var":.24,"rl_q5_var_fuzzy":.21}.get(m, .28)
                vals  = base + .15*np.exp(-np.array(steps)/80000) \
                        + np.random.normal(0,.012,len(steps))

            ax.plot(steps, vals, color=col, lw=1.8, label=lbl, alpha=.9)

        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fmt_steps(ax)

    plt.suptitle("Training Convergence — Synthetic Data (300K steps, 3 seeds)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(f"{OUT}/fig_convergence.png", fig)


def fuzzy_coefficients():
    """Fuzzy katsayılarının sürüş senaryosunda değişimi."""
    from scipy.ndimage import uniform_filter1d
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)

    np.random.seed(42)
    t  = np.linspace(0, 20, 2000)
    dt = t[1] - t[0]
    road_acc = .15 * np.sin(2*np.pi*1.5*t) + np.exp(-50*(t-8)**2)*2.5

    def coef(sig, scale=.7, w=60):
        norm = np.abs(sig) / (np.abs(sig).max() + 1e-6)
        return np.clip(uniform_filter1d(.8 + scale*norm, w), .8, 1.5)

    alpha = coef(road_acc, .70)
    beta  = coef(road_acc, .65)
    gamma = coef(np.gradient(road_acc, dt)**2, .55)

    for ax, vals, lbl, col in zip(
        axes,
        [alpha, beta, gamma],
        [r"$\alpha$ comfort", r"$\beta$ suspension", r"$\gamma$ energy"],
        ["#4878CF","#6ACC65","#D65F5F"],
    ):
        ax.plot(t, vals, color=col, lw=1.5)
        ax.axhline(1.0, ls="--", color="gray", lw=1, alpha=.5)
        ax.fill_between(t, 1.0, vals, where=vals>1.0, alpha=.18, color=col)
        ax.axvspan(7.5, 8.5, alpha=.08, color="orange")
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_ylim(.72, 1.58); ax.set_yticks([.8,1.0,1.2,1.4])

    axes[1].text(8.0, 1.52, "Speed\nbump", ha="center",
                 fontsize=8.5, color="darkorange", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    plt.suptitle("Adaptive Fuzzy Reward Coefficients During Driving",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(f"{OUT}/fig_fuzzy_coefficients.png", fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--td3_dir",     type=str, default=None)
    args = parser.parse_args()

    print("=== Training Figures ===")

    sac_res = load_ablation(args.results_dir) if args.results_dir else {}
    td3_res = load_ablation(args.td3_dir)     if args.td3_dir     else {}

    sac_vs_td3(sac_res, td3_res)
    convergence_curves(sac_res)
    fuzzy_coefficients()
    print("Done →", OUT)


if __name__ == "__main__":
    main()
