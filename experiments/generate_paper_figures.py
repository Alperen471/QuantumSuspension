from pathlib import Path
import matplotlib.pyplot as plt

from configs.evaluation_config import (
    EVALUATION_CONFIG,
    TEST_ROADS,
    CONTROLLER_CONFIGS,
)
from evaluation.benchmark_runner import evaluate_multiple_controllers
from evaluation.benchmark_summary import build_benchmark_report


PLOT_DIR = Path("results") / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_controller_comparison_with_errorbars(
    report: dict,
    metric_mean_key: str,
    metric_std_key: str,
    ylabel: str,
    filename: str,
):
    roads = ["random", "sinusoidal", "bump", "pothole"]
    controllers = sorted({key[0] for key in report.keys()})

    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(range(len(roads)))
    width = 0.35 if len(controllers) <= 2 else 0.25

    if len(controllers) == 1:
        offsets = [0.0]
    elif len(controllers) == 2:
        offsets = [-width / 2, width / 2]
    else:
        start = -width * (len(controllers) - 1) / 2
        offsets = [start + i * width for i in range(len(controllers))]

    for controller, offset in zip(controllers, offsets):
        means = []
        stds = []

        for road in roads:
            key = (controller, road)
            if key in report:
                means.append(report[key][metric_mean_key])
                stds.append(report[key][metric_std_key])
            else:
                means.append(0.0)
                stds.append(0.0)

        ax.bar(
            [xi + offset for xi in x],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=controller,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(roads)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Across Controllers and Roads")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    save_path = PLOT_DIR / filename
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main():
    dt = EVALUATION_CONFIG["dt"]
    episode_length = EVALUATION_CONFIG["episode_length"]
    seeds = EVALUATION_CONFIG["seeds"]

    all_summaries = evaluate_multiple_controllers(
        controller_configs=CONTROLLER_CONFIGS,
        roads=TEST_ROADS,
        seeds=seeds,
        dt=dt,
        episode_length=episode_length,
        save_results=False,
        verbose=False,
    )

    report = build_benchmark_report(all_summaries)

    p1 = plot_controller_comparison_with_errorbars(
        report=report,
        metric_mean_key="rms_body_acc_mean",
        metric_std_key="rms_body_acc_std",
        ylabel="RMS Body Acceleration",
        filename="controller_comparison_rms_body_acc_errorbar.png",
    )

    p2 = plot_controller_comparison_with_errorbars(
        report=report,
        metric_mean_key="band_1_3Hz_energy_mean",
        metric_std_key="band_1_3Hz_energy_std",
        ylabel="1–3 Hz Band Energy",
        filename="controller_comparison_band_1_3Hz_errorbar.png",
    )

    p3 = plot_controller_comparison_with_errorbars(
        report=report,
        metric_mean_key="rms_control_force_mean",
        metric_std_key="rms_control_force_std",
        ylabel="RMS Control Force",
        filename="controller_comparison_rms_force_errorbar.png",
    )

    p4 = plot_controller_comparison_with_errorbars(
        report=report,
        metric_mean_key="total_psd_energy_mean",
        metric_std_key="total_psd_energy_std",
        ylabel="Total PSD Energy",
        filename="controller_comparison_total_psd_errorbar.png",
    )

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")


if __name__ == "__main__":
    main()