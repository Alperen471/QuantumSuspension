from pathlib import Path
import matplotlib.pyplot as plt


def ensure_plot_dir():
    plot_dir = Path("results") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_pareto_tradeoff(report: dict, save_name: str = "skyhook_pareto_tradeoff.png"):
    """
    x-axis  : aggregated RMS control force
    y-axis  : aggregated RMS body acceleration

    report:
        build_repeated_tuning_report(...) çıktısı
    """
    aggregated = report["aggregated"]
    pareto_gains = set(report["pareto_gains"])
    knee_result = report["knee_result"]

    gains = sorted(aggregated.keys())

    x = [aggregated[g]["rms_control_force"] for g in gains]
    y = [aggregated[g]["rms_body_acc"] for g in gains]

    fig, ax = plt.subplots(figsize=(9, 6))

    # tüm noktalar
    ax.scatter(x, y, s=80)

    # gain etiketleri
    for g in gains:
        ax.annotate(
            f"{int(g)}",
            (aggregated[g]["rms_control_force"], aggregated[g]["rms_body_acc"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    # Pareto noktalarını ayrıca çevrele
    pareto_x = [aggregated[g]["rms_control_force"] for g in gains if g in pareto_gains]
    pareto_y = [aggregated[g]["rms_body_acc"] for g in gains if g in pareto_gains]

    ax.scatter(
        pareto_x,
        pareto_y,
        s=220,
        facecolors="none",
        linewidths=1.8,
        label="Pareto-optimal gains",
    )

    # knee-point işaretle
    if knee_result is not None:
        knee_gain, knee_score = knee_result
        knee_x = aggregated[knee_gain]["rms_control_force"]
        knee_y = aggregated[knee_gain]["rms_body_acc"]

        ax.scatter(
            [knee_x],
            [knee_y],
            s=260,
            marker="*",
            label=f"Knee-point (c_sky={int(knee_gain)})",
        )

        ax.annotate(
            f"Knee-point\nc_sky={int(knee_gain)}",
            (knee_x, knee_y),
            textcoords="offset points",
            xytext=(10, -18),
        )

    ax.set_xlabel("Aggregated RMS Control Force")
    ax.set_ylabel("Aggregated RMS Body Acceleration")
    ax.set_title("Skyhook Gain Pareto Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_dir = ensure_plot_dir()
    save_path = plot_dir / save_name
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_band_energy_tradeoff(report: dict, save_name: str = "skyhook_band_tradeoff.png"):
    """
    x-axis  : aggregated RMS control force
    y-axis  : aggregated 1-3 Hz band energy
    """
    aggregated = report["aggregated"]
    pareto_gains = set(report["pareto_gains"])
    knee_result = report["knee_result"]

    gains = sorted(aggregated.keys())

    x = [aggregated[g]["rms_control_force"] for g in gains]
    y = [aggregated[g]["band_1_3Hz_energy"] for g in gains]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x, y, s=80)

    for g in gains:
        ax.annotate(
            f"{int(g)}",
            (aggregated[g]["rms_control_force"], aggregated[g]["band_1_3Hz_energy"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    pareto_x = [aggregated[g]["rms_control_force"] for g in gains if g in pareto_gains]
    pareto_y = [aggregated[g]["band_1_3Hz_energy"] for g in gains if g in pareto_gains]

    ax.scatter(
        pareto_x,
        pareto_y,
        s=220,
        facecolors="none",
        linewidths=1.8,
        label="Pareto-optimal gains",
    )

    if knee_result is not None:
        knee_gain, knee_score = knee_result
        knee_x = aggregated[knee_gain]["rms_control_force"]
        knee_y = aggregated[knee_gain]["band_1_3Hz_energy"]

        ax.scatter(
            [knee_x],
            [knee_y],
            s=260,
            marker="*",
            label=f"Knee-point (c_sky={int(knee_gain)})",
        )

        ax.annotate(
            f"Knee-point\nc_sky={int(knee_gain)}",
            (knee_x, knee_y),
            textcoords="offset points",
            xytext=(10, -18),
        )

    ax.set_xlabel("Aggregated RMS Control Force")
    ax.set_ylabel("Aggregated 1–3 Hz Band Energy")
    ax.set_title("Skyhook Gain Comfort-Band Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_dir = ensure_plot_dir()
    save_path = plot_dir / save_name
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path