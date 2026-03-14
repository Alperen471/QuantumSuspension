from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from configs.evaluation_config import EVALUATION_CONFIG
from experiments.run_experiment import run_single_experiment
from analysis.frequency_domain_metrics import compute_frequency_domain_metrics


PLOT_DIR = Path("results") / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def generate_controller_result(
    controller_type: str,
    controller_params: dict,
    road_type: str,
    road_amplitude: float,
    road_frequency: float,
    episode_length: int,
    dt: float,
    seed: int,
):
    results = run_single_experiment(
        controller_type=controller_type,
        controller_params=controller_params,
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
        seed=seed,
    )

    freq_metrics = compute_frequency_domain_metrics(results, dt=dt)

    return {
        "controller_type": controller_type,
        "controller_params": controller_params,
        "freqs": freq_metrics["psd_freqs"],
        "psd": freq_metrics["psd_body_acc"],
    }


def plot_psd_comparison(
    controller_results: list[dict],
    road_type: str,
    road_amplitude: float,
    road_frequency: float,
    filename: str,
    max_freq: float = 20.0,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in controller_results:
        freqs = np.asarray(result["freqs"])
        psd = np.asarray(result["psd"])

        mask = freqs <= max_freq
        freqs = freqs[mask]
        psd = psd[mask]

        label = result["controller_type"]

        params = result.get("controller_params", {})
        if result["controller_type"] == "skyhook" and "c_sky" in params:
            label = f"skyhook (c_sky={int(params['c_sky'])})"

        ax.plot(freqs, psd, label=label)

    # 1–3 Hz comfort band
    ax.axvspan(1.0, 3.0, alpha=0.2, label="1–3 Hz comfort band")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD of Body Acceleration")
    ax.set_title(
        f"PSD Comparison - road={road_type}, amp={road_amplitude}, freq={road_frequency}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path = PLOT_DIR / filename
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main():
    dt = EVALUATION_CONFIG["dt"]
    episode_length = EVALUATION_CONFIG["episode_length"]

    # Şimdilik tek seed ile figür üretelim.
    # İstersen sonra mean PSD ekleriz.
    seed = 1

    controller_configs = [
        {
            "controller_type": "passive",
            "controller_params": {},
        },
        {
            "controller_type": "skyhook",
            "controller_params": {
                "c_sky": 4000.0,
                "max_force": 3000.0,
            },
        },
    ]

    road_configs = [
        {
            "road_type": "random",
            "road_amplitude": 0.01,
            "road_frequency": 1.0,
            "filename": "psd_comparison_random.png",
        },
        {
            "road_type": "sinusoidal",
            "road_amplitude": 0.01,
            "road_frequency": 1.0,
            "filename": "psd_comparison_sinusoidal.png",
        },
        {
            "road_type": "bump",
            "road_amplitude": 0.02,
            "road_frequency": 1.0,
            "filename": "psd_comparison_bump.png",
        },
        {
            "road_type": "pothole",
            "road_amplitude": 0.02,
            "road_frequency": 1.0,
            "filename": "psd_comparison_pothole.png",
        },
    ]

    for road_cfg in road_configs:
        controller_results = []

        for ctrl_cfg in controller_configs:
            result = generate_controller_result(
                controller_type=ctrl_cfg["controller_type"],
                controller_params=ctrl_cfg["controller_params"],
                road_type=road_cfg["road_type"],
                road_amplitude=road_cfg["road_amplitude"],
                road_frequency=road_cfg["road_frequency"],
                episode_length=episode_length,
                dt=dt,
                seed=seed,
            )
            controller_results.append(result)

        save_path = plot_psd_comparison(
            controller_results=controller_results,
            road_type=road_cfg["road_type"],
            road_amplitude=road_cfg["road_amplitude"],
            road_frequency=road_cfg["road_frequency"],
            filename=road_cfg["filename"],
            max_freq=20.0,
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()