from experiments.run_experiment import summarize_scenario, pretty_print_summary
from analysis.result_annotations import add_metric_labels
from analysis.results_io import save_summary_json, append_summary_csv
from analysis.repeated_tuning_analysis import (
    build_repeated_tuning_report,
    print_repeated_tuning_summary,
    print_repeated_pareto_report,
)
from analysis.pareto_plot import (
    plot_pareto_tradeoff,
    plot_band_energy_tradeoff,
)


def run_repeated_skyhook_tuning():
    dt = 0.01
    episode_length = 2000

    gain_values = [1000.0, 2000.0, 2500.0, 4000.0, 6000.0]
    seeds = list(range(1, 11))  # 10 tekrar

    road_configs = [
        {
            "road_type": "random",
            "road_amplitude": 0.01,
            "road_frequency": 1.0,
        },
        {
            "road_type": "sinusoidal",
            "road_amplitude": 0.01,
            "road_frequency": 1.0,
        },
    ]

    all_summaries = []

    for gain in gain_values:
        for seed in seeds:
            for road_cfg in road_configs:
                summary = summarize_scenario(
                    controller_type="skyhook",
                    controller_params={
                        "c_sky": gain,
                        "max_force": 3000.0,
                    },
                    road_type=road_cfg["road_type"],
                    road_amplitude=road_cfg["road_amplitude"],
                    road_frequency=road_cfg["road_frequency"],
                    episode_length=episode_length,
                    dt=dt,
                    seed=seed,
                )

                summary = add_metric_labels(summary)

                pretty_print_summary(summary)

                json_path = save_summary_json(summary)
                csv_path = append_summary_csv(summary)

                print(f"JSON saved to: {json_path}")
                print(f"CSV updated : {csv_path}")

                all_summaries.append(summary)

    report = build_repeated_tuning_report(
        all_summaries,
        road_weights={
            "random": 0.6,
            "sinusoidal": 0.4,
        },
    )

    print_repeated_tuning_summary(report)
    print_repeated_pareto_report(report)

    pareto_plot_path = plot_pareto_tradeoff(report)
    band_plot_path = plot_band_energy_tradeoff(report)

    print(f"Pareto plot saved to    :{pareto_plot_path}")
    print(f"Band-energy plot saved to   :{band_plot_path}")

    return all_summaries, report


if __name__ == "__main__":
    run_repeated_skyhook_tuning()