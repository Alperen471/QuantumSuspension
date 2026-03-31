from experiments.run_experiment import run_skyhook_gain_tuning
from analysis.pareto_selection import build_pareto_report, print_pareto_report


def main():
    all_summaries = run_skyhook_gain_tuning()

    report = build_pareto_report(
        all_summaries,
        road_weights={
            "random": 0.6,
            "sinusoidal": 0.4,
        },
    )

    print_pareto_report(report)


if __name__ == "__main__":
    main()