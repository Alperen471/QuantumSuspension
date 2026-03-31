from configs.evaluation_config import (
    EVALUATION_CONFIG,
    TEST_ROADS,
    CONTROLLER_CONFIGS,
)
from evaluation.benchmark_runner import evaluate_multiple_controllers
from evaluation.benchmark_summary import (
    build_benchmark_report,
    print_benchmark_report,
)
from evaluation.benchmark_export import (
    save_benchmark_report_csv,
    save_benchmark_report_json,
)


def main():
    dt = EVALUATION_CONFIG["dt"]
    episode_length = EVALUATION_CONFIG["episode_length"]
    seeds = EVALUATION_CONFIG["seeds"]

    print("=" * 100)
    print("RUNNING FULL BENCHMARK")
    print("=" * 100)
    print(f"dt              : {dt}")
    print(f"episode_length  : {episode_length}")
    print(f"seeds           : {seeds}")
    print(f"num_roads       : {len(TEST_ROADS)}")
    print(f"num_controllers : {len(CONTROLLER_CONFIGS)}")
    print("=" * 100)

    all_summaries = evaluate_multiple_controllers(
        controller_configs=CONTROLLER_CONFIGS,
        roads=TEST_ROADS,
        seeds=seeds,
        dt=dt,
        episode_length=episode_length,
        save_results=True,
        verbose=True,
    )

    print(f"\nToplam summary sayısı: {len(all_summaries)}")

    report = build_benchmark_report(all_summaries)

    print(f"Aggregate report key sayısı: {len(report)}")

    print_benchmark_report(report)

    csv_path = save_benchmark_report_csv(report)
    json_path = save_benchmark_report_json(report)

    print(f"\nAggregate CSV saved to : {csv_path}")
    print(f"Aggregate JSON saved to: {json_path}")

    return all_summaries, report


if __name__ == "__main__":
    main()