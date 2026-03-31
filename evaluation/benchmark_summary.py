import math
from collections import defaultdict


SUMMARY_METRICS = [
    "rms_body_acc",
    "band_1_3Hz_energy",
    "rms_control_force",
    "max_abs_body_acc",
    "total_psd_energy",
    "rms_suspension_deflection",
    "rms_tire_deflection",
]


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def group_by_controller_and_road(all_summaries: list[dict]):
    grouped = defaultdict(list)

    for s in all_summaries:
        key = (
            s["controller_type"],
            s["road_type"],
        )
        grouped[key].append(s)

    return dict(grouped)


def build_benchmark_report(all_summaries: list[dict]):
    grouped = group_by_controller_and_road(all_summaries)
    report = {}

    for (controller_type, road_type), entries in grouped.items():
        key = (controller_type, road_type)

        report[key] = {
            "controller_type": controller_type,
            "road_type": road_type,
            "num_runs": len(entries),
        }

        # controller params özet olarak ilk entry'den alınabilir
        report[key]["controller_params"] = entries[0].get("controller_params", {})

        for metric in SUMMARY_METRICS:
            values = [float(e[metric]) for e in entries]
            report[key][f"{metric}_mean"] = mean(values)
            report[key][f"{metric}_std"] = std(values)

    return report


def print_benchmark_report(report: dict):
    print("\n" + "=" * 220)
    print("FULL BENCHMARK SUMMARY (MEAN ± STD)")
    print("=" * 220)

    header = (
        f"{'controller':<12}"
        f"{'road':<12}"
        f"{'runs':>8}"
        f"{'rms_body_mean':>16}"
        f"{'rms_body_std':>14}"
        f"{'band_mean':>14}"
        f"{'band_std':>12}"
        f"{'force_mean':>14}"
        f"{'force_std':>12}"
        f"{'max_body_mean':>16}"
        f"{'max_body_std':>14}"
        f"{'total_psd_mean':>16}"
    )
    print(header)
    print("-" * 220)

    for key in sorted(report.keys(), key=lambda x: (x[1], x[0])):
        s = report[key]

        row = (
            f"{s['controller_type']:<12}"
            f"{s['road_type']:<12}"
            f"{s['num_runs']:>8}"
            f"{s['rms_body_acc_mean']:>16.6f}"
            f"{s['rms_body_acc_std']:>14.6f}"
            f"{s['band_1_3Hz_energy_mean']:>14.6f}"
            f"{s['band_1_3Hz_energy_std']:>12.6f}"
            f"{s['rms_control_force_mean']:>14.6f}"
            f"{s['rms_control_force_std']:>12.6f}"
            f"{s['max_abs_body_acc_mean']:>16.6f}"
            f"{s['max_abs_body_acc_std']:>14.6f}"
            f"{s['total_psd_energy_mean']:>16.6f}"
        )
        print(row)

    print("=" * 220)