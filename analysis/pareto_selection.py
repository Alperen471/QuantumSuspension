import math
from collections import defaultdict


PRIMARY_METRICS = [
    "rms_body_acc",
    "band_1_3Hz_energy",
    "rms_control_force",
    "max_abs_body_acc",
]


def group_summaries_by_gain(all_summaries: list[dict], controller_type: str = "skyhook"):
    grouped = defaultdict(list)

    for summary in all_summaries:
        if summary.get("controller_type") != controller_type:
            continue

        c_sky = summary.get("controller_params", {}).get("c_sky", None)
        if c_sky is None:
            continue

        grouped[float(c_sky)].append(summary)

    return dict(grouped)


def aggregate_gain_metrics(grouped_by_gain: dict, road_weights: dict | None = None):
    """
    Her gain için farklı road sonuçlarını tek objective vektörüne indirger.

    road_weights örnek:
        {
            "random": 0.6,
            "sinusoidal": 0.4
        }
    """
    if road_weights is None:
        road_weights = {
            "random": 0.5,
            "sinusoidal": 0.5,
        }

    aggregated = {}

    for gain, summaries in grouped_by_gain.items():
        gain_result = {"c_sky": gain}

        for metric in PRIMARY_METRICS:
            weighted_sum = 0.0
            total_weight = 0.0

            for s in summaries:
                road_type = s["road_type"]
                weight = road_weights.get(road_type, 0.0)

                if weight <= 0:
                    continue

                weighted_sum += weight * float(s[metric])
                total_weight += weight

            if total_weight == 0.0:
                gain_result[metric] = float("inf")
            else:
                gain_result[metric] = weighted_sum / total_weight

        # yardımcı rapor bilgileri
        gain_result["roads_present"] = sorted([s["road_type"] for s in summaries])

        aggregated[gain] = gain_result

    return aggregated


def min_max_normalize(values: list[float]):
    vmin = min(values)
    vmax = max(values)

    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def normalize_aggregated_metrics(aggregated: dict):
    """
    Her metrik için gain'ler arasında min-max normalize eder.
    Tüm metrikler minimization objective kabul edilir.
    """
    gains = sorted(aggregated.keys())

    normalized = {g: {"c_sky": g} for g in gains}

    for metric in PRIMARY_METRICS:
        values = [aggregated[g][metric] for g in gains]
        norm_values = min_max_normalize(values)

        for g, nv in zip(gains, norm_values):
            normalized[g][metric] = nv

    return normalized


def dominates(a: dict, b: dict, metrics: list[str]):
    """
    a, b'yi domine eder mi?
    Tüm metrikler minimize ediliyor kabul edilir.
    """
    not_worse_in_all = all(a[m] <= b[m] for m in metrics)
    strictly_better_in_at_least_one = any(a[m] < b[m] for m in metrics)
    return not_worse_in_all and strictly_better_in_at_least_one


def find_pareto_front(normalized_aggregated: dict):
    gains = sorted(normalized_aggregated.keys())
    pareto_gains = []

    for g in gains:
        candidate = normalized_aggregated[g]
        dominated = False

        for other_g in gains:
            if other_g == g:
                continue

            other = normalized_aggregated[other_g]

            if dominates(other, candidate, PRIMARY_METRICS):
                dominated = True
                break

        if not dominated:
            pareto_gains.append(g)

    return pareto_gains


def euclidean_norm_of_metrics(entry: dict, metrics: list[str]):
    return math.sqrt(sum((entry[m] ** 2) for m in metrics))


def select_knee_point(normalized_aggregated: dict, pareto_gains: list[float]):
    """
    Basit knee-point seçimi:
    Normalize objective uzayında orijine en yakın Pareto çözümü.
    """
    if not pareto_gains:
        return None

    best_gain = None
    best_score = float("inf")

    for g in pareto_gains:
        score = euclidean_norm_of_metrics(normalized_aggregated[g], PRIMARY_METRICS)
        if score < best_score:
            best_score = score
            best_gain = g

    return best_gain, best_score


def build_pareto_report(all_summaries: list[dict], road_weights: dict | None = None):
    grouped = group_summaries_by_gain(all_summaries, controller_type="skyhook")
    aggregated = aggregate_gain_metrics(grouped, road_weights=road_weights)
    normalized = normalize_aggregated_metrics(aggregated)
    pareto_gains = find_pareto_front(normalized)
    knee_result = select_knee_point(normalized, pareto_gains)

    report = {
        "grouped": grouped,
        "aggregated": aggregated,
        "normalized": normalized,
        "pareto_gains": pareto_gains,
        "knee_result": knee_result,
    }

    return report


def print_pareto_report(report: dict):
    aggregated = report["aggregated"]
    normalized = report["normalized"]
    pareto_gains = report["pareto_gains"]
    knee_result = report["knee_result"]

    print("\n" + "=" * 160)
    print("SKYHOOK GAIN PARETO ANALYSIS")
    print("=" * 160)

    header = (
        f"{'c_sky':>10}"
        f"{'rms_body':>14}"
        f"{'band_1_3Hz':>14}"
        f"{'rms_force':>14}"
        f"{'max_body':>14}"
        f"{'n_rms_body':>14}"
        f"{'n_band':>14}"
        f"{'n_force':>14}"
        f"{'n_max_body':>14}"
        f"{'pareto':>10}"
    )
    print(header)
    print("-" * 160)

    for g in sorted(aggregated.keys()):
        agg = aggregated[g]
        norm = normalized[g]
        is_pareto = "yes" if g in pareto_gains else "no"

        row = (
            f"{g:>10.1f}"
            f"{agg['rms_body_acc']:>14.6f}"
            f"{agg['band_1_3Hz_energy']:>14.6f}"
            f"{agg['rms_control_force']:>14.6f}"
            f"{agg['max_abs_body_acc']:>14.6f}"
            f"{norm['rms_body_acc']:>14.6f}"
            f"{norm['band_1_3Hz_energy']:>14.6f}"
            f"{norm['rms_control_force']:>14.6f}"
            f"{norm['max_abs_body_acc']:>14.6f}"
            f"{is_pareto:>10}"
        )
        print(row)

    print("=" * 160)

    print("\nPareto-optimal gains:", pareto_gains)

    if knee_result is not None:
        best_gain, best_score = knee_result
        print(f"Knee-point gain     : {best_gain}")
        print(f"Knee-point score    : {best_score:.6f}")
    else:
        print("Knee-point gain     : None")

    print("=" * 160)