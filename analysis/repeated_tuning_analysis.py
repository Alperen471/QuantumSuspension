import math
from collections import defaultdict


PRIMARY_METRICS = [
    "rms_body_acc",
    "band_1_3Hz_energy",
    "rms_control_force",
    "max_abs_body_acc",
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


def group_by_gain_and_road(all_summaries: list[dict]):
    grouped = defaultdict(list)

    for s in all_summaries:
        if s.get("controller_type") != "skyhook":
            continue

        gain = float(s.get("controller_params", {}).get("c_sky"))
        road = s.get("road_type")
        grouped[(gain, road)].append(s)

    return dict(grouped)


def aggregate_repeated_stats(all_summaries: list[dict]):
    grouped = group_by_gain_and_road(all_summaries)

    stats = {}

    for (gain, road), entries in grouped.items():
        key = (gain, road)
        stats[key] = {
            "c_sky": gain,
            "road_type": road,
            "num_runs": len(entries),
        }

        for metric in PRIMARY_METRICS:
            values = [float(e[metric]) for e in entries]
            stats[key][f"{metric}_mean"] = mean(values)
            stats[key][f"{metric}_std"] = std(values)

    return stats


def aggregate_gain_objectives(stats: dict, road_weights=None):
    """
    Gain bazında random + sinusoidal sonuçlarını birleştirir.
    Mean metrikler üzerinden aggregate objective üretir.
    """
    if road_weights is None:
        road_weights = {
            "random": 0.5,
            "sinusoidal": 0.5,
        }

    gains = sorted({key[0] for key in stats.keys()})
    aggregated = {}

    for gain in gains:
        aggregated[gain] = {"c_sky": gain}

        for metric in PRIMARY_METRICS:
            weighted_sum = 0.0
            total_weight = 0.0

            for road, weight in road_weights.items():
                key = (gain, road)
                if key not in stats:
                    continue

                weighted_sum += weight * stats[key][f"{metric}_mean"]
                total_weight += weight

            if total_weight == 0.0:
                aggregated[gain][metric] = float("inf")
            else:
                aggregated[gain][metric] = weighted_sum / total_weight

        # robustness göstergesi: std’lerin ortalaması
        for metric in PRIMARY_METRICS:
            weighted_std_sum = 0.0
            total_weight = 0.0

            for road, weight in road_weights.items():
                key = (gain, road)
                if key not in stats:
                    continue

                weighted_std_sum += weight * stats[key][f"{metric}_std"]
                total_weight += weight

            if total_weight == 0.0:
                aggregated[gain][f"{metric}_std"] = float("inf")
            else:
                aggregated[gain][f"{metric}_std"] = weighted_std_sum / total_weight

    return aggregated


def min_max_normalize(values):
    vmin = min(values)
    vmax = max(values)

    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def normalize_aggregated(aggregated: dict):
    gains = sorted(aggregated.keys())
    normalized = {g: {"c_sky": g} for g in gains}

    for metric in PRIMARY_METRICS:
        values = [aggregated[g][metric] for g in gains]
        nvalues = min_max_normalize(values)

        for g, nv in zip(gains, nvalues):
            normalized[g][metric] = nv

    return normalized


def dominates(a: dict, b: dict, metrics):
    not_worse = all(a[m] <= b[m] for m in metrics)
    strictly_better = any(a[m] < b[m] for m in metrics)
    return not_worse and strictly_better


def find_pareto_front(normalized: dict):
    gains = sorted(normalized.keys())
    pareto = []

    for g in gains:
        candidate = normalized[g]
        dominated = False

        for other_g in gains:
            if other_g == g:
                continue

            other = normalized[other_g]
            if dominates(other, candidate, PRIMARY_METRICS):
                dominated = True
                break

        if not dominated:
            pareto.append(g)

    return pareto


def euclidean_norm(entry: dict, metrics):
    return math.sqrt(sum(entry[m] ** 2 for m in metrics))


def select_knee_point(normalized: dict, aggregated: dict, pareto_gains: list[float]):
    """
    Basit ama daha sağlam seçim:
    - önce Pareto kümesi
    - sonra normalize objective normu
    - eşitlik yakınsa daha düşük control force std tercih edilir
    """
    if not pareto_gains:
        return None

    best_gain = None
    best_score = float("inf")
    best_tiebreak = float("inf")

    for g in pareto_gains:
        score = euclidean_norm(normalized[g], PRIMARY_METRICS)
        tiebreak = aggregated[g]["rms_control_force_std"]

        if score < best_score:
            best_gain = g
            best_score = score
            best_tiebreak = tiebreak
        elif math.isclose(score, best_score, rel_tol=1e-9, abs_tol=1e-9):
            if tiebreak < best_tiebreak:
                best_gain = g
                best_tiebreak = tiebreak

    return best_gain, best_score


def build_repeated_tuning_report(all_summaries: list[dict], road_weights=None):
    stats = aggregate_repeated_stats(all_summaries)
    aggregated = aggregate_gain_objectives(stats, road_weights=road_weights)
    normalized = normalize_aggregated(aggregated)
    pareto = find_pareto_front(normalized)
    knee_result = select_knee_point(normalized, aggregated, pareto)

    return {
        "stats": stats,
        "aggregated": aggregated,
        "normalized": normalized,
        "pareto_gains": pareto,
        "knee_result": knee_result,
    }


def print_repeated_tuning_summary(report: dict):
    stats = report["stats"]

    print("\n" + "=" * 190)
    print("REPEATED SKYHOOK TUNING SUMMARY (MEAN ± STD)")
    print("=" * 190)

    header = (
        f"{'road':<12}"
        f"{'c_sky':>10}"
        f"{'runs':>8}"
        f"{'rms_body_mean':>16}"
        f"{'rms_body_std':>14}"
        f"{'band_mean':>14}"
        f"{'band_std':>12}"
        f"{'force_mean':>14}"
        f"{'force_std':>12}"
        f"{'max_body_mean':>16}"
        f"{'max_body_std':>14}"
    )
    print(header)
    print("-" * 190)

    for (gain, road) in sorted(stats.keys(), key=lambda x: (x[1], x[0])):
        s = stats[(gain, road)]
        row = (
            f"{road:<12}"
            f"{gain:>10.1f}"
            f"{s['num_runs']:>8}"
            f"{s['rms_body_acc_mean']:>16.6f}"
            f"{s['rms_body_acc_std']:>14.6f}"
            f"{s['band_1_3Hz_energy_mean']:>14.6f}"
            f"{s['band_1_3Hz_energy_std']:>12.6f}"
            f"{s['rms_control_force_mean']:>14.6f}"
            f"{s['rms_control_force_std']:>12.6f}"
            f"{s['max_abs_body_acc_mean']:>16.6f}"
            f"{s['max_abs_body_acc_std']:>14.6f}"
        )
        print(row)

    print("=" * 190)


def print_repeated_pareto_report(report: dict):
    aggregated = report["aggregated"]
    normalized = report["normalized"]
    pareto_gains = report["pareto_gains"]
    knee_result = report["knee_result"]

    print("\n" + "=" * 190)
    print("REPEATED-RUN PARETO ANALYSIS")
    print("=" * 190)

    header = (
        f"{'c_sky':>10}"
        f"{'agg_rms_body':>16}"
        f"{'agg_band':>14}"
        f"{'agg_force':>14}"
        f"{'agg_max_body':>16}"
        f"{'std_force':>12}"
        f"{'n_rms_body':>14}"
        f"{'n_band':>12}"
        f"{'n_force':>12}"
        f"{'n_max_body':>14}"
        f"{'pareto':>10}"
    )
    print(header)
    print("-" * 190)

    for g in sorted(aggregated.keys()):
        agg = aggregated[g]
        norm = normalized[g]
        is_pareto = "yes" if g in pareto_gains else "no"

        row = (
            f"{g:>10.1f}"
            f"{agg['rms_body_acc']:>16.6f}"
            f"{agg['band_1_3Hz_energy']:>14.6f}"
            f"{agg['rms_control_force']:>14.6f}"
            f"{agg['max_abs_body_acc']:>16.6f}"
            f"{agg['rms_control_force_std']:>12.6f}"
            f"{norm['rms_body_acc']:>14.6f}"
            f"{norm['band_1_3Hz_energy']:>12.6f}"
            f"{norm['rms_control_force']:>12.6f}"
            f"{norm['max_abs_body_acc']:>14.6f}"
            f"{is_pareto:>10}"
        )
        print(row)

    print("=" * 190)
    print("Pareto-optimal gains:", pareto_gains)

    if knee_result is not None:
        best_gain, best_score = knee_result
        print(f"Knee-point gain     : {best_gain}")
        print(f"Knee-point score    : {best_score:.6f}")
    else:
        print("Knee-point gain     : None")

    print("=" * 190)