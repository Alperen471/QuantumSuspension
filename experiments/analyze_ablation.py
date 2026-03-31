"""
experiments/analyze_ablation.py
--------------------------------
Ablation sonuçlarını okuyup karşılaştırma tablosu üretir.

Çalıştırma:
    python -m experiments.analyze_ablation --dir results/ablation_XXXX
    python -m experiments.analyze_ablation --dir results/ablation_XXXX --latex
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path


def load_results(results_dir: str) -> dict:
    """JSON dosyalarını yükle, model bazında grupla."""
    results_dir = Path(results_dir)
    json_files  = list(results_dir.glob("*.json"))

    if not json_files:
        # Alt dizinde ara
        json_files = list(results_dir.rglob("*.json"))

    grouped = {}
    for f in json_files:
        if "ablation_summary" in f.name:
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            model = data.get("model_name", "unknown")
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(data)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    return grouped


def compute_model_stats(runs: list) -> dict:
    """Bir modelin tüm run'larından ortalama ve std hesapla."""
    metrics = [
        "rms_body_acc_mean",
        "iso_weighted_rms_mean",
        "rms_susp_defl_mean",
        "rms_ctrl_force_mean",
        "rms_tire_defl_mean",
        "max_abs_body_acc_mean",
        "band_1_3Hz_mean",
        "total_psd_energy_mean",
    ]

    stats = {"n_runs": len(runs)}

    for m in metrics:
        vals = []
        for run in runs:
            fm = run.get("final_metrics", {})
            v  = fm.get(m)
            if v is not None:
                vals.append(float(v))
        if vals:
            stats[f"{m}_avg"] = float(np.mean(vals))
            stats[f"{m}_std"] = float(np.std(vals))
        else:
            stats[f"{m}_avg"] = float("nan")
            stats[f"{m}_std"] = float("nan")

    # Road tipi bazında kır
    road_stats = {}
    for run in runs:
        road = run.get("road_type", "unknown")
        if road not in road_stats:
            road_stats[road] = []
        fm = run.get("final_metrics", {})
        v  = fm.get("rms_body_acc_mean")
        if v is not None:
            road_stats[road].append(float(v))

    stats["by_road"] = {
        road: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for road, vals in road_stats.items()
    }

    return stats


def print_summary_table(all_stats: dict, baseline_model: str = "rl_baseline"):
    """Ana karşılaştırma tablosu."""
    print(f"\n{'='*90}")
    print(f"ABLATION SONUÇ TABLOSU")
    print(f"{'='*90}")
    print(
        f"{'Model':<22} {'n':>3} {'rms_body':>10} {'±':>6} "
        f"{'iso_rms':>9} {'±':>6} {'ctrl_N':>9} {'susp_defl':>10}"
    )
    print(f"{'-'*90}")

    # Baseline değerleri
    baseline = all_stats.get(baseline_model, {})
    b_rms    = baseline.get("rms_body_acc_mean_avg", float("nan"))

    for model, stats in sorted(all_stats.items()):
        rms  = stats.get("rms_body_acc_mean_avg", float("nan"))
        rms_s= stats.get("rms_body_acc_mean_std", float("nan"))
        iso  = stats.get("iso_weighted_rms_mean_avg", float("nan"))
        iso_s= stats.get("iso_weighted_rms_mean_std", float("nan"))
        ctrl = stats.get("rms_ctrl_force_mean_avg", float("nan"))
        susp = stats.get("rms_susp_defl_mean_avg", float("nan"))
        n    = stats.get("n_runs", 0)

        # Baseline'a göre iyileşme
        if not np.isnan(rms) and not np.isnan(b_rms) and b_rms > 0:
            impr = (b_rms - rms) / b_rms * 100
            impr_str = f"({impr:+.1f}%)" if model != baseline_model else ""
        else:
            impr_str = ""

        print(
            f"  {model:<20} {n:>3} {rms:>10.4f} {rms_s:>6.4f} "
            f"{iso:>9.4f} {iso_s:>6.4f} {ctrl:>9.2f} {susp:>10.6f} "
            f"{impr_str}"
        )

    print(f"{'='*90}")


def print_road_breakdown(all_stats: dict):
    """Road tipi bazında kırılım tablosu."""
    print(f"\n{'='*75}")
    print(f"ROAD TİPİ BAZINDA rms_body_acc")
    print(f"{'='*75}")

    road_types = ["random", "sinusoidal", "bump", "pothole"]
    header = f"{'Model':<22}"
    for r in road_types:
        header += f" {r:>12}"
    print(header)
    print(f"{'-'*75}")

    for model, stats in sorted(all_stats.items()):
        row = f"  {model:<20}"
        by_road = stats.get("by_road", {})
        for r in road_types:
            v = by_road.get(r, {}).get("mean", float("nan"))
            row += f" {v:>12.4f}"
        print(row)

    print(f"{'='*75}")


def print_improvement_matrix(all_stats: dict, baseline: str = "rl_baseline"):
    """Her modelin baseline'a göre iyileşme matrisi."""
    b_stats = all_stats.get(baseline, {})
    b_rms   = b_stats.get("rms_body_acc_mean_avg", float("nan"))
    b_iso   = b_stats.get("iso_weighted_rms_mean_avg", float("nan"))

    if np.isnan(b_rms):
        print(f"\n[WARN] Baseline '{baseline}' bulunamadı.")
        return

    print(f"\n{'='*65}")
    print(f"BASELINE'A GÖRE İYİLEŞME")
    print(f"Baseline: {baseline}  (rms_body={b_rms:.4f}, iso={b_iso:.4f})")
    print(f"{'='*65}")
    print(f"{'Model':<22} {'rms iyileşme':>14} {'iso iyileşme':>14} {'karar':>10}")
    print(f"{'-'*65}")

    for model, stats in sorted(all_stats.items()):
        if model == baseline:
            continue
        rms = stats.get("rms_body_acc_mean_avg", float("nan"))
        iso = stats.get("iso_weighted_rms_mean_avg", float("nan"))

        if np.isnan(rms):
            continue

        rms_impr = (b_rms - rms) / b_rms * 100
        iso_impr = (b_iso - iso) / b_iso * 100

        if rms_impr > 5 and iso_impr > 5:
            karar = "✓ BAŞARILI"
        elif rms_impr > 0:
            karar = "~ KISMI"
        else:
            karar = "✗ KÖTÜ"

        print(
            f"  {model:<20} {rms_impr:>+13.1f}% {iso_impr:>+13.1f}% "
            f"{karar:>10}"
        )

    print(f"{'='*65}")


def print_latex_table(all_stats: dict):
    """LaTeX tablosu — makaleye direkt kopyala."""
    print(f"\n% LaTeX Tablosu")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Ablation Study Results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Model & RMS Body Acc & ISO-2631 RMS & RMS Ctrl Force & Susp. Defl \\")
    print(r"\hline")

    for model, stats in sorted(all_stats.items()):
        rms  = stats.get("rms_body_acc_mean_avg", float("nan"))
        rms_s= stats.get("rms_body_acc_mean_std", float("nan"))
        iso  = stats.get("iso_weighted_rms_mean_avg", float("nan"))
        iso_s= stats.get("iso_weighted_rms_mean_std", float("nan"))
        ctrl = stats.get("rms_ctrl_force_mean_avg", float("nan"))
        susp = stats.get("rms_susp_defl_mean_avg", float("nan"))

        name = model.replace("_", r"\_")
        print(
            f"{name} & "
            f"${rms:.4f} \\pm {rms_s:.4f}$ & "
            f"${iso:.4f} \\pm {iso_s:.4f}$ & "
            f"${ctrl:.2f}$ & "
            f"${susp:.6f}$ \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Analizi")
    parser.add_argument("--dir",     type=str, required=True,
                        help="Ablation sonuç dizini")
    parser.add_argument("--baseline",type=str, default="rl_baseline")
    parser.add_argument("--latex",   action="store_true",
                        help="LaTeX tablosu üret")
    args = parser.parse_args()

    print(f"\nSonuç dizini: {args.dir}")

    # Yükle
    grouped = load_results(args.dir)

    if not grouped:
        print("Sonuç bulunamadı!")
        return

    print(f"Bulunan modeller: {sorted(grouped.keys())}")
    print(f"Run sayıları: { {m: len(r) for m,r in grouped.items()} }")

    # İstatistik hesapla
    all_stats = {
        model: compute_model_stats(runs)
        for model, runs in grouped.items()
    }

    # Tablolar
    print_summary_table(all_stats, baseline=args.baseline)
    print_road_breakdown(all_stats)
    print_improvement_matrix(all_stats, baseline=args.baseline)

    if args.latex:
        print_latex_table(all_stats)


if __name__ == "__main__":
    main()
