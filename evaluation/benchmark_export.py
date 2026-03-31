import csv
import json
from pathlib import Path
from datetime import datetime


RESULTS_ROOT = Path("results")
CSV_DIR = RESULTS_ROOT / "csv"
JSON_DIR = RESULTS_ROOT / "json"


def ensure_export_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_benchmark_report_csv(report: dict, filename: str = "full_benchmark_summary.csv"):
    ensure_export_dirs()

    path = CSV_DIR / filename

    rows = []
    for key in sorted(report.keys(), key=lambda x: (x[1], x[0])):
        s = report[key]
        row = {
            "controller_type": s["controller_type"],
            "road_type": s["road_type"],
            "num_runs": s["num_runs"],
            "controller_params": json.dumps(s.get("controller_params", {}), ensure_ascii=False),
            "rms_body_acc_mean": s["rms_body_acc_mean"],
            "rms_body_acc_std": s["rms_body_acc_std"],
            "band_1_3Hz_energy_mean": s["band_1_3Hz_energy_mean"],
            "band_1_3Hz_energy_std": s["band_1_3Hz_energy_std"],
            "rms_control_force_mean": s["rms_control_force_mean"],
            "rms_control_force_std": s["rms_control_force_std"],
            "max_abs_body_acc_mean": s["max_abs_body_acc_mean"],
            "max_abs_body_acc_std": s["max_abs_body_acc_std"],
            "total_psd_energy_mean": s["total_psd_energy_mean"],
            "total_psd_energy_std": s["total_psd_energy_std"],
            "rms_suspension_deflection_mean": s["rms_suspension_deflection_mean"],
            "rms_suspension_deflection_std": s["rms_suspension_deflection_std"],
            "rms_tire_deflection_mean": s["rms_tire_deflection_mean"],
            "rms_tire_deflection_std": s["rms_tire_deflection_std"],
        }
        rows.append(row)

    if not rows:
        return path

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_benchmark_report_json(report: dict, filename: str = "full_benchmark_summary.json"):
    ensure_export_dirs()

    path = JSON_DIR / filename

    serializable = {}
    for key, value in report.items():
        serializable[str(key)] = value

    payload = {
        "timestamp": build_timestamp(),
        "report": serializable,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path