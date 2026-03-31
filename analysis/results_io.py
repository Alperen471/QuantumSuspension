import csv
import json
from pathlib import Path
from datetime import datetime


RESULTS_ROOT = Path("results")
JSON_DIR = RESULTS_ROOT / "json"
CSV_DIR = RESULTS_ROOT / "csv"
PLOTS_DIR = RESULTS_ROOT / "plots"


def ensure_results_dirs():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_experiment_filename(summary: dict, timestamp: str):
    controller = summary["controller_type"]
    road = summary["road_type"]
    seed = summary.get("seed","na")

    params = summary.get("controller_params",{})
    c_sky = params.get("c_sky","na")
    return f"{controller}_{road}_gain{c_sky}_seed{seed}_{timestamp}"


def save_summary_json(summary: dict):
    ensure_results_dirs()

    timestamp = build_timestamp()
    filename = build_experiment_filename(summary, timestamp)
    path = JSON_DIR / f"{filename}.json"

    payload = dict(summary)
    payload["timestamp"] = timestamp

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def append_summary_csv(summary: dict):
    ensure_results_dirs()

    path = CSV_DIR / "benchmark_summary.csv"

    row = {
        "seed": summary.get("seed",""),
        "controller_type": summary["controller_type"],
        "c_sky": summary.get("controller_params",{}).get("c_sky",""),
        "max_force": summary.get("controller_params", {}).get("max_force", ""),
        "road_type": summary["road_type"],
        "road_amplitude": summary["road_amplitude"],
        "road_frequency": summary["road_frequency"],
        "num_steps": summary["num_steps"],
        "mean_reward": summary["mean_reward"],
        "rms_body_acc": summary["rms_body_acc"],
        "rms_body_acc_label": summary["labels"]["rms_body_acc"],
        "max_abs_body_acc": summary["max_abs_body_acc"],
        "max_abs_body_acc_label": summary["labels"]["max_abs_body_acc"],
        "rms_wheel_acc": summary["rms_wheel_acc"],
        "max_abs_wheel_acc": summary["max_abs_wheel_acc"],
        "rms_control_force": summary["rms_control_force"],
        "rms_control_force_label": summary["labels"]["rms_control_force"],
        "max_abs_control_force": summary["max_abs_control_force"],
        "rms_suspension_deflection": summary["rms_suspension_deflection"],
        "rms_suspension_label": summary["labels"]["rms_suspension_deflection"],
        "max_abs_suspension_deflection": summary["max_abs_suspension_deflection"],
        "rms_tire_deflection": summary["rms_tire_deflection"],
        "rms_tire_label": summary["labels"]["rms_tire_deflection"],
        "max_abs_tire_deflection": summary["max_abs_tire_deflection"],
        "total_psd_energy": summary["total_psd_energy"],
        "total_psd_label": summary["labels"]["total_psd_energy"],
        "band_1_3Hz_energy": summary["band_1_3Hz_energy"],
        "band_1_3Hz_label": summary["labels"]["band_1_3Hz_energy"],
        "band_0_5_5Hz_energy": summary["band_0_5_5Hz_energy"],
        "dominant_frequency": summary["dominant_frequency"],
        "iso_weighted_rms_body_acc": summary["iso_weighted_rms_body_acc"],
        "iso_weighted_rms_body_acc_label": summary["labels"]["iso_weighted_rms_body_acc"],
        
    }

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path