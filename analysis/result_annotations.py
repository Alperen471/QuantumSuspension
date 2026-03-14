from analysis.metric_labels import (
    label_rms_body_acc,
    label_max_body_acc,
    label_suspension,
    label_tire,
    label_control_force,
    label_band_energy,
    label_total_psd,
)
from analysis.metric_labels import label_iso_weighted_rms


def add_metric_labels(summary: dict) -> dict:
    annotated = dict(summary)

    annotated["labels"] = {
        "rms_body_acc": label_rms_body_acc(summary["rms_body_acc"]),
        "max_abs_body_acc": label_max_body_acc(summary["max_abs_body_acc"]),
        "rms_suspension_deflection": label_suspension(summary["rms_suspension_deflection"]),
        "rms_tire_deflection": label_tire(summary["rms_tire_deflection"]),
        "rms_control_force": label_control_force(summary["rms_control_force"]),
        "band_1_3Hz_energy": label_band_energy(summary["band_1_3Hz_energy"]),
        "total_psd_energy": label_total_psd(summary["total_psd_energy"]),
        "iso_weighted_rms_body_acc": label_iso_weighted_rms(summary["iso_weighted_rms_body_acc"]),
    }

    return annotated