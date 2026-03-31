import numpy as np
from scipy.signal import welch
from analysis.iso2631_metrics import compute_iso2631_weighted_metrics


def band_energy(freqs, psd, f_low, f_high):
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(psd[mask], freqs[mask]))


def dominant_frequency(freqs, psd):
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    if freqs.size == 0 or psd.size == 0:
        return 0.0

    idx = int(np.argmax(psd))
    return float(freqs[idx])


def compute_psd(signal, fs, nperseg=None):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        return np.array([]), np.array([])

    if nperseg is None:
        nperseg = min(256, signal.size)

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def compute_frequency_domain_metrics(results: dict, dt: float) -> dict:
    infos = results.get("infos", [])

    body_acc = np.array([info["body_acc"] for info in infos], dtype=np.float64)

    fs = 1.0 / dt
    freqs, psd = compute_psd(body_acc, fs=fs)

    total_energy = float(np.trapezoid(psd, freqs)) if freqs.size > 0 else 0.0
    band_1_3_energy = band_energy(freqs, psd, 1.0, 3.0)
    band_0_5_5_energy = band_energy(freqs, psd, 0.5, 5.0)
    dom_freq = dominant_frequency(freqs, psd)

    iso_metrics = compute_iso2631_weighted_metrics(freqs, psd)

    return {
        "psd_freqs": freqs,
        "psd_body_acc": psd,
        "total_psd_energy": total_energy,
        "band_1_3Hz_energy": band_1_3_energy,
        "band_0_5_5Hz_energy": band_0_5_5_energy,
        "dominant_frequency": dom_freq,
        "iso_weighted_rms_body_acc": iso_metrics["iso_weighted_rms_body_acc"],
        "iso_weighting": iso_metrics["iso_weighting"],
    }