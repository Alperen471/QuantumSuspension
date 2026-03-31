import numpy as np


def vertical_comfort_weighting(freqs: np.ndarray) -> np.ndarray:
    """
    ISO 2631-uyumlu tam resmi weighting yerine,
    düşey titreşim konforu için araştırma-uygun bir frekans ağırlıklandırması.

    Amaç:
    - çok düşük frekansları baskılamak
    - insan duyarlılığının yüksek olduğu orta bantları vurgulamak
    - yüksek frekansları tekrar azaltmak
    """
    freqs = np.asarray(freqs, dtype=np.float64)

    w = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        if f <= 0.0:
            w[i] = 0.0
        elif f < 0.5:
            # çok düşük frekansları zayıf tut
            w[i] = 0.2 * (f / 0.5)
        elif f < 2.0:
            # duyarlılık bandına yüksel
            w[i] = 0.2 + 0.8 * ((f - 0.5) / 1.5)
        elif f < 8.0:
            # ana konfor bandında yüksek ağırlık
            w[i] = 1.0
        elif f < 20.0:
            # yüksek frekanslarda kademeli düşüş
            w[i] = 1.0 - 0.7 * ((f - 8.0) / 12.0)
        else:
            # çok yüksek frekanslarda düşük etki
            w[i] = 0.3

    return w


def compute_weighted_rms_from_psd(freqs: np.ndarray, psd: np.ndarray, weighting: np.ndarray) -> float:
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    weighting = np.asarray(weighting, dtype=np.float64)

    if freqs.size == 0 or psd.size == 0 or weighting.size == 0:
        return 0.0

    weighted_psd = (weighting ** 2) * psd
    value = np.trapezoid(weighted_psd, freqs)
    return float(np.sqrt(max(value, 0.0)))


def compute_iso2631_weighted_metrics(freqs: np.ndarray, psd: np.ndarray) -> dict:
    w = vertical_comfort_weighting(freqs)
    weighted_rms = compute_weighted_rms_from_psd(freqs, psd, w)

    return {
        "iso_weighting": w,
        "iso_weighted_rms_body_acc": weighted_rms,
    }