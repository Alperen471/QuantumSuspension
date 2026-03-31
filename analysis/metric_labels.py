def label_rms_body_acc(v):
    if v < 1.0:
        return "çok iyi"
    elif v < 2.0:
        return "iyi"
    elif v < 3.0:
        return "orta"
    elif v < 4.5:
        return "kötü"
    return "çok kötü"


def label_max_body_acc(v):
    if v < 3.0:
        return "çok iyi"
    elif v < 6.0:
        return "iyi"
    elif v < 9.0:
        return "orta"
    elif v < 13.0:
        return "kötü"
    return "çok kötü"


def label_suspension(v):
    if v < 0.005:
        return "çok iyi"
    elif v < 0.015:
        return "iyi"
    elif v < 0.03:
        return "orta"
    elif v < 0.05:
        return "kötü"
    return "çok kötü"


def label_tire(v):
    if v < 0.005:
        return "çok iyi"
    elif v < 0.015:
        return "iyi"
    elif v < 0.03:
        return "orta"
    elif v < 0.05:
        return "kötü"
    return "çok kötü"


def label_control_force(v):
    if v < 100.0:
        return "çok iyi"
    elif v < 400.0:
        return "iyi"
    elif v < 900.0:
        return "orta"
    elif v < 1500.0:
        return "kötü"
    return "çok kötü"


def label_band_energy(v):
    if v < 0.01:
        return "çok iyi"
    elif v < 0.03:
        return "iyi"
    elif v < 0.07:
        return "orta"
    elif v < 0.12:
        return "kötü"
    return "çok kötü"


def label_total_psd(v):
    if v < 1.0:
        return "çok iyi"
    elif v < 3.0:
        return "iyi"
    elif v < 6.0:
        return "orta"
    elif v < 10.0:
        return "kötü"
    return "çok kötü"

def label_iso_weighted_rms(v):
    if v < 0.315:
        return "çok iyi"
    elif v < 0.63:
        return "iyi"
    elif v < 1.0:
        return "orta"
    elif v < 1.6:
        return "kötü"
    return "çok kötü"