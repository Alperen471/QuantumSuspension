"""
dataset/normalizer.py
---------------------
Dataset üzerinden normalizasyon istatistiklerini hesaplar ve
JSON olarak diske kaydeder / diskten yükler.

Neden ayrı bir dosya?
---------------------
Normalizasyon istatistikleri (mean, std, min, max) hem quantum
encoder hem fuzzy membership kalibrasyonu tarafından kullanılır.
Tek bir kaynaktan üretilip her iki modüle aktarılması tutarlılığı
garanti eder — iki modül farklı ölçekler görmez.

Hangi değişkenler normalize edilir?
------------------------------------
  STATE_COLS   : quantum encoder girişi — [-1,+1] gerekli
  FUZZY_COLS   : fuzzy membership kalibrasyonu — Low/Med/High sınırları
  ALL_COLS     : ikisininde birleşimi
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# ------------------------------------------------------------------ #
#  Kolon grupları                                                      #
# ------------------------------------------------------------------ #

# Quantum encoder'a gidecek state değişkenleri
STATE_COLS = ["z_s", "z_u", "z_s_dot", "z_u_dot", "z_r"]

# Fuzzy sisteminin membership sınırlarını belirlemek için kullanılacak
# fiziksel metrikler — bunlar reward fonksiyonunun girdileri
FUZZY_COLS = ["body_acc", "susp_defl", "ctrl_force"]

# Replay buffer için tüm sürekli değişkenler
ALL_COLS = STATE_COLS + FUZZY_COLS + ["action", "reward", "tire_defl"]


# ------------------------------------------------------------------ #
#  İstatistik hesaplama                                                #
# ------------------------------------------------------------------ #

def compute_statistics(
    records: List[Dict[str, Any]],
    columns: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Verilen kayıt listesinden her kolon için
    mean, std, min, max, p05, p95 hesaplar.

    p05/p95 neden var?
    Fuzzy membership sınırları için outlier'lara karşı dayanıklı
    sınırlar gerekiyor. min/max yerine percentile kullanmak daha
    sağlam sınırlar verir — örneğin body_acc'ta nadir spike'lar
    "High" sınırını anlamsız derecede yukarı çekmesin.
    """
    if columns is None:
        columns = ALL_COLS

    stats: Dict[str, Dict[str, float]] = {}

    for col in columns:
        values = np.array([r[col] for r in records if col in r], dtype=np.float64)

        if values.size == 0:
            stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0,
                          "p05": 0.0, "p95": 1.0}
            continue

        stats[col] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)) if np.std(values) > 1e-8 else 1.0,
            "min":  float(np.min(values)),
            "max":  float(np.max(values)),
            "p05":  float(np.percentile(values, 5)),
            "p95":  float(np.percentile(values, 95)),
        }

    return stats


# ------------------------------------------------------------------ #
#  Fuzzy sınır önerileri                                               #
# ------------------------------------------------------------------ #

def suggest_fuzzy_boundaries(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    İstatistiklerden fuzzy Low/Medium/High sınırlarını otomatik önerir.

    Fuzzy sisteminde body_acc, susp_defl, ctrl_force MUTlak değer
    üzerinden değerlendirilir — negatif değerler anlamsız olur.
    Bu yüzden p05 yerine 0'dan başlayan, p95_abs üzerinden bölünen
    sınırlar kullanılır.

    Strateji:
      low_high  = p95_abs * 0.33   (alt üçte bir)
      med_high  = p95_abs * 0.67   (orta üçte bir)
      high_upper= p95_abs          (üst üçte bir)
    """
    boundaries: Dict[str, Dict[str, float]] = {}

    for col in FUZZY_COLS:
        if col not in stats:
            continue

        s = stats[col]

        # Mutlak değer üst sınırı — max(|p05|, |p95|)
        p95_abs = max(abs(s["p05"]), abs(s["p95"]))

        if p95_abs < 1e-8:
            p95_abs = abs(s["max"]) + 1e-8

        low_high   = p95_abs * 0.33
        med_high   = p95_abs * 0.67
        high_upper = p95_abs

        boundaries[col] = {
            # Membership fonksiyonu için [0, max_abs] aralığı
            "low_center":   0.0,
            "low_high":     low_high,
            "med_low":      low_high * 0.5,   # overlap için
            "med_center":   p95_abs * 0.50,
            "med_high":     med_high,
            "high_low":     med_high * 0.85,  # overlap için
            "high_center":  high_upper,
            "high_upper":   abs(s["max"]),
            # Kullanışlı özet
            "p95_abs":      p95_abs,
            "max_abs":      abs(s["max"]),
        }

    return boundaries


# ------------------------------------------------------------------ #
#  Kayıt / Yükleme                                                     #
# ------------------------------------------------------------------ #

def save_statistics(
    stats: Dict[str, Dict[str, float]],
    fuzzy_boundaries: Dict[str, Dict[str, float]],
    path: str | Path = "dataset/statistics.json",
) -> Path:
    """
    İstatistik ve fuzzy sınırlarını JSON'a yazar.

    Neden JSON?
    Human-readable — makale yazarken değerlere bakıp
    "membership sınırlarımız şunlardır" diyebilirsin.
    Hem quantum hem fuzzy modülü bu dosyayı yükler.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "statistics": stats,
        "fuzzy_boundaries": fuzzy_boundaries,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def load_statistics(path: str | Path = "dataset/statistics.json") -> tuple[dict, dict]:
    """
    Kaydedilmiş istatistikleri yükler.
    Döndürür: (stats, fuzzy_boundaries)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"İstatistik dosyası bulunamadı: {path}\n"
            "Önce 'experiments/run_dataset_collection.py' çalıştır."
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload["statistics"], payload["fuzzy_boundaries"]


# ------------------------------------------------------------------ #
#  Normalizasyon yardımcıları                                          #
# ------------------------------------------------------------------ #

class StateNormalizer:
    """
    State vektörünü [-1, +1] aralığına normalize eder.

    Neden [-1, +1]?
    RY(π·x) gate'i bu aralıkta tam Bloch küresi rotasyonu sağlar.
    Dışarıda kalan değerler rotasyonun doymasına neden olur.

    z-score değil min-max neden?
    Quantum encoding açı tabanlı — Gaussian dağılım varsayımı yok.
    Min-max, değerin [-1,+1] içinde kalmasını garanti eder.
    """

    def __init__(self, stats: Dict[str, Dict[str, float]]):
        self.stats = stats

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        state: [z_s, z_u, z_s_dot, z_u_dot, z_r] — ham değerler
        Döndürür: aynı shape, [-1,+1] aralığında
        """
        result = np.zeros_like(state, dtype=np.float32)

        for i, col in enumerate(["z_s", "z_u", "z_s_dot", "z_u_dot", "z_r"]):
            if col not in self.stats:
                result[i] = state[i]
                continue

            s = self.stats[col]
            # p05-p95 aralığını [-1,+1]'e map et
            rng = s["p95"] - s["p05"]
            if rng < 1e-8:
                result[i] = 0.0
            else:
                result[i] = np.clip(
                    2.0 * (state[i] - s["p05"]) / rng - 1.0,
                    -1.0, 1.0,
                )

        return result

    def normalize_batch(self, states: np.ndarray) -> np.ndarray:
        """states: (N, 5) — batch normalizasyon"""
        return np.stack([self.normalize(s) for s in states])


class FuzzyNormalizer:
    """
    Fuzzy giriş değişkenlerini [0, 1] aralığına normalize eder.

    Fuzzy membership fonksiyonları [0,1] üzerinde tanımlı.
    Negatif değerler anlamsız olur (body_acc absolute alınacak zaten).
    """

    def __init__(self, stats: Dict[str, Dict[str, float]]):
        self.stats = stats

    def normalize(self, col: str, value: float) -> float:
        """Tek bir değeri [0,1]'e normalize et."""
        if col not in self.stats:
            return float(value)

        s = self.stats[col]
        rng = s["p95"] - s["p05"]

        if rng < 1e-8:
            return 0.0

        return float(np.clip((abs(value) - s["p05"]) / rng, 0.0, 1.0))