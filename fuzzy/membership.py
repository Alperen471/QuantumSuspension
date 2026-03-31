"""
fuzzy/membership.py
-------------------
Üçgen ve trapezoid membership fonksiyonları.

Neden üçgen?
------------
Her RL step'te çağrılır — hesaplama hızı kritik.
Üçgen membership O(1) — sadece iki karşılaştırma ve
doğrusal interpolasyon. Gaussian'dan ~10x hızlı.
Yorumlanabilirliği yüksek — makale şekilleri net.

Membership fonksiyonu anatomisi (üçgen):
           1.0
            /\
           /  \
          /    \
    -----/      \-----
    a    b    c
    a: sol ayak (üyelik başlar)
    b: tepe (tam üyelik = 1.0)
    c: sağ ayak (üyelik biter)
"""

import numpy as np
from typing import NamedTuple


# ------------------------------------------------------------------ #
#  Membership fonksiyon tipleri                                       #
# ------------------------------------------------------------------ #

class TriMF(NamedTuple):
    """Üçgen membership fonksiyonu parametreleri."""
    a: float  # sol ayak
    b: float  # tepe
    c: float  # sağ ayak


class TrapMF(NamedTuple):
    """Trapezoid membership fonksiyonu parametreleri."""
    a: float  # sol ayak başı
    b: float  # sol tepe başı
    c: float  # sağ tepe sonu
    d: float  # sağ ayak sonu


# ------------------------------------------------------------------ #
#  Temel hesaplama fonksiyonları                                      #
# ------------------------------------------------------------------ #

def trimf(x: float, mf: TriMF) -> float:
    """
    Üçgen membership değeri hesapla.

    x: giriş değeri (mutlak değer, ≥ 0)
    Döndürür: [0, 1] aralığında üyelik derecesi
    """
    a, b, c = mf.a, mf.b, mf.c

    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        # Sol taraf — yükselen
        return (x - a) / (b - a) if (b - a) > 1e-10 else 1.0
    else:
        # Sağ taraf — düşen
        return (c - x) / (c - b) if (c - b) > 1e-10 else 1.0


def trapmf(x: float, mf: TrapMF) -> float:
    """
    Trapezoid membership değeri hesapla.
    Low gibi sınır değerleri için kullanılır —
    sıfırdan tam üyeliğe kadar düz bir bant var.
    """
    a, b, c, d = mf.a, mf.b, mf.c, mf.d

    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if (b - a) > 1e-10 else 1.0
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if (d - c) > 1e-10 else 1.0


def trimf_batch(x: np.ndarray, mf: TriMF) -> np.ndarray:
    """Vektörleştirilmiş üçgen membership — batch hesaplama için."""
    a, b, c = mf.a, mf.b, mf.c
    result = np.zeros_like(x, dtype=np.float32)

    left_mask  = (x > a) & (x <= b)
    right_mask = (x > b) & (x < c)

    denom_l = b - a
    denom_r = c - b

    if denom_l > 1e-10:
        result[left_mask]  = (x[left_mask]  - a) / denom_l
    else:
        result[left_mask]  = 1.0

    if denom_r > 1e-10:
        result[right_mask] = (c - x[right_mask]) / denom_r
    else:
        result[right_mask] = 1.0

    return result


# ------------------------------------------------------------------ #
#  Değişken başına membership seti                                    #
# ------------------------------------------------------------------ #

class MembershipSet:
    """
    Tek bir fuzzy giriş değişkeni için Low/Med/High membership seti.

    statistics.json'dan gelen p95_abs değeri ile otomatik kalibre edilir.
    Her üç seviye çakışmalı (overlapping) üçgenlerden oluşur.

    Neden çakışmalı?
    Keskin sınırlar reward'da ani sıçramalara yol açar.
    Çakışma, geçişlerin yumuşak (smooth) olmasını sağlar.
    """

    def __init__(
        self,
        name    : str,
        p95_abs : float,
        max_abs : float | None = None,
        overlap : float = 0.3,
    ):
        """
        Args:
            name     : değişken adı (loglama için)
            p95_abs  : dataset %95 mutlak değer üst sınırı
            max_abs  : dataset mutlak maksimum — High üçgeninin
                       sağ ayağını belirler, kör bölge oluşmaz.
                       None ise p95_abs * 5.0 kullanılır.
            overlap  : Low-Med ve Med-High çakışma oranı [0,1]
        """
        self.name    = name
        self.p95_abs = p95_abs
        self.max_abs = max_abs if max_abs is not None else p95_abs * 5.0
        self.overlap = overlap

        self._build_membership_functions()

    def _build_membership_functions(self):
        """
        p95_abs ve max_abs'den Low/Med/High üçgenlerini oluşturur.

        Tepe noktaları p95_abs'e göre belirlenir — veride sıkça
        görülen değerlerin tam üyeliğini sağlar.

        High üçgeninin sağ ayağı max_abs * 1.5 ile uzatılır.
        Bu sayede tümsek/çukur gibi ani olaylarda p95 aşılsa bile
        sistem kör kalmaz — her değer bir membership seviyesine girer.
        """
        p  = self.p95_abs
        mx = self.max_abs
        ov = self.overlap

        # Tepe noktaları
        low_peak  = 0.0
        med_peak  = p * 0.5
        high_peak = p

        # High sağ ayak: max_abs * 1.5 — kör bölge yok
        right_tail = max(mx * 1.5, p * 4.0)

        self.low = TriMF(
            a = -p * 0.1,
            b = low_peak,
            c = p * (0.33 + ov),
        )
        self.med = TriMF(
            a = p * (0.33 - ov * 0.5),
            b = med_peak,
            c = p * (0.67 + ov * 0.5),
        )
        self.high = TriMF(
            a = p * (0.67 - ov),
            b = high_peak,
            c = right_tail,
        )

    def evaluate(self, x: float) -> dict:
        """
        Mutlak değer x için tüm membership derecelerini döndür.

        Returns: {"low": float, "med": float, "high": float}
        """
        x_abs = abs(x)
        return {
            "low":  trimf(x_abs, self.low),
            "med":  trimf(x_abs, self.med),
            "high": trimf(x_abs, self.high),
        }

    def __repr__(self) -> str:
        return (
            f"MembershipSet('{self.name}', p95={self.p95_abs:.4f}, "
            f"Low={self.low}, Med={self.med}, High={self.high})"
        )


# ------------------------------------------------------------------ #
#  Çıktı membership seti (α, β, γ için)                              #
# ------------------------------------------------------------------ #

class OutputMembershipSet:
    """
    Fuzzy çıktı değişkeni için membership seti.

    α, β, γ katsayıları [0.5, 5.0] aralığında.
    Üç seviye: Low (küçük ceza), Med (orta), High (büyük ceza)

    Neden [0.5, 5.0]?
    - 0.5: tamamen sıfır ceza istemiyoruz — her metrik her zaman önemli
    - 5.0: çok büyük katsayı → training instability riski
    - Baseline α=1.0, β=10.0 (ama normalize edilmiş hali 1.0 alıyoruz)
    """

    # Bounded aralık — adil ablation karşılaştırması için
    # Klasik RL baseline α=1.0 etrafında ±0.35 modülasyon
    # Maksimum 1.5x fark → reward ölçeği etkisi izole edilebilir
    LOW_OUTPUT  = 0.8
    MED_OUTPUT  = 1.15
    HIGH_OUTPUT = 1.5

    def __init__(self, name: str, mode: str = "bounded"):
        """
        Args:
            name : katsayı adı (alpha, beta, gamma)
            mode : "bounded" → [0.8, 1.5] adil karşılaştırma (default)
                   "raw"     → [0.5, 5.0] ham fuzzy, etki büyüklüğü analizi
        """
        self.name = name
        self.mode = mode

        if mode == "raw":
            low, med, high = 0.5, 2.0, 5.0
            self.low  = TriMF(a=0.1, b=low,  c=1.5)
            self.med  = TriMF(a=0.8, b=med,  c=3.5)
            self.high = TriMF(a=2.5, b=high, c=6.0)
        else:
            # bounded — default
            low, med, high = self.LOW_OUTPUT, self.MED_OUTPUT, self.HIGH_OUTPUT
            self.low  = TriMF(a=0.5,  b=low,  c=1.0)
            self.med  = TriMF(a=0.9,  b=med,  c=1.35)
            self.high = TriMF(a=1.1,  b=high, c=1.8)

        # Centroid grid — mode'a göre doğru aralıkta
        if mode == "raw":
            self._output_range = np.linspace(0.1, 6.0, 200)
            self._default_out  = 2.0
        else:
            self._output_range = np.linspace(0.4, 2.0, 200)
            self._default_out  = self.MED_OUTPUT

        self._low_vals  = trimf_batch(self._output_range, self.low)
        self._med_vals  = trimf_batch(self._output_range, self.med)
        self._high_vals = trimf_batch(self._output_range, self.high)

    def centroid(
        self,
        low_strength: float,
        med_strength: float,
        high_strength: float,
    ) -> float:
        """
        Mamdani centroid defuzzification.

        Her çıktı seviyesini aktivasyon gücüyle kırparak (clip),
        birleşik membership fonksiyonu oluşturur ve ağırlık merkezi hesaplar.

        Neden centroid?
        Pürüzsüz, sürekli çıktı üretir. Training sırasında
        reward'da ani sıçramalar olmaz.
        """
        # Kırpılmış membership'leri birleştir (Mamdani min-max)
        clipped = np.maximum(
            np.minimum(self._low_vals,  low_strength),
            np.maximum(
                np.minimum(self._med_vals,  med_strength),
                np.minimum(self._high_vals, high_strength),
            )
        )

        # Centroid: Σ(x·μ) / Σ(μ)
        total = np.sum(clipped)
        if total < 1e-10:
            return self._default_out

        return float(np.sum(self._output_range * clipped) / total)