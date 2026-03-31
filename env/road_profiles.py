"""
env/road_profiles.py
--------------------
Yol bozulma profillerini üretir.

DEĞİŞİKLİK:
  v1 → random_road her step bağımsız Gaussian (beyaz gürültü).
  v2 → RandomRoadGenerator: Gauss-Markov (OU) süreci.
       ISO 8608 yol standardı S(f)~1/f² davranışını yaklar.
       Gerçek yollar zamansal korelasyon içerir — ani sıçrama olmaz.

Gauss-Markov modeli:
    z_r[k] = α * z_r[k-1] + σ * w[k],   w ~ N(0,1)
    α = exp(-dt / T_corr)
    σ = amplitude * sqrt(1 - α²)
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Gauss-Markov yardımcı                                              #
# ------------------------------------------------------------------ #

def _gm_params(amplitude: float, dt: float, correlation_time: float = 0.5):
    """
    α ve σ parametrelerini hesapla.
    correlation_time=0.5s → kentsel yol (ISO 8608 Class C/D arası).
    """
    alpha = float(np.exp(-dt / correlation_time))
    sigma = amplitude * float(np.sqrt(max(1.0 - alpha ** 2, 0.0)))
    return alpha, sigma


# ------------------------------------------------------------------ #
#  Stateful üretici — environment bu sınıfı kullanır                  #
# ------------------------------------------------------------------ #

class RandomRoadGenerator:
    """
    Episode boyunca zamansal korelasyon koruyan yol profili.

    Kullanım:
        gen = RandomRoadGenerator(amplitude=0.01, dt=0.01)
        gen.reset()
        z_r = gen.step(rng)
    """

    def __init__(
        self,
        amplitude: float = 0.01,
        dt: float = 0.01,
        correlation_time: float = 0.5,
    ):
        self.alpha, self.sigma = _gm_params(amplitude, dt, correlation_time)
        self._z_r = 0.0

    def reset(self):
        self._z_r = 0.0

    def step(self, rng) -> float:
        w = float(rng.standard_normal())
        self._z_r = self.alpha * self._z_r + self.sigma * w
        return self._z_r


# ------------------------------------------------------------------ #
#  Stateless fonksiyonlar                                             #
# ------------------------------------------------------------------ #

def sinusoidal_road(t: float, amplitude: float = 0.01, frequency: float = 1.0) -> float:
    return float(amplitude * np.sin(2.0 * np.pi * frequency * t))


def bump_road(t: float, amplitude: float = 0.02, start: float = 1.0, end: float = 1.2) -> float:
    return float(amplitude) if start <= t <= end else 0.0


def pothole_road(t: float, amplitude: float = 0.02, start: float = 1.0, end: float = 1.2) -> float:
    return float(-amplitude) if start <= t <= end else 0.0


def multi_bump_road(t: float, amplitude: float = 0.02, bumps: list | None = None) -> float:
    """Birden fazla tümsek — kentsel yol senaryosu."""
    if bumps is None:
        bumps = [(1.0, 1.2), (3.0, 3.2), (5.0, 5.2)]
    for start, end in bumps:
        if start <= t <= end:
            return float(amplitude)
    return 0.0