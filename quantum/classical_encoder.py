"""
quantum/classical_encoder.py
-----------------------------
Klasik quantum-inspired encoder'lar — karşılaştırma baseline'ı.

Neden bu dosya?
---------------
Ablation matrisimizde "gerçek quantum devre" ile
"quantum-inspired klasik encoder" karşılaştırması var.
Bu karşılaştırma olmadan quantum devrenin katkısını
izole edemeyiz — aynı feature boyutunda klasik bir
encoding ne kadar iyi? Fark gerçek quantum etkisini gösterir.

QAngle Encoder:
  x → [sin(π·x), cos(π·x)] her boyut için
  5D → 10D
  Fourier feature map benzeri projeksiyon.
  Periyodik nonlinear temsil.

QAmp Encoder:
  x → L2 normalize → outer product → üst üçgen flatten
  5D → 15D (5×5 matrisin üst üçgeni)
  Quadratic feature interaction — z_s * z_u gibi çarpımlar explicit.
  Kuantum amplitüd kodlamasından ilham.
"""

import numpy as np
import torch
from typing import Optional

from quantum.base_encoder import BaseEncoder
from dataset.normalizer import StateNormalizer, load_statistics


class QAngleEncoder(BaseEncoder):
    """
    Trigonometrik angle encoding.

    Her state değişkeni için [sin(π·x), cos(π·x)] üretir.
    5D → 10D.

    Neden sin+cos?
    Tek sin yerine hem sin hem cos kullanmak faz bilgisini korur.
    sin(π·x) ve cos(π·x) birlikte x'i tam olarak temsil eder.
    Fourier feature map'in ilk harmonikleri — periyodik kernel
    approximation sağlar.
    """

    def __init__(self, stats_path: str = "dataset/statistics.json"):
        try:
            stats, _ = load_statistics(stats_path)
            self._normalizer = StateNormalizer(stats)
        except FileNotFoundError:
            self._normalizer = None

    @property
    def output_dim(self) -> int:
        return 10  # 5 × [sin, cos]

    @property
    def n_params(self) -> int:
        return 0  # sabit encoder

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        state: (5,) ham state
        Returns: (10,) encoded features
        """
        if self._normalizer is not None:
            x = self._normalizer.normalize(state)
        else:
            x = np.clip(state, -1.0, 1.0)

        features = np.zeros(10, dtype=np.float32)
        for i in range(5):
            features[2 * i]     = np.sin(np.pi * x[i])
            features[2 * i + 1] = np.cos(np.pi * x[i])

        return features

    def encode_torch(self, state: torch.Tensor) -> torch.Tensor:
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
        return torch.tensor(self.encode(state_np), dtype=torch.float32)


class QAmpEncoder(BaseEncoder):
    """
    Amplitude encoding — quadratic feature interaction.

    state → L2 normalize → outer product → üst üçgen flatten
    5D → 15D (C(5,2) + 5 = 10 + 5 = 15)

    Neden outer product?
    Kuantum amplitüd kodlamasında |ψ⟩ = Σαᵢ|i⟩ ve
    density matrix ρ = |ψ⟩⟨ψ| = αᵢαⱼ matrisini üretir.
    Bu matrisin üst üçgeni z_s*z_u, z_s*z_s_dot gibi
    ikili etkileşimleri explicit olarak encode eder.
    Policy network bunları ayrıca öğrenmek zorunda kalmaz.
    """

    def __init__(self, stats_path: str = "dataset/statistics.json"):
        try:
            stats, _ = load_statistics(stats_path)
            self._normalizer = StateNormalizer(stats)
        except FileNotFoundError:
            self._normalizer = None

        # Üst üçgen indekslerini önceden hesapla
        self._triu_indices = np.triu_indices(5)

    @property
    def output_dim(self) -> int:
        return 15  # 5×5 matrisin üst üçgeni: C(5,2) + 5 = 15

    @property
    def n_params(self) -> int:
        return 0

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        state: (5,) ham state
        Returns: (15,) encoded features
        """
        if self._normalizer is not None:
            x = self._normalizer.normalize(state).astype(np.float64)
        else:
            x = np.clip(state, -1.0, 1.0).astype(np.float64)

        # L2 normalize — amplitüd normalizasyonu
        norm = np.linalg.norm(x)
        if norm > 1e-8:
            alpha = x / norm
        else:
            alpha = x

        # Outer product = density matrix
        outer = np.outer(alpha, alpha)

        # Üst üçgen flatten
        features = outer[self._triu_indices].astype(np.float32)
        return features

    def encode_torch(self, state: torch.Tensor) -> torch.Tensor:
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
        return torch.tensor(self.encode(state_np), dtype=torch.float32)


class IdentityEncoder(BaseEncoder):
    """
    Ham state'i direkt geçirir — normalize ederek.
    Ablation için "encoder yok" baseline'ı.
    5D → 5D.
    """

    def __init__(self, stats_path: str = "dataset/statistics.json"):
        try:
            stats, _ = load_statistics(stats_path)
            self._normalizer = StateNormalizer(stats)
        except FileNotFoundError:
            self._normalizer = None

    @property
    def output_dim(self) -> int:
        return 5

    @property
    def n_params(self) -> int:
        return 0

    def encode(self, state: np.ndarray) -> np.ndarray:
        if self._normalizer is not None:
            return self._normalizer.normalize(state)
        return np.clip(state, -1.0, 1.0).astype(np.float32)

    def encode_torch(self, state: torch.Tensor) -> torch.Tensor:
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
        return torch.tensor(self.encode(state_np), dtype=torch.float32)


# ------------------------------------------------------------------ #
#  Encoder fabrikası — ablation için                                   #
# ------------------------------------------------------------------ #

def build_encoder(
    encoder_type: str,
    n_qubits    : int = 5,
    variational : bool = True,
    frozen      : bool = True,
    device_name : str = "lightning.gpu",
    stats_path  : str = "dataset/statistics.json",
):
    """
    Encoder tipine göre doğru encoder'ı oluşturur.

    encoder_type seçenekleri:
        "none"    → IdentityEncoder (5D)
        "qangle"  → QAngleEncoder (10D)
        "qamp"    → QAmpEncoder (15D)
        "quantum" → QNodeEncoder (n_qubits + 3 D)

    frozen=True  → QFM modu: sabit parametreler, ~100x hızlı (önerilen)
    frozen=False → VQC modu: öğrenilebilir parametreler (araştırma)
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "none":
        return IdentityEncoder(stats_path=stats_path)

    elif encoder_type == "qangle":
        return QAngleEncoder(stats_path=stats_path)

    elif encoder_type == "qamp":
        return QAmpEncoder(stats_path=stats_path)

    elif encoder_type == "quantum":
        from quantum.qnode_encoder import QNodeEncoder
        return QNodeEncoder(
            n_qubits    = n_qubits,
            variational = variational,
            frozen      = frozen,
            device_name = device_name,
            stats_path  = stats_path,
        )

    else:
        raise ValueError(
            f"Bilinmeyen encoder tipi: {encoder_type}\n"
            f"Geçerli: 'none', 'qangle', 'qamp', 'quantum'"
        )
