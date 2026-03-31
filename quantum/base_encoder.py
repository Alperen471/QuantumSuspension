"""
quantum/base_encoder.py
-----------------------
Tüm encoder'ların implement etmesi gereken soyut arayüz.

Neden soyut arayüz?
-------------------
RL training döngüsü encoder tipini bilmek zorunda değil.
Sadece encode(state) → feature_vector çağrısını yapıyor.
Bu sayede ablation için encoder'ı değiştirmek tek satır:

    encoder = QNodeEncoder(n_qubits=10, ...)
    encoder = ClassicalEncoder(mode="qangle")
    encoder = ClassicalEncoder(mode="qamp")

Training kodu hiç değişmiyor.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseEncoder(ABC):
    """
    Tüm state encoder'larının temel sınıfı.

    Her encoder şu kontratı yerine getirmek zorunda:
      - encode(state)       : tek state vektörü → feature vektörü (numpy)
      - encode_batch(states): batch state → batch feature (numpy)
      - encode_torch(state) : torch tensor döndür — RL için
      - output_dim          : çıktı boyutu (int) — policy network input size
      - n_params            : öğrenilebilir parametre sayısı
      - get_params()        : mevcut parametreler
      - set_params(params)  : parametreleri güncelle
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Policy network'e gidecek feature vektörünün boyutu."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Öğrenilebilir parametre sayısı. Sabit encoder için 0."""
        pass

    @abstractmethod
    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Tek state vektörünü encode eder.

        Args:
            state: (5,) numpy array — [z_s, z_u, z_s_dot, z_u_dot, z_r]
                   Normalize edilmiş, [-1, +1] aralığında olmalı.

        Returns:
            features: (output_dim,) numpy array
        """
        pass

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Batch state encode. Default: döngü ile çağır.
        Performans kritikse alt sınıflar override edebilir.

        Args:
            states: (N, 5) numpy array

        Returns:
            features: (N, output_dim) numpy array
        """
        return np.stack([self.encode(s) for s in states])

    def encode_torch(self, state: np.ndarray) -> torch.Tensor:
        """
        Torch tensor döndür — RL policy network için.
        Gradient akışı encoder parametreleri varsa burada sağlanır.
        """
        features = self.encode(state)
        return torch.tensor(features, dtype=torch.float32)

    def get_params(self) -> np.ndarray | None:
        """
        Variational parametreleri döndür.
        Sabit encoder için None döner.
        """
        return None

    def set_params(self, params: np.ndarray) -> None:
        """
        Variational parametreleri güncelle.
        Sabit encoder için no-op.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_dim={self.output_dim}, "
            f"n_params={self.n_params})"
        )