"""
quantum/qnode_encoder.py
------------------------
PennyLane QNode'unu BaseEncoder arayüzüne bağlar.

Neden ayrı bir wrapper?
-----------------------
circuit.py saf PennyLane kodu — cihazdan, parametrelerden bağımsız.
QNodeEncoder bu devreyi:
  1. Normalizatör ile entegre eder (ham state → normalize → devre)
  2. PyTorch ile uyumlu hale getirir (gradient akışı)
  3. RL optimizer'ına bağlar (parametre güncelleme)
  4. Ablation için tek değiştirilen sınıf haline getirir

Gradient Akışı:
  SAC optimizer → QNodeEncoder.params (torch.nn.Parameter)
  → PennyLane parameter-shift rule
  → devreye uygulanır
  → ⟨Z⟩ beklenti değerleri
  → policy network'e giriş
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional

from quantum.base_encoder import BaseEncoder
from quantum.circuit import build_circuit, print_circuit_summary
from dataset.normalizer import StateNormalizer, load_statistics


class QNodeEncoder(BaseEncoder, nn.Module):
    """
    Fizik-güdümlü Quantum Feature Map (QFM) encoder.

    İki mod:
    ─────────────────────────────────────────────────────
    frozen=True  (QFM — önerilen, ~100x hızlı):
        Variational parametreler sabit kalır.
        Gradient hesabı yoktur — sadece forward pass.
        Akademik adı: Quantum Feature Map.
        SAC/TD3 sadece klasik ağları öğrenir.
        Literatürde yaygın: Schuld et al. 2021, Cerezo et al. 2021.

    frozen=False (VQC — yavaş, araştırma amaçlı):
        Adjoint diff ile parametreler öğrenilir.
        n=5 için ~7ms/circuit — batch ile toplam ağır.
        Sadece qubit ablation makalesinde anlamlı.
    ─────────────────────────────────────────────────────

    Args:
        n_qubits      : toplam qubit sayısı (5'in katı: 5,10,15,20,25)
        variational   : True → RZ+RX katmanı eklenir (frozen=True ise sabit)
        frozen        : True → QFM modu, gradient yok (önerilen)
        device_name   : "lightning.gpu" veya "default.qubit"
        stats_path    : normalizasyon istatistikleri dosyası
        init_std      : variational parametre başlangıç std'si
    """

    def __init__(
        self,
        n_qubits: int = 5,
        variational: bool = True,
        frozen: bool = True,
        device_name: str = "lightning.gpu",
        stats_path: str = "dataset/statistics.json",
        init_std: float = 0.1,
    ):
        BaseEncoder.__init__(self)
        nn.Module.__init__(self)

        if n_qubits % 5 != 0:
            raise ValueError(f"n_qubits 5'in katı olmalı. Verilen: {n_qubits}")

        self.n_qubits    = n_qubits
        self.variational = variational
        self.frozen      = frozen
        self.device_name = device_name

        self._circuit, self._n_params, self._output_dim = build_circuit(
            n_qubits=n_qubits,
            variational=variational,
            device_name=device_name,
        )

        if variational and self._n_params > 0:
            init_params = torch.zeros(self._n_params)
            if frozen:
                # QFM: sabit tensor — gradient yok, optimizer'a verilmez
                self.register_buffer("_params_data", init_params)
                self._params = None   # nn.Parameter değil
            else:
                # VQC: öğrenilebilir parametre
                self._params = nn.Parameter(init_params)
        else:
            self._params = None
            self.register_buffer("_params_data", torch.zeros(0))

        # Normalizatör — ham state'i [-1,+1]'e çevirir
        try:
            stats, _ = load_statistics(stats_path)
            self._normalizer = StateNormalizer(stats)
        except FileNotFoundError:
            print(
                f"[WARNING] {stats_path} bulunamadı. "
                "Normalizasyon olmadan devam ediliyor. "
                "Önce run_dataset_collection.py çalıştır."
            )
            self._normalizer = None

    # ---------------------------------------------------------------- #
    #  BaseEncoder arayüzü                                              #
    # ---------------------------------------------------------------- #

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def n_params(self) -> int:
        return self._n_params

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Numpy state → numpy feature vektörü.
        Gradient hesabı yapılmaz — inference için.
        """
        with torch.no_grad():
            t = self.encode_torch(torch.tensor(state, dtype=torch.float32))
        return t.detach().cpu().numpy()

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch numpy encode — inference için."""
        with torch.no_grad():
            results = []
            for s in states:
                t = self.encode_torch(torch.tensor(s, dtype=torch.float32))
                results.append(t.detach().cpu().numpy())
        return np.stack(results)

    def _get_params(self):
        """Mevcut parametre tensörünü döndür (frozen veya learnable)."""
        if self._params is not None:
            return self._params           # VQC: nn.Parameter
        if hasattr(self, "_params_data") and self._params_data.numel() > 0:
            return self._params_data      # QFM: sabit buffer
        return None

    def encode_torch(self, state: torch.Tensor) -> torch.Tensor:
        """
        Torch tensor → torch tensor.
        QFM modunda no_grad ile ~100x hızlı çalışır.
        """
        params = self._get_params()

        # QFM: gradient hesabı gereksiz — no_grad ile çok hızlı
        if self.frozen:
            with torch.no_grad():
                return self._encode_batch(state, params)
        else:
            return self._encode_batch(state, params)

    def _run_single_circuit(self, state_np: np.ndarray, params) -> np.ndarray:
        """
        Tek state için circuit çalıştır — thread-safe.
        Her thread kendi normalizer kopyasını kullanır.
        """
        if self._normalizer is not None:
            norm = self._normalizer.normalize(state_np)
        else:
            norm = np.clip(state_np, -1.0, 1.0)

        result = self._circuit(norm, params)

        if isinstance(result, (list, tuple)):
            return np.array([float(r) for r in result], dtype=np.float32)
        elif isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy().astype(np.float32)
        else:
            return np.array(result, dtype=np.float32)

    def _encode_batch(self, state: torch.Tensor, params) -> torch.Tensor:
        """
        Batch encode — sıralı, thread-safe.

        PennyLane qnode'lar thread-safe değil — birden fazla thread
        aynı devreyi çağırırsa ölçüm sırası bozulur.
        Sıralı çalıştırma daha yavaş görünse de:
        - lightning.gpu zaten GPU'yu tam kullanıyor
        - Pre-encode buffer sayesinde training'de 0 circuit call var
        - Bu encode sadece 1 kez yapılıyor
        """
        is_single = state.dim() == 1
        states_np = state.detach().cpu().numpy()
        if is_single:
            states_np = states_np.reshape(1, -1)
        states_np = states_np.astype(np.float64)
        n = len(states_np)

        if params is not None:
            if isinstance(params, torch.Tensor):
                params_np = params.detach().cpu().numpy()
            else:
                params_np = np.array(params)
        else:
            params_np = None

        results = [self._run_single_circuit(states_np[i], params_np)
                   for i in range(n)]

        out = torch.from_numpy(np.stack(results))
        if is_single:
            out = out.squeeze(0)
        return out.to(state.device)

    def _normalize_output(self, out: torch.Tensor) -> torch.Tensor:
        """Encoder output'unu [-1,1] aralığına normalize et."""
        # Her feature bağımsız normalize edilir
        # max(abs) ile böl — işaret korunur
        abs_max = out.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        return out / abs_max

    def _encode_single(self, state: torch.Tensor) -> torch.Tensor:
        """Geriye dönük uyumluluk."""
        return self.encode_torch(state)

    # ---------------------------------------------------------------- #
    #  Parametre yönetimi                                               #
    # ---------------------------------------------------------------- #

    def get_params(self) -> Optional[np.ndarray]:
        if self._params is None:
            return None
        return self._params.detach().cpu().numpy()

    def set_params(self, params: np.ndarray) -> None:
        if self._params is not None:
            with torch.no_grad():
                self._params.copy_(torch.tensor(params, dtype=torch.float32))

    def parameters_for_optimizer(self) -> list:
        """
        SAC optimizer'ına verilecek parametre listesi.

        Kullanım:
            optimizer = torch.optim.Adam([
                *policy.parameters(),
                *encoder.parameters_for_optimizer(),
            ], lr=3e-4)
        """
        if self._params is not None:
            return [self._params]
        return []

    # ---------------------------------------------------------------- #
    #  Yardımcılar                                                      #
    # ---------------------------------------------------------------- #

    def summary(self) -> None:
        """Devre özetini yazdır."""
        print_circuit_summary(self.n_qubits, self.variational)

    def __repr__(self) -> str:
        mode = "QFM-frozen" if self.frozen else "VQC-learnable"
        return (
            f"QNodeEncoder("
            f"n_qubits={self.n_qubits}, "
            f"mode={mode}, "
            f"device={self.device_name}, "
            f"output_dim={self.output_dim}, "
            f"n_params={self.n_params})"
        )