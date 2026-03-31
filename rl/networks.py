"""
rl/networks.py
--------------
SAC Actor ve Critic ağ mimarileri.

Tasarım Kararları
-----------------
1. ENCODER AYRIMI
   Encoder (quantum/klasik) ağlardan ayrı tutulur.
   Policy/Critic sadece encoder çıktısını görür.
   Bu sayede ablation için encoder swap = tek satır.

2. DOUBLE-Q CRITIC
   SAC iki ayrı Q-network kullanır, minimum alır.
   Bu overestimation bias'ını önler — kararlı training.

3. TANH SQUASHING
   Actor çıktısı tanh ile [-1,+1] sınırlandırılır.
   Environment'ta max_force ile ölçeklenir.
   Log-probability hesabı için squashing correction uygulanır.

4. LAYER NORMALIZATION
   Quantum encoder çıktısının ölçeği değişken olabilir.
   LayerNorm bu ölçek farklarını dengeleyerek training
   kararlılığını artırır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON     = 1e-6


# ------------------------------------------------------------------ #
#  Yardımcı: MLP bloğu                                               #
# ------------------------------------------------------------------ #

def mlp(
    input_dim : int,
    hidden_dims: list[int],
    output_dim : int,
    activation : nn.Module = nn.ReLU,
    layer_norm : bool = False,
) -> nn.Sequential:
    """
    Çok katmanlı perceptron oluşturur.

    layer_norm=True: Her hidden layer'dan sonra LayerNorm.
    Quantum encoder çıktısının ölçek varyasyonlarını dengelemek için.
    """
    layers = []
    dims = [input_dim] + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(activation())

    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ #
#  Actor                                                              #
# ------------------------------------------------------------------ #

class GaussianActor(nn.Module):
    """
    SAC Gaussian Policy.

    Encoder çıktısından mean ve log_std üretir.
    Aksiyon tanh ile sınırlandırılır.

    Args:
        input_dim  : encoder output boyutu
        action_dim : aksiyon boyutu (1)
        hidden_dims: gizli katman boyutları
        layer_norm : LayerNorm kullan mı
    """

    def __init__(
        self,
        input_dim  : int,
        action_dim : int = 1,
        hidden_dims: list[int] = None,
        layer_norm : bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Ortak feature extraction
        self.net = mlp(
            input_dim  = input_dim,
            hidden_dims= hidden_dims,
            output_dim = hidden_dims[-1],
            layer_norm = layer_norm,
        )
        # Son ReLU'yu kaldır — mean/std katmanları öncesi
        # net'in son elemanı Linear olmalı
        # mlp sonuna activation eklediğimiz için düzeltelim
        self.net = nn.Sequential(*list(self.net.children())[:-1])

        self.mean_layer    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        # Ağırlık başlatma
        self._init_weights()

    def _init_weights(self):
        """Son katman küçük ağırlıklarla başlat — kararlı başlangıç."""
        nn.init.uniform_(self.mean_layer.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias,      -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias,   -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean   : (B, action_dim)
            log_std: (B, action_dim)
        """
        features = self.net(x)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features)
        log_std  = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aksiyonu örnekle ve log probability hesapla.

        Tanh squashing correction:
          log π(a|s) = log N(u|s) - Σ log(1 - tanh²(u_i))

        Returns:
            action  : (B, action_dim) — tanh ile [-1,+1]
            log_prob: (B, 1)
        """
        mean, log_std = self.forward(x)
        std  = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterization trick
        u = dist.rsample()

        # Tanh squashing
        action = torch.tanh(u)

        # Log probability with squashing correction
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, x: torch.Tensor) -> np.ndarray:
        """
        Deterministic action — evaluation için.
        Gradient hesabı yapılmaz.
        """
        with torch.no_grad():
            mean, _ = self.forward(x)
            action  = torch.tanh(mean)
        return action.cpu().numpy()


# ------------------------------------------------------------------ #
#  TD3 Deterministik Actor                                            #
# ------------------------------------------------------------------ #

class DeterministicActor(nn.Module):
    """
    TD3 için deterministik policy ağı.

    SAC'tan farkı: Gaussian dağılım yok, direkt aksiyon üretir.
    Exploration için training sırasında Gaussian gürültü eklenir.

    TD3'ün üç yeniliği:
      1. Clipped Double-Q → overestimation önler
      2. Delayed policy update → her 2 critic adımında 1 actor adımı
      3. Target policy smoothing → hedef aksiyona gürültü ekle

    Args:
        input_dim  : encoder output boyutu
        action_dim : aksiyon boyutu
        hidden_dims: gizli katman boyutları
        layer_norm : LayerNorm kullan mı
    """

    def __init__(
        self,
        input_dim  : int,
        action_dim : int = 1,
        hidden_dims: list[int] = None,
        layer_norm : bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.net = mlp(
            input_dim   = input_dim,
            hidden_dims = hidden_dims,
            output_dim  = action_dim,
            layer_norm  = layer_norm,
        )

        # Son katman küçük ağırlıkla başlat
        last_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_linear.bias,   -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            action: (B, action_dim) — tanh ile [-1,+1]
        """
        return torch.tanh(self.net(x))

    def get_action(self, x: torch.Tensor) -> np.ndarray:
        """Gradient yok — inference için."""
        with torch.no_grad():
            action = self.forward(x)
        return action.cpu().numpy()


# ------------------------------------------------------------------ #
#  Critic (Double-Q)                                                  #
# ------------------------------------------------------------------ #

class DoubleQCritic(nn.Module):
    """
    SAC Double-Q Critic.

    İki bağımsız Q-network — minimum alınarak overestimation önlenir.

    Args:
        input_dim  : encoder output + action_dim
        hidden_dims: gizli katman boyutları
        layer_norm : LayerNorm kullan mı
    """

    def __init__(
        self,
        input_dim  : int,
        hidden_dims: list[int] = None,
        layer_norm : bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.q1 = mlp(input_dim, hidden_dims, 1, layer_norm=layer_norm)
        self.q2 = mlp(input_dim, hidden_dims, 1, layer_norm=layer_norm)

    def forward(
        self,
        state : torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q1: (B, 1)
            q2: (B, 1)
        """
        x  = torch.cat([state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def q_min(
        self,
        state : torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Minimum Q değeri — target hesabı için."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)