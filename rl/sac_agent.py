"""
rl/sac_agent.py
---------------
Soft Actor-Critic (SAC) implementasyonu.
Quantum encoder ve fuzzy reward ile entegre.

SAC Özeti
---------
Objective: max E[Σ r_t + α·H(π)]
  - r_t: reward
  - H(π): policy entropy
  - α: entropy katsayısı (auto-tune)

Güncelleme adımları:
  1. Critic update: Bellman hatası minimize et
  2. Actor update: Q + entropy maximize et
  3. Entropy update: target entropy'e yaklaştır
  4. Target network: soft update (polyak averaging)
  5. Encoder update: actor loss ile birlikte (variational ise)

Neden entropy regularization?
Süspansiyon kontrolü stochastic bir ortam — farklı yol
profilleri, farklı başlangıç koşulları. Entropy term
policy'nin keşif yapmasını teşvik eder, local optima'ya
sıkışmayı önler.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from rl.networks import GaussianActor, DoubleQCritic
from rl.replay_buffer import ReplayBuffer, Batch
from quantum.base_encoder import BaseEncoder
from quantum.classical_encoder import IdentityEncoder


class SACAgent:
    """
    SAC agent — encoder ve fuzzy entegreli.

    Args:
        state_dim    : ham state boyutu (5)
        action_dim   : aksiyon boyutu (1)
        max_action   : maksimum aksiyon değeri (3000.0 N)
        encoder      : BaseEncoder instance — None ise Identity kullanılır
        device       : "cuda" veya "cpu"
        lr_actor     : actor learning rate
        lr_critic    : critic learning rate
        lr_encoder   : encoder learning rate (variational için)
        gamma        : discount factor
        tau          : soft update katsayısı
        alpha        : başlangıç entropy katsayısı
        auto_entropy : entropy katsayısı otomatik ayarlansın mı
        hidden_dims  : gizli katman boyutları
        buffer_capacity: replay buffer kapasitesi
    """

    def __init__(
        self,
        state_dim      : int = 5,
        action_dim     : int = 1,
        max_action     : float = 3000.0,
        encoder        : Optional[BaseEncoder] = None,
        device         : str = "cuda",
        lr_actor       : float = 3e-4,
        lr_critic      : float = 3e-4,
        lr_encoder     : float = 1e-4,
        gamma          : float = 0.99,
        tau            : float = 0.005,
        alpha          : float = 0.2,
        auto_entropy   : bool = True,
        hidden_dims    : list[int] = None,
        buffer_capacity: int = 100_000,
    ):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.max_action  = max_action
        self.gamma       = gamma
        self.tau         = tau
        self.auto_entropy= auto_entropy

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # ------------------------------------------------------------ #
        #  Encoder                                                       #
        # ------------------------------------------------------------ #
        self.encoder = encoder if encoder is not None else IdentityEncoder()
        feature_dim  = self.encoder.output_dim

        # Encoder torch modülü ise GPU'ya taşı
        if isinstance(self.encoder, nn.Module):
            self.encoder = self.encoder.to(self.device)

        # ------------------------------------------------------------ #
        #  Actor                                                         #
        # ------------------------------------------------------------ #
        self.actor = GaussianActor(
            input_dim   = feature_dim,
            action_dim  = action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)

        # ------------------------------------------------------------ #
        #  Critic (double-Q)                                             #
        # ------------------------------------------------------------ #
        self.critic = DoubleQCritic(
            input_dim   = feature_dim + action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)

        # Target critic — soft update ile güncellenir
        self.critic_target = DoubleQCritic(
            input_dim   = feature_dim + action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target gradient hesabı yok
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------ #
        #  Optimizerlar                                                  #
        # ------------------------------------------------------------ #
        # Encoder parametreleri varsa ayrı lr ile ekle
        encoder_params = []
        if isinstance(self.encoder, nn.Module):
            encoder_params = list(self.encoder.parameters())

        if encoder_params:
            self.actor_optimizer = torch.optim.Adam([
                {"params": self.actor.parameters(), "lr": lr_actor},
                {"params": encoder_params,          "lr": lr_encoder},
            ])
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr_actor
            )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        # ------------------------------------------------------------ #
        #  Entropy katsayısı (auto-tune)                                 #
        # ------------------------------------------------------------ #
        # Target entropy: -action_dim (SAC paper'dan)
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True,
                                          device=self.device)
        self.alpha_val      = alpha
        self.alpha_optimizer= torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # ------------------------------------------------------------ #
        #  Replay buffer — QFM modunda encoder ile init et            #
        # ------------------------------------------------------------ #
        # QFM modunda buffer add() sırasında encode eder →
        # training'de _encode() çağrısı sadece features döndürür
        qfm_encoder = None
        if (isinstance(self.encoder, nn.Module) and
                hasattr(self.encoder, "frozen") and
                self.encoder.frozen):
            qfm_encoder = self.encoder

        self.replay_buffer = ReplayBuffer(
            state_dim   = state_dim,
            action_dim  = action_dim,
            capacity    = buffer_capacity,
            device      = str(self.device),
            encoder     = qfm_encoder,
        )
        self._qfm_mode = qfm_encoder is not None

        # ------------------------------------------------------------ #
        #  Sayaçlar                                                       #
        # ------------------------------------------------------------ #
        self.total_steps   = 0
        self.update_count  = 0
        self._train_losses : list[dict] = []

    # ---------------------------------------------------------------- #
    #  Encode                                                            #
    # ---------------------------------------------------------------- #

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Ham state → feature tensor.
        QFM modunda: buffer zaten encoded → direkt döndür.
        VQC / klasik modda: encoder çalıştır.
        """
        if self._qfm_mode:
            # Buffer'dan gelen tensor zaten encoded feature
            return state
        if isinstance(self.encoder, nn.Module):
            return self.encoder.encode_torch(state)
        else:
            state_np = state.cpu().numpy()
            if state.dim() == 1:
                return torch.FloatTensor(
                    self.encoder.encode(state_np)).to(self.device)
            else:
                return torch.FloatTensor(
                    self.encoder.encode_batch(state_np)).to(self.device)

    # ---------------------------------------------------------------- #
    #  Aksiyon seçimi                                                    #
    # ---------------------------------------------------------------- #

    def select_action(
        self,
        state    : np.ndarray,
        evaluate : bool = False,
    ) -> np.ndarray:
        """
        Tek state için aksiyon seç.

        evaluate=True  → deterministic (test için)
        evaluate=False → stochastic (training için)

        Returns: numpy array (action_dim,), ölçeklenmiş N aralığında
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # QFM modunda select_action için encode gerekli
            # (buffer'daki encoded state değil, ham env state geliyor)
            if self._qfm_mode and hasattr(self.encoder, "encode_torch"):
                features = self.encoder.encode_torch(state_t)
            else:
                features = self._encode(state_t)

            if evaluate:
                action = self.actor.get_action(features)
            else:
                action, _ = self.actor.sample(features)
                action     = action.cpu().numpy()

        # [-1,+1] → [-max_action, +max_action]
        return action.flatten() * self.max_action

    # ---------------------------------------------------------------- #
    #  SAC güncelleme adımı                                              #
    # ---------------------------------------------------------------- #

    def update(self, batch_size: int = 256) -> dict:
        """
        Tek SAC güncelleme adımı.

        Returns: loss değerleri dict'i (loglama için)
        """
        if not self.replay_buffer.is_ready(batch_size):
            return {}

        batch = self.replay_buffer.sample(batch_size)
        alpha = self.log_alpha.exp().detach() if self.auto_entropy \
                else torch.tensor(self.alpha_val)

        # -------------------------------------------------------- #
        # Encode — batch.states ve batch.next_states SADECE 1 kez  #
        # encode edilir, sonra yeniden kullanılır.                  #
        # Quantum circuit: 32 call → 64 call (states+next_states)  #
        # Eskiden: 96 call (3x states + 1x next_states)            #
        # -------------------------------------------------------- #
        with torch.no_grad():
            curr_features_cached = self._encode(batch.states)
            next_features        = self._encode(batch.next_states)

        # -------------------------------------------------------- #
        # 1. Critic güncelleme                                      #
        # -------------------------------------------------------- #
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_features)
            target_q = self.critic_target.q_min(next_features, next_action)
            target_q = batch.rewards + self.gamma * (1 - batch.dones) * \
                       (target_q - alpha * next_log_prob)

        actions_norm = batch.actions / self.max_action
        q1, q2 = self.critic(curr_features_cached, actions_norm)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # -------------------------------------------------------- #
        # 2. Actor + Encoder güncelleme                             #
        # Cached features kullan — encoder variational değilse     #
        # gradient gerekmez, direkt cached kullan.                 #
        # -------------------------------------------------------- #
        if isinstance(self.encoder, nn.Module) and self.encoder.n_params > 0:
            # Variational encoder: gradient gerekli — yeniden encode et
            features_actor = self._encode(batch.states)
        else:
            # Fixed encoder veya klasik: cached yeterli
            features_actor = curr_features_cached

        new_action, log_prob = self.actor.sample(features_actor)

        q_val = self.critic.q_min(features_actor.detach(), new_action)
        actor_loss = (alpha * log_prob - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        # Encoder gradient clip — sadece VQC modunda (QFM frozen'da parametre yok)
        if isinstance(self.encoder, nn.Module) and not getattr(self.encoder, "frozen", True):
            enc_params = list(self.encoder.parameters())
            if enc_params:
                nn.utils.clip_grad_norm_(enc_params, 0.5)
        self.actor_optimizer.step()

        # -------------------------------------------------------- #
        # 3. Entropy katsayısı güncelleme                           #
        # -------------------------------------------------------- #
        entropy_loss = torch.tensor(0.0)
        if self.auto_entropy:
            entropy_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()
            self.alpha_val = self.log_alpha.exp().item()

        # -------------------------------------------------------- #
        # 4. Target network soft update                             #
        # -------------------------------------------------------- #
        for p, p_target in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            p_target.data.copy_(
                self.tau * p.data + (1 - self.tau) * p_target.data
            )

        self.update_count += 1

        return {
            "critic_loss" : critic_loss.item(),
            "actor_loss"  : actor_loss.item(),
            "entropy_loss": entropy_loss.item() if self.auto_entropy else 0.0,
            "alpha"       : self.alpha_val,
            "q1_mean"     : q1.mean().item(),
        }

    # ---------------------------------------------------------------- #
    #  Kayıt / Yükleme                                                   #
    # ---------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Model ağırlıklarını kaydet."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "actor"        : self.actor.state_dict(),
            "critic"       : self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha"    : self.log_alpha.data,
            "total_steps"  : self.total_steps,
            "update_count" : self.update_count,
        }

        if isinstance(self.encoder, nn.Module) and \
           self.encoder.n_params > 0:
            state["encoder"] = self.encoder.state_dict()

        torch.save(state, path)

    def load(self, path: str) -> None:
        """Kaydedilmiş model ağırlıklarını yükle."""
        state = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.log_alpha.data = state["log_alpha"]
        self.total_steps    = state.get("total_steps", 0)
        self.update_count   = state.get("update_count", 0)

        if "encoder" in state and isinstance(self.encoder, nn.Module):
            self.encoder.load_state_dict(state["encoder"])

    def __repr__(self) -> str:
        return (
            f"SACAgent(\n"
            f"  encoder   = {self.encoder}\n"
            f"  actor_dim = {self.encoder.output_dim} → [256,256] → {self.action_dim}\n"
            f"  device    = {self.device}\n"
            f"  steps     = {self.total_steps}\n"
            f")"
        )
