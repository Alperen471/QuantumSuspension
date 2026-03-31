"""
rl/td3_agent.py
---------------
Twin Delayed Deep Deterministic Policy Gradient (TD3).

SAC ile farklar:
  SAC: Stochastic policy + entropy regularization + auto-tuned α
  TD3: Deterministic policy + exploration noise + delayed updates

TD3'ün Üç Yeniliği (Fujimoto et al., 2018):
  1. Clipped Double-Q Critic: min(Q1,Q2) ile overestimation önlenir
  2. Delayed Policy Update: Actor her policy_delay adımda bir güncellenir
     Critic önce stabilize olur, actor stabil critica göre öğrenir
  3. Target Policy Smoothing: Target aksiyona gürültü ekle
     Q(s',π(s')+ε) — overfit eden Q'ya karşı koruma

Süspansiyon için SAC vs TD3 farkı:
  SAC: Entropy terimi exploration'ı teşvik eder — farklı road
       koşullarında daha iyi genelleme potansiyeli
  TD3: Deterministik — daha stabil training, daha az hyperparameter
  → Hangisi daha iyi? Bunu ablation'da ölçeceğiz.

Ortak arayüz: SACAgent ile aynı metodlar — training loop değişmez.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from rl.networks import DeterministicActor, DoubleQCritic
from rl.replay_buffer import ReplayBuffer, Batch
from quantum.base_encoder import BaseEncoder
from quantum.classical_encoder import IdentityEncoder


class TD3Agent:
    """
    TD3 agent — encoder entegreli, SACAgent ile ortak arayüz.

    Args:
        state_dim      : ham state boyutu (5)
        action_dim     : aksiyon boyutu (1)
        max_action     : maksimum aksiyon değeri (3000.0 N)
        encoder        : BaseEncoder instance
        device         : "cuda" veya "cpu"
        lr_actor       : actor learning rate
        lr_critic      : critic learning rate
        lr_encoder     : encoder learning rate
        gamma          : discount factor
        tau            : soft update katsayısı
        policy_noise   : target policy smoothing gürültüsü std
        noise_clip     : gürültü kırpma sınırı
        expl_noise     : exploration gürültüsü std
        policy_delay   : actor update gecikmesi (critic / actor adım oranı)
        hidden_dims    : gizli katman boyutları
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
        policy_noise   : float = 0.2,
        noise_clip     : float = 0.5,
        expl_noise     : float = 0.1,
        policy_delay   : int = 2,
        hidden_dims    : list[int] = None,
        buffer_capacity: int = 100_000,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.max_action   = max_action
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.expl_noise   = expl_noise
        self.policy_delay = policy_delay

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

        if isinstance(self.encoder, nn.Module):
            self.encoder = self.encoder.to(self.device)

        # ------------------------------------------------------------ #
        #  Actor (deterministik)                                         #
        # ------------------------------------------------------------ #
        self.actor = DeterministicActor(
            input_dim   = feature_dim,
            action_dim  = action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)

        self.actor_target = DeterministicActor(
            input_dim   = feature_dim,
            action_dim  = action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # ------------------------------------------------------------ #
        #  Critic (Double-Q)                                             #
        # ------------------------------------------------------------ #
        self.critic = DoubleQCritic(
            input_dim   = feature_dim + action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)

        self.critic_target = DoubleQCritic(
            input_dim   = feature_dim + action_dim,
            hidden_dims = hidden_dims,
            layer_norm  = True,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target gradient yok
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------ #
        #  Optimizerlar                                                  #
        # ------------------------------------------------------------ #
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
        #  Replay buffer                                                 #
        # ------------------------------------------------------------ #
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
        self.total_steps  = 0
        self.update_count = 0

    # ---------------------------------------------------------------- #
    #  Encode                                                            #
    # ---------------------------------------------------------------- #

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        if self._qfm_mode:
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
        state   : np.ndarray,
        evaluate: bool = False,
    ) -> np.ndarray:
        """
        evaluate=False → deterministik + Gaussian gürültü (training)
        evaluate=True  → deterministik, gürültüsüz (test)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._qfm_mode and hasattr(self.encoder, "encode_torch"):
                features = self.encoder.encode_torch(state_t)
            else:
                features = self._encode(state_t)
            action = self.actor.get_action(features)

        if not evaluate:
            # Exploration gürültüsü
            noise  = np.random.normal(0, self.expl_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action.flatten() * self.max_action

    # ---------------------------------------------------------------- #
    #  TD3 güncelleme adımı                                              #
    # ---------------------------------------------------------------- #

    def update(self, batch_size: int = 256) -> dict:
        """
        TD3 güncelleme adımı.
        Critic her adımda, Actor her policy_delay adımda güncellenir.
        """
        if not self.replay_buffer.is_ready(batch_size):
            return {}

        batch = self.replay_buffer.sample(batch_size)

        # -------------------------------------------------------- #
        # 1. Critic güncelleme                                      #
        # -------------------------------------------------------- #
        with torch.no_grad():
            next_features = self._encode(batch.next_states)

            # Target policy smoothing — gürültü ekle
            noise = torch.randn_like(batch.actions / self.max_action) \
                    * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_features) + noise) \
                          .clamp(-1.0, 1.0)

            # Clipped Double-Q target
            target_q = self.critic_target.q_min(next_features, next_action)
            target_q = batch.rewards + self.gamma * (1 - batch.dones) * target_q

        curr_features = self._encode(batch.states)
        actions_norm  = batch.actions / self.max_action
        q1, q2 = self.critic(curr_features.detach(), actions_norm)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # -------------------------------------------------------- #
        # 2. Delayed Actor + Encoder güncelleme                     #
        # -------------------------------------------------------- #
        actor_loss_val = 0.0

        if self.update_count % self.policy_delay == 0:
            features_actor = self._encode(batch.states)
            actor_action   = self.actor(features_actor)
            actor_loss     = -self.critic.q_min(
                features_actor.detach(), actor_action
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            if isinstance(self.encoder, nn.Module):
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Soft update — actor ve critic target
            for p, pt in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

            for p, pt in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        self.update_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss" : actor_loss_val,
            "q1_mean"    : q1.mean().item(),
        }

    # ---------------------------------------------------------------- #
    #  Kayıt / Yükleme                                                   #
    # ---------------------------------------------------------------- #

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "actor"        : self.actor.state_dict(),
            "actor_target" : self.actor_target.state_dict(),
            "critic"       : self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "total_steps"  : self.total_steps,
            "update_count" : self.update_count,
        }
        if isinstance(self.encoder, nn.Module) and \
           self.encoder.n_params > 0:
            state["encoder"] = self.encoder.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.total_steps  = state.get("total_steps", 0)
        self.update_count = state.get("update_count", 0)
        if "encoder" in state and isinstance(self.encoder, nn.Module):
            self.encoder.load_state_dict(state["encoder"])

    def __repr__(self) -> str:
        return (
            f"TD3Agent(\n"
            f"  encoder     = {self.encoder}\n"
            f"  actor_dim   = {self.encoder.output_dim} → [256,256] → {self.action_dim}\n"
            f"  policy_delay= {self.policy_delay}\n"
            f"  device      = {self.device}\n"
            f")"
        )
