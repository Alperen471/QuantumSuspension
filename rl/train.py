"""
rl/train.py
-----------
SAC training loop — tek model için.

run_ablation.py bu fonksiyonu farklı konfigürasyonlarla çağırır.

Training Akışı:
  1. Environment ve agent oluştur
  2. Replay buffer'ı dataset ile preload et (opsiyonel)
  3. Episode döngüsü:
     a. Env reset
     b. Step: action seç → env step → buffer'a ekle
     c. Her adımda update (buffer hazır ise)
     d. Episode sonu: metrikleri kaydet
  4. Belirli aralıklarla evaluate et (5 seed, deterministic)
  5. En iyi modeli kaydet
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from env.quarter_car_env import QuarterCarEnv
from rl.sac_agent import SACAgent
from rl.env_wrapper import FuzzyRewardWrapper
from quantum.classical_encoder import build_encoder
from analysis.time_domain_metrics import compute_time_domain_metrics
from analysis.frequency_domain_metrics import compute_frequency_domain_metrics


# ------------------------------------------------------------------ #
#  Training konfigürasyonu                                            #
# ------------------------------------------------------------------ #

@dataclass
class TrainingConfig:
    """Tek model için tam training konfigürasyonu."""

    # Model kimliği
    model_name   : str = "rl_baseline"
    algorithm    : str = "sac"
    encoder_type : str = "none"
    n_qubits     : int = 5
    variational  : bool = True
    frozen       : bool = True    # True=QFM (hızlı), False=VQC (yavaş)
    fuzzy_mode   : str = "none"
    device       : str = "cuda"

    # Environment
    road_type    : str = "random"
    road_amplitude: float = 0.01
    road_frequency: float = 1.0
    episode_length: int = 2000
    dt           : float = 0.01

    # Training
    total_steps  : int = 200_000
    batch_size   : int = 256
    buffer_capacity: int = 100_000
    warmup_steps : int = 1000        # random action ile ısınma
    update_freq  : int = 1           # her kaç adımda bir update
    eval_freq    : int = 10_000      # her kaç adımda bir evaluate
    eval_episodes: int = 5           # evaluate için kaç episode
    save_freq    : int = 50_000      # model kaydetme sıklığı

    # SAC hyperparameters
    lr_actor     : float = 3e-4
    lr_critic    : float = 3e-4
    lr_encoder   : float = 1e-4
    gamma        : float = 0.99
    tau          : float = 0.005
    alpha        : float = 0.2
    auto_entropy : bool = True
    hidden_dims  : list[int] = field(default_factory=lambda: [256, 256])

    # TD3-specific hyperparameters
    policy_noise : float = 0.2
    noise_clip   : float = 0.5
    expl_noise   : float = 0.1
    policy_delay : int   = 2

    # Paths
    save_dir     : str = "results/models"
    log_dir      : str = "results/training_logs"
    stats_path   : str = "dataset/statistics.json"

    # Seed
    seed         : int = 42


@dataclass
class TrainingResult:
    """Training sonuçları."""
    model_name      : str
    config          : TrainingConfig
    episode_rewards : list[float]
    eval_metrics    : list[dict]
    training_losses : list[dict]
    total_time_s    : float
    best_eval_reward: float
    final_metrics   : dict


# ------------------------------------------------------------------ #
#  Yardımcı: environment oluştur                                      #
# ------------------------------------------------------------------ #

def build_environment(cfg: TrainingConfig, seed: Optional[int] = None):
    """Konfigürasyona göre environment oluştur."""
    env = QuarterCarEnv(
        road_type      = cfg.road_type,
        road_amplitude = cfg.road_amplitude,
        road_frequency = cfg.road_frequency,
        episode_length = cfg.episode_length,
        dt             = cfg.dt,
        seed           = seed or cfg.seed,
    )

    if cfg.fuzzy_mode != "none":
        env = FuzzyRewardWrapper(
            env,
            mode       = cfg.fuzzy_mode,
            stats_path = cfg.stats_path,
        )

    return env


# ------------------------------------------------------------------ #
#  Yardımcı: agent oluştur                                            #
# ------------------------------------------------------------------ #

def build_agent(cfg: TrainingConfig):
    """Konfigürasyona göre SAC veya TD3 agent oluştur."""
    encoder = build_encoder(
        encoder_type = cfg.encoder_type,
        n_qubits     = cfg.n_qubits,
        variational  = cfg.variational,
        frozen       = cfg.frozen,
        device_name  = "lightning.gpu" if cfg.device == "cuda" else "default.qubit",
        stats_path   = cfg.stats_path,
    )

    if cfg.algorithm.lower() == "td3":
        from rl.td3_agent import TD3Agent
        agent = TD3Agent(
            state_dim      = 5,
            action_dim     = 1,
            max_action     = 3000.0,
            encoder        = encoder,
            device         = cfg.device,
            lr_actor       = cfg.lr_actor,
            lr_critic      = cfg.lr_critic,
            lr_encoder     = cfg.lr_encoder,
            gamma          = cfg.gamma,
            tau            = cfg.tau,
            policy_noise   = cfg.policy_noise,
            noise_clip     = cfg.noise_clip,
            expl_noise     = cfg.expl_noise,
            policy_delay   = cfg.policy_delay,
            hidden_dims    = cfg.hidden_dims,
            buffer_capacity= cfg.buffer_capacity,
        )
    else:
        # SAC (default)
        agent = SACAgent(
            state_dim      = 5,
            action_dim     = 1,
            max_action     = 3000.0,
            encoder        = encoder,
            device         = cfg.device,
            lr_actor       = cfg.lr_actor,
            lr_critic      = cfg.lr_critic,
            lr_encoder     = cfg.lr_encoder,
            gamma          = cfg.gamma,
            tau            = cfg.tau,
            alpha          = cfg.alpha,
            auto_entropy   = cfg.auto_entropy,
            hidden_dims    = cfg.hidden_dims,
            buffer_capacity= cfg.buffer_capacity,
        )

    return agent


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate_agent(
    agent   : SACAgent,
    cfg     : TrainingConfig,
    n_episodes: int = 5,
) -> dict:
    """
    Deterministic policy ile evaluate et.
    Quantum encoder için hafifletilmiş mod otomatik devreye girer.
    """
    # Quantum encoder için evaluation'ı hafiflet
    is_quantum = (cfg.encoder_type == "quantum")
    road_types  = ["random", "sinusoidal", "bump", "pothole"]
    n_eps       = n_episodes
    ep_len      = cfg.episode_length

    all_metrics = []

    for road_type in road_types:
        eval_cfg             = TrainingConfig(**vars(cfg))
        eval_cfg.road_type   = road_type
        eval_cfg.episode_length = ep_len

        for ep in range(n_eps):
            env = build_environment(eval_cfg, seed=1000 + ep)
            state, _ = env.reset()

            states_log  = []
            rewards_log = []
            infos_log   = []
            done = truncated = False

            while not (done or truncated):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, truncated, info = env.step(action)

                states_log.append(next_state)
                rewards_log.append(reward)
                infos_log.append(info)
                state = next_state

            env.close()

            results = {
                "states" : np.array(states_log),
                "rewards": np.array(rewards_log),
                "infos"  : infos_log,
            }

            td = compute_time_domain_metrics(results)
            fd = compute_frequency_domain_metrics(results, dt=cfg.dt)

            all_metrics.append({
                "road_type"         : road_type,
                "episode"           : ep,
                "mean_reward"       : float(np.mean(rewards_log)),
                "rms_body_acc"      : td["rms_body_acc"],
                "max_abs_body_acc"  : td["max_abs_body_acc"],
                "rms_susp_defl"     : td["rms_suspension_deflection"],
                "rms_tire_defl"     : td["rms_tire_deflection"],
                "rms_ctrl_force"    : td["rms_control_force"],
                "total_psd_energy"  : fd["total_psd_energy"],
                "band_1_3Hz"        : fd["band_1_3Hz_energy"],
                "iso_weighted_rms"  : fd["iso_weighted_rms_body_acc"],
            })

    # Tüm metriklerin ortalaması
    keys = [k for k in all_metrics[0] if k not in ["road_type", "episode"]]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if isinstance(m[k], (int, float))]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))

    return summary


# ------------------------------------------------------------------ #
#  Ana training fonksiyonu                                            #
# ------------------------------------------------------------------ #

def train(cfg: TrainingConfig, verbose: bool = True) -> TrainingResult:
    """
    Tek model için tam training döngüsü.

    Args:
        cfg    : TrainingConfig
        verbose: terminal çıktısı

    Returns:
        TrainingResult
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    t_start = time.time()

    # Dizinler
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {cfg.model_name}")
        print(f"  Encoder  : {cfg.encoder_type} (n_qubits={cfg.n_qubits})")
        print(f"  Fuzzy    : {cfg.fuzzy_mode}")
        print(f"  Steps    : {cfg.total_steps:,}")
        print(f"  Device   : {cfg.device}")
        print(f"{'='*60}")

    # Environment ve Agent
    env   = build_environment(cfg)
    agent = build_agent(cfg)

    # Quantum encoder için batch_size ve update_freq otomatik ayarla
    batch_size   = cfg.batch_size
    update_freq  = cfg.update_freq

    if cfg.encoder_type == "quantum":
        n = cfg.n_qubits
        is_frozen = cfg.frozen  # QFM mi VQC mi?

        if is_frozen:
            # QFM: gradient yok, normal batch boyutu kullanılabilir
            batch_size   = cfg.batch_size   # 256
            update_freq  = cfg.update_freq  # 1
            if verbose:
                print(f"  [QFM n={n}q] frozen=True → "
                      f"batch={batch_size}, update_freq={update_freq} "
                      f"(baseline kadar hızlı)")
        else:
            # VQC: gradient var, kademeli yavaşlama
            if n <= 5:
                batch_size, update_freq = 32, 5
            elif n <= 10:
                batch_size, update_freq = 16, 10
            else:
                batch_size, update_freq = 8, 20
            if verbose:
                print(f"  [VQC n={n}q] frozen=False → "
                      f"batch={batch_size}, update_freq={update_freq}")

        eval_freq_actual = max(cfg.eval_freq, cfg.total_steps // 20) \
                           if not is_frozen else cfg.eval_freq
    else:
        eval_freq_actual = cfg.eval_freq

    if verbose:
        print(f"\n{agent}")

    # Replay buffer preload — her zaman ham dataset, encoder live çalışır
    std_npz = Path("dataset/data/full_dataset.npz")
    if std_npz.exists():
        npz_data = dict(np.load(std_npz))
        agent.replay_buffer.add_batch(npz_data)
        if verbose:
            print(f"  Buffer preloaded: {agent.replay_buffer.size:,} adım (live encode)")

    # Loglama
    episode_rewards : list[float] = []
    eval_metrics    : list[dict]  = []
    training_losses : list[dict]  = []
    best_eval_reward = -np.inf
    episode_reward   = 0.0
    episode_count    = 0
    recent_rms       = 0.0
    recent_iso       = 0.0

    state, _ = env.reset()

    # tqdm progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(
            total      = cfg.total_steps,
            desc       = cfg.model_name[:30],
            unit       = "step",
            dynamic_ncols= True,
            bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )
        use_tqdm = True
    except ImportError:
        print("\n[INFO] tqdm kurulu değil: pip install tqdm")
        print("Training başlıyor...")
        use_tqdm = False
        pbar = None

    for step in range(cfg.total_steps):

        # Aksiyon seç
        if step < cfg.warmup_steps and agent.replay_buffer.size < cfg.batch_size:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Environment adımı
        next_state, reward, done, truncated, info = env.step(action)

        # Buffer'a ekle
        agent.replay_buffer.add(
            state, action, reward, next_state, float(done or truncated)
        )

        episode_reward += reward
        state           = next_state
        agent.total_steps += 1

        # SAC/TD3 güncelleme
        if step % update_freq == 0 and agent.replay_buffer.is_ready(batch_size):
            loss_dict = agent.update(batch_size)
            if loss_dict:
                training_losses.append({**loss_dict, "step": step})

        # Episode sonu
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_count  += 1
            episode_reward  = 0.0
            state, _        = env.reset()

            # Her 10 episode'da verbose çıktı (tqdm yoksa)
            if verbose and not use_tqdm and episode_count % 10 == 0:
                elapsed = time.time() - t_start
                recent_mean = np.mean(episode_rewards[-10:])
                print(
                    f"  Episode {episode_count:4d} | "
                    f"Step {step:6d}/{cfg.total_steps} | "
                    f"Reward {recent_mean:8.2f} | "
                    f"Time {elapsed:6.0f}s"
                )

        # Evaluation
        if step % eval_freq_actual == 0 and step > 0:
            eval_result = evaluate_agent(agent, cfg, cfg.eval_episodes)
            eval_result["step"] = step
            eval_metrics.append(eval_result)

            current_reward = eval_result.get("mean_reward_mean", -np.inf)
            recent_rms     = eval_result.get("rms_body_acc_mean", 0.0)
            recent_iso     = eval_result.get("iso_weighted_rms_mean", 0.0)

            if current_reward > best_eval_reward:
                best_eval_reward = current_reward
                agent.save(f"{cfg.save_dir}/{cfg.model_name}_best.pt")

            if verbose and not use_tqdm:
                print(
                    f"  [EVAL step={step}] "
                    f"rms_body_acc={recent_rms:.4f} | "
                    f"iso_rms={recent_iso:.4f} | "
                    f"reward={current_reward:.2f}"
                )

        # Progress bar güncelle
        if use_tqdm and pbar is not None:
            pbar.update(1)
            # Her 500 step'te postfix güncelle
            if step % 500 == 0:
                elapsed  = time.time() - t_start
                eta_s    = (elapsed / max(step, 1)) * (cfg.total_steps - step)
                eta_min  = eta_s / 60
                pbar.set_postfix(
                    rms   = f"{recent_rms:.3f}",
                    iso   = f"{recent_iso:.3f}",
                    ep    = episode_count,
                    ETA   = f"{eta_min:.0f}m",
                    refresh=True,
                )

        # Model kaydet
        if step % cfg.save_freq == 0 and step > 0:
            agent.save(f"{cfg.save_dir}/{cfg.model_name}_step{step}.pt")

    if use_tqdm and pbar is not None:
        pbar.close()

        # Model kaydet
        if step % cfg.save_freq == 0 and step > 0:
            agent.save(f"{cfg.save_dir}/{cfg.model_name}_step{step}.pt")

    # Son evaluation
    final_metrics = evaluate_agent(agent, cfg, cfg.eval_episodes)
    agent.save(f"{cfg.save_dir}/{cfg.model_name}_final.pt")

    env.close()

    total_time = time.time() - t_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training tamamlandı: {cfg.model_name}")
        print(f"  Toplam süre    : {total_time/60:.1f} dakika")
        print(f"  Final rms_body : {final_metrics.get('rms_body_acc_mean', 0):.4f}")
        print(f"  Final iso_rms  : {final_metrics.get('iso_weighted_rms_mean', 0):.4f}")
        print(f"{'='*60}")

    return TrainingResult(
        model_name       = cfg.model_name,
        config           = cfg,
        episode_rewards  = episode_rewards,
        eval_metrics     = eval_metrics,
        training_losses  = training_losses,
        total_time_s     = total_time,
        best_eval_reward = best_eval_reward,
        final_metrics    = final_metrics,
    )
