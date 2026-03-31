"""
dataset/collector.py
--------------------
Passive ve Skyhook controller'ları çalıştırarak ham geçiş verisi toplar.

Neden bu dosya var?
-------------------
Fuzzy membership sınırları ve quantum encoder normalizasyonu keyfi
belirlenemez — gerçek sistem dağılımına dayanmalı.
Bu collector, o istatistiklerin hesaplanacağı ham veriyi üretir.

Her step için şunlar kaydedilir:
  state       : [z_s, z_u, z_s_dot, z_u_dot, z_r]
  action      : [u]
  reward      : skaler
  next_state  : [z_s, z_u, z_s_dot, z_u_dot, z_r]
  done        : bool
  body_acc    : z_s_ddot  (fuzzy girişi)
  susp_defl   : z_s - z_u (fuzzy girişi)
  tire_defl   : z_u - z_r
  ctrl_force  : u          (fuzzy girişi)
  road_type   : str
  controller  : str
  episode_id  : int
  step        : int
"""

import numpy as np
from typing import List, Dict, Any

from env.quarter_car_env import QuarterCarEnv
from controllers.passive_controller import PassiveController
from controllers.skyhook_controller import SkyhookController


# ------------------------------------------------------------------ #
#  Sabitler                                                            #
# ------------------------------------------------------------------ #

# Hangi road tipleri ve parametreleri kullanılacak
ROAD_CONFIGS: List[Dict[str, Any]] = [
    {"road_type": "random",      "road_amplitude": 0.01, "road_frequency": 1.0},
    {"road_type": "sinusoidal",  "road_amplitude": 0.01, "road_frequency": 1.0},
    {"road_type": "bump",        "road_amplitude": 0.02, "road_frequency": 1.0},
    {"road_type": "pothole",     "road_amplitude": 0.02, "road_frequency": 1.0},
]

# Hangi controller'lar çalıştırılacak
CONTROLLER_CONFIGS: List[Dict[str, Any]] = [
    {"controller_type": "passive",  "params": {}},
    {"controller_type": "skyhook",  "params": {"c_sky": 4000.0, "max_force": 3000.0}},
]


# ------------------------------------------------------------------ #
#  Yardımcı: controller fabrikası                                      #
# ------------------------------------------------------------------ #

def _build_controller(controller_type: str, params: dict):
    if controller_type == "passive":
        return PassiveController()
    elif controller_type == "skyhook":
        return SkyhookController(
            c_sky=params.get("c_sky", 4000.0),
            max_force=params.get("max_force", 3000.0),
        )
    else:
        raise ValueError(f"Bilinmeyen controller: {controller_type}")


# ------------------------------------------------------------------ #
#  Tek episode collector                                               #
# ------------------------------------------------------------------ #

def collect_episode(
    controller_type: str,
    controller_params: dict,
    road_type: str,
    road_amplitude: float,
    road_frequency: float,
    episode_length: int = 2000,
    dt: float = 0.01,
    seed: int | None = None,
    episode_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Tek bir episode çalıştırır, her step için bir dict döndürür.

    Neden liste of dict?
    Pandas / Parquet'e direkt dönüştürülebilir.
    Her step bağımsız bir veri noktası — replay buffer gibi kullanılabilir.
    """
    env = QuarterCarEnv(
        road_type=road_type,
        road_amplitude=road_amplitude,
        road_frequency=road_frequency,
        episode_length=episode_length,
        dt=dt,
        seed=seed,
    )
    controller = _build_controller(controller_type, controller_params)

    state, _ = env.reset()
    records = []
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        action = controller.act(state)
        next_state, reward, done, truncated, info = env.step(action)

        records.append({
            # State bileşenleri — normalizasyon için ayrı ayrı tutulur
            "z_s":        float(state[0]),
            "z_u":        float(state[1]),
            "z_s_dot":    float(state[2]),
            "z_u_dot":    float(state[3]),
            "z_r":        float(state[4]),

            # Aksiyon
            "action":     float(action[0]),

            # Reward
            "reward":     float(reward),

            # Next state — replay buffer için
            "z_s_next":   float(next_state[0]),
            "z_u_next":   float(next_state[1]),
            "z_s_dot_next": float(next_state[2]),
            "z_u_dot_next": float(next_state[3]),
            "z_r_next":   float(next_state[4]),

            # Terminal flag
            "done":       bool(done or truncated),

            # Fiziksel metrikler — fuzzy giriş değişkenleri
            "body_acc":   float(info["body_acc"]),
            "susp_defl":  float(info["suspension_deflection"]),
            "tire_defl":  float(info["tire_deflection"]),
            "ctrl_force": float(info["control_force"]),
            "road_input": float(info["road_input"]),

            # Meta
            "road_type":  road_type,
            "controller": controller_type,
            "episode_id": episode_id,
            "step":       step,
            "time":       float(info["time"]),
        })

        state = next_state
        step += 1

    env.close()
    return records


# ------------------------------------------------------------------ #
#  Ana toplama fonksiyonu                                              #
# ------------------------------------------------------------------ #

def collect_dataset(
    road_configs: List[Dict[str, Any]] | None = None,
    controller_configs: List[Dict[str, Any]] | None = None,
    episodes_per_combo: int = 5,
    episode_length: int = 2000,
    dt: float = 0.01,
    base_seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Tüm controller × road kombinasyonlarını çalıştırır.

    episodes_per_combo: Her kombinasyon için kaç farklı seed ile tekrar.
    Neden 5?  Dağılım istatistiklerinin kararlı tahmin edilmesi için
    yeterli — 5 × 4 road × 2 controller × 2000 step = 80 000 veri noktası.

    Döndürür: flat liste — her eleman tek bir step kaydı.
    """
    if road_configs is None:
        road_configs = ROAD_CONFIGS
    if controller_configs is None:
        controller_configs = CONTROLLER_CONFIGS

    all_records: List[Dict[str, Any]] = []
    episode_id = 0

    total_combos = len(controller_configs) * len(road_configs) * episodes_per_combo
    combo_count = 0

    for ctrl_cfg in controller_configs:
        for road_cfg in road_configs:
            for ep in range(episodes_per_combo):
                seed = base_seed + episode_id

                records = collect_episode(
                    controller_type=ctrl_cfg["controller_type"],
                    controller_params=ctrl_cfg["params"],
                    road_type=road_cfg["road_type"],
                    road_amplitude=road_cfg["road_amplitude"],
                    road_frequency=road_cfg["road_frequency"],
                    episode_length=episode_length,
                    dt=dt,
                    seed=seed,
                    episode_id=episode_id,
                )

                all_records.extend(records)
                episode_id += 1
                combo_count += 1

                if verbose:
                    pct = 100 * combo_count / total_combos
                    print(
                        f"[{pct:5.1f}%] {ctrl_cfg['controller_type']:<10} "
                        f"{road_cfg['road_type']:<12} ep={ep}  "
                        f"steps={len(records)}  total={len(all_records)}"
                    )

    if verbose:
        print(f"\nToplam kayıt: {len(all_records):,}")
        print(f"Episode sayısı: {episode_id}")

    return all_records
