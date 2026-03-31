"""
rl/env_wrapper.py
-----------------
Environment wrapper'ları — reward injection için.

Neden wrapper?
--------------
Mevcut QuarterCarEnv koduna dokunmadan fuzzy reward
inject edebiliyoruz. Ablation için tek değiştirilen
şey wrapper parametresi.

Kullanım:
    # Klasik RL
    env = QuarterCarEnv(road_type="random")

    # Fuzzy RL — bounded
    env = FuzzyRewardWrapper(
        QuarterCarEnv(road_type="random"),
        mode="bounded"
    )

    # Fuzzy RL — raw
    env = FuzzyRewardWrapper(
        QuarterCarEnv(road_type="random"),
        mode="raw"
    )
"""

import numpy as np
import gymnasium as gym
from fuzzy.fuzzy_reward import FuzzyRewardShaper


class FuzzyRewardWrapper(gym.Wrapper):
    """
    QuarterCarEnv'in reward fonksiyonunu fuzzy ile değiştirir.

    Environment'ın step() metodunu override eder:
    - Orijinal reward hesaplanır (ama kullanılmaz)
    - info dict'ten fiziksel değerler alınır
    - FuzzyRewardShaper ile yeni reward hesaplanır

    Args:
        env       : wrap edilecek QuarterCarEnv instance'ı
        mode      : "bounded" veya "raw"
        stats_path: normalizasyon istatistikleri
    """

    def __init__(
        self,
        env       : gym.Env,
        mode      : str = "bounded",
        stats_path: str = "dataset/statistics.json",
    ):
        super().__init__(env)
        self.fuzzy_shaper = FuzzyRewardShaper(
            stats_path=stats_path,
            mode=mode,
        )
        self.mode = mode

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Fiziksel değerlerden fuzzy reward hesapla
        fuzzy_reward = self.fuzzy_shaper.compute_reward(
            body_acc   = info["body_acc"],
            susp_defl  = info["suspension_deflection"],
            ctrl_force = info["control_force"],
            tire_defl  = info["tire_deflection"],
        )

        # Info'ya fuzzy katsayıları ekle — analiz için
        coeffs = self.fuzzy_shaper.get_coefficients(
            body_acc   = info["body_acc"],
            susp_defl  = info["suspension_deflection"],
            ctrl_force = info["control_force"],
            tire_defl  = info["tire_deflection"],
        )
        info["fuzzy_alpha"] = coeffs.alpha
        info["fuzzy_beta"]  = coeffs.beta
        info["fuzzy_gamma"] = coeffs.gamma
        info["fuzzy_mode"]  = self.mode

        return obs, fuzzy_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.fuzzy_shaper.reset_history()
        return self.env.reset(**kwargs)