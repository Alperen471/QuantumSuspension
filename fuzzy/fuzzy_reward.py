"""
fuzzy/fuzzy_reward.py
---------------------
Fuzzy reward shaping sistemi — environment'a plug-in.

Kullanım:
    # RL training başında bir kez oluştur
    fuzzy = FuzzyRewardShaper(stats_path="dataset/statistics.json")

    # Her step'te environment'dan gelen info ile çağır
    reward = fuzzy.compute_reward(
        body_acc=info["body_acc"],
        susp_defl=info["suspension_deflection"],
        ctrl_force=info["control_force"],
        tire_defl=info["tire_deflection"],
    )

Ablation için:
    # Fuzzy kapalı — sabit katsayılarla klasik reward
    reward = FuzzyRewardShaper.classical_reward(body_acc, susp_defl, ctrl_force)

    # Fuzzy açık — adaptif katsayılar
    reward = fuzzy.compute_reward(...)
"""

import numpy as np
from typing import Optional

from fuzzy.membership import MembershipSet, OutputMembershipSet
from fuzzy.inference import MamdaniInference, FuzzyInputs, FuzzyOutputs
from fuzzy.rules import get_rules
from dataset.normalizer import load_statistics


# ------------------------------------------------------------------ #
#  Klasik (sabit katsayılı) reward sabitleri                         #
# ------------------------------------------------------------------ #

# Baseline reward katsayıları — mevcut quarter_car_env.py'deki değerler
ALPHA_BASE = 1.0      # body_acc ceza çarpanı
BETA_BASE  = 10.0     # susp_defl ceza çarpanı
GAMMA_BASE = 1e-6     # ctrl_force ceza çarpanı


# ------------------------------------------------------------------ #
#  Ana sınıf                                                          #
# ------------------------------------------------------------------ #

class FuzzyRewardShaper:
    """
    Fuzzy Mamdani reward shaping sistemi.

    Adım adım ne yapıyor:
    1. statistics.json'dan dataset istatistiklerini yükler
    2. Her giriş değişkeni için MembershipSet oluşturur (p95_abs ile kalibre)
    3. Her reward katsayısı (α,β,γ) için OutputMembershipSet oluşturur
    4. Kural setini yükler
    5. MamdaniInference motorunu başlatır

    Her step'te:
    6. Fiziksel değerleri alır → fuzzy çıkarım → α,β,γ üretir
    7. R = -(α·body_acc² + β·susp_defl² + γ·ctrl_force²) hesaplar

    Args:
        stats_path: dataset/statistics.json yolu
        alpha_scale: α katsayısını ölçekle (default 1.0 = normalize)
        beta_scale:  β katsayısını ölçekle
        gamma_scale: γ katsayısını ölçekle
    """

    def __init__(
        self,
        stats_path  : str = "dataset/statistics.json",
        alpha_scale : float = 1.0,
        beta_scale  : float = 10.0,
        gamma_scale : float = 1e-6,
        mode        : str = "bounded",
    ):
        """
        Args:
            mode: "bounded" → α ∈ [0.8, 1.5] adil ablation karşılaştırması
                  "raw"     → α ∈ [0.5, 5.0] ham fuzzy, etki büyüklüğü analizi
        """
        self.alpha_scale = alpha_scale
        self.beta_scale  = beta_scale
        self.gamma_scale = gamma_scale
        self.mode        = mode

        # İstatistikleri yükle
        try:
            stats, fuzzy_bounds = load_statistics(stats_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{stats_path} bulunamadı.\n"
                "Önce: python -m experiments.run_dataset_collection"
            )

        # Giriş membership setleri — p95_abs ve max_abs ile kalibre
        def _p95(col):
            s = stats[col]
            return max(abs(s["p05"]), abs(s["p95"]))

        def _max(col):
            s = stats[col]
            return max(abs(s["min"]), abs(s["max"]))

        self.input_mfs = {
            "body_acc":   MembershipSet("body_acc",
                p95_abs=_p95("body_acc"),
                max_abs=_max("body_acc")),
            "susp_defl":  MembershipSet("susp_defl",
                p95_abs=_p95("susp_defl"),
                max_abs=_max("susp_defl")),
            "ctrl_force": MembershipSet("ctrl_force",
                p95_abs=_p95("ctrl_force"),
                max_abs=_max("ctrl_force")),
            "tire_defl":  MembershipSet("tire_defl",
                p95_abs=_p95("tire_defl"),
                max_abs=_max("tire_defl")),
        }

        # Çıktı membership setleri — mode'a göre bounded veya raw
        self.output_mfs = {
            "alpha": OutputMembershipSet("alpha", mode=self.mode),
            "beta":  OutputMembershipSet("beta",  mode=self.mode),
            "gamma": OutputMembershipSet("gamma", mode=self.mode),
        }

        # Çıkarım motoru
        self.engine = MamdaniInference(
            input_mfs  = self.input_mfs,
            output_mfs = self.output_mfs,
            rules      = get_rules(),
        )

        # İstatistik kayıt (analiz için)
        self._step_count  = 0
        self._alpha_history: list[float] = []
        self._beta_history : list[float] = []
        self._gamma_history: list[float] = []

    # ---------------------------------------------------------------- #
    #  Reward hesaplama                                                 #
    # ---------------------------------------------------------------- #

    def compute_reward(
        self,
        body_acc  : float,
        susp_defl : float,
        ctrl_force: float,
        tire_defl : float,
        record    : bool = False,
    ) -> float:
        """
        Fuzzy adaptif reward hesaplar.

        Args:
            body_acc   : gövde ivmesi (m/s²)
            susp_defl  : süspansiyon defleksiyonu (m)
            ctrl_force : kontrol kuvveti (N)
            tire_defl  : tire defleksiyonu (m)
            record     : True → katsayı geçmişini kaydet (analiz için)

        Returns:
            reward: negatif skaler
        """
        # Fuzzy çıkarım
        inputs  = FuzzyInputs(body_acc, susp_defl, ctrl_force, tire_defl)
        outputs = self.engine.infer(inputs)

        # Katsayıları ölçekle — baseline değerleri etrafında modülasyon
        alpha = outputs.alpha * self.alpha_scale
        beta  = outputs.beta  * self.beta_scale
        gamma = outputs.gamma * self.gamma_scale

        # Reward
        reward = -(
            alpha * (body_acc   ** 2) +
            beta  * (susp_defl  ** 2) +
            gamma * (ctrl_force ** 2)
        )

        # Kayıt
        if record:
            self._alpha_history.append(outputs.alpha)
            self._beta_history.append(outputs.beta)
            self._gamma_history.append(outputs.gamma)
        self._step_count += 1

        return float(reward)

    def get_coefficients(
        self,
        body_acc  : float,
        susp_defl : float,
        ctrl_force: float,
        tire_defl : float,
    ) -> FuzzyOutputs:
        """
        Sadece α, β, γ katsayılarını döndür — reward hesaplamadan.
        Analiz ve görselleştirme için.
        """
        inputs = FuzzyInputs(body_acc, susp_defl, ctrl_force, tire_defl)
        return self.engine.infer(inputs)

    # ---------------------------------------------------------------- #
    #  Klasik (sabit katsayılı) reward — ablation baseline            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def classical_reward(
        body_acc  : float,
        susp_defl : float,
        ctrl_force: float,
    ) -> float:
        """
        Sabit katsayılı klasik reward — fuzzy kapalı ablation.
        quarter_car_env.py'deki orijinal formül.
        """
        return float(
            -(ALPHA_BASE * (body_acc   ** 2) +
              BETA_BASE  * (susp_defl  ** 2) +
              GAMMA_BASE * (ctrl_force ** 2))
        )

    # ---------------------------------------------------------------- #
    #  İstatistik ve analiz                                             #
    # ---------------------------------------------------------------- #

    def reset_history(self) -> None:
        """Episode başında geçmişi sıfırla."""
        self._alpha_history.clear()
        self._beta_history.clear()
        self._gamma_history.clear()
        self._step_count = 0

    def get_coefficient_stats(self) -> dict:
        """
        Kaydedilen katsayı geçmişinin istatistiklerini döndürür.
        Eğitim sonrası analiz için.
        """
        if not self._alpha_history:
            return {}

        def _stats(arr):
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "std" : float(np.std(a)),
                "min" : float(np.min(a)),
                "max" : float(np.max(a)),
            }

        return {
            "alpha": _stats(self._alpha_history),
            "beta" : _stats(self._beta_history),
            "gamma": _stats(self._gamma_history),
            "n_steps": self._step_count,
        }

    def debug_step(
        self,
        body_acc  : float,
        susp_defl : float,
        ctrl_force: float,
        tire_defl : float,
    ) -> dict:
        """
        Tek adım için tam debug çıktısı.
        Makale analizleri ve görselleştirmeler için.
        """
        inputs = FuzzyInputs(body_acc, susp_defl, ctrl_force, tire_defl)
        debug  = self.engine.infer_debug(inputs)

        # Ölçeklenmiş katsayılar
        debug["scaled_alpha"] = debug["outputs"]["alpha"] * self.alpha_scale
        debug["scaled_beta"]  = debug["outputs"]["beta"]  * self.beta_scale
        debug["scaled_gamma"] = debug["outputs"]["gamma"] * self.gamma_scale

        # Karşılaştırma için klasik reward
        debug["fuzzy_reward"] = self.compute_reward(
            body_acc, susp_defl, ctrl_force, tire_defl
        )
        debug["classical_reward"] = self.classical_reward(
            body_acc, susp_defl, ctrl_force
        )

        return debug

    def print_summary(self) -> None:
        """Fuzzy sistemin özetini yazdır."""
        print(f"\n{'='*55}")
        print(f"Fuzzy Reward Shaper Özeti")
        print(f"{'='*55}")
        print(f"  Kural sayısı    : {len(self.engine.rules)}")
        print(f"  Giriş değişkeni : {list(self.input_mfs.keys())}")
        print(f"  Çıktı katsayısı : α, β, γ")
        print(f"\n  Membership sınırları (p95_abs):")
        for name, mfs in self.input_mfs.items():
            print(f"    {name:<12}: p95={mfs.p95_abs:.4f}")
        print(f"\n  Ölçek faktörleri:")
        print(f"    alpha_scale = {self.alpha_scale}")
        print(f"    beta_scale  = {self.beta_scale}")
        print(f"    gamma_scale = {self.gamma_scale}")
        print(f"{'='*55}\n")