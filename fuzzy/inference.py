"""
fuzzy/inference.py
------------------
Mamdani fuzzy çıkarım motoru.

Akış:
  1. Fuzzification : giriş değerlerini membership derecelerine çevir
  2. Rule firing   : her kural için aktivasyon gücü hesapla (AND = min)
  3. Aggregation   : aynı çıktı seviyesine gelen kuralları birleştir (max)
  4. Defuzzification: centroid ile crisp çıktı üret

Neden Mamdani?
--------------
Takagi-Sugeno'ya kıyasla çıktı membership fonksiyonları da tanımlı.
Bu sayede sonuçlar görselleştirilebilir — makale figürleri için ideal.
Yorumlanabilirlik Q1 için önemli bir argüman.
"""

import numpy as np
from typing import NamedTuple

from fuzzy.membership import MembershipSet, OutputMembershipSet
from fuzzy.rules import FuzzyRule, get_rules


# ------------------------------------------------------------------ #
#  Çıkarım giriş/çıkış yapıları                                      #
# ------------------------------------------------------------------ #

class FuzzyInputs(NamedTuple):
    """Fuzzy sisteme giren fiziksel değerler (ham, mutlak değer alınacak)."""
    body_acc  : float
    susp_defl : float
    ctrl_force: float
    tire_defl : float


class FuzzyOutputs(NamedTuple):
    """Fuzzy sistemden çıkan reward katsayıları."""
    alpha: float  # body_acc ceza çarpanı
    beta : float  # susp_defl ceza çarpanı
    gamma: float  # ctrl_force ceza çarpanı


# ------------------------------------------------------------------ #
#  Mamdani çıkarım motoru                                             #
# ------------------------------------------------------------------ #

class MamdaniInference:
    """
    Mamdani fuzzy çıkarım sistemi.

    Kurulumda membership setleri ve kural seti yüklenir.
    Her step'te infer() çağrılır — hızlı, numpy vektörleştirilmiş.

    Args:
        input_mfs  : her giriş değişkeni için MembershipSet dict
        output_mfs : her çıktı değişkeni için OutputMembershipSet dict
        rules      : FuzzyRule listesi
    """

    def __init__(
        self,
        input_mfs : dict[str, MembershipSet],
        output_mfs: dict[str, OutputMembershipSet],
        rules     : list[FuzzyRule],
    ):
        self.input_mfs  = input_mfs
        self.output_mfs = output_mfs
        self.rules      = rules

        # Kural sayısı ve çıktı değişken isimleri
        self.n_rules     = len(rules)
        self.output_keys = list(output_mfs.keys())

    # ---------------------------------------------------------------- #
    #  Fuzzification                                                    #
    # ---------------------------------------------------------------- #

    def _fuzzify(self, inputs: FuzzyInputs) -> dict[str, dict[str, float]]:
        """
        Ham girişleri membership derecelerine çevirir.

        Döndürür:
            {
              "body_acc":   {"low": 0.2, "med": 0.8, "high": 0.0},
              "susp_defl":  {"low": 0.9, "med": 0.1, "high": 0.0},
              "ctrl_force": {"low": 0.5, "med": 0.5, "high": 0.0},
              "tire_defl":  {"low": 1.0, "med": 0.0, "high": 0.0},
            }
        """
        return {
            "body_acc":   self.input_mfs["body_acc"].evaluate(inputs.body_acc),
            "susp_defl":  self.input_mfs["susp_defl"].evaluate(inputs.susp_defl),
            "ctrl_force": self.input_mfs["ctrl_force"].evaluate(inputs.ctrl_force),
            "tire_defl":  self.input_mfs["tire_defl"].evaluate(inputs.tire_defl),
        }

    # ---------------------------------------------------------------- #
    #  Kural ateşleme                                                   #
    # ---------------------------------------------------------------- #

    def _get_level_strength(
        self,
        memberships: dict[str, float],
        level: str,
    ) -> float:
        """
        Bir değişkenin belirli seviyedeki üyelik derecesini döndürür.
        "any" için 1.0 döner — koşul her zaman sağlanır.
        """
        if level == "any":
            return 1.0
        return memberships.get(level, 0.0)

    def _fire_rules(
        self,
        fuzzified: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Her kuralı ateşler, çıktı seviyesi başına maksimum gücü toplar.

        AND operatörü = min (Mamdani standardı)
        Aggregation   = max (birden fazla kural aynı çıktıya varırsa)

        Döndürür:
            {
              "alpha": {"low": 0.0, "med": 0.3, "high": 0.7},
              "beta":  {"low": 0.1, "med": 0.6, "high": 0.0},
              "gamma": {"low": 0.5, "med": 0.2, "high": 0.0},
            }
        """
        # Başlangıçta tüm çıktı seviyeleri sıfır
        aggregated = {
            key: {"low": 0.0, "med": 0.0, "high": 0.0}
            for key in self.output_keys
        }

        for rule in self.rules:
            # Antecedent: AND = min
            strength = min(
                self._get_level_strength(fuzzified["body_acc"],   rule.body_level),
                self._get_level_strength(fuzzified["susp_defl"],  rule.susp_level),
                self._get_level_strength(fuzzified["ctrl_force"], rule.force_level),
                self._get_level_strength(fuzzified["tire_defl"],  rule.tire_level),
            ) * rule.weight

            if strength < 1e-6:
                continue  # ateşlenmeyen kural

            # Consequent: aggregation = max
            aggregated["alpha"][rule.alpha_out] = max(
                aggregated["alpha"][rule.alpha_out], strength
            )
            aggregated["beta"][rule.beta_out] = max(
                aggregated["beta"][rule.beta_out], strength
            )
            aggregated["gamma"][rule.gamma_out] = max(
                aggregated["gamma"][rule.gamma_out], strength
            )

        return aggregated

    # ---------------------------------------------------------------- #
    #  Defuzzification                                                  #
    # ---------------------------------------------------------------- #

    def _defuzzify(
        self,
        aggregated: dict[str, dict[str, float]],
    ) -> FuzzyOutputs:
        """
        Centroid defuzzification — her çıktı için crisp değer üretir.
        """
        results = {}

        for key in self.output_keys:
            agg   = aggregated[key]
            omfs  = self.output_mfs[key]
            crisp = omfs.centroid(
                low_strength  = agg["low"],
                med_strength  = agg["med"],
                high_strength = agg["high"],
            )
            results[key] = crisp

        return FuzzyOutputs(
            alpha = results["alpha"],
            beta  = results["beta"],
            gamma = results["gamma"],
        )

    # ---------------------------------------------------------------- #
    #  Ana çıkarım fonksiyonu                                           #
    # ---------------------------------------------------------------- #

    def infer(self, inputs: FuzzyInputs) -> FuzzyOutputs:
        """
        Tam Mamdani çıkarım döngüsü.

        Args:
            inputs: FuzzyInputs(body_acc, susp_defl, ctrl_force, tire_defl)

        Returns:
            FuzzyOutputs(alpha, beta, gamma) — reward katsayıları
        """
        fuzzified  = self._fuzzify(inputs)
        aggregated = self._fire_rules(fuzzified)
        outputs    = self._defuzzify(aggregated)
        return outputs

    def infer_debug(self, inputs: FuzzyInputs) -> dict:
        """
        Debug modu — ara değerleri de döndürür.
        Makale figürleri ve analiz için kullanılır.
        """
        fuzzified  = self._fuzzify(inputs)
        aggregated = self._fire_rules(fuzzified)
        outputs    = self._defuzzify(aggregated)

        return {
            "inputs"    : inputs._asdict(),
            "fuzzified" : fuzzified,
            "aggregated": aggregated,
            "outputs"   : outputs._asdict(),
        }