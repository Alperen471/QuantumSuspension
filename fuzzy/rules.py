"""
fuzzy/rules.py
--------------
Fuzzy IF-THEN kural seti.

Kural Tasarım Felsefesi
------------------------
Kurallar üç hedefi dengeler:
  1. KONFOR    : body_acc düşük tutulsun (yolcu konforu)
  2. MEKANİK   : susp_defl + tire_defl sınırda kalmasın
  3. ENERJİ    : ctrl_force gereksiz yere büyük olmasın

Kurallar fizik sezgisine dayanır:
  - Yüksek gövde ivmesi → konfor kritik → α artır
  - Yüksek süspansiyon seyahati → mekanik kısıt → β artır
  - Yüksek kontrol kuvveti ama düşük ivme → enerji israfı → γ artır
  - Düz yol (hepsi Low) → dengeli ağırlıklar, enerji tasarrufu

Kural formatı:
    (body_acc_level, susp_defl_level, ctrl_force_level, tire_defl_level)
    → (alpha_output, beta_output, gamma_output)

Level değerleri: "low", "med", "high"
Output değerleri: "low", "med", "high"
"""

from typing import NamedTuple


# ------------------------------------------------------------------ #
#  Kural yapısı                                                       #
# ------------------------------------------------------------------ #

class FuzzyRule(NamedTuple):
    """
    Tek bir fuzzy kural.

    Antecedent (öncül) — giriş koşulları:
      body_level  : "low" | "med" | "high" | "any"
      susp_level  : "low" | "med" | "high" | "any"
      force_level : "low" | "med" | "high" | "any"
      tire_level  : "low" | "med" | "high" | "any"

    Consequent (ardıl) — çıktı katsayı seviyeleri:
      alpha_out : "low" | "med" | "high"   (body_acc ceza çarpanı)
      beta_out  : "low" | "med" | "high"   (susp_defl ceza çarpanı)
      gamma_out : "low" | "med" | "high"   (ctrl_force ceza çarpanı)

    weight: kural ağırlığı [0,1] — önem derecesi
    """
    body_level : str
    susp_level : str
    force_level: str
    tire_level : str
    alpha_out  : str
    beta_out   : str
    gamma_out  : str
    weight     : float = 1.0


# ------------------------------------------------------------------ #
#  Kural seti                                                         #
# ------------------------------------------------------------------ #

FUZZY_RULES = [

    # ================================================================ #
    # GRUP 1: Düz yol — hepsi düşük                                   #
    # Konfor iyi, kuvvet düşük tut                                     #
    # ================================================================ #
    FuzzyRule(
        body_level="low", susp_level="low", force_level="low", tire_level="low",
        alpha_out="low", beta_out="low", gamma_out="low",
        weight=1.0,
    ),

    # ================================================================ #
    # GRUP 2: Yüksek gövde ivmesi — konfor kritik                      #
    # ================================================================ #
    FuzzyRule(
        body_level="high", susp_level="any", force_level="any", tire_level="any",
        alpha_out="high", beta_out="med", gamma_out="low",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="high", susp_level="high", force_level="any", tire_level="any",
        alpha_out="high", beta_out="high", gamma_out="med",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="med", susp_level="low", force_level="low", tire_level="low",
        alpha_out="med", beta_out="low", gamma_out="low",
        weight=0.8,
    ),

    # ================================================================ #
    # GRUP 3: Yüksek süspansiyon seyahati — mekanik kısıt             #
    # ================================================================ #
    FuzzyRule(
        body_level="any", susp_level="high", force_level="any", tire_level="any",
        alpha_out="med", beta_out="high", gamma_out="med",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="low", susp_level="high", force_level="low", tire_level="any",
        alpha_out="low", beta_out="high", gamma_out="low",
        weight=0.9,
    ),

    # ================================================================ #
    # GRUP 4: Yüksek tire defleksiyonu — tekerlek kopma riski         #
    # ================================================================ #
    FuzzyRule(
        body_level="any", susp_level="any", force_level="any", tire_level="high",
        alpha_out="med", beta_out="high", gamma_out="med",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="low", susp_level="low", force_level="any", tire_level="high",
        alpha_out="low", beta_out="high", gamma_out="low",
        weight=0.9,
    ),

    # ================================================================ #
    # GRUP 5: Yüksek kontrol kuvveti — enerji israfı                  #
    # ================================================================ #
    FuzzyRule(
        body_level="low", susp_level="low", force_level="high", tire_level="low",
        alpha_out="low", beta_out="low", gamma_out="high",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="med", susp_level="low", force_level="high", tire_level="low",
        alpha_out="med", beta_out="low", gamma_out="high",
        weight=0.9,
    ),

    # ================================================================ #
    # GRUP 6: Dengeli orta durum                                       #
    # ================================================================ #
    FuzzyRule(
        body_level="med", susp_level="med", force_level="med", tire_level="med",
        alpha_out="med", beta_out="med", gamma_out="med",
        weight=0.8,
    ),
    FuzzyRule(
        body_level="med", susp_level="med", force_level="low", tire_level="low",
        alpha_out="med", beta_out="med", gamma_out="low",
        weight=0.8,
    ),

    # ================================================================ #
    # GRUP 7: Kritik durum — her şey yüksek                           #
    # ================================================================ #
    FuzzyRule(
        body_level="high", susp_level="high", force_level="high", tire_level="high",
        alpha_out="high", beta_out="high", gamma_out="high",
        weight=1.0,
    ),
    FuzzyRule(
        body_level="high", susp_level="high", force_level="high", tire_level="any",
        alpha_out="high", beta_out="high", gamma_out="med",
        weight=0.9,
    ),

    # ================================================================ #
    # GRUP 8: Tümsek/çukur geçişi — ani ivme, kuvvet orta              #
    # ================================================================ #
    FuzzyRule(
        body_level="high", susp_level="med", force_level="med", tire_level="med",
        alpha_out="high", beta_out="med", gamma_out="med",
        weight=0.9,
    ),
    FuzzyRule(
        body_level="high", susp_level="low", force_level="med", tire_level="low",
        alpha_out="high", beta_out="low", gamma_out="med",
        weight=0.8,
    ),
]


# ------------------------------------------------------------------ #
#  Kural yardımcıları                                                  #
# ------------------------------------------------------------------ #

def get_rules() -> list[FuzzyRule]:
    """Kural listesini döndür."""
    return FUZZY_RULES


def print_rules() -> None:
    """Kural setini okunabilir formatta yazdır."""
    print(f"\n{'='*70}")
    print(f"FUZZY KURAL SETİ — {len(FUZZY_RULES)} kural")
    print(f"{'='*70}")
    print(f"{'#':<4} {'body':<6} {'susp':<6} {'force':<6} {'tire':<6} "
          f"{'→α':<6} {'→β':<6} {'→γ':<6} {'w':>5}")
    print(f"{'-'*70}")
    for i, r in enumerate(FUZZY_RULES):
        print(
            f"{i+1:<4} {r.body_level:<6} {r.susp_level:<6} "
            f"{r.force_level:<6} {r.tire_level:<6} "
            f"{r.alpha_out:<6} {r.beta_out:<6} {r.gamma_out:<6} "
            f"{r.weight:>5.1f}"
        )
    print(f"{'='*70}\n")