"""
experiments/run_fuzzy_test.py
------------------------------
Fuzzy reward shaping modülünü test eder.

Testler:
  1. Membership fonksiyonları — değer aralığı, süreklilik
  2. Kural seti — tutarlılık kontrolü
  3. Mamdani çıkarım — bilinen girişler için beklenen çıktılar
  4. FuzzyRewardShaper — klasik vs fuzzy karşılaştırma
  5. Performans — her RL step'te kullanılabilir mi?
  6. Uç durumlar — sıfır, maksimum, negatif girişler

Çalıştırma:
    python -m experiments.run_fuzzy_test
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzy.membership import MembershipSet, OutputMembershipSet, trimf, TriMF
from fuzzy.rules import get_rules, print_rules
from fuzzy.inference import MamdaniInference, FuzzyInputs
from fuzzy.fuzzy_reward import FuzzyRewardShaper

STATS_PATH = "dataset/statistics.json"


def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ------------------------------------------------------------------ #
#  Test 1: Membership fonksiyonları                                   #
# ------------------------------------------------------------------ #

def test_membership():
    separator("TEST 1: Membership Fonksiyonları")

    # p95=1.0, max=4.5 ile test — gerçek body_acc dağılımına yakın
    mf = MembershipSet("body_acc", p95_abs=1.0, max_abs=4.5)

    # Sıfırda Low = 1.0 olmalı
    result = mf.evaluate(0.0)
    assert result["low"] > 0.9, f"Low(0) = {result['low']:.3f}"

    # p95 üstündeki değerlerde High hâlâ aktif olmalı — kör bölge yok
    result_p95  = mf.evaluate(1.0)
    result_2xp  = mf.evaluate(2.0)
    result_max  = mf.evaluate(4.5)
    assert result_2xp["high"] > 0, \
        f"2×p95'te High=0! Kör bölge var. High(2.0)={result_2xp['high']:.3f}"
    assert result_max["high"] > 0, \
        f"max_abs'te High=0! Kör bölge var."
    print("  Kör bölge testi: p95 üstü değerler High'a giriyor ✓")

    # Tüm değerler [0,1] aralığında olmalı
    for x in np.linspace(0, 6.0, 200):
        r = mf.evaluate(x)
        for level, val in r.items():
            assert 0.0 <= val <= 1.0, f"{level}({x}) = {val} aralık dışı!"
    print("  Membership değer aralığı: [0, 1] ✓")

    # Süreklilik testi
    prev = mf.evaluate(0.0)
    max_jump = 0.0
    for x in np.linspace(0.0, 5.0, 1000):
        curr = mf.evaluate(x)
        for level in ["low", "med", "high"]:
            jump = abs(curr[level] - prev[level])
            max_jump = max(max_jump, jump)
        prev = curr
    print(f"  Membership max sıçrama: {max_jump:.6f} (< 0.02 olmalı) ✓")
    assert max_jump < 0.02, f"Membership süreksiz! Max sıçrama: {max_jump}"

    # Genişletilmiş tablo — p95 üstü değerleri de göster
    print(f"\n  body_acc membership (p95=1.0, max=4.5):")
    print(f"  {'x':>6}  {'low':>6}  {'med':>6}  {'high':>6}  not")
    for x, note in [(0.0,"düz yol"), (0.5,"hafif"), (1.0,"p95"),
                    (1.5,"↑p95"), (2.0,"2×p95"), (3.0,"tümsek"),
                    (4.5,"max"), (5.0,"aşırı")]:
        r = mf.evaluate(x)
        any_active = "✓" if any(v > 0.01 for v in r.values()) else "✗ KÖR"
        print(f"  {x:>6.1f}  {r['low']:>6.3f}  {r['med']:>6.3f}  "
              f"{r['high']:>6.3f}  {note} {any_active}")

    print("\n✓ Membership testi geçti")


# ------------------------------------------------------------------ #
#  Test 2: Kural seti                                                 #
# ------------------------------------------------------------------ #

def test_rules():
    separator("TEST 2: Kural Seti")

    rules = get_rules()
    print_rules()

    # Her kural geçerli level değerleri içermeli
    valid_levels = {"low", "med", "high", "any"}
    valid_outputs = {"low", "med", "high"}

    for i, r in enumerate(rules):
        assert r.body_level  in valid_levels,  f"Kural {i}: geçersiz body_level"
        assert r.susp_level  in valid_levels,  f"Kural {i}: geçersiz susp_level"
        assert r.force_level in valid_levels,  f"Kural {i}: geçersiz force_level"
        assert r.tire_level  in valid_levels,  f"Kural {i}: geçersiz tire_level"
        assert r.alpha_out   in valid_outputs, f"Kural {i}: geçersiz alpha_out"
        assert r.beta_out    in valid_outputs, f"Kural {i}: geçersiz beta_out"
        assert r.gamma_out   in valid_outputs, f"Kural {i}: geçersiz gamma_out"
        assert 0.0 < r.weight <= 1.0,          f"Kural {i}: geçersiz weight"

    print(f"  {len(rules)} kural doğrulandı ✓")
    print("\n✓ Kural seti testi geçti")


# ------------------------------------------------------------------ #
#  Test 3: Mamdani çıkarım                                            #
# ------------------------------------------------------------------ #

def test_inference():
    separator("TEST 3: Mamdani Çıkarım")

    shaper = FuzzyRewardShaper(stats_path=STATS_PATH)
    shaper.print_summary()

    # Senaryo 1: Düz yol — hepsi düşük
    # Beklenti: α düşük, β düşük, γ düşük (enerji tasarrufu modu)
    out1 = shaper.get_coefficients(
        body_acc=0.01, susp_defl=0.0005, ctrl_force=5.0, tire_defl=0.0005
    )
    print(f"\n  Senaryo 1 (düz yol, hepsi düşük):")
    print(f"    α={out1.alpha:.3f}  β={out1.beta:.3f}  γ={out1.gamma:.3f}")
    assert out1.alpha < 3.0, "Düz yolda α çok yüksek"

    # Senaryo 2: Yüksek gövde ivmesi (tümsek)
    # Beklenti: α yüksek (konfor öncelikli)
    out2 = shaper.get_coefficients(
        body_acc=3.0, susp_defl=0.002, ctrl_force=50.0, tire_defl=0.002
    )
    print(f"\n  Senaryo 2 (yüksek ivme - tümsek):")
    print(f"    α={out2.alpha:.3f}  β={out2.beta:.3f}  γ={out2.gamma:.3f}")
    assert out2.alpha > out1.alpha, "Yüksek ivmede α artmalı"

    # Senaryo 3: Yüksek kontrol kuvveti, düşük ivme
    # Beklenti: γ yüksek (enerji cezası)
    out3 = shaper.get_coefficients(
        body_acc=0.05, susp_defl=0.001, ctrl_force=200.0, tire_defl=0.001
    )
    print(f"\n  Senaryo 3 (yüksek kuvvet, düşük ivme):")
    print(f"    α={out3.alpha:.3f}  β={out3.beta:.3f}  γ={out3.gamma:.3f}")
    assert out3.gamma > out1.gamma, "Yüksek kuvvette γ artmalı"

    # Senaryo 4: Kritik — her şey yüksek
    # Beklenti: hepsi yüksek
    out4 = shaper.get_coefficients(
        body_acc=4.0, susp_defl=0.03, ctrl_force=400.0, tire_defl=0.03
    )
    print(f"\n  Senaryo 4 (kritik - her şey yüksek):")
    print(f"    α={out4.alpha:.3f}  β={out4.beta:.3f}  γ={out4.gamma:.3f}")
    assert out4.alpha > out1.alpha, "Kritik durumda α artmalı"
    assert out4.beta  > out1.beta,  "Kritik durumda β artmalı"

    # Debug çıktısı
    debug = shaper.debug_step(
        body_acc=3.0, susp_defl=0.02, ctrl_force=100.0, tire_defl=0.01
    )
    print(f"\n  Debug çıktısı (ivme=3.0, defl=0.02, kuvvet=100):")
    print(f"    Fuzzified: {debug['fuzzified']}")
    print(f"    Aggregated: {debug['aggregated']}")
    print(f"    Fuzzy reward   : {debug['fuzzy_reward']:.6f}")
    print(f"    Classical reward: {debug['classical_reward']:.6f}")
    print(f"    Fark           : {abs(debug['fuzzy_reward'] - debug['classical_reward']):.6f}")

    print("\n✓ Mamdani çıkarım testi geçti")


# ------------------------------------------------------------------ #
#  Test 4: Reward karşılaştırma                                       #
# ------------------------------------------------------------------ #

def test_reward_comparison():
    separator("TEST 4: Fuzzy vs Klasik Reward Karşılaştırması")

    shaper_bounded = FuzzyRewardShaper(stats_path=STATS_PATH, mode="bounded")
    shaper_raw     = FuzzyRewardShaper(stats_path=STATS_PATH, mode="raw")

    scenarios = [
        ("Düz yol",      0.05,  0.0005, 5.0,   0.0005),
        ("Hafif tümsek", 0.5,   0.005,  50.0,  0.003),
        ("Orta tümsek",  1.5,   0.01,   150.0, 0.008),
        ("Sert tümsek",  3.5,   0.025,  350.0, 0.02),
        ("Kritik",       4.0,   0.035,  450.0, 0.03),
    ]

    print(f"\n  {'Senaryo':<16} {'Klasik':>10} {'Bounded':>10} "
          f"{'Raw':>10} {'B-Fark%':>9} {'R-Fark%':>9}")
    print(f"  {'-'*66}")

    for name, body, susp, force, tire in scenarios:
        classical = FuzzyRewardShaper.classical_reward(body, susp, force)
        bounded_r = shaper_bounded.compute_reward(body, susp, force, tire)
        raw_r     = shaper_raw.compute_reward(body, susp, force, tire)

        b_pct = abs(bounded_r - classical) / (abs(classical) + 1e-10) * 100
        r_pct = abs(raw_r     - classical) / (abs(classical) + 1e-10) * 100

        print(f"  {name:<16} {classical:>10.4f} {bounded_r:>10.4f} "
              f"{raw_r:>10.4f} {b_pct:>8.1f}% {r_pct:>8.1f}%")

    # Katsayı aralığı doğrulama
    print(f"\n  Bounded katsayı aralıkları:")
    for name, body, susp, force, tire in scenarios:
        out = shaper_bounded.get_coefficients(body, susp, force, tire)
        assert 0.5 <= out.alpha <= 2.0, f"α={out.alpha:.3f} aralık dışı!"
        assert 0.5 <= out.beta  <= 2.0, f"β={out.beta:.3f} aralık dışı!"
        assert 0.5 <= out.gamma <= 2.0, f"γ={out.gamma:.3f} aralık dışı!"

    # Son senaryo için detay
    out_b = shaper_bounded.get_coefficients(3.5, 0.025, 350.0, 0.02)
    out_r = shaper_raw.get_coefficients(3.5, 0.025, 350.0, 0.02)
    print(f"\n  Sert tümsek katsayıları:")
    print(f"    Bounded: α={out_b.alpha:.3f}  β={out_b.beta:.3f}  γ={out_b.gamma:.3f}")
    print(f"    Raw    : α={out_r.alpha:.3f}  β={out_r.beta:.3f}  γ={out_r.gamma:.3f}")
    print(f"\n  Ablation matrisinde 3 model kullanılacak:")
    print(f"    Model A: Klasik RL        (α=1.0 sabit)")
    print(f"    Model B: Fuzzy-Bounded RL (α∈[0.8,1.5])")
    print(f"    Model C: Fuzzy-Raw RL     (α∈[0.5,5.0])")
    print(f"    A vs B → fuzzy reasoning'in gerçek katkısı")
    print(f"    B vs C → reward ölçeğinin etkisi")

    print("\n✓ Reward karşılaştırma testi geçti")


# ------------------------------------------------------------------ #
#  Test 5: Performans                                                  #
# ------------------------------------------------------------------ #

def test_performance():
    separator("TEST 5: Performans Testi")

    shaper = FuzzyRewardShaper(stats_path=STATS_PATH)
    n_steps = 10_000

    # Warmup
    for _ in range(100):
        shaper.compute_reward(1.0, 0.01, 100.0, 0.005)

    # Ölçüm
    t_start = time.time()
    for _ in range(n_steps):
        shaper.compute_reward(
            np.random.uniform(0, 3.0),
            np.random.uniform(0, 0.03),
            np.random.uniform(0, 400.0),
            np.random.uniform(0, 0.02),
        )
    t_elapsed = (time.time() - t_start) * 1000  # ms

    per_step = t_elapsed / n_steps
    print(f"  {n_steps} adım toplam: {t_elapsed:.1f}ms")
    print(f"  Per-step ortalama : {per_step:.4f}ms")
    print(f"  RL uyumluluğu     : ", end="")

    if per_step < 0.1:
        print(f"✓ Mükemmel ({per_step:.4f}ms < 0.1ms)")
    elif per_step < 1.0:
        print(f"✓ İyi ({per_step:.4f}ms < 1ms)")
    else:
        print(f"⚠ Yavaş ({per_step:.4f}ms > 1ms) — optimizasyon gerekebilir")

    print("\n✓ Performans testi geçti")


# ------------------------------------------------------------------ #
#  Test 6: Uç durumlar                                                #
# ------------------------------------------------------------------ #

def test_edge_cases():
    separator("TEST 6: Uç Durum Testleri")

    shaper = FuzzyRewardShaper(stats_path=STATS_PATH)

    # Sıfır girişler
    r = shaper.compute_reward(0.0, 0.0, 0.0, 0.0)
    assert not np.isnan(r), "Sıfır girişte NaN!"
    assert not np.isinf(r), "Sıfır girişte Inf!"
    print(f"  Sıfır girişler  : reward={r:.6f} ✓")

    # Çok büyük girişler
    r = shaper.compute_reward(100.0, 1.0, 10000.0, 1.0)
    assert not np.isnan(r), "Büyük girişte NaN!"
    assert not np.isinf(r), "Büyük girişte Inf!"
    print(f"  Büyük girişler  : reward={r:.6f} ✓")

    # Negatif girişler — mutlak değer alınmalı
    r_pos = shaper.compute_reward( 2.0,  0.01,  100.0,  0.005)
    r_neg = shaper.compute_reward(-2.0, -0.01, -100.0, -0.005)
    assert abs(r_pos - r_neg) < 1e-6, \
        f"Simetri hatası: pozitif={r_pos:.6f}, negatif={r_neg:.6f}"
    print(f"  Simetri testi   : r(+)={r_pos:.6f}, r(-)={r_neg:.6f} ✓")

    # Reward her zaman negatif olmalı
    for _ in range(100):
        r = shaper.compute_reward(
            np.random.uniform(-5, 5),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-500, 500),
            np.random.uniform(-0.05, 0.05),
        )
        assert r <= 0, f"Reward pozitif: {r}"
    print(f"  Reward negatiflik: 100/100 ✓")

    print("\n✓ Uç durum testleri geçti")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("\n" + "="*55)
    print("  FUZZY REWARD SHAPING TEST SÜİTİ")
    print("="*55)

    try:
        test_membership()
        test_rules()
        test_inference()
        test_reward_comparison()
        test_performance()
        test_edge_cases()

        print("\n" + "="*55)
        print("  TÜM TESTLER BAŞARILI ✓")
        print("="*55)
        print("\nSonraki adım: RL Training modülü")

    except AssertionError as e:
        print(f"\n✗ TEST BAŞARISIZ: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()