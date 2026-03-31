"""
experiments/run_quantum_test.py
--------------------------------
Quantum encoder modülünü test eder.

Testler:
  1. Devre özeti — her n_qubits için yapı kontrolü
  2. Klasik encoder'lar — QAngle, QAmp, Identity
  3. QNodeEncoder — sabit ve variational, her n_qubits
  4. lightning.gpu performans testi
  5. Gradient akışı testi — variational parametreler öğrenilebilir mi?

Çalıştırma:
    python -m experiments.run_quantum_test
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum.circuit import print_circuit_summary, build_circuit
from quantum.classical_encoder import QAngleEncoder, QAmpEncoder, IdentityEncoder, build_encoder
from quantum.qnode_encoder import QNodeEncoder


STATS_PATH = "dataset/statistics.json"
TEST_STATE = np.array([0.02, -0.01, 0.15, -0.05, 0.008], dtype=np.float32)


def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ------------------------------------------------------------------ #
#  Test 1: Devre özeti                                                 #
# ------------------------------------------------------------------ #

def test_circuit_summary():
    separator("TEST 1: Devre Yapısı Özeti")

    for n in [5, 10, 15, 20, 25]:
        print_circuit_summary(n_qubits=n, variational=True)

    print("✓ Devre özeti testi geçti")


# ------------------------------------------------------------------ #
#  Test 2: Klasik encoder'lar                                          #
# ------------------------------------------------------------------ #

def test_classical_encoders():
    separator("TEST 2: Klasik Encoder'lar")

    encoders = {
        "Identity (5D)": IdentityEncoder(stats_path=STATS_PATH),
        "QAngle  (10D)": QAngleEncoder(stats_path=STATS_PATH),
        "QAmp    (15D)": QAmpEncoder(stats_path=STATS_PATH),
    }

    for name, enc in encoders.items():
        features = enc.encode(TEST_STATE)
        t_features = enc.encode_torch(torch.tensor(TEST_STATE))

        assert features.shape == (enc.output_dim,), \
            f"{name}: shape hatası {features.shape} != ({enc.output_dim},)"
        assert not np.any(np.isnan(features)), f"{name}: NaN var!"
        assert not np.any(np.isinf(features)), f"{name}: Inf var!"

        print(f"  {name}: output={features.shape}  "
              f"range=[{features.min():.3f}, {features.max():.3f}]  ✓")

    # build_encoder fabrikası testi
    for etype in ["none", "qangle", "qamp"]:
        enc = build_encoder(etype, stats_path=STATS_PATH)
        f = enc.encode(TEST_STATE)
        print(f"  build_encoder('{etype}'): output_dim={enc.output_dim}  ✓")

    print("\n✓ Klasik encoder testleri geçti")


# ------------------------------------------------------------------ #
#  Test 3: QNodeEncoder — tüm n_qubits değerleri                      #
# ------------------------------------------------------------------ #

def test_qnode_encoder():
    separator("TEST 3: QNodeEncoder — n_qubits Ablation")

    qubit_configs = [5, 10, 15, 20, 25]

    for n_qubits in qubit_configs:
        for variational in [False, True]:
            print(f"\n  n_qubits={n_qubits}, variational={variational}")

            enc = QNodeEncoder(
                n_qubits=n_qubits,
                variational=variational,
                device_name="lightning.gpu",
                stats_path=STATS_PATH,
            )

            print(f"    {enc}")

            # JIT warmup — ilk çağrı derleme süresi ölçüme girmesin
            for _ in range(3):
                enc.encode(TEST_STATE)

            # Gerçek encode süresi ölçümü
            t_start = time.time()
            features = enc.encode(TEST_STATE)
            t_elapsed = time.time() - t_start

            assert features.shape == (enc.output_dim,), \
                f"Shape hatası: {features.shape} != ({enc.output_dim},)"
            assert not np.any(np.isnan(features)), "NaN var!"
            assert not np.any(np.isinf(features)), "Inf var!"

            # Değer aralığı kontrolü — ZZ dahil [-1,+1] dışına çıkabilir ama sınırlı olmalı
            assert np.all(np.abs(features) <= 2.0), \
                f"Değerler aralık dışı: {features}"

            print(
                f"    output={features.shape}  "
                f"range=[{features.min():.3f}, {features.max():.3f}]  "
                f"süre={t_elapsed*1000:.1f}ms  ✓"
            )

            # RL kullanılabilirlik uyarısı
            # 1M step × t_per_step = toplam süre tahmini
            estimated_hours = (t_elapsed * 1_000_000) / 3600
            if estimated_hours > 10:
                print(
                    f"    [UYARI] 1M RL step için tahmini süre: "
                    f"{estimated_hours:.0f} saat — "
                    f"bu n_qubits için episode sayısını azaltmayı düşün"
                )

    print("\n✓ QNodeEncoder testleri geçti")


# ------------------------------------------------------------------ #
#  Test 4: GPU performans karşılaştırması                              #
# ------------------------------------------------------------------ #

def test_gpu_performance():
    separator("TEST 4: GPU Performans Karşılaştırması")

    n_steps   = 100
    n_warmup  = 5   # JIT derleme için yeterli warmup adımı

    print(f"  {n_warmup} warmup + {n_steps} ölçüm adımı\n")
    print(f"  {'n_qubits':<12} {'default.qubit':>16} {'lightning.gpu':>16} {'hızlanma':>12}")
    print(f"  {'-'*58}")

    for n_qubits in [5, 10, 15, 20]:
        times = {}

        for device_name in ["default.qubit", "lightning.gpu"]:
            # Encoder bir kez oluştur — JIT warmup encoder ömrüyle bağlı
            enc = QNodeEncoder(
                n_qubits=n_qubits,
                variational=True,
                device_name=device_name,
                stats_path=STATS_PATH,
            )

            # Çoklu warmup — JIT derleme tamamen tamamlansın
            for _ in range(n_warmup):
                enc.encode(TEST_STATE)

            # Gerçek ölçüm
            t_start = time.time()
            for _ in range(n_steps):
                enc.encode(TEST_STATE)
            t_elapsed = (time.time() - t_start) * 1000  # ms toplam
            times[device_name] = t_elapsed

        per_step_cpu = times["default.qubit"] / n_steps
        per_step_gpu = times["lightning.gpu"]  / n_steps
        speedup = per_step_cpu / max(per_step_gpu, 0.001)

        print(
            f"  {n_qubits:<12} "
            f"{per_step_cpu:>13.2f}ms  "
            f"{per_step_gpu:>13.2f}ms  "
            f"{speedup:>10.1f}x"
        )

    print("\n  Not: Süre per-step ortalaması (warmup hariç)")
    print("\n✓ GPU performans testi geçti")


# ------------------------------------------------------------------ #
#  Test 5: Gradient akışı testi                                        #
# ------------------------------------------------------------------ #

def test_gradient_flow():
    separator("TEST 5: Gradient Akışı — Variational Parametreler")

    enc = QNodeEncoder(
        n_qubits=5,
        variational=True,
        device_name="lightning.gpu",
        stats_path=STATS_PATH,
    )

    print(f"  Encoder: {enc}")
    print(f"  Parametre sayısı: {enc.n_params}")

    # Torch state'i
    state_t = torch.tensor(TEST_STATE, dtype=torch.float32)

    # Forward pass — gradient izleme açık
    features = enc.encode_torch(state_t)
    assert features.shape == (enc.output_dim,), "Shape hatası"

    # Dummy loss — gerçek training'de reward-based olacak
    loss = features.sum()
    loss.backward()

    # Gradient var mı?
    if enc._params is not None and enc._params.grad is not None:
        grad_norm = enc._params.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.6f}")
        assert grad_norm > 0, "Gradient sıfır — barren plateau!"
        print(f"  ✓ Gradient akışı sağlıklı")
    else:
        print(f"  [WARNING] Gradient hesaplanamadı — parameter-shift kontrol et")

    # Optimizer testi
    optimizer = torch.optim.Adam(enc.parameters_for_optimizer(), lr=1e-3)
    params_before = enc._params.data.clone()

    for _ in range(3):
        optimizer.zero_grad()
        features = enc.encode_torch(state_t)
        loss = -features.sum()  # minimize et
        loss.backward()
        optimizer.step()

    params_after = enc._params.data
    param_change = (params_after - params_before).abs().max().item()
    print(f"  3 optimizer adımı sonrası max param değişimi: {param_change:.6f}")
    assert param_change > 0, "Parametreler güncellenmiyor!"
    print(f"  ✓ Parametreler başarıyla güncellendi")

    print("\n✓ Gradient akış testi geçti")


# ------------------------------------------------------------------ #
#  Test 6: build_encoder fabrikası — quantum                          #
# ------------------------------------------------------------------ #

def test_build_encoder_factory():
    separator("TEST 6: build_encoder Fabrikası")

    for n_qubits in [5, 10]:
        for var in [False, True]:
            enc = build_encoder(
                "quantum",
                n_qubits=n_qubits,
                variational=var,
                stats_path=STATS_PATH,
            )
            f = enc.encode(TEST_STATE)
            print(f"  quantum n={n_qubits} var={var}: "
                  f"output_dim={enc.output_dim} shape={f.shape}  ✓")

    print("\n✓ Fabrika testi geçti")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("\n" + "="*55)
    print("  QUANTUM ENCODER TEST SÜİTİ")
    print("="*55)
    print(f"  Test state: {TEST_STATE}")
    print(f"  Stats path: {STATS_PATH}")

    try:
        test_circuit_summary()
        test_classical_encoders()
        test_qnode_encoder()
        test_gpu_performance()
        test_gradient_flow()
        test_build_encoder_factory()

        print("\n" + "="*55)
        print("  TÜM TESTLER BAŞARILI ✓")
        print("="*55)
        print("\nSonraki adım: Fuzzy Reward Shaping modülü")

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