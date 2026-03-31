"""
experiments/run_dataset_collection.py
--------------------------------------
Dataset toplama pipeline'ını çalıştırır.

Çalıştırma:
    cd QuantumSuspension
    python -m experiments.run_dataset_collection

Üretilen çıktılar:
    dataset/data/full_dataset.parquet  — pandas analizi için
    dataset/data/full_dataset.npz      — RL replay buffer için
    dataset/statistics.json            — normalizasyon + fuzzy sınırları

Bu dosya bir kez çalıştırılır. Sonraki tüm modüller (fuzzy, quantum, rl)
dataset/statistics.json'ı yükleyerek kalibre olur.
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.collector import collect_dataset
from dataset.normalizer import (
    compute_statistics,
    suggest_fuzzy_boundaries,
    save_statistics,
    ALL_COLS,
    FUZZY_COLS,
    STATE_COLS,
)
from dataset.storage import (
    save_parquet,
    save_numpy,
    print_dataset_summary,
)


def main():
    print("=" * 60)
    print("DATASET COLLECTION PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------------- #
    # 1. Veri toplama                                                    #
    # ---------------------------------------------------------------- #
    print("\n[1/4] Veri toplanıyor...")
    print("      Passive + Skyhook × 4 road × 5 episode = 40 episode")
    print("      Her episode 2000 step → ~80 000 kayıt bekleniyor\n")

    records = collect_dataset(
        episodes_per_combo=5,
        episode_length=2000,
        dt=0.01,
        base_seed=42,
        verbose=True,
    )

    # ---------------------------------------------------------------- #
    # 2. Özet raporu                                                     #
    # ---------------------------------------------------------------- #
    print("\n[2/4] Dataset özeti hesaplanıyor...")
    print_dataset_summary(records)

    # ---------------------------------------------------------------- #
    # 3. İstatistik hesaplama                                            #
    # ---------------------------------------------------------------- #
    print("[3/4] Normalizasyon istatistikleri hesaplanıyor...")
    stats = compute_statistics(records, columns=ALL_COLS)

    print("\n      State değişkeni istatistikleri:")
    for col in STATE_COLS:
        s = stats[col]
        print(
            f"      {col:<12}: mean={s['mean']:>8.4f}  std={s['std']:>8.4f}"
            f"  p05={s['p05']:>8.4f}  p95={s['p95']:>8.4f}"
        )

    print("\n      Fuzzy giriş değişkeni istatistikleri:")
    for col in FUZZY_COLS:
        s = stats[col]
        print(
            f"      {col:<12}: mean={s['mean']:>8.4f}  std={s['std']:>8.4f}"
            f"  p05={s['p05']:>8.4f}  p95={s['p95']:>8.4f}"
        )

    # Fuzzy sınır önerileri
    fuzzy_boundaries = suggest_fuzzy_boundaries(stats)
    print("\n      Fuzzy membership sınırı önerileri:")
    for col, b in fuzzy_boundaries.items():
        print(
            f"      {col:<12}: "
            f"Low=[{b['low_center']:.4f},{b['low_high']:.4f}]  "
            f"Med=[{b['med_low']:.4f},{b['med_high']:.4f}]  "
            f"High=[{b['high_low']:.4f},{b['high_center']:.4f}]"
        )

    # Kaydet
    stats_path = save_statistics(stats, fuzzy_boundaries, path="dataset/statistics.json")
    print(f"\n      İstatistikler kaydedildi: {stats_path}")

    # ---------------------------------------------------------------- #
    # 4. Veri kaydetme                                                   #
    # ---------------------------------------------------------------- #
    print("\n[4/4] Veri kaydediliyor...")

    try:
        pq_path = save_parquet(records, tag="full_dataset")
        print(f"      Parquet : {pq_path}")
    except ImportError:
        print("      Parquet atlandı (pandas/pyarrow kurulu değil)")

    npz_path = save_numpy(records, tag="full_dataset")
    print(f"      Numpy   : {npz_path}")

    # ---------------------------------------------------------------- #
    # Tamamlandı                                                         #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print("=" * 60)
    print(f"  Toplam kayıt      : {len(records):,}")
    print(f"  İstatistik dosyası: dataset/statistics.json")
    print(f"  Numpy dataset     : dataset/data/full_dataset.npz")
    print("\nSonraki adım:")
    print("  Fuzzy modülü dataset/statistics.json'ı okuyarak kalibre olacak.")
    print("  Quantum encoder aynı dosyadan normalizasyon parametrelerini alacak.")
    print("=" * 60)


if __name__ == "__main__":
    main()
