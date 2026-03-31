"""
experiments/run_ablation.py
-----------------------------
Tam ablation çalışması — tüm model varyantları.

Ablation Matrisi:
  Baseline     : Passive, Skyhook (benchmark pipeline'dan)
  RL varyantları:
    RL-Baseline          : encoder=none,  fuzzy=none
    RL-Fuzzy-Bounded     : encoder=none,  fuzzy=bounded
    RL-Fuzzy-Raw         : encoder=none,  fuzzy=raw
    RL-QAngle            : encoder=qangle, fuzzy=none
    RL-QAmp              : encoder=qamp,  fuzzy=none
    RL-Q5-Fixed          : encoder=quantum(5,  var=False), fuzzy=none
    RL-Q5-Var            : encoder=quantum(5,  var=True),  fuzzy=none
    RL-Q10-Var           : encoder=quantum(10, var=True),  fuzzy=none
    RL-Q15-Var           : encoder=quantum(15, var=True),  fuzzy=none
    RL-Q20-Var           : encoder=quantum(20, var=True),  fuzzy=none
    RL-Q5-Var-Fuzzy      : encoder=quantum(5,  var=True),  fuzzy=bounded
    RL-Q10-Var-Fuzzy     : encoder=quantum(10, var=True),  fuzzy=bounded
    RL-Q20-Var-Fuzzy     : encoder=quantum(20, var=True),  fuzzy=bounded

Her varyant: 5 seed × 4 road = 20 run

Çalıştırma:
    python -m experiments.run_ablation --quick   # hızlı test (50K step)
    python -m experiments.run_ablation           # tam (200K step)
    python -m experiments.run_ablation --models rl_baseline rl_fuzzy_bounded
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.train import train, TrainingConfig


# ------------------------------------------------------------------ #
#  Ablation model konfigürasyonları                                   #
# ------------------------------------------------------------------ #

def get_algorithm_comparison_configs(total_steps: int = 100_000) -> list[TrainingConfig]:
    """
    SAC vs TD3 karşılaştırması — ön çalışma.
    Hangisi süspansiyon probleminde daha iyi? Bu sonuca göre
    ana ablation için algoritma seçimi yapılır.
    """
    base = dict(
        total_steps    = total_steps,
        encoder_type   = "none",
        fuzzy_mode     = "none",
        eval_freq      = max(5_000, total_steps // 20),
        save_freq      = total_steps,
        device         = "cuda",
        stats_path     = "dataset/statistics.json",
    )

    return [
        TrainingConfig(model_name="sac_baseline", algorithm="sac", **base),
        TrainingConfig(model_name="td3_baseline", algorithm="td3", **base),
        TrainingConfig(model_name="td3_fuzzy",    algorithm="td3",
                       **{**base, "fuzzy_mode": "bounded"}),
    ]


def get_ablation_configs(total_steps: int = 200_000, algorithm: str = "sac") -> list[TrainingConfig]:
    """Tüm ablation model konfigürasyonlarını döndür."""

    base = dict(
        algorithm      = algorithm,
        eval_freq      = max(5_000, total_steps // 40),
        save_freq      = max(50_000, total_steps // 4),
        device         = "cuda",
        stats_path     = "dataset/statistics.json",
    )

    # QFM pre-encoded buffer ile daha az step'te converge olur
    # Qubit ↑ → feature_dim ↑ → ağ daha yavaş öğrenir → daha az step yeterli
    q_steps_5  = min(total_steps, 150_000)
    q_steps_10 = min(total_steps, 150_000)
    q_steps_15 = min(total_steps, 150_000)
    q_steps_20 = min(total_steps, 100_000)
    q_steps_25 = min(total_steps,  75_000)

    configs = [

        # ── Baseline RL ──────────────────────────────────────────── #
        TrainingConfig(model_name="rl_baseline",
                       encoder_type="none", fuzzy_mode="none",
                       total_steps=total_steps, **base),

        # ── Fuzzy RL ─────────────────────────────────────────────── #
        TrainingConfig(model_name="rl_fuzzy_bounded",
                       encoder_type="none", fuzzy_mode="bounded",
                       total_steps=total_steps, **base),
        TrainingConfig(model_name="rl_fuzzy_raw",
                       encoder_type="none", fuzzy_mode="raw",
                       total_steps=total_steps, **base),

        # ── Klasik Encoder'lar ───────────────────────────────────── #
        TrainingConfig(model_name="rl_qangle",
                       encoder_type="qangle", fuzzy_mode="none",
                       total_steps=total_steps, **base),
        TrainingConfig(model_name="rl_qamp",
                       encoder_type="qamp", fuzzy_mode="none",
                       total_steps=total_steps, **base),

        # ── Quantum Fixed (ablation: variational etkisi) ─────────── #
        TrainingConfig(model_name="rl_q5_fixed",
                       encoder_type="quantum", n_qubits=5,
                       variational=False, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_5,
                       eval_freq=max(2500, q_steps_5//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),

        # ── Quantum Variational QFM — qubit ablation ─────────────── #
        TrainingConfig(model_name="rl_q5_var",
                       encoder_type="quantum", n_qubits=5,
                       variational=True, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_5,
                       eval_freq=max(2500, q_steps_5//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q10_var",
                       encoder_type="quantum", n_qubits=10,
                       variational=True, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_10,
                       eval_freq=max(2500, q_steps_10//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q15_var",
                       encoder_type="quantum", n_qubits=15,
                       variational=True, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_10,
                       eval_freq=max(2500, q_steps_10//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q20_var",
                       encoder_type="quantum", n_qubits=20,
                       variational=True, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_20,
                       eval_freq=max(2500, q_steps_20//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q25_var",
                       encoder_type="quantum", n_qubits=25,
                       variational=True, frozen=True, fuzzy_mode="none",
                       total_steps=q_steps_25,
                       eval_freq=max(2500, q_steps_25//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),

        # ── Quantum + Fuzzy Hybrid ───────────────────────────────── #
        TrainingConfig(model_name="rl_q5_var_fuzzy",
                       encoder_type="quantum", n_qubits=5,
                       variational=True, frozen=True, fuzzy_mode="bounded",
                       total_steps=q_steps_5,
                       eval_freq=max(2500, q_steps_5//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q10_var_fuzzy",
                       encoder_type="quantum", n_qubits=10,
                       variational=True, frozen=True, fuzzy_mode="bounded",
                       total_steps=q_steps_10,
                       eval_freq=max(2500, q_steps_10//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
        TrainingConfig(model_name="rl_q20_var_fuzzy",
                       encoder_type="quantum", n_qubits=20,
                       variational=True, frozen=True, fuzzy_mode="bounded",
                       total_steps=q_steps_20,
                       eval_freq=max(2500, q_steps_20//40),
                       **{k:v for k,v in base.items() if k not in ['eval_freq']}),
    ]

    return configs


# ------------------------------------------------------------------ #
#  Ablation runner                                                     #
# ------------------------------------------------------------------ #

def run_ablation(
    configs    : list[TrainingConfig],
    seeds      : list[int] = None,
    road_types : list[str] = None,
    verbose    : bool = True,
) -> dict:
    """
    Tüm konfigürasyonları çalıştır, sonuçları topla.

    Her (config, seed, road) kombinasyonu için ayrı training.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if road_types is None:
        road_types = ["random", "sinusoidal", "bump", "pothole"]

    total_runs  = len(configs) * len(seeds) * len(road_types)
    run_count   = 0
    all_results = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/ablation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"ABLATION ÇALIŞMASI")
    print(f"{'='*65}")
    print(f"  Konfigürasyon sayısı : {len(configs)}")
    print(f"  Seed sayısı          : {len(seeds)}")
    print(f"  Road tipi sayısı     : {len(road_types)}")
    print(f"  Toplam run           : {total_runs}")
    print(f"  Sonuç dizini         : {results_dir}")
    print(f"{'='*65}\n")

    # Ablation genel progress bar
    try:
        from tqdm import tqdm as tqdm_outer
        ablation_bar = tqdm_outer(
            total      = total_runs,
            desc       = "Ablation",
            unit       = "run",
            position   = 0,
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}] {postfix}",
        )
    except ImportError:
        ablation_bar = None

    for cfg in configs:
        model_results = []

        for seed in seeds:
            for road_type in road_types:
                run_count += 1

                # Bu run için konfigürasyon
                run_cfg = TrainingConfig(**vars(cfg))
                run_cfg.road_type  = road_type
                run_cfg.seed       = seed
                run_cfg.model_name = f"{cfg.model_name}_road{road_type}_seed{seed}"
                run_cfg.save_dir   = str(results_dir / "models")
                run_cfg.log_dir    = str(results_dir / "logs")

                print(f"\n[{run_count}/{total_runs}] {run_cfg.model_name}")

                try:
                    result = train(run_cfg, verbose=verbose)
                    model_results.append({
                        "model_name" : cfg.model_name,
                        "road_type"  : road_type,
                        "seed"       : seed,
                        "final_metrics": result.final_metrics,
                        "total_time_s" : result.total_time_s,
                        "episode_count": len(result.episode_rewards),
                    })

                    # Her run sonucunu kaydet
                    run_json = results_dir / f"{run_cfg.model_name}.json"
                    with open(run_json, "w") as f:
                        json.dump(model_results[-1], f, indent=2)

                except Exception as e:
                    print(f"  HATA: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    if ablation_bar is not None:
                        rms = model_results[-1]["final_metrics"].get(
                            "rms_body_acc_mean", 0) if model_results else 0
                        ablation_bar.set_postfix(
                            model=cfg.model_name[:15],
                            road=road_type,
                            rms=f"{rms:.3f}",
                        )
                        ablation_bar.update(1)

        all_results[cfg.model_name] = model_results

    if ablation_bar is not None:
        ablation_bar.close()

    # Özet raporu kaydet
    summary_path = results_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"ABLATION TAMAMLANDI")
    print(f"  Özet: {summary_path}")
    print(f"{'='*65}")

    return all_results


# ------------------------------------------------------------------ #
#  Özet tablo                                                          #
# ------------------------------------------------------------------ #

def print_ablation_summary(all_results: dict) -> None:
    """Ablation sonuçlarını terminal'de tablo halinde göster."""
    print(f"\n{'='*90}")
    print(f"ABLATION ÖZET TABLOSU")
    print(f"{'='*90}")
    print(
        f"{'Model':<25} {'rms_body':>10} {'iso_rms':>10} "
        f"{'susp_defl':>10} {'ctrl_force':>12}"
    )
    print(f"{'-'*90}")

    for model_name, runs in all_results.items():
        if not runs:
            continue

        rms_body   = np.mean([r["final_metrics"].get("rms_body_acc_mean", 0)   for r in runs])
        iso_rms    = np.mean([r["final_metrics"].get("iso_weighted_rms_mean", 0) for r in runs])
        susp_defl  = np.mean([r["final_metrics"].get("rms_susp_defl_mean", 0)  for r in runs])
        ctrl_force = np.mean([r["final_metrics"].get("rms_ctrl_force_mean", 0) for r in runs])

        print(
            f"{model_name:<25} {rms_body:>10.4f} {iso_rms:>10.4f} "
            f"{susp_defl:>10.6f} {ctrl_force:>12.4f}"
        )

    print(f"{'='*90}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Ablation Çalışması")
    parser.add_argument("--quick", action="store_true",
                        help="Hızlı test: 50K step, 2 seed, 2 road")
    parser.add_argument("--algo-compare", action="store_true",
                        help="SAC vs TD3 ön karşılaştırması (100K step)")
    parser.add_argument("--algorithm", type=str, default="sac",
                        choices=["sac", "td3"],
                        help="Ana ablation için kullanılacak algoritma")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds",  nargs="+", type=int, default=None)
    parser.add_argument("--roads",  nargs="+", default=None)
    args = parser.parse_args()

    # SAC vs TD3 ön karşılaştırması
    if args.algo_compare:
        total_steps = args.steps or 100_000
        seeds       = args.seeds or [42, 43, 44]
        road_types  = args.roads or ["random", "sinusoidal", "bump", "pothole"]

        configs = get_algorithm_comparison_configs(total_steps)
        print("\n=== SAC vs TD3 Algoritma Karşılaştırması ===")
        print("Sonuca göre ana ablation için algoritma seçilecek.\n")
        results = run_ablation(configs, seeds=seeds, road_types=road_types)
        print_ablation_summary(results)

        # Tavsiye
        sac_rms = np.mean([
            r["final_metrics"].get("rms_body_acc_mean", 99)
            for r in results.get("sac_baseline", [])
        ])
        td3_rms = np.mean([
            r["final_metrics"].get("rms_body_acc_mean", 99)
            for r in results.get("td3_baseline", [])
        ])
        winner = "sac" if sac_rms <= td3_rms else "td3"
        print(f"\n  SAC rms_body_acc: {sac_rms:.4f}")
        print(f"  TD3 rms_body_acc: {td3_rms:.4f}")
        print(f"  Tavsiye: Ana ablation için '{winner}' kullan")
        print(f"  Komut : python -m experiments.run_ablation --algorithm {winner}")
        return

    # Quick mode
    if args.quick:
        total_steps = args.steps or 50_000
        seeds       = args.seeds or [42, 43]
        road_types  = args.roads or ["random", "sinusoidal"]
    else:
        total_steps = args.steps or 200_000
        seeds       = args.seeds or [42, 43, 44, 45, 46]
        road_types  = args.roads or ["random", "sinusoidal", "bump", "pothole"]

    configs = get_ablation_configs(total_steps, algorithm=args.algorithm)

    # Tüm mevcut model isimleri
    all_names = [c.model_name for c in configs]

    if args.models:
        # Trim + lower karşılaştırma — terminal tırnak/boşluk sorunları için
        requested = [m.strip() for m in args.models]
        configs   = [c for c in configs if c.model_name in requested]
        not_found = [m for m in requested if m not in all_names]
        if not_found:
            print(f"\n[UYARI] Bulunamayan modeller: {not_found}")
            print(f"  Mevcut model isimleri: {all_names}")
        if not configs:
            print(f"\n[HATA] Hiç model seçilmedi!")
            print(f"  --models ile şunları kullanabilirsin:")
            for name in all_names:
                print(f"    {name}")
            return
        print(f"Seçili modeller: {[c.model_name for c in configs]}")

    results = run_ablation(configs, seeds=seeds, road_types=road_types)
    print_ablation_summary(results)


if __name__ == "__main__":
    main()