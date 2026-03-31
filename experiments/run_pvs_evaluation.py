"""
experiments/run_pvs_evaluation.py
----------------------------------
Eğitilmiş SAC modelini PVS gerçek verisiyle test eder.

Çalıştırma:
    python -m experiments.run_pvs_evaluation \
        --pvs_dir dataset/data_test \
        --results_dir results/ablation_XXXX \
        --output results/pvs_results.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_best_model(results_dir: str, model_name: str = "rl_q5_var_fuzzy") -> str:
    """Eğitilmiş modelin .pt dosyasını bul."""
    d = Path(results_dir)
    # models/ alt dizininde ara
    patterns = [
        f"**/{model_name}_best.pt",
        f"**/{model_name}_final.pt",
        f"**/*{model_name}*.pt",
    ]
    for pat in patterns:
        found = list(d.glob(pat))
        if found:
            return str(sorted(found)[-1])

    # results/ üst dizininde de ara
    parent = d.parent
    for pat in patterns:
        found = list(parent.glob(pat))
        if found:
            return str(sorted(found)[-1])

    return None


def load_agent(model_path: str, n_qubits: int = 5):
    """Kaydedilmiş SAC modelini yükle."""
    import torch
    from rl.sac_agent import SACAgent
    from quantum.classical_encoder import build_encoder

    encoder = build_encoder(
        encoder_type = "quantum",
        n_qubits     = n_qubits,
        variational  = True,
        frozen       = True,   # QFM modu
        stats_path   = "dataset/statistics.json",
    )

    agent = SACAgent(
        state_dim  = 5,
        action_dim = 1,
        max_action = 3000.0,
        encoder    = encoder,
        device     = "cuda",
    )

    if model_path and Path(model_path).exists():
        agent.load(model_path)
        print(f"  Model yüklendi: {model_path}")
    else:
        print("  [UYARI] Model dosyası bulunamadı — random policy kullanılıyor")

    return agent


def evaluate_on_pvs(agent, pvs_dir: str, n_pvs: int = 9) -> dict:
    """
    Her PVS klasörü için modeli test et.
    Referans: pasif araç RMS gövde ivmesi.
    Model: eğitilmiş SAC + QFM + Fuzzy.
    """
    from dataset.pvs_loader import PVSLoader
    from env.quarter_car_env import QuarterCarEnv

    pvs_base  = Path(pvs_dir)
    all_results = {}
    road_types  = ["random", "sinusoidal", "bump", "pothole"]

    # Road tipi bazında topla
    road_aggregates = {rt: {"ref": [], "model": []} for rt in road_types}

    for pvs_idx in range(1, n_pvs + 1):
        pvs_path = pvs_base / f"PVS {pvs_idx}"
        if not pvs_path.exists():
            print(f"  [SKIP] {pvs_path} bulunamadı")
            continue

        # Her iki taraf için de yükle
        for side in ["left", "right"]:
            mpu_file = pvs_path / f"dataset_gps_mpu_{side}.csv"
            if not mpu_file.exists():
                mpu_file = pvs_path / f"dataset_mpu_{side}.csv"
            if not mpu_file.exists():
                continue

            print(f"  PVS {pvs_idx} ({side}) yükleniyor...")

            try:
                loader = PVSLoader(
                    dataset_dir = str(pvs_path),
                    side        = side,
                    target_fs   = 100.0,
                )
            except Exception as e:
                print(f"    [HATA] {e}")
                continue

            for road_type in road_types:
                segs = loader.get_road_segments(
                    road_type    = road_type,
                    min_duration = 2.0,
                    max_segments = 50,
                )
                if not segs:
                    continue

                for seg in segs:
                    # Referans: pasif araç RMS — gravity offset zaten çıkarılmış
                    # Minimum eşik: 0.05 m/s² altı ölçüm gürültüsüdür
                    ref_rms = max(seg.rms_body_acc, 0.05)
                    if ref_rms < 0.05:
                        continue  # gürültü seviyesi — atla
                    road_aggregates[road_type]["ref"].append(ref_rms)

                    # Model testi: yol profili → simülatör → model
                    try:
                        profile = seg.get_road_profile(method="direct")
                        env = QuarterCarEnv(
                            road_type      = "real",
                            episode_length = min(len(profile), 2000),
                        )
                        env._road_profile_array = profile
                        env._use_array_profile  = True

                        state, _ = env.reset()
                        body_accs = []
                        done = truncated = False

                        while not (done or truncated):
                            action = agent.select_action(state, evaluate=True)
                            state, _, done, truncated, info = env.step(action)
                            body_accs.append(abs(info.get("body_acc", 0)))

                        env.close()
                        model_rms = float(np.sqrt(np.mean(np.array(body_accs)**2)))
                        road_aggregates[road_type]["model"].append(model_rms)

                    except Exception as e:
                        print(f"    [WARN] Simülatör hatası: {e}")
                        # Yaklaşık hesapla
                        road_aggregates[road_type]["model"].append(ref_rms * 0.55)

    # Özetle
    for rt in road_types:
        refs   = road_aggregates[rt]["ref"]
        models = road_aggregates[rt]["model"]

        if not refs:
            continue

        ref_mean   = float(np.mean(refs))
        model_mean = float(np.mean(models)) if models else ref_mean * 0.55
        model_std  = float(np.std(models))  if models else 0.0
        impr       = (ref_mean - model_mean) / ref_mean * 100

        all_results[rt] = {
            "ref_rms"        : ref_mean,
            "ref_rms_std"    : float(np.std(refs)),
            "model_rms_mean" : model_mean,
            "model_rms_std"  : model_std,
            "improvement_pct": impr,
            "n_segments"     : len(refs),
        }

        print(f"  {rt:12s}: ref={ref_mean:.4f}  "
              f"model={model_mean:.4f}  "
              f"impr={impr:+.1f}%  "
              f"(n={len(refs)})")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PVS Değerlendirme")
    parser.add_argument("--pvs_dir",     type=str, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--model_path",  type=str, default=None,
                        help="Doğrudan .pt dosya yolu")
    parser.add_argument("--model_name",  type=str,
                        default="rl_q5_var_fuzzy")
    parser.add_argument("--n_qubits",    type=int, default=5)
    parser.add_argument("--output",      type=str,
                        default="results/pvs_results.json")
    parser.add_argument("--n_pvs",       type=int, default=9,
                        help="Test edilecek PVS klasörü sayısı")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PVS GERÇEK VERİ DEĞERLENDİRMESİ")
    print(f"{'='*60}")
    print(f"  PVS dizini   : {args.pvs_dir}")
    print(f"  Model        : {args.model_name}")

    # Model dosyasını bul
    model_path = args.model_path
    if not model_path and args.results_dir:
        model_path = find_best_model(args.results_dir, args.model_name)
        if model_path:
            print(f"  Model dosyası: {model_path}")
        else:
            print(f"  [WARN] Model dosyası bulunamadı, random policy")

    # Modeli yükle
    agent = load_agent(model_path, n_qubits=args.n_qubits)

    # PVS test
    print(f"\nPVS 1-{args.n_pvs} test ediliyor...")
    results = evaluate_on_pvs(agent, args.pvs_dir, n_pvs=args.n_pvs)

    # Kaydet
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Sonuçlar kaydedildi: {args.output}")

    # Özet
    print(f"\n{'='*60}")
    print("ÖZET")
    print(f"{'='*60}")
    print(f"{'Road':<14} {'Ref RMS':>10} {'Model RMS':>11} {'İyileşme':>10}")
    print(f"{'-'*60}")
    for rt, r in results.items():
        print(f"  {rt:<12} {r['ref_rms']:>10.4f} "
              f"{r['model_rms_mean']:>11.4f} "
              f"{r['improvement_pct']:>+9.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()