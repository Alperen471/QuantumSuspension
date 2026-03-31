"""
experiments/run_training.py
----------------------------
Tek model training — hızlı test ve debug için.

Tam ablation için run_ablation.py kullan.

Çalıştırma örnekleri:
    # Baseline RL
    python -m experiments.run_training

    # Fuzzy RL
    python -m experiments.run_training --fuzzy bounded

    # Quantum RL (5 qubit)
    python -m experiments.run_training --encoder quantum --qubits 5

    # Tam hybrid
    python -m experiments.run_training --encoder quantum --qubits 10 --fuzzy bounded
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.train import train, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SAC Training")
    parser.add_argument("--encoder",  type=str,  default="none",
                        choices=["none", "qangle", "qamp", "quantum"],
                        help="Encoder tipi")
    parser.add_argument("--qubits",   type=int,  default=5,
                        help="Qubit sayısı (quantum encoder için)")
    parser.add_argument("--variational", action="store_true", default=True,
                        help="Variational quantum devre")
    parser.add_argument("--fuzzy",    type=str,  default="none",
                        choices=["none", "bounded", "raw"],
                        help="Fuzzy reward modu")
    parser.add_argument("--steps",    type=int,  default=200_000,
                        help="Training adım sayısı")
    parser.add_argument("--road",     type=str,  default="random",
                        choices=["random", "sinusoidal", "bump", "pothole"],
                        help="Road tipi")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--device",   type=str,  default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Model adını otomatik oluştur
    name_parts = ["rl"]
    if args.encoder != "none":
        name_parts.append(f"{args.encoder}")
        if args.encoder == "quantum":
            name_parts.append(f"{args.qubits}q")
            if args.variational:
                name_parts.append("var")
    if args.fuzzy != "none":
        name_parts.append(f"fuzzy_{args.fuzzy}")
    model_name = "_".join(name_parts)

    cfg = TrainingConfig(
        model_name    = model_name,
        encoder_type  = args.encoder,
        n_qubits      = args.qubits,
        variational   = args.variational,
        fuzzy_mode    = args.fuzzy,
        road_type     = args.road,
        total_steps   = args.steps,
        device        = args.device,
        seed          = args.seed,
    )

    result = train(cfg, verbose=True)

    print(f"\nFinal Sonuçlar:")
    print(f"  Model          : {result.model_name}")
    print(f"  Toplam süre    : {result.total_time_s/60:.1f} dakika")
    print(f"  rms_body_acc   : {result.final_metrics.get('rms_body_acc_mean', 0):.4f}")
    print(f"  iso_weighted   : {result.final_metrics.get('iso_weighted_rms_mean', 0):.4f}")
    print(f"  rms_ctrl_force : {result.final_metrics.get('rms_ctrl_force_mean', 0):.4f}")


if __name__ == "__main__":
    main()
