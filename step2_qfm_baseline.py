"""Adım 2: QFM encode — 5/10/20 qubit"""
import sys, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, ".")
from quantum.qnode_encoder import QNodeEncoder

QUBITS = [5, 10, 20]

print("\n=== ADIM 2: QFM ENCODE ===")

raw = dict(np.load("dataset/data/full_dataset.npz"))
states      = raw["states"].astype(np.float32)
next_states = raw["next_states"].astype(np.float32)
n = len(states)

for nq in QUBITS:
    out = Path(f"dataset/data/qfm_{nq}q_encoded.npz")
    if out.exists():
        existing = len(dict(np.load(out)).get("states", []))
        if existing >= n:
            print(f"✓ QFM-{nq}q zaten mevcut ({existing:,} state) — atlanıyor")
            continue

    print(f"\nQFM-{nq}q encode başlıyor ({n:,} state)...")
    enc = QNodeEncoder(n_qubits=nq, variational=True, frozen=True,
                       device_name="lightning.gpu",
                       stats_path="dataset/statistics.json")
    enc.eval()
    chunk = 64
    es  = np.zeros((n, enc.output_dim), dtype=np.float32)
    ens = np.zeros((n, enc.output_dim), dtype=np.float32)

    for start in tqdm(range(0, n, chunk), desc=f"QFM-{nq}q"):
        end = min(start+chunk, n)
        with torch.no_grad():
            es[start:end]  = enc.encode_torch(torch.FloatTensor(states[start:end])).cpu().numpy()
            ens[start:end] = enc.encode_torch(torch.FloatTensor(next_states[start:end])).cpu().numpy()

    np.savez_compressed(out, states=es, next_states=ens,
                        actions=raw["actions"], rewards=raw["rewards"], dones=raw["dones"])
    print(f"✓ QFM-{nq}q kaydedildi: {out}")

print("\n✓ ADIM 2 TAMAMLANDI — step3 çalıştırabilirsiniz\n")
