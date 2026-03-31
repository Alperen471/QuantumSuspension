"""
quantum/circuit.py — v2
Amplitude encoding: RY(arcsin(x)) + RZ(arccos(x^2)) — lineer, saturation yok
"""
import numpy as np
import pennylane as qml

BLOCK_ZS  = 0
BLOCK_ZU  = 1
BLOCK_ZSD = 2
BLOCK_ZUD = 3
BLOCK_ZR  = 4

PHYSICS_CNOT_PAIRS = [
    (BLOCK_ZS,  BLOCK_ZU),
    (BLOCK_ZSD, BLOCK_ZUD),
    (BLOCK_ZU,  BLOCK_ZR),
    (BLOCK_ZS,  BLOCK_ZSD),
]

PHYSICS_ZZ_PAIRS = [
    (BLOCK_ZS,  BLOCK_ZU),
    (BLOCK_ZSD, BLOCK_ZUD),
    (BLOCK_ZU,  BLOCK_ZR),
]

def _global_idx(block, local):
    return block * 5 + local

def _apply_encoding(state, n_qubits):
    # Hadamard ile başla — |0> yerine |+> durumu
    # Sıfır state için <Z> = 0 (denge noktası)
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for q in range(n_qubits):
        x = float(np.clip(state[q % 5], -1.0, 1.0))
        angle = float(np.pi * x)  # tam aralık
        qml.RZ(angle, wires=q)

def _apply_entanglement(n_qubits):
    n_blocks = n_qubits // 5
    for block in range(n_blocks):
        for ctrl_local, tgt_local in PHYSICS_CNOT_PAIRS:
            qml.CNOT(wires=[_global_idx(block, ctrl_local),
                            _global_idx(block, tgt_local)])

def _apply_variational(n_qubits, params):
    if params is None or len(params) == 0:
        return
    pq = len(params) // n_qubits
    for q in range(n_qubits):
        idx = q * pq
        if pq >= 1: qml.RZ(params[idx],     wires=q)
        if pq >= 2: qml.RX(params[idx + 1], wires=q)

def build_circuit(n_qubits, variational=True):
    n_blocks   = max(1, n_qubits // 5)
    n_params   = n_qubits * 2 if variational else 0
    zz_pairs   = [(_global_idx(b, zs), _global_idx(b, zu))
                  for b in range(n_blocks)
                  for zs, zu in PHYSICS_ZZ_PAIRS]
    output_dim = n_qubits + len(zz_pairs)
    dev        = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(state, params):
        _apply_encoding(state, n_qubits)
        _apply_entanglement(n_qubits)
        _apply_variational(n_qubits, params)
        meas = []
        for q in range(n_qubits):
            meas.append(qml.expval(qml.PauliX(q)))
        for q1, q2 in zz_pairs:
            meas.append(qml.expval(qml.PauliX(q1) @ qml.PauliX(q2)))
        return meas

    return circuit, n_params, output_dim

def print_circuit_summary(n_qubits, variational=True):
    _, n_params, output_dim = build_circuit(n_qubits, variational)
    print(f"  Devre: {n_qubits}q → {output_dim}D, {n_params} param")
    print(f"  Encoding: RY(arcsin) + RZ(arccos²) — lineer, saturation yok")
