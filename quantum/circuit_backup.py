"""
quantum/circuit.py
------------------
Fizik-güdümlü, katmanlı variational quantum circuit.

Mimari Kararları ve Gerekçeleri
--------------------------------

1. BLOK TABANLI ORGANİZASYON
   Her 5 qubit bir "fiziksel blok" oluşturur.
   n_qubits = 5k → k blok.
   Bu sayede qubit sayısı arttıkça hem daha derin
   hem daha geniş bir devre elde edilir.
   Ablation: qubit sayısı ile performans ilişkisi sistematik ölçülür.

2. FİZİK-GÜDÜMLÜ ENTANGLEMENT
   CNOT bağlantıları quarter-car dinamik denkleminden türetildi:
     q_zs  ↔ q_zu     : süspansiyon defleksiyonu (z_s - z_u)
     q_zsd ↔ q_zud    : göreli hız (z_s_dot - z_u_dot)
     q_zu  ↔ q_zr     : tire defleksiyonu (z_u - z_r)
     q_zs  ↔ q_zsd    : pozisyon-hız ilişkisi
   Rastgele entanglement değil — fizik denkleminin bağımlılıklarını yansıtır.

3. CROSS-ENTANGLEMENT
   Bloklar arası CNOT'lar aynı fiziksel değişkenin farklı
   temsil katmanlarını birbirine bağlar. Bu sayede daha derin
   devrelerde bilgi bloklar arasında akabilir.

4. VARİATIONAL KATMAN
   RZ(θ) + RX(φ) her qubit'te öğrenilebilir parametre.
   Parameter-shift rule ile gradient hesaplanır.
   RL optimizer (SAC/Adam) bu parametreleri günceller.

5. ÖLÇÜM STRATEJİSİ
   - Tek qubit: ⟨Zᵢ⟩ — qubit'in ortalama spin beklentisi
   - Çift qubit: ⟨ZᵢZⱼ⟩ — fiziksel çiftler için korelasyon ölçümü
     ⟨Z_zs · Z_zu⟩   → defleksiyon korelasyonu
     ⟨Z_zsd · Z_zud⟩ → göreli hız korelasyonu
     ⟨Z_zu · Z_zr⟩   → tire korelasyonu
   Toplam çıktı: n_qubits + 3 boyut (her blok için 3 ZZ çifti,
   ama bloklar arası çakışmayı önlemek için sadece ilk bloğun ZZ'leri)

6. JOSEPHSON KÖPRÜSÜ UYUMLULUĞU
   RY, RZ, RX → superconducting transmon qubit'lerde
                microwave pulse olarak uygulanır.
   CNOT       → iki transmon arasında kapasitif bağlaşım.
   Bu devre IBM Quantum / Google Sycamore donanımına transpile edilebilir.
"""

import numpy as np
import pennylane as qml
from typing import Optional

# ------------------------------------------------------------------ #
#  Sabitler                                                            #
# ------------------------------------------------------------------ #

# State değişkenlerinin blok içi qubit indeksleri
# Her blok 5 qubit: [z_s, z_u, z_s_dot, z_u_dot, z_r]
BLOCK_ZS  = 0
BLOCK_ZU  = 1
BLOCK_ZSD = 2
BLOCK_ZUD = 3
BLOCK_ZR  = 4

# Fizik-güdümlü CNOT çiftleri (blok içi relatif indeksler)
PHYSICS_CNOT_PAIRS = [
    (BLOCK_ZS,  BLOCK_ZU),   # süspansiyon defleksiyonu
    (BLOCK_ZSD, BLOCK_ZUD),  # göreli hız
    (BLOCK_ZU,  BLOCK_ZR),   # tire defleksiyonu
    (BLOCK_ZS,  BLOCK_ZSD),  # pozisyon-hız ilişkisi
]

# ZZ ölçüm çiftleri (blok içi relatif indeksler)
PHYSICS_ZZ_PAIRS = [
    (BLOCK_ZS,  BLOCK_ZU),   # defleksiyon korelasyonu
    (BLOCK_ZSD, BLOCK_ZUD),  # göreli hız korelasyonu
    (BLOCK_ZU,  BLOCK_ZR),   # tire korelasyonu
]


# ------------------------------------------------------------------ #
#  Yardımcı: blok indekslerini global qubit indekslerine çevir        #
# ------------------------------------------------------------------ #

def _global_idx(block: int, local: int) -> int:
    """
    Blok ve lokal indeksten global qubit indeksi.
    block=0, local=2 → qubit 2
    block=1, local=2 → qubit 7
    """
    return block * 5 + local


# ------------------------------------------------------------------ #
#  Devre bileşenleri                                                   #
# ------------------------------------------------------------------ #

def _apply_ry_encoding(state: np.ndarray, n_qubits: int) -> None:
    """
    RY encoding katmanı.

    state: normalize edilmiş [-1,+1] aralığında 5 boyutlu vektör.
    Her qubit'e state[i % 5] * π açısıyla RY rotasyonu uygulanır.

    Neden π ile çarp?
    RY(θ) Bloch küresinde Y ekseninde θ açısı döndürür.
    [-1,+1] → [-π,+π] mapping: tam Bloch küresi taranır.
    """
    for q in range(n_qubits):
        angle = float(state[q % 5]) * (np.pi / 4) + (np.pi / 2)
        qml.RY(angle, wires=q)


def _apply_physics_entanglement(n_qubits: int) -> None:
    """
    Fizik-güdümlü CNOT katmanı.

    Her blok kendi içinde quarter-car dinamiğini yansıtan
    4 CNOT bağlantısı alır.
    """
    n_blocks = n_qubits // 5

    for block in range(n_blocks):
        for ctrl_local, tgt_local in PHYSICS_CNOT_PAIRS:
            ctrl = _global_idx(block, ctrl_local)
            tgt  = _global_idx(block, tgt_local)
            qml.CNOT(wires=[ctrl, tgt])


def _apply_cross_entanglement(n_qubits: int) -> None:
    """
    Bloklar arası cross-entanglement.

    Blok i'nin z_s qubit'i → Blok (i+1)'in z_s qubit'i şeklinde
    aynı fiziksel değişkenin farklı temsil katmanları bağlanır.

    Neden cross-entanglement?
    Tek bir blok sadece mevcut state'i encode eder.
    Cross-entanglement ile bilgi bloklar arasında akabilir —
    daha derin devrelerde daha zengin temsil sağlanır.
    """
    n_blocks = n_qubits // 5

    if n_blocks < 2:
        return

    for block in range(n_blocks - 1):
        for local in range(5):
            ctrl = _global_idx(block,     local)
            tgt  = _global_idx(block + 1, local)
            qml.CNOT(wires=[ctrl, tgt])


def _apply_variational(params: np.ndarray, n_qubits: int) -> None:
    """
    Variational RZ + RX katmanı.

    params: (n_qubits * 2,) array — [θ₀,φ₀, θ₁,φ₁, ..., θₙ,φₙ]
    Her qubit için RZ(θ) ardından RX(φ) uygulanır.

    Neden RZ + RX?
    İki ardışık rotasyon, Bloch küresinde herhangi bir tek-qubit
    dönüşümünü yaklaşık olarak gerçekleştirebilir (universal single-qubit gate).
    Bu expressibility'yi maksimize eder.

    Gradient hesabı için parameter-shift rule kullanılır:
    ∂f/∂θ = [f(θ+π/2) - f(θ-π/2)] / 2
    """
    for q in range(n_qubits):
        theta = params[2 * q]
        phi   = params[2 * q + 1]
        qml.RZ(theta, wires=q)
        qml.RX(phi,   wires=q)


# ------------------------------------------------------------------ #
#  Ölçüm                                                               #
# ------------------------------------------------------------------ #

def _build_observables(n_qubits: int) -> list:
    """
    Ölçülecek observable listesini oluşturur.

    Returns:
        list of qml.expval observables

    Çıktı boyutu:
        n_qubits (tek qubit Z) + 3 (ZZ çiftleri, sadece ilk blok)
        = n_qubits + 3

    Neden sadece ilk bloğun ZZ çiftleri?
    Tüm bloklar için ZZ eklemek output boyutunu şişirir.
    İlk bloğun ZZ'leri zaten fiziksel anlamı taşıyor.
    Ablation'da ZZ sayısını artırmak ek bir boyut olarak eklenebilir.
    """
    observables = []

    # Tek qubit Z beklenti değerleri
    for q in range(n_qubits):
        observables.append(qml.expval(qml.PauliZ(q)))

    # ZZ çiftleri — sadece ilk bloğun fiziksel çiftleri
    for local_i, local_j in PHYSICS_ZZ_PAIRS:
        gi = _global_idx(0, local_i)
        gj = _global_idx(0, local_j)
        observables.append(qml.expval(qml.PauliZ(gi) @ qml.PauliZ(gj)))

    return observables


# ------------------------------------------------------------------ #
#  Ana devre fabrikası                                                 #
# ------------------------------------------------------------------ #

def build_circuit(
    n_qubits: int = 5,
    variational: bool = True,
    device_name: str = "lightning.gpu",
) -> tuple:
    """
    PennyLane QNode devresini oluşturur ve döndürür.

    Args:
        n_qubits    : toplam qubit sayısı, 5'in katı olmalı
        variational : True → öğrenilebilir RZ+RX katmanı eklenir
        device_name : "lightning.gpu" (4090) veya "default.qubit" (debug)

    Returns:
        (circuit_fn, n_params, output_dim)
        circuit_fn  : (state, params) → list[float]
        n_params    : variational parametre sayısı (0 eğer variational=False)
        output_dim  : çıktı vektörü boyutu

    Raises:
        ValueError : n_qubits 5'in katı değilse
    """
    if n_qubits % 5 != 0:
        raise ValueError(
            f"n_qubits 5'in katı olmalı. Verilen: {n_qubits}\n"
            f"Geçerli değerler: 5, 10, 15, 20, 25"
        )

    n_params   = n_qubits * 2 if variational else 0
    output_dim = n_qubits + 3  # Z ölçümleri + 3 ZZ çifti

    # Device oluştur
    try:
        dev = qml.device(device_name, wires=n_qubits)
    except Exception:
        # lightning.gpu yoksa default.qubit'e fall back
        print(f"[WARNING] {device_name} kullanılamıyor, default.qubit kullanılıyor.")
        dev = qml.device("default.qubit", wires=n_qubits)

    # QNode tanımı
    # adjoint: lightning.gpu'nun native differentiator'ı
    # parameter-shift'ten 10-100x hızlı, O(params) yerine O(1) pass
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(state: np.ndarray, params: Optional[np.ndarray] = None):
        """
        Args:
            state  : (5,) normalize edilmiş state vektörü
            params : (n_params,) variational parametreler — None ise atlanır
        """
        # 1. RY Encoding
        _apply_ry_encoding(state, n_qubits)

        # 2. Fizik-güdümlü entanglement
        _apply_physics_entanglement(n_qubits)

        # 3. Cross-entanglement (n_qubits > 5 ise)
        _apply_cross_entanglement(n_qubits)

        # 4. Variational katman (opsiyonel)
        if variational and params is not None:
            _apply_variational(params, n_qubits)

        # 5. Ölçüm — tek tensor olarak döndür (batch uyumluluğu için)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)] + \
               [qml.expval(qml.PauliZ(_global_idx(0, i)) @ qml.PauliZ(_global_idx(0, j)))
                for i, j in PHYSICS_ZZ_PAIRS]

    return circuit, n_params, output_dim


# ------------------------------------------------------------------ #
#  Devre görselleştirme yardımcısı                                    #
# ------------------------------------------------------------------ #

def print_circuit_summary(n_qubits: int = 5, variational: bool = True) -> None:
    """
    Devre yapısını terminal'de özetler.
    Debug ve makale için.
    """
    n_blocks   = n_qubits // 5
    n_params   = n_qubits * 2 if variational else 0
    output_dim = n_qubits + 3

    print(f"\n{'='*55}")
    print(f"Quantum Circuit Özeti")
    print(f"{'='*55}")
    print(f"  Qubit sayısı       : {n_qubits}")
    print(f"  Blok sayısı        : {n_blocks}")
    print(f"  Variational        : {variational}")
    print(f"  Parametre sayısı   : {n_params}")
    print(f"  Çıktı boyutu       : {output_dim}")
    print(f"\n  Katmanlar:")
    print(f"    1. RY Encoding   : {n_qubits} gate")
    print(f"    2. Blok CNOT     : {n_blocks * 4} gate ({n_blocks} blok × 4 çift)")
    cross = (n_blocks - 1) * 5 if n_blocks > 1 else 0
    print(f"    3. Cross CNOT    : {cross} gate")
    if variational:
        print(f"    4. RZ+RX         : {n_qubits * 2} gate")
    print(f"    5. Ölçüm         : {n_qubits} × ⟨Z⟩ + 3 × ⟨ZZ⟩")
    print(f"\n  ZZ Çiftleri (ilk blok):")
    print(f"    ⟨Z_zs · Z_zu⟩   → defleksiyon korelasyonu")
    print(f"    ⟨Z_zsd · Z_zud⟩ → göreli hız korelasyonu")
    print(f"    ⟨Z_zu · Z_zr⟩   → tire korelasyonu")
    print(f"{'='*55}\n")