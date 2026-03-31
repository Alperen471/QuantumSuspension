"""
dataset/storage.py
------------------
Ham veriyi diske yazar ve diskten okur.

İki format desteklenir:
  parquet  : Pandas ile hızlı sütunsal okuma, küçük boyut, metadata desteği.
             Analiz ve fuzzy kalibrasyon için tercih edilir.
  numpy    : SAC replay buffer'ına direkt yüklemek için.
             Array formatı: states (N,5), actions (N,1), rewards (N,), vb.

Neden iki format?
  Parquet → fuzzy membership kalibrasyonu, istatistik hesaplama, pandas analizi
  Numpy   → RL training — buffer'a tek seferde yükle, tekrar toplamaya gerek yok
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Parquet için pandas — isteğe bağlı import
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Sabitler                                                            #
# ------------------------------------------------------------------ #

DEFAULT_DATA_DIR = Path("dataset/data")

# Numpy array'e dönüştürülecek kolonların sırası — sabit tutulmalı
STATE_KEYS     = ["z_s", "z_u", "z_s_dot", "z_u_dot", "z_r"]
NEXT_STATE_KEYS = ["z_s_next", "z_u_next", "z_s_dot_next", "z_u_dot_next", "z_r_next"]
ACTION_KEYS    = ["action"]
REWARD_KEY     = "reward"
DONE_KEY       = "done"
FUZZY_KEYS     = ["body_acc", "susp_defl", "ctrl_force", "tire_defl"]


# ------------------------------------------------------------------ #
#  Parquet                                                             #
# ------------------------------------------------------------------ #

def save_parquet(
    records: List[Dict[str, Any]],
    path: str | Path | None = None,
    tag: str = "dataset",
) -> Path:
    """
    Kayıt listesini Parquet formatında kaydeder.

    records : collect_dataset() çıktısı
    tag     : dosya adı öneki — örn. "passive_random_ep0"
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("Parquet için pandas kurulu olmalı: pip install pandas pyarrow")

    path = Path(path) if path else DEFAULT_DATA_DIR / f"{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        # pyarrow/fastparquet yoksa CSV olarak kaydet
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    return path


def load_parquet(path: str | Path) -> "pd.DataFrame":
    if not _PANDAS_AVAILABLE:
        raise ImportError("Parquet için pandas kurulu olmalı.")
    return pd.read_parquet(path)


def load_all_parquet(data_dir: str | Path = DEFAULT_DATA_DIR) -> "pd.DataFrame":
    """
    Bir dizindeki tüm .parquet dosyalarını birleştirir.
    Analiz ve istatistik hesaplama için kullanışlı.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("Parquet için pandas kurulu olmalı.")

    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"Parquet dosyası bulunamadı: {data_dir}")

    frames = [pd.read_parquet(f) for f in sorted(files)]
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ #
#  Numpy                                                               #
# ------------------------------------------------------------------ #

def records_to_numpy(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Kayıt listesini numpy array'lerine dönüştürür.

    Döndürülen dict:
      states      : (N, 5)
      actions     : (N, 1)
      rewards     : (N,)
      next_states : (N, 5)
      dones       : (N,)  float
      body_acc    : (N,)
      susp_defl   : (N,)
      ctrl_force  : (N,)
      tire_defl   : (N,)
    """
    N = len(records)

    states      = np.zeros((N, 5),  dtype=np.float32)
    next_states = np.zeros((N, 5),  dtype=np.float32)
    actions     = np.zeros((N, 1),  dtype=np.float32)
    rewards     = np.zeros(N,       dtype=np.float32)
    dones       = np.zeros(N,       dtype=np.float32)
    body_acc    = np.zeros(N,       dtype=np.float32)
    susp_defl   = np.zeros(N,       dtype=np.float32)
    ctrl_force  = np.zeros(N,       dtype=np.float32)
    tire_defl   = np.zeros(N,       dtype=np.float32)

    for i, r in enumerate(records):
        states[i]      = [r[k] for k in STATE_KEYS]
        next_states[i] = [r[k] for k in NEXT_STATE_KEYS]
        actions[i]     = [r["action"]]
        rewards[i]     = r["reward"]
        dones[i]       = float(r["done"])
        body_acc[i]    = r["body_acc"]
        susp_defl[i]   = r["susp_defl"]
        ctrl_force[i]  = r["ctrl_force"]
        tire_defl[i]   = r["tire_defl"]

    return {
        "states":      states,
        "actions":     actions,
        "rewards":     rewards,
        "next_states": next_states,
        "dones":       dones,
        "body_acc":    body_acc,
        "susp_defl":   susp_defl,
        "ctrl_force":  ctrl_force,
        "tire_defl":   tire_defl,
    }


def save_numpy(
    records: List[Dict[str, Any]],
    path: str | Path | None = None,
    tag: str = "dataset",
) -> Path:
    """
    Numpy formatında .npz olarak kaydeder.
    RL training sırasında replay buffer preloading için.
    """
    path = Path(path) if path else DEFAULT_DATA_DIR / f"{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = records_to_numpy(records)
    np.savez_compressed(path, **arrays)

    return path


def load_numpy(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ------------------------------------------------------------------ #
#  Dataset özet raporu                                                 #
# ------------------------------------------------------------------ #

def print_dataset_summary(records: List[Dict[str, Any]]) -> None:
    """
    Toplanan veri hakkında özet istatistikler basar.
    Normalizasyon kararlarını doğrulamak için kullanılır.
    """
    N = len(records)
    print(f"\n{'='*60}")
    print(f"DATASET ÖZET")
    print(f"{'='*60}")
    print(f"Toplam kayıt (step) : {N:,}")

    # Controller dağılımı
    controllers = {}
    for r in records:
        c = r.get("controller", "?")
        controllers[c] = controllers.get(c, 0) + 1
    print(f"\nController dağılımı:")
    for c, cnt in sorted(controllers.items()):
        print(f"  {c:<12}: {cnt:>8,} ({100*cnt/N:.1f}%)")

    # Road dağılımı
    roads = {}
    for r in records:
        rd = r.get("road_type", "?")
        roads[rd] = roads.get(rd, 0) + 1
    print(f"\nRoad dağılımı:")
    for rd, cnt in sorted(roads.items()):
        print(f"  {rd:<12}: {cnt:>8,} ({100*cnt/N:.1f}%)")

    # Fiziksel değişken aralıkları
    print(f"\nDeğişken aralıkları (ham, normalize edilmemiş):")
    cols = ["z_s", "z_u", "z_s_dot", "z_u_dot", "z_r",
            "body_acc", "susp_defl", "ctrl_force"]
    header = f"  {'değişken':<14} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}"
    print(header)
    print(f"  {'-'*54}")
    for col in cols:
        vals = np.array([r[col] for r in records if col in r])
        print(
            f"  {col:<14} {vals.min():>10.4f} {vals.max():>10.4f} "
            f"{vals.mean():>10.4f} {vals.std():>10.4f}"
        )
    print(f"{'='*60}\n")
