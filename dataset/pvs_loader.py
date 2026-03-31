"""
dataset/pvs_loader.py
---------------------
PVS (Passive Vehicular Sensors) veri setini yükler.

Desteklenen dosya isimleri (öncelik sırasıyla):
  dataset_gps_mpu_left.csv   ← GPS + MPU birleşik (sizin formatınız)
  dataset_gps_mpu_right.csv
  dataset_mpu_left.csv       ← Sadece MPU
  dataset_mpu_right.csv

Sütun isimleri otomatik algılanır.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------------ #
#  Road tipi eşleme                                                   #
# ------------------------------------------------------------------ #

SIM_TO_PVS = {
    "random"     : ["cobblestone_road"],
    "sinusoidal" : ["asphalt_road", "paved_road"],
    "bump"       : ["speed_bump_asphalt", "speed_bump"],
    "pothole"    : ["unpaved_road", "dirt_road"],
}


# ------------------------------------------------------------------ #
#  Veri yapısı                                                        #
# ------------------------------------------------------------------ #

@dataclass
class RoadSegment:
    road_type  : str
    pvs_label  : str
    timestamps : np.ndarray
    body_acc   : np.ndarray
    wheel_acc  : np.ndarray
    susp_acc   : np.ndarray
    speed_mps  : np.ndarray
    dt         : float
    fs         : float
    duration   : float
    n_samples  : int
    quality    : str = "unknown"
    side       : str = "left"

    @property
    def rms_body_acc(self) -> float:
        return float(np.sqrt(np.mean(self.body_acc ** 2)))

    @property
    def rms_wheel_acc(self) -> float:
        return float(np.sqrt(np.mean(self.wheel_acc ** 2)))

    def resample(self, target_fs: float = 100.0) -> "RoadSegment":
        if abs(self.fs - target_fs) < 1.0:
            return self
        from scipy.signal import resample_poly
        from math import gcd
        p = int(target_fs); q = int(self.fs)
        g = gcd(p, q);      p, q = p // g, q // g
        n_new = int(self.n_samples * target_fs / self.fs)
        def _r(x): return resample_poly(x, p, q)[:n_new]
        return RoadSegment(
            road_type=self.road_type, pvs_label=self.pvs_label,
            timestamps=np.linspace(self.timestamps[0],self.timestamps[-1],n_new),
            body_acc=_r(self.body_acc), wheel_acc=_r(self.wheel_acc),
            susp_acc=_r(self.susp_acc), speed_mps=_r(self.speed_mps),
            dt=1.0/target_fs, fs=target_fs, duration=self.duration,
            n_samples=n_new, quality=self.quality, side=self.side,
        )

    def get_road_profile(self, method: str = "direct") -> np.ndarray:
        if method == "direct":
            acc = self.wheel_acc - np.mean(self.wheel_acc)
            return acc
        # double_int
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.5 / (self.fs / 2), btype="high")
        acc  = filtfilt(b, a, self.wheel_acc)
        vel  = np.cumsum(acc) * self.dt
        disp = np.cumsum(vel) * self.dt
        rng  = np.ptp(disp)
        if rng > 1e-6:
            disp = disp / rng * 0.02
        return disp


# ------------------------------------------------------------------ #
#  PVSLoader                                                          #
# ------------------------------------------------------------------ #

class PVSLoader:
    """
    PVS klasöründen (PVS 1 … PVS 9) veri yükler.

    Otomatik olarak şu dosyaları dener:
      dataset_gps_mpu_{side}.csv  (GPS+MPU birleşik — sizin formatınız)
      dataset_mpu_{side}.csv      (sadece MPU)
    """

    # Olası ivme sütun isimleri (en yaygın → en az yaygın sırasıyla)
    _BODY_COLS  = ["acc_z_dashboard", "accZ_dashboard",
                   "body_acc_z", "acc_z_body", "az_dashboard"]
    _WHEEL_COLS = ["acc_z_below_suspension", "accZ_below_suspension",
                   "wheel_acc_z", "acc_z_wheel", "az_below"]
    _SUSP_COLS  = ["acc_z_above_suspension", "accZ_above_suspension",
                   "susp_acc_z", "acc_z_susp", "az_above"]
    _SPEED_COLS = ["speed_meters_per_second", "speed_mps",
                   "gps_speed", "speed"]
    _TS_COLS    = ["timestamp", "time_ms", "time", "ts"]

    def __init__(
        self,
        dataset_dir : str,
        side        : str = "left",
        target_fs   : Optional[float] = 100.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.side        = side
        self.target_fs   = target_fs
        self._df   : Optional[pd.DataFrame] = None
        self._labels: Optional[pd.DataFrame] = None
        self._fs   : float = 100.0
        self._cols : dict  = {}
        self._load()

    # ── yükleme ─────────────────────────────────────────────────────── #

    def _load(self):
        df = self._find_and_load_mpu()
        if df is None:
            raise FileNotFoundError(
                f"MPU/GPS dosyası bulunamadı: {self.dataset_dir}\n"
                f"Aranan: dataset_gps_mpu_{self.side}.csv, "
                f"dataset_mpu_{self.side}.csv"
            )

        self._df  = df
        self._fs  = self._detect_fs(df)
        self._cols = self._detect_columns(df)

        label_file = self.dataset_dir / "dataset_labels.csv"
        if label_file.exists():
            ldf = pd.read_csv(label_file)
            n = min(len(ldf), len(df))
            self._labels = ldf.iloc[:n].reset_index(drop=True)
            self._df     = df.iloc[:n].reset_index(drop=True)

        print(f"  PVS {self.dataset_dir.name} ({self.side}): "
              f"{len(self._df):,} satır @ {self._fs:.0f} Hz")

    def _find_and_load_mpu(self) -> Optional[pd.DataFrame]:
        """GPS+MPU veya MPU dosyasını bul ve yükle."""
        candidates = [
            self.dataset_dir / f"dataset_gps_mpu_{self.side}.csv",
            self.dataset_dir / f"dataset_mpu_{self.side}.csv",
            # alt klasörlerde de ara
            *self.dataset_dir.glob(f"*gps*mpu*{self.side}*.csv"),
            *self.dataset_dir.glob(f"*mpu*{self.side}*.csv"),
        ]
        for f in candidates:
            if f.exists():
                return pd.read_csv(f)
        return None

    def _detect_fs(self, df: pd.DataFrame) -> float:
        """Timestamp sütunundan örnekleme frekansını hesapla."""
        ts_col = self._find_col(df, self._TS_COLS)
        if ts_col is None:
            return 100.0
        ts = df[ts_col].values.astype(float)
        dt = np.median(np.diff(ts))
        # ms mi s mi?
        if dt > 0.5:          # ms cinsinden (tipik: 1ms @ 1000Hz)
            dt /= 1000.0
        return float(1.0 / max(dt, 1e-6))

    def _detect_columns(self, df: pd.DataFrame) -> dict:
        """Sütun isimlerini otomatik algıla."""
        return {
            "ts"   : self._find_col(df, self._TS_COLS)    or df.columns[0],
            "body" : self._find_col(df, self._BODY_COLS)  or df.columns[1],
            "wheel": self._find_col(df, self._WHEEL_COLS) or df.columns[2],
            "susp" : self._find_col(df, self._SUSP_COLS)  or df.columns[min(3,len(df.columns)-1)],
            "speed": self._find_col(df, self._SPEED_COLS),
        }

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    # ── segmentasyon ─────────────────────────────────────────────────── #

    def _label_to_road_type(self, row: pd.Series) -> tuple:
        for label, check in [
            ("speed_bump_asphalt",    "bump"),
            ("speed_bump_cobblestone","bump"),
            ("cobblestone_road",      "random"),
            ("dirt_road",             "pothole"),
            ("unpaved_road",          "pothole"),
            ("asphalt_road",          "sinusoidal"),
            ("paved_road",            "sinusoidal"),
        ]:
            val = row.get(label, 0)
            if val == 1 or str(val).strip() == "1":
                return check, label
        return "unknown", "unknown"

    def get_road_segments(
        self,
        road_type    : str = "all",
        min_duration : float = 3.0,
        max_segments : int = 20,
        min_speed    : float = 1.0,
    ) -> list:
        df   = self._df
        cols = self._cols

        ts        = df[cols["ts"]].values.astype(float)
        body_acc  = df[cols["body"]].values.astype(float)
        wheel_acc = df[cols["wheel"]].values.astype(float)
        susp_acc  = df[cols["susp"]].values.astype(float)

        # Gravity çıkar
        body_acc  -= np.median(body_acc)
        wheel_acc -= np.median(wheel_acc)
        susp_acc  -= np.median(susp_acc)

        # Hız
        if cols["speed"] and cols["speed"] in df.columns:
            speed = df[cols["speed"]].values.astype(float)
        else:
            speed = np.full(len(df), 10.0)

        # Timestamp ms → s dönüşüm
        dt_raw = np.median(np.diff(ts))
        ts_s   = ts / 1000.0 if dt_raw > 0.5 else ts

        # Label yoksa tüm veriyi 1 segment olarak döndür
        if self._labels is None:
            return [self._make_seg(
                road_type if road_type != "all" else "random",
                "unknown", ts_s, body_acc, wheel_acc, susp_acc, speed
            )]

        # Label bazlı segmentasyon
        n   = min(len(df), len(self._labels))
        rts = []; quals = []
        for i in range(n):
            rt, q = self._label_to_road_type(self._labels.iloc[i])
            rts.append(rt); quals.append(q)
        rts   = np.array(rts)
        quals = np.array(quals)

        segments = []
        i = 0
        while i < n and len(segments) < max_segments:
            rt = rts[i]
            if rt == "unknown" or (road_type != "all" and rt != road_type):
                i += 1; continue

            j = i + 1
            while j < n and rts[j] == rt:
                j += 1

            dur = ts_s[j-1] - ts_s[i]
            if dur < min_duration or np.mean(speed[i:j]) < min_speed:
                i = j; continue

            seg = self._make_seg(
                rt, quals[i],
                ts_s[i:j], body_acc[i:j],
                wheel_acc[i:j], susp_acc[i:j], speed[i:j]
            )
            if seg:
                segments.append(seg)
            i = j

        return segments

    def _make_seg(self, road_type, pvs_label,
                  timestamps, body_acc, wheel_acc, susp_acc, speed):
        n = len(timestamps)
        if n < 10:
            return None
        dt  = float(np.median(np.diff(timestamps))) if n > 1 else 0.01
        if dt <= 0:
            dt = 0.01
        fs  = 1.0 / dt
        seg = RoadSegment(
            road_type=road_type, pvs_label=pvs_label,
            timestamps=timestamps, body_acc=body_acc,
            wheel_acc=wheel_acc, susp_acc=susp_acc,
            speed_mps=speed, dt=dt, fs=fs,
            duration=n*dt, n_samples=n, side=self.side,
        )
        if self.target_fs and abs(fs - self.target_fs) > 1.0:
            seg = seg.resample(self.target_fs)
        return seg

    def get_summary(self) -> dict:
        if self._df is None:
            return {}
        return {
            "total_samples": len(self._df),
            "fs_hz"        : self._fs,
            "duration_min" : len(self._df) / self._fs / 60,
            "columns"      : list(self._df.columns),
            "detected_cols": self._cols,
        }