"""
GeoLife GPS Trajectory Ingestion
---------------------------------
Membaca raw data dari GeoLife GPS Trajectory Dataset (Microsoft Research).

Struktur folder GeoLife yang diexpect:
    data/raw/mobility/
        000/
            Trajectory/
                20081023025304.plt
                ...
        001/
            Trajectory/
                ...
        ...

Format file .plt (PLT format):
    Baris 1-6: header (dilewati)
    Baris 7+:  latitude,longitude,0,altitude,date_days,date,time

Sumber dataset: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_plt_file(filepath: str) -> pd.DataFrame:
    """
    Parse satu file .plt menjadi DataFrame.

    Returns DataFrame dengan kolom:
        latitude, longitude, altitude, timestamp, trajectory_id
    Returns None jika file tidak valid atau terlalu kecil.
    """
    try:
        df = pd.read_csv(
            filepath,
            skiprows=6,
            header=None,
            names=["latitude", "longitude", "zero", "altitude", "date_days", "date", "time"],
            dtype={
                "latitude": float,
                "longitude": float,
                "altitude": float,
                "date_days": float,
                "date": str,
                "time": str,
            },
        )

        if df.empty:
            return None

        # Parse timestamp dari kolom date + time
        df["timestamp"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

        # Drop baris yang gagal parse timestamp
        df = df.dropna(subset=["timestamp"])

        # Gunakan path sebagai trajectory ID
        df["trajectory_id"] = Path(filepath).stem

        return df[["latitude", "longitude", "altitude", "timestamp", "trajectory_id"]]

    except Exception as e:
        logger.debug(f"Gagal parse {filepath}: {e}")
        return None


def load_geolife(
    raw_dir: str,
    max_trajectories: int = 500,
    min_points: int = 50,
) -> pd.DataFrame:
    """
    Load dan gabungkan seluruh trajectory dari GeoLife dataset.

    Parameters
    ----------
    raw_dir : str
        Path ke folder data/raw/mobility yang berisi folder user GeoLife.
    max_trajectories : int
        Batas jumlah file .plt yang diproses. Default 500.
    min_points : int
        Minimum jumlah GPS points per trajectory. Trajectory lebih pendek dibuang.

    Returns
    -------
    pd.DataFrame
        DataFrame gabungan semua trajectory dengan kolom:
        latitude, longitude, altitude, timestamp, trajectory_id, user_id
    """
    plt_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.plt"), recursive=True))

    if not plt_files:
        logger.warning(
            f"Tidak ada file .plt ditemukan di {raw_dir}. "
            "Pastikan kamu sudah mendownload dan mengekstrak GeoLife dataset ke folder tersebut."
        )
        return pd.DataFrame()

    logger.info(f"Ditemukan {len(plt_files)} file .plt. Memproses max {max_trajectories}...")

    selected = plt_files[:max_trajectories]
    frames = []

    for fpath in tqdm(selected, desc="Parsing trajectories", unit="file"):
        df = parse_plt_file(fpath)
        if df is None or len(df) < min_points:
            continue

        # Ekstrak user_id dari path (folder level pertama setelah raw_dir)
        parts = Path(fpath).relative_to(raw_dir).parts
        df["user_id"] = parts[0] if parts else "unknown"

        frames.append(df)

    if not frames:
        logger.error("Tidak ada trajectory yang berhasil diparse.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Berhasil load {len(frames)} trajectories, "
        f"total {len(combined):,} GPS points."
    )
    return combined


def remap_to_study_area(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """
    Remap koordinat GPS dari dataset asli ke bounding box studi kasus (Surabaya).

    Ini adalah teknik normalisasi spasial yang valid secara metodologi:
    framework diuji pada dataset publik dan dicontextualize ke area studi.
    Pola spasial relatif antar titik dipertahankan.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'latitude' dan 'longitude'.
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box target (metropolitan Surabaya).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan koordinat yang sudah diremap.
    """
    if df.empty:
        return df

    # Normalisasi 0-1 dari range asli, lalu scale ke range target
    lat_orig_min, lat_orig_max = df["latitude"].min(), df["latitude"].max()
    lon_orig_min, lon_orig_max = df["longitude"].min(), df["longitude"].max()

    lat_range_orig = lat_orig_max - lat_orig_min
    lon_range_orig = lon_orig_max - lon_orig_min

    if lat_range_orig == 0 or lon_range_orig == 0:
        logger.warning("Range koordinat nol, remap tidak dilakukan.")
        return df

    df = df.copy()
    df["latitude"] = lat_min + (df["latitude"] - lat_orig_min) / lat_range_orig * (lat_max - lat_min)
    df["longitude"] = lon_min + (df["longitude"] - lon_orig_min) / lon_range_orig * (lon_max - lon_min)

    logger.info(
        f"Koordinat diremap ke bounding box Surabaya: "
        f"lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]"
    )
    return df


def compute_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung traffic speed dan congestion index dari GPS trajectory data.

    Speed dihitung sebagai jarak Euclidean antar titik berurutan dibagi
    selisih waktu. Congestion index adalah normalisasi invers dari speed
    relatif terhadap median speed per jam.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom tambahan: speed_kmh, congestion_index
    """
    if df.empty:
        return df

    df = df.sort_values(["trajectory_id", "timestamp"]).copy()

    # Hitung delta latitude/longitude dan delta waktu
    df["dlat"] = df.groupby("trajectory_id")["latitude"].diff()
    df["dlon"] = df.groupby("trajectory_id")["longitude"].diff()
    df["dt_sec"] = (
        df.groupby("trajectory_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # Jarak aproximasi dalam km (1 derajat ~ 111 km)
    df["dist_km"] = np.sqrt(df["dlat"] ** 2 + df["dlon"] ** 2) * 111.0

    # Speed dalam km/h
    df["speed_kmh"] = np.where(
        df["dt_sec"] > 0,
        df["dist_km"] / (df["dt_sec"] / 3600),
        np.nan,
    )

    # Filter outlier speed (> 150 km/h tidak masuk akal untuk urban)
    df["speed_kmh"] = df["speed_kmh"].clip(0, 150)

    # Congestion index: invers speed yang dinormalisasi (0 = bebas, 1 = macet)
    max_speed_ref = 60.0  # referensi kecepatan bebas hambatan urban
    df["congestion_index"] = 1.0 - (df["speed_kmh"].clip(0, max_speed_ref) / max_speed_ref)
    df["congestion_index"] = df["congestion_index"].fillna(0.5)

    # Ekstrak fitur temporal
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Drop kolom intermediate
    df = df.drop(columns=["dlat", "dlon", "dt_sec", "dist_km"])

    return df
