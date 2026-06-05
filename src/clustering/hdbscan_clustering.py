"""
Stage 1 - Urban Mobility Mapping via HDBSCAN Clustering
---------------------------------------------------------
Tujuan: Mengidentifikasi pola mobilitas kawasan urban — di mana pergerakan
terjadi, seberapa padat, dan di mana hotspot kemacetan terbentuk.

HDBSCAN dipilih karena:
    - Tidak memerlukan jumlah cluster yang ditetapkan di awal (berbeda dengan K-Means)
    - Robust terhadap noise (titik GPS outlier diberi label -1)
    - Mampu mendeteksi cluster dengan kepadatan dan bentuk yang bervariasi
    - Secara natural menghasilkan hierarki cluster yang sesuai untuk urban data

Output utama:
    - Cluster label per titik GPS
    - Cluster summary: centroid, ukuran, kepadatan
    - Hotspot kandidat (top-N cluster berdasarkan kepadatan)
"""

import logging

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_hdbscan_clustering(
    df: pd.DataFrame,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.003,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Jalankan HDBSCAN clustering pada koordinat GPS.

    Parameters
    ----------
    df : pd.DataFrame
        Harus memiliki kolom 'latitude' dan 'longitude'.
    min_cluster_size : int
        Ukuran cluster minimum. Semakin kecil, semakin banyak cluster kecil
        yang terdeteksi. Nilai 15 cocok untuk data urban skala kota.
    min_samples : int
        Jumlah sample minimum di sekitar titik untuk dianggap core point.
        Mengendalikan konservatisme clustering.
    cluster_selection_epsilon : float
        Jarak minimum antar cluster (dalam derajat). 0.003 ~ 300 meter.
    metric : str
        Metrik jarak. 'euclidean' untuk koordinat yang sudah diremap.
        Gunakan 'haversine' jika bekerja dengan koordinat GPS asli (dalam radian).

    Returns
    -------
    pd.DataFrame
        Input DataFrame dengan kolom tambahan:
        cluster_label, cluster_probability
    """
    if df.empty:
        logger.error("DataFrame kosong, clustering tidak dijalankan.")
        return df

    coords = df[["latitude", "longitude"]].values

    # Standarisasi koordinat untuk HDBSCAN euclidean
    # Tidak menggunakan StandardScaler karena kita ingin preservasi
    # rasio aspek spasial — hanya scale ke range yang komparable
    lat_range = coords[:, 0].max() - coords[:, 0].min()
    lon_range = coords[:, 1].max() - coords[:, 1].min()

    if lat_range == 0 or lon_range == 0:
        logger.error("Range koordinat nol. Periksa data input.")
        return df

    coords_norm = np.column_stack([
        (coords[:, 0] - coords[:, 0].min()) / lat_range,
        (coords[:, 1] - coords[:, 1].min()) / lon_range,
    ])

    logger.info(
        f"Menjalankan HDBSCAN pada {len(coords_norm):,} titik GPS... "
        f"(min_cluster_size={min_cluster_size}, epsilon={cluster_selection_epsilon})"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        cluster_selection_method="eom",  # Excess of Mass - lebih stabil untuk urban data
        prediction_data=True,
    )

    labels = clusterer.fit_predict(coords_norm)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    logger.info(
        f"HDBSCAN selesai: {n_clusters} cluster teridentifikasi, "
        f"{n_noise:,} noise points ({n_noise/len(labels)*100:.1f}%)"
    )

    df = df.copy()
    df["cluster_label"] = labels
    df["cluster_probability"] = probabilities

    return df


def compute_cluster_summary(df: pd.DataFrame, hotspot_percentile: float = 75) -> pd.DataFrame:
    """
    Hitung statistik per cluster untuk identifikasi hotspot.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari run_hdbscan_clustering.
    hotspot_percentile : float
        Percentile kepadatan di atas mana cluster dianggap hotspot.

    Returns
    -------
    pd.DataFrame
        Satu baris per cluster dengan kolom:
        cluster_label, centroid_lat, centroid_lon, point_count,
        density_score, avg_speed_kmh, avg_congestion, is_hotspot
    """
    # Exclude noise points (label -1)
    clustered = df[df["cluster_label"] >= 0].copy()

    if clustered.empty:
        logger.warning("Tidak ada cluster yang valid (semua noise). Periksa parameter HDBSCAN.")
        return pd.DataFrame()

    # Aggregasi per cluster
    agg_dict = {
        "latitude": ["mean", "std", "count"],
        "longitude": ["mean", "std"],
        "cluster_probability": "mean",
    }

    if "speed_kmh" in clustered.columns:
        agg_dict["speed_kmh"] = "mean"
    if "congestion_index" in clustered.columns:
        agg_dict["congestion_index"] = "mean"

    summary = clustered.groupby("cluster_label").agg(agg_dict)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    # Rename kolom utama
    rename_map = {
        "latitude_mean": "centroid_lat",
        "latitude_std": "lat_spread",
        "latitude_count": "point_count",
        "longitude_mean": "centroid_lon",
        "longitude_std": "lon_spread",
        "cluster_probability_mean": "avg_probability",
    }
    if "speed_kmh_mean" in summary.columns:
        rename_map["speed_kmh_mean"] = "avg_speed_kmh"
    if "congestion_index_mean" in summary.columns:
        rename_map["congestion_index_mean"] = "avg_congestion"

    summary = summary.rename(columns=rename_map)

    # Density score: titik per unit area (spread)
    area_spread = (summary["lat_spread"].fillna(0.001) * summary["lon_spread"].fillna(0.001))
    summary["density_score"] = summary["point_count"] / (area_spread + 1e-9)

    # Normalisasi density score ke 0-1
    max_density = summary["density_score"].max()
    if max_density > 0:
        summary["density_score"] = summary["density_score"] / max_density

    # Tandai hotspot
    threshold = np.percentile(summary["density_score"], hotspot_percentile)
    summary["is_hotspot"] = summary["density_score"] >= threshold

    n_hotspots = summary["is_hotspot"].sum()
    logger.info(
        f"Cluster summary: {len(summary)} cluster, "
        f"{n_hotspots} hotspot (density >= percentile {hotspot_percentile})"
    )

    return summary


def extract_baseline_mobility_pattern(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> dict:
    """
    Ekstrak baseline mobility pattern: distribusi pergerakan per jam
    dan per cluster. Digunakan sebagai referensi untuk deteksi anomali
    temporal di Stage 2.

    Returns
    -------
    dict dengan keys:
        'hourly_volume': pd.Series (index=hour, values=point_count)
        'cluster_hourly': pd.DataFrame (cluster x hour matrix)
        'mobility_entropy': float (entropy distribusi spasial)
    """
    result = {}

    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour

    # Volume per jam (baseline temporal)
    hourly = df.groupby("hour").size()
    result["hourly_volume"] = hourly

    # Volume per cluster per jam (untuk deteksi anomali spatio-temporal)
    if "cluster_label" in df.columns:
        cluster_hourly = (
            df[df["cluster_label"] >= 0]
            .groupby(["cluster_label", "hour"])
            .size()
            .unstack(fill_value=0)
        )
        result["cluster_hourly"] = cluster_hourly

    # Mobility entropy: seberapa merata distribusi pergerakan antar cluster
    if not cluster_summary.empty:
        counts = cluster_summary["point_count"].values.astype(float)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        result["mobility_entropy"] = float(entropy)
        logger.info(f"Mobility entropy: {entropy:.3f} (semakin tinggi = semakin merata)")

    return result
