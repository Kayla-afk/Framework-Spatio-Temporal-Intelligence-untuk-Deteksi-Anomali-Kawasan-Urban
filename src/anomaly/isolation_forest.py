"""
Stage 2 - Spatio-Temporal Anomaly Detection via Isolation Forest
-----------------------------------------------------------------
Tujuan: Mendeteksi perubahan abnormal pada pola mobilitas berdasarkan
kombinasi dimensi spasial dan temporal.

Isolation Forest dipilih karena:
    - Tidak membuat asumsi distribusi data (non-parametric)
    - Efisien pada dataset besar (kompleksitas O(n log n))
    - Robust terhadap multi-dimensional feature space
    - Contamination parameter memungkinkan kontrol proporsi anomali

Feature engineering untuk anomaly detection:
    - Dimensi spasial: latitude, longitude, cluster_label, density_score
    - Dimensi temporal: hour, day_of_week, speed deviation dari baseline
    - Dimensi behavioral: congestion index, trajectory density per window

Output:
    - anomaly_score per titik GPS (-1 = anomali, 1 = normal dalam IF convention)
    - normalized_anomaly_score (0-1, semakin tinggi = semakin anomali)
    - anomalous_zone: agregasi anomali ke level area spasial
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def build_anomaly_features(
    df: pd.DataFrame,
    cluster_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Bangun feature matrix untuk Isolation Forest.

    Features yang digunakan:
        1. lat_norm, lon_norm: posisi spasial ternormalisasi
        2. hour_sin, hour_cos: encoding sirkular jam (preservasi periodicitas)
        3. dow_sin, dow_cos: encoding sirkular hari dalam minggu
        4. speed_kmh: kecepatan (jika tersedia)
        5. congestion_index: tingkat kemacetan
        6. cluster_density: kepadatan cluster lokal (dari cluster_summary)
        7. point_density_window: kepadatan titik GPS per window spasio-temporal

    Returns
    -------
    pd.DataFrame
        Feature matrix siap untuk Isolation Forest.
        Baris yang tidak valid (NaN) di-drop.
    """
    df = df.copy()

    # Pastikan kolom temporal tersedia
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Normalisasi spasial ke 0-1
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    df["lat_norm"] = (df["latitude"] - lat_min) / (lat_max - lat_min + 1e-9)
    df["lon_norm"] = (df["longitude"] - lon_min) / (lon_max - lon_min + 1e-9)

    # Encoding sirkular untuk jam dan hari (menghindari discontinuity 23->0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Feature dasar yang selalu ada
    feature_cols = ["lat_norm", "lon_norm", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    # Speed dan congestion jika tersedia
    if "speed_kmh" in df.columns:
        df["speed_kmh"] = df["speed_kmh"].fillna(df["speed_kmh"].median())
        feature_cols.append("speed_kmh")

    if "congestion_index" in df.columns:
        df["congestion_index"] = df["congestion_index"].fillna(0.5)
        feature_cols.append("congestion_index")

    # Merge cluster density dari cluster_summary
    if cluster_summary is not None and not cluster_summary.empty and "cluster_label" in df.columns:
        density_map = cluster_summary.set_index("cluster_label")["density_score"].to_dict()
        df["cluster_density"] = df["cluster_label"].map(density_map).fillna(0)
        feature_cols.append("cluster_density")

    # Point density per spatio-temporal window
    # Approximasi: hitung jumlah titik dalam grid sel 0.01 deg x 1 jam
    df["lat_grid"] = (df["lat_norm"] * 30).astype(int)
    df["lon_grid"] = (df["lon_norm"] * 30).astype(int)
    df["hour_grid"] = df["hour"]

    density_window = (
        df.groupby(["lat_grid", "lon_grid", "hour_grid"])
        .size()
        .reset_index(name="point_density_window")
    )
    df = df.merge(density_window, on=["lat_grid", "lon_grid", "hour_grid"], how="left")
    df["point_density_window"] = df["point_density_window"].fillna(1)

    # Log-transform density (distribusi sangat skewed)
    df["point_density_log"] = np.log1p(df["point_density_window"])
    feature_cols.append("point_density_log")

    # Drop temporary columns
    df = df.drop(columns=["lat_grid", "lon_grid", "hour_grid", "point_density_window"])

    return df, feature_cols


def run_isolation_forest(
    df: pd.DataFrame,
    cluster_summary: Optional[pd.DataFrame] = None,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Jalankan Isolation Forest untuk deteksi anomali spatio-temporal.

    Parameters
    ----------
    df : pd.DataFrame
        GPS trajectory data (output Stage 1).
    cluster_summary : pd.DataFrame, optional
        Output dari compute_cluster_summary, untuk feature cluster density.
    contamination : float
        Proporsi estimasi anomali. 0.05 = ekspektasi 5% data adalah anomali.
        Sesuaikan berdasarkan domain knowledge (urban anomaly rate).
    n_estimators : int
        Jumlah isolation trees. 200 memberikan stabilitas yang baik.

    Returns
    -------
    pd.DataFrame
        Input DataFrame dengan kolom tambahan:
        anomaly_prediction (-1=anomali, 1=normal),
        anomaly_score_raw, normalized_anomaly_score
    """
    if df.empty:
        logger.error("DataFrame kosong.")
        return df

    df_feat, feature_cols = build_anomaly_features(df, cluster_summary)

    X = df_feat[feature_cols].values

    # RobustScaler: lebih stabil dari StandardScaler untuk data dengan outlier
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        f"Menjalankan Isolation Forest: {len(X_scaled):,} titik, "
        f"{len(feature_cols)} features, contamination={contamination}"
    )

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,  # gunakan semua CPU
    )

    predictions = iso_forest.fit_predict(X_scaled)
    # decision_function: semakin negatif = semakin anomali
    scores_raw = iso_forest.decision_function(X_scaled)

    # Normalisasi score ke 0-1 (1 = paling anomali)
    score_min, score_max = scores_raw.min(), scores_raw.max()
    normalized_scores = 1.0 - (scores_raw - score_min) / (score_max - score_min + 1e-9)

    df_feat = df_feat.copy()
    df_feat["anomaly_prediction"] = predictions
    df_feat["anomaly_score_raw"] = scores_raw
    df_feat["normalized_anomaly_score"] = normalized_scores

    n_anomalies = (predictions == -1).sum()
    logger.info(
        f"Isolation Forest selesai: {n_anomalies:,} anomali terdeteksi "
        f"({n_anomalies/len(predictions)*100:.1f}%)"
    )

    return df_feat


def aggregate_anomaly_zones(
    df: pd.DataFrame,
    grid_resolution: float = 0.01,
) -> pd.DataFrame:
    """
    Agregasi anomali dari level titik GPS ke level zona spasial.

    Membagi area studi ke grid sel, lalu hitung:
        - Proporsi anomali per sel
        - Score anomali rata-rata per sel
        - Cluster kepadatan per sel

    Parameters
    ----------
    grid_resolution : float
        Ukuran grid sel dalam derajat. 0.01 deg ~ 1.1 km.

    Returns
    -------
    pd.DataFrame
        Satu baris per grid sel dengan kolom:
        grid_lat, grid_lon, anomaly_rate, avg_anomaly_score,
        point_count, is_anomalous_zone
    """
    if df.empty or "normalized_anomaly_score" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["grid_lat"] = (df["latitude"] / grid_resolution).round() * grid_resolution
    df["grid_lon"] = (df["longitude"] / grid_resolution).round() * grid_resolution

    agg = df.groupby(["grid_lat", "grid_lon"]).agg(
        point_count=("normalized_anomaly_score", "count"),
        avg_anomaly_score=("normalized_anomaly_score", "mean"),
        max_anomaly_score=("normalized_anomaly_score", "max"),
        anomaly_count=("anomaly_prediction", lambda x: (x == -1).sum()),
    ).reset_index()

    agg["anomaly_rate"] = agg["anomaly_count"] / agg["point_count"]

    # Zona dianggap anomali jika rate >= 30% ATAU avg score >= 0.65
    agg["is_anomalous_zone"] = (
        (agg["anomaly_rate"] >= 0.30) |
        (agg["avg_anomaly_score"] >= 0.65)
    )

    n_zones = agg["is_anomalous_zone"].sum()
    logger.info(
        f"Zone aggregation: {len(agg)} grid cells, "
        f"{n_zones} anomalous zones teridentifikasi."
    )

    return agg
