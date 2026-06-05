"""
Stage 4 - Urban Anomaly Intelligence Integration
-------------------------------------------------
Tujuan: Mengintegrasikan hasil analisis mobilitas (Stage 1-2) dan
transaksi digital (Stage 3) menjadi satu unified urban anomaly score
per kawasan.

Pendekatan integrasi:
    Weighted composite scoring dengan tiga komponen:
        1. Mobility anomaly score (dari Isolation Forest, Stage 2)
        2. Spatial cluster density (dari HDBSCAN, Stage 1)
        3. Transaction contextual score (dari Stage 3)

    Bobot dikonfigurasi di config.yaml dan dapat disesuaikan berdasarkan
    kepentingan relatif data yang tersedia.

Klasifikasi kawasan:
    HIGH RISK   : composite score >= 0.65
    MEDIUM RISK : 0.35 <= score < 0.65
    LOW RISK    : score < 0.35

Output:
    - urban_anomaly_score per zona spatial
    - risk_classification per zona
    - kawasan_prioritas_monitoring (top-N high risk zones)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def spatial_join_mobility_transaction(
    mobility_zones: pd.DataFrame,
    transaction_indicator: pd.DataFrame,
    max_distance_deg: float = 0.08,
) -> pd.DataFrame:
    """
    Spatial join antara anomaly zones (mobilitas) dan transaction areas.

    Karena dua dataset ini tidak memiliki area ID yang sama,
    kita gunakan nearest-neighbor spatial matching berdasarkan koordinat.

    Parameters
    ----------
    mobility_zones : pd.DataFrame
        Output dari aggregate_anomaly_zones (Stage 2).
        Harus memiliki: grid_lat, grid_lon, avg_anomaly_score.
    transaction_indicator : pd.DataFrame
        Output dari build_contextual_indicator (Stage 3).
        Harus memiliki: area_lat, area_lon, contextual_activity_score.
    max_distance_deg : float
        Jarak maksimum untuk matching (dalam derajat). 0.02 deg ~ 2 km.

    Returns
    -------
    pd.DataFrame
        Mobility zones dengan kolom transaction yang di-join.
    """
    if mobility_zones.empty:
        logger.warning("Mobility zones kosong, integrasi dilewati.")
        return pd.DataFrame()

    if transaction_indicator.empty:
        logger.warning("Transaction indicator kosong. Integrasi tanpa data transaksi.")
        mob = mobility_zones.copy()
        mob["contextual_activity_score"] = 0.0
        mob["area_name"] = "Surabaya Analytical Zone (Unmatched)"
        return mob

    # Build KD-tree dari transaction area coords
    tx_coords = transaction_indicator[["area_lat", "area_lon"]].values
    mob_coords = mobility_zones[["grid_lat", "grid_lon"]].values

    tree = cKDTree(tx_coords)
    distances, indices = tree.query(mob_coords, k=1)

    # Mask: hanya ambil match dalam jarak maksimum
    valid_mask = distances <= max_distance_deg

    logger.info(
        "Spatial join distance statistics | "
        f"min={distances.min():.4f}, "
        f"median={np.median(distances):.4f}, "
        f"p75={np.percentile(distances, 75):.4f}, "
        f"max={distances.max():.4f}"
    )

    logger.info(
        f"Spatial join coverage: "
        f"{valid_mask.sum()}/{len(valid_mask)} zones "
        f"({100 * valid_mask.mean():.1f}%) matched "
        f"within threshold {max_distance_deg:.3f}"
    )

    mob = mobility_zones.copy()

    # Default nilai jika tidak ada match
    mob["contextual_activity_score"] = 0.0
    mob["area_name"] = [f"Surabaya Urban Zone {i+1:02d}" for i in range(len(mob))]
    mob["fraud_rate"] = 0.0
    mob["transaction_intensity"] = 0.0

    for col in ["contextual_activity_score", "area_name", "fraud_rate", "transaction_intensity"]:
        if col in transaction_indicator.columns:
            matched_values = transaction_indicator.iloc[indices][col].values
            mob.loc[valid_mask, col] = matched_values[valid_mask]

    n_matched = valid_mask.sum()
    logger.info(
        f"Spatial join: {n_matched}/{len(mob)} mobility zones berhasil "
        f"di-match ke transaction area (max distance={max_distance_deg} deg)"
    )

    return mob


def compute_urban_anomaly_score(
    integrated_df: pd.DataFrame,
    weight_mobility: float = 0.45,
    weight_density: float = 0.25,
    weight_transaction: float = 0.30,
) -> pd.DataFrame:
    """
    Hitung composite urban anomaly score per zona.

    Parameters
    ----------
    weight_mobility : float
        Bobot untuk avg_anomaly_score dari Isolation Forest.
    weight_density : float
        Bobot untuk anomaly_rate (kepadatan anomali) di zona.
    weight_transaction : float
        Bobot untuk contextual_activity_score dari transaksi.

    Returns
    -------
    pd.DataFrame dengan kolom:
        urban_anomaly_score (0-1),
        risk_level ('HIGH', 'MEDIUM', 'LOW')
    """
    assert abs(weight_mobility + weight_density + weight_transaction - 1.0) < 0.01, \
        "Bobot harus berjumlah 1.0"

    df = integrated_df.copy()

    # Pastikan semua komponen ada dan dalam range 0-1
    mob_score = df.get("avg_anomaly_score", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)
    density_score = df.get("anomaly_rate", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)
    tx_score = df.get("contextual_activity_score", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)

    df["urban_anomaly_score"] = (
        weight_mobility * mob_score +
        weight_density * density_score +
        weight_transaction * tx_score
    ).clip(0, 1)

    # Klasifikasi risiko
    def classify_risk(score):
        if score >= 0.65:
            return "HIGH"
        elif score >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    df["risk_level"] = df["urban_anomaly_score"].apply(classify_risk)

    risk_counts = df["risk_level"].value_counts()
    logger.info(
        f"Urban anomaly scoring selesai:\n"
        f"  HIGH  : {risk_counts.get('HIGH', 0)} zona\n"
        f"  MEDIUM: {risk_counts.get('MEDIUM', 0)} zona\n"
        f"  LOW   : {risk_counts.get('LOW', 0)} zona"
    )

    return df


def identify_priority_monitoring_areas(
    scored_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identifikasi kawasan prioritas monitoring berdasarkan composite score.

    Kriteria tambahan selain score tinggi:
        - Zona dengan kombinasi anomali mobilitas DAN transaksi (double-signal)
        - Zona dengan titik GPS padat (bukan sparse area)

    Returns
    -------
    pd.DataFrame (top_n kawasan prioritas)
    """
    if scored_df.empty:
        return pd.DataFrame()

    df = scored_df.copy()

    # Double-signal bonus: anomali di kedua dimensi
    mob_anomalous = df.get("is_anomalous_zone", pd.Series(False, index=df.index))
    tx_high = df.get("contextual_activity_score", pd.Series(0, index=df.index)) >= 0.5

    df["double_signal"] = (mob_anomalous & tx_high).astype(float)

    # Priority score: anomaly score + bonus double signal
    df["priority_score"] = df["urban_anomaly_score"] + 0.10 * df["double_signal"]
    df["priority_score"] = df["priority_score"].clip(0, 1)

    # Filter ke area dengan minimal data yang cukup
    min_points = df["point_count"].quantile(0.25) if "point_count" in df.columns else 0
    sufficient_data = df.get("point_count", pd.Series(1, index=df.index)) >= min_points

    priority_areas = (
        df[sufficient_data]
        .sort_values("priority_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    priority_areas["monitoring_rank"] = priority_areas.index + 1

    logger.info(
        f"Kawasan prioritas monitoring: top {len(priority_areas)} area, "
        f"score range: [{priority_areas['priority_score'].min():.3f}, "
        f"{priority_areas['priority_score'].max():.3f}]"
    )

    return priority_areas


def generate_anomaly_correlation_pattern(
    scored_df: pd.DataFrame,
) -> dict:
    """
    Analisis korelasi antara anomali mobilitas dan transaksi di seluruh zona.

    Returns
    -------
    dict dengan key:
        'pearson_r': korelasi Pearson antara mobility_score dan tx_score
        'high_risk_zones': jumlah zona HIGH RISK
        'double_signal_rate': proporsi zona dengan sinyal ganda
        'spatial_concentration': Gini index distribusi anomali (0=merata, 1=terpusat)
    """
    from scipy.stats import pearsonr

    result = {}

    mob_scores = scored_df.get("avg_anomaly_score", pd.Series()).fillna(0)
    tx_scores = scored_df.get("contextual_activity_score", pd.Series()).fillna(0)

    if len(mob_scores) >= 3 and len(tx_scores) >= 3:
        r, p_val = pearsonr(mob_scores, tx_scores)
        result["pearson_r"] = float(r)
        result["pearson_p"] = float(p_val)
    else:
        result["pearson_r"] = 0.0
        result["pearson_p"] = 1.0

    result["high_risk_zones"] = int((scored_df.get("risk_level", pd.Series()) == "HIGH").sum())
    result["medium_risk_zones"] = int((scored_df.get("risk_level", pd.Series()) == "MEDIUM").sum())
    result["total_zones"] = len(scored_df)

    # Double signal rate
    mob_anom = scored_df.get("is_anomalous_zone", pd.Series(False)).fillna(False)
    tx_high = scored_df.get("contextual_activity_score", pd.Series(0)) >= 0.5
    result["double_signal_rate"] = float((mob_anom & tx_high).mean()) if len(scored_df) > 0 else 0.0

    # Gini index dari urban_anomaly_score
    scores = scored_df.get("urban_anomaly_score", pd.Series()).fillna(0).sort_values().values
    if len(scores) > 1:
        n = len(scores)
        cumsum = np.cumsum(scores)
        gini = (2 * np.sum((np.arange(1, n + 1)) * scores) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-9)
        result["spatial_concentration_gini"] = float(gini)
    else:
        result["spatial_concentration_gini"] = 0.0

    logger.info(
        f"Anomaly correlation: r={result['pearson_r']:.3f}, "
        f"double_signal={result['double_signal_rate']:.1%}, "
        f"gini={result['spatial_concentration_gini']:.3f}"
    )

    return result
