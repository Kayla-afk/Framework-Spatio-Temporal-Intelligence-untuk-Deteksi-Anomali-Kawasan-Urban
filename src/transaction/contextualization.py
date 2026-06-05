"""
Stage 3 - Transaction Contextualization
-----------------------------------------
Tujuan: Menganalisis pola transaksi digital sebagai konteks tambahan
terhadap anomali mobilitas yang terdeteksi di Stage 2.

Posisi metodologi yang tepat:
    Transaksi digital tidak digunakan untuk mendeteksi fraud.
    Transaksi digunakan sebagai contextual urban activity indicator:
    ketika mobilitas menunjukkan anomali di suatu zona, apakah pola
    transaksi digital di zona tersebut juga menunjukkan penyimpangan?
    Korelasi ini memperkuat atau melemahkan hipotesis anomali kawasan.

Output:
    - Contextual urban activity indicator per area
    - Transaction irregularity score per area
    - Temporal transaction fluctuation pattern
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_transaction_irregularity(
    tx_area_agg: pd.DataFrame,
    zscore_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Hitung irregularity score untuk setiap area berdasarkan deviasi
    dari pola transaksi normal.

    Variabel yang dipertimbangkan:
        - transaction_intensity: volume relatif terhadap rata-rata semua area
        - fraud_rate: proporsi transaksi berpotensi fraud
        - temporal_irregularity: volatilitas volume per jam
        - amount_zscore_max: nilai transaksi tertinggi dibanding baseline

    Returns
    -------
    pd.DataFrame
        Input DataFrame dengan kolom tambahan:
        transaction_irregularity_score (0-1),
        irregularity_flag (bool)
    """
    if tx_area_agg.empty:
        return tx_area_agg

    df = tx_area_agg.copy()

    # Komponen 1: Volume anomali (z-score transaction intensity)
    intensity_zscore = np.abs(stats.zscore(df["transaction_intensity"].fillna(0)))
    df["intensity_anomaly"] = np.clip(intensity_zscore / zscore_threshold, 0, 1)

    # Komponen 2: Fraud rate sebagai risk signal
    max_fraud = df["fraud_rate"].max()
    df["fraud_signal"] = df["fraud_rate"] / max_fraud if max_fraud > 0 else 0

    # Komponen 3: Temporal irregularity (sudah dinormalisasi 0-1)
    df["temporal_signal"] = df["temporal_irregularity"].fillna(0)

    # Komponen 4: Nilai transaksi ekstrem
    if "amount_zscore_max" in df.columns:
        max_z = df["amount_zscore_max"].replace([np.inf, -np.inf], np.nan).fillna(0).abs().max()
        df["value_signal"] = df["amount_zscore_max"].abs().clip(0, max_z) / max_z if max_z > 0 else 0
    else:
        df["value_signal"] = 0

    # Composite transaction irregularity score (weighted sum)
    df["transaction_irregularity_score"] = (
        0.35 * df["intensity_anomaly"] +
        0.25 * df["fraud_signal"] +
        0.25 * df["temporal_signal"] +
        0.15 * df["value_signal"]
    )

    # Clip ke 0-1
    df["transaction_irregularity_score"] = df["transaction_irregularity_score"].clip(0, 1)

    # Flag area dengan irregularity signifikan
    threshold = df["transaction_irregularity_score"].quantile(0.75)
    df["irregularity_flag"] = df["transaction_irregularity_score"] >= threshold

    n_flagged = df["irregularity_flag"].sum()
    logger.info(
        f"Transaction irregularity: {n_flagged}/{len(df)} area terflag "
        f"(threshold={threshold:.3f})"
    )

    return df


def compute_temporal_transaction_profile(
    df_transactions: pd.DataFrame,
    n_hours: int = 24,
) -> pd.DataFrame:
    """
    Bangun profil temporal transaksi per area (volume per jam).

    Profil ini digunakan untuk:
        1. Memvalidasi pola yang dideteksi Isolation Forest
        2. Mengidentifikasi waktu-waktu dengan aktivitas tidak wajar

    Returns
    -------
    pd.DataFrame (pivot table):
        Index: area_id
        Kolom: jam 0-23
        Nilai: jumlah transaksi
    """
    if df_transactions.empty or "hour_of_day" not in df_transactions.columns:
        return pd.DataFrame()

    pivot = (
        df_transactions.groupby(["area_id", "hour_of_day"])
        .size()
        .reset_index(name="count")
        .pivot_table(index="area_id", columns="hour_of_day", values="count", fill_value=0)
    )

    # Pastikan semua 24 jam ada (isi 0 jika tidak ada transaksi)
    for h in range(n_hours):
        if h not in pivot.columns:
            pivot[h] = 0

    pivot = pivot[sorted(pivot.columns)]

    logger.info(f"Temporal profile: {len(pivot)} area x {len(pivot.columns)} jam")
    return pivot


def identify_transaction_anomaly_hours(
    temporal_profile: pd.DataFrame,
    zscore_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Identifikasi jam-jam dengan volume transaksi anomali per area.

    Anomali temporal: jam di mana volume transaksi menyimpang lebih dari
    zscore_threshold dari rata-rata jam lain di area yang sama.

    Returns
    -------
    pd.DataFrame dengan kolom:
        area_id, anomaly_hours (list), n_anomaly_hours, peak_anomaly_hour
    """
    if temporal_profile.empty:
        return pd.DataFrame()

    records = []
    for area_id, row in temporal_profile.iterrows():
        values = row.values.astype(float)
        mean_val = values.mean()
        std_val = values.std()

        if std_val < 1e-6:
            # Distribusi seragam — tidak ada anomali temporal
            records.append({
                "area_id": area_id,
                "anomaly_hours": [],
                "n_anomaly_hours": 0,
                "peak_anomaly_hour": -1,
            })
            continue

        zscores = np.abs((values - mean_val) / std_val)
        anomaly_hours = [int(h) for h, z in enumerate(zscores) if z >= zscore_threshold]

        peak_hour = int(np.argmax(values)) if len(anomaly_hours) > 0 else -1

        records.append({
            "area_id": area_id,
            "anomaly_hours": anomaly_hours,
            "n_anomaly_hours": len(anomaly_hours),
            "peak_anomaly_hour": peak_hour,
        })

    return pd.DataFrame(records)


def build_contextual_indicator(
    tx_area_agg: pd.DataFrame,
    anomaly_hours_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gabungkan semua komponen menjadi Contextual Urban Activity Indicator
    per area — siap untuk diintegrasi di Stage 4.

    Returns
    -------
    pd.DataFrame dengan kolom kunci:
        area_id, area_lat, area_lon, area_name,
        transaction_irregularity_score,
        n_anomaly_hours,
        contextual_activity_score (composite 0-1)
    """
    if tx_area_agg.empty:
        return pd.DataFrame()

    df = tx_area_agg.copy()

    if not anomaly_hours_df.empty:
        df = df.merge(anomaly_hours_df[["area_id", "n_anomaly_hours", "peak_anomaly_hour"]],
                      on="area_id", how="left")
        df["n_anomaly_hours"] = df["n_anomaly_hours"].fillna(0)
    else:
        df["n_anomaly_hours"] = 0
        df["peak_anomaly_hour"] = -1

    # Contextual activity score: gabungan irregularity + temporal anomali
    max_anom_hours = df["n_anomaly_hours"].max()
    temporal_component = df["n_anomaly_hours"] / max_anom_hours if max_anom_hours > 0 else 0

    irregularity_component = df.get("transaction_irregularity_score", pd.Series(0, index=df.index))

    df["contextual_activity_score"] = (
        0.60 * irregularity_component +
        0.40 * temporal_component
    ).clip(0, 1)

    cols = [
        "area_id", "area_lat", "area_lon", "area_name",
        "total_transactions", "avg_amount", "fraud_rate",
        "transaction_intensity", "transaction_irregularity_score",
        "n_anomaly_hours", "peak_anomaly_hour",
        "contextual_activity_score",
    ]
    cols = [c for c in cols if c in df.columns]

    logger.info(
        f"Contextual indicator siap: {len(df)} area, "
        f"avg contextual score: {df['contextual_activity_score'].mean():.3f}"
    )

    return df[cols]
