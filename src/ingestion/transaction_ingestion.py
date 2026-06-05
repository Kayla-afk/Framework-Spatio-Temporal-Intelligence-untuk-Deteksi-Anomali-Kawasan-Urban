"""
Transaction Data Ingestion (Proxy Dataset)
-------------------------------------------
Menggunakan IEEE-CIS Fraud Detection Dataset sebagai proxy untuk
pola perilaku transaksi digital urban.

Posisi metodologi:
    Dataset ini tidak merepresentasikan transaksi Surabaya secara literal.
    Yang digunakan adalah distribusi temporal dan behavioral pattern-nya
    sebagai konteks aktivitas digital kawasan urban.

Sumber dataset:
    IEEE-CIS: https://www.kaggle.com/competitions/ieee-fraud-detection/data
    Alternatif: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Format yang diexpect (IEEE-CIS):
    File: train_transaction.csv
    Kolom minimal: TransactionID, TransactionDT, TransactionAmt, isFraud

Format alternatif (Credit Card Fraud):
    File: creditcard.csv
    Kolom minimal: Time, Amount, Class
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping dari format Credit Card Fraud (Kaggle alternatif) ke format internal
CREDITCARD_COLUMN_MAP = {
    "Time": "TransactionDT",
    "Amount": "TransactionAmt",
    "Class": "isFraud",
}


def detect_dataset_format(filepath: str) -> str:
    """
    Deteksi otomatis format dataset berdasarkan nama kolom.

    Returns: 'ieee_cis', 'creditcard', atau 'unknown'
    """
    try:
        header = pd.read_csv(filepath, nrows=0)
        cols = set(header.columns)

        if "TransactionDT" in cols and "TransactionAmt" in cols:
            return "ieee_cis"
        elif "Time" in cols and "Amount" in cols and "Class" in cols:
            return "creditcard"
        else:
            return "unknown"
    except Exception:
        return "unknown"


def load_transaction_dataset(raw_dir: str) -> pd.DataFrame:
    """
    Load transaction dataset dari raw_dir. Deteksi format otomatis.

    Mencari file berikut secara berurutan:
        1. train_transaction.csv (IEEE-CIS)
        2. creditcard.csv (Credit Card Fraud Kaggle)

    Returns
    -------
    pd.DataFrame dengan kolom standar internal:
        transaction_dt, amount, is_fraud
    """
    candidates = [
        ("train_transaction.csv", "ieee_cis"),
        ("creditcard.csv", "creditcard"),
    ]

    for filename, expected_fmt in candidates:
        fpath = os.path.join(raw_dir, filename)
        if os.path.exists(fpath):
            logger.info(f"Ditemukan file: {filename} (format: {expected_fmt})")
            return _load_and_normalize(fpath, expected_fmt)

    logger.warning(
        f"Tidak ada file transaction dataset ditemukan di {raw_dir}.\n"
        "Download salah satu dari:\n"
        "  - IEEE-CIS: https://www.kaggle.com/competitions/ieee-fraud-detection\n"
        "    (rename train_transaction.csv, letakkan di data/raw/transactions/)\n"
        "  - Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "    (file creditcard.csv, letakkan di data/raw/transactions/)\n"
        "Menggunakan synthetic transaction data sebagai fallback."
    )
    return pd.DataFrame()


def _load_and_normalize(filepath: str, fmt: str) -> pd.DataFrame:
    """Load dan normalisasi ke format internal."""
    try:
        if fmt == "ieee_cis":
            # IEEE-CIS bisa sangat besar, ambil subset yang relevan
            df = pd.read_csv(
                filepath,
                usecols=lambda c: c in ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"],
                dtype={"TransactionAmt": float, "isFraud": float},
            )
            df = df.rename(columns={
                "TransactionDT": "transaction_dt",
                "TransactionAmt": "amount",
                "isFraud": "is_fraud",
            })

        elif fmt == "creditcard":
            df = pd.read_csv(filepath, usecols=["Time", "Amount", "Class"])
            df = df.rename(columns={
                "Time": "transaction_dt",
                "Amount": "amount",
                "Class": "is_fraud",
            })
        else:
            return pd.DataFrame()

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

        logger.info(f"Loaded {len(df):,} transaksi dari {os.path.basename(filepath)}")
        return df

    except Exception as e:
        logger.error(f"Gagal load {filepath}: {e}")
        return pd.DataFrame()


def assign_spatial_context(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_areas: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign lokasi spasial simulasi ke setiap transaksi.

    Karena dataset transaksi tidak memiliki koordinat GPS,
    kita membuat grid area urban dan assign transaksi secara
    probabilistik — distribusi tidak seragam untuk mensimulasikan
    konsentrasi aktivitas di pusat kota vs pinggiran.

    Parameters
    ----------
    n_areas : int
        Jumlah zona urban yang disimulasikan.
    seed : int
        Random seed untuk reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom tambahan: area_id, area_lat, area_lon, area_name
    """
    if df.empty:
        return df

    rng = np.random.default_rng(seed)

    # Generate zona urban — lebih padat di tengah (distribusi normal)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    area_lats = np.clip(
        rng.normal(center_lat, (lat_max - lat_min) * 0.25, n_areas),
        lat_min, lat_max,
    )
    area_lons = np.clip(
        rng.normal(center_lon, (lon_max - lon_min) * 0.25, n_areas),
        lon_min, lon_max,
    )

    # Analytical urban zones generated inside the Surabaya metropolitan
    # bounding box. The zones are synthetic spatial units used for
    # contextual integration of mobility and transaction datasets.

    area_names = [
      f"Surabaya Analytical Zone {i+1:02d}"
      for i in range(n_areas)
    ]

    # Assign area ke transaksi — probabilitas lebih tinggi untuk area pusat
    # menggunakan jarak ke center sebagai weight
    distances = np.sqrt((area_lats - center_lat) ** 2 + (area_lons - center_lon) ** 2)
    weights = 1.0 / (distances + 0.01)
    weights = weights / weights.sum()

    area_ids = rng.choice(n_areas, size=len(df), p=weights)

    df = df.copy()
    df["area_id"] = area_ids
    df["area_lat"] = area_lats[area_ids]
    df["area_lon"] = area_lons[area_ids]
    df["area_name"] = [area_names[i] for i in area_ids]

    return df


def build_temporal_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun fitur temporal dari kolom transaction_dt.

    transaction_dt pada IEEE-CIS adalah unix timestamp relatif (detik dari
    referensi awal). Pada Credit Card Fraud, Time adalah detik dari transaksi pertama.

    Fitur yang dihasilkan:
        hour_of_day, transaction_count_per_hour_area, amount_zscore_per_area
    """
    if df.empty:
        return df

    df = df.copy()

    # Konversi ke jam dalam sehari (modulo 86400 detik)
    df["hour_of_day"] = (df["transaction_dt"] % 86400) // 3600
    df["hour_of_day"] = df["hour_of_day"].astype(int)

    # Hitung volume transaksi per area per jam
    volume = (
        df.groupby(["area_id", "hour_of_day"])
        .size()
        .reset_index(name="transaction_count")
    )
    df = df.merge(volume, on=["area_id", "hour_of_day"], how="left")

    # Z-score amount per area (deteksi spike nilai transaksi)
    df["amount_mean_area"] = df.groupby("area_id")["amount"].transform("mean")
    df["amount_std_area"] = df.groupby("area_id")["amount"].transform("std").fillna(1)
    df["amount_zscore"] = (df["amount"] - df["amount_mean_area"]) / df["amount_std_area"]

    return df


def aggregate_transaction_by_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction features ke level area untuk integrasi dengan
    hasil analisis mobilitas.

    Returns
    -------
    pd.DataFrame dengan satu baris per area:
        area_id, area_lat, area_lon, area_name,
        total_transactions, avg_amount, fraud_rate,
        peak_hour, transaction_intensity, temporal_irregularity
    """
    if df.empty:
        return pd.DataFrame()

    agg_spec = {
        "total_transactions": ("amount", "count"),
        "avg_amount": ("amount", "mean"),
        "fraud_rate": ("is_fraud", "mean"),
    }

    if "hour_of_day" in df.columns:
        agg_spec["peak_hour"] = ("hour_of_day", lambda x: x.mode().iloc[0] if len(x) > 0 else 0)

    if "amount_zscore" in df.columns:
        agg_spec["amount_zscore_max"] = ("amount_zscore", "max")

    if "transaction_count" in df.columns:
        agg_spec["transaction_count_max"] = ("transaction_count", "max")

    agg = df.groupby(["area_id", "area_lat", "area_lon", "area_name"]).agg(
        **agg_spec
    ).reset_index()

    # Transaction intensity: normalisasi volume relatif terhadap semua area
    max_tx = agg["total_transactions"].max()
    agg["transaction_intensity"] = agg["total_transactions"] / max_tx if max_tx > 0 else 0

    # Temporal irregularity: std volume transaksi per jam per area
    if "hour_of_day" in df.columns:
        hourly_counts = (
            df.groupby(["area_id", "hour_of_day"])
            .size()
            .reset_index(name="_hourly_count")
        )
        temporal_std = (
            hourly_counts.groupby("area_id")["_hourly_count"]
            .std()
            .fillna(0)
            .reset_index(name="temporal_irregularity")
        )
    else:
        temporal_std = agg[["area_id"]].copy()
        temporal_std["temporal_irregularity"] = 0.0

    # Normalisasi 0-1
    max_irr = temporal_std["temporal_irregularity"].max()
    if max_irr > 0:
        temporal_std["temporal_irregularity"] /= max_irr

    agg = agg.merge(temporal_std, on="area_id", how="left")
    agg["temporal_irregularity"] = agg["temporal_irregularity"].fillna(0)

    return agg
