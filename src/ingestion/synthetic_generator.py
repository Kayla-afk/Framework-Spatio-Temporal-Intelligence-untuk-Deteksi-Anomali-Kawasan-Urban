"""
Synthetic Data Generator
-------------------------
Fallback generator untuk menjalankan pipeline ketika dataset asli
(GeoLife / IEEE-CIS) belum didownload.

Data yang digenerate mencerminkan karakteristik statistik yang realistis
untuk kawasan metropolitan:
    - Pola pergerakan dengan rush hour pagi dan sore
    - Hotspot konsentrasi tinggi di pusat kota dan koridor utama
    - Anomali mobilitas yang disengaja untuk demonstrasi deteksi
    - Pola transaksi dengan spike temporal

CATATAN: Data ini hanya untuk keperluan demonstrasi framework.
Untuk analisis riil, gunakan dataset asli yang tertera di README.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_mobility(
    lat_min: float = -7.4,
    lat_max: float = -7.1,
    lon_min: float = 112.6,
    lon_max: float = 112.85,
    n_trajectories: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic GPS trajectory data yang mensimulasikan
    pola mobilitas urban Surabaya.

    Pola yang disimulasikan:
        - Koridor utama (Jl. A. Yani, Protokol Tol)
        - Hotspot pusat kota (Tunjungan, Gubeng)
        - Rush hour 07:00-09:00 dan 16:00-19:00
        - Anomali: kepadatan tidak wajar di beberapa zona malam hari
    """
    rng = np.random.default_rng(seed)

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Definisi koridor dan hotspot berbobot
    # Format: (center_lat, center_lon, spread_lat, spread_lon, weight)
    movement_zones = [
        # Pusat kota - kepadatan tinggi
        (center_lat + 0.02, center_lon + 0.02, 0.02, 0.02, 0.25),
        # Koridor barat-timur (simulasi jalan arteri)
        (center_lat, center_lon, 0.04, 0.08, 0.20),
        # Zona selatan (Wonokromo - terminal)
        (center_lat - 0.08, center_lon - 0.01, 0.02, 0.02, 0.15),
        # Zona industri barat
        (center_lat + 0.05, center_lon - 0.08, 0.03, 0.03, 0.10),
        # Zona timur (Rungkut - kawasan industri)
        (center_lat - 0.05, center_lon + 0.08, 0.03, 0.03, 0.10),
        # Zona suburban tersebar
        (center_lat - 0.10, center_lon + 0.05, 0.05, 0.05, 0.20),
    ]

    records = []
    traj_id = 0

    for _ in range(n_trajectories):
        # Pilih zona asal dan tujuan
        zone_weights = np.array([z[4] for z in movement_zones])
        zone_weights /= zone_weights.sum()

        origin_zone = movement_zones[rng.choice(len(movement_zones), p=zone_weights)]
        dest_zone = movement_zones[rng.choice(len(movement_zones), p=zone_weights)]

        # Jumlah titik per trajectory
        n_points = rng.integers(50, 200)

        # Titik asal dan tujuan
        start_lat = rng.normal(origin_zone[0], origin_zone[2])
        start_lon = rng.normal(origin_zone[1], origin_zone[3])
        end_lat = rng.normal(dest_zone[0], dest_zone[2])
        end_lon = rng.normal(dest_zone[1], dest_zone[3])

        # Interpolasi dengan sedikit noise (simulasi jalan tidak lurus)
        t = np.linspace(0, 1, n_points)
        lats = start_lat + t * (end_lat - start_lat) + rng.normal(0, 0.002, n_points)
        lons = start_lon + t * (end_lon - start_lon) + rng.normal(0, 0.002, n_points)

        # Clip ke bounding box
        lats = np.clip(lats, lat_min, lat_max)
        lons = np.clip(lons, lon_min, lon_max)

        # Timestamp: distribusi rush hour
        # 30% pagi, 30% sore, 40% tersebar
        hour_roll = rng.random()
        if hour_roll < 0.30:
            base_hour = rng.integers(7, 10)  # rush pagi
        elif hour_roll < 0.60:
            base_hour = rng.integers(16, 20)  # rush sore
        else:
            base_hour = rng.integers(0, 24)   # tersebar

        # Durasi perjalanan 5-45 menit
        duration_sec = rng.integers(300, 2700)
        base_ts = pd.Timestamp("2013-01-01") + pd.Timedelta(hours=int(base_hour))
        timestamps = [base_ts + pd.Timedelta(seconds=int(s)) for s in np.linspace(0, duration_sec, n_points)]

        for i in range(n_points):
            records.append({
                "latitude": lats[i],
                "longitude": lons[i],
                "altitude": rng.uniform(0, 50),
                "timestamp": timestamps[i],
                "trajectory_id": f"traj_{traj_id:04d}",
                "user_id": f"user_{traj_id // 3:03d}",
            })

        traj_id += 1

    # Injeksi anomali: konsentrasi tidak wajar di zona tertentu tengah malam
    n_anomaly_traj = 20
    anomaly_center_lat = lat_min + 0.05
    anomaly_center_lon = lon_min + 0.05

    for i in range(n_anomaly_traj):
        n_pts = 80
        lats = rng.normal(anomaly_center_lat, 0.005, n_pts)
        lons = rng.normal(anomaly_center_lon, 0.005, n_pts)
        base_ts = pd.Timestamp("2013-01-01 02:00:00")
        timestamps = [base_ts + pd.Timedelta(seconds=int(s)) for s in np.linspace(0, 600, n_pts)]

        for j in range(n_pts):
            records.append({
                "latitude": float(np.clip(lats[j], lat_min, lat_max)),
                "longitude": float(np.clip(lons[j], lon_min, lon_max)),
                "altitude": 5.0,
                "timestamp": timestamps[j],
                "trajectory_id": f"traj_anomaly_{i:03d}",
                "user_id": f"user_anomaly_{i:02d}",
            })

        traj_id += 1

    df = pd.DataFrame(records)
    logger.info(
        f"[SYNTHETIC] Generated {n_trajectories + n_anomaly_traj} trajectories, "
        f"{len(df):,} GPS points (termasuk {n_anomaly_traj} trajectory anomali)."
    )
    return df


def generate_synthetic_transactions(
    lat_min: float = -7.4,
    lat_max: float = -7.1,
    lon_min: float = 112.6,
    lon_max: float = 112.85,
    n_transactions: int = 50000,
    n_areas: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data untuk demonstrasi pipeline.

    Mensimulasikan pola transaksi digital urban:
        - Volume lebih tinggi di jam kerja dan malam (commerce)
        - Spike anomali pada beberapa area dan waktu tertentu
        - Fraud rate lebih tinggi pada transaksi nilai besar malam hari
    """
    rng = np.random.default_rng(seed)

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    area_names = [
        "Pusat Kota", "Gubeng", "Wonokromo", "Rungkut", "Mulyorejo",
        "Kenjeran", "Benowo", "Lakarsantri", "Genteng", "Tegalsari",
        "Simokerto", "Semampir", "Pabean Cantian", "Bubutan", "Krembangan",
        "Sawahan", "Dukuh Pakis", "Wiyung", "Gayungan", "Jambangan",
    ][:n_areas]

    # Generate area centers dengan distribusi normal dari pusat kota
    area_lats = np.clip(rng.normal(center_lat, 0.06, n_areas), lat_min, lat_max)
    area_lons = np.clip(rng.normal(center_lon, 0.06, n_areas), lon_min, lon_max)

    # Weight area berdasarkan jarak ke center (pusat lebih ramai)
    dist_to_center = np.sqrt((area_lats - center_lat) ** 2 + (area_lons - center_lon) ** 2)
    area_weights = 1.0 / (dist_to_center + 0.01)
    area_weights /= area_weights.sum()

    # Assign area ke transaksi
    area_ids = rng.choice(n_areas, size=n_transactions, p=area_weights)

    # Timestamp: unix seconds, simulasi 30 hari
    total_seconds = 30 * 24 * 3600
    transaction_dt = rng.integers(0, total_seconds, n_transactions)
    hour_of_day = (transaction_dt % 86400) // 3600

    # Amount: lognormal, spike di beberapa area
    base_amount = rng.lognormal(mean=3.5, sigma=1.2, size=n_transactions)
    # 5% transaksi bernilai sangat tinggi (potensi fraud)
    high_value_mask = rng.random(n_transactions) < 0.05
    base_amount[high_value_mask] *= rng.uniform(5, 20, high_value_mask.sum())

    # Fraud: lebih mungkin pada transaksi nilai besar di malam hari
    fraud_prob = 0.02 + 0.05 * (base_amount > np.percentile(base_amount, 90)).astype(float)
    fraud_prob += 0.03 * ((hour_of_day >= 22) | (hour_of_day <= 4)).astype(float)
    is_fraud = (rng.random(n_transactions) < fraud_prob).astype(int)

    df = pd.DataFrame({
        "transaction_dt": transaction_dt,
        "amount": base_amount,
        "is_fraud": is_fraud,
        "area_id": area_ids,
        "area_lat": area_lats[area_ids],
        "area_lon": area_lons[area_ids],
        "area_name": [area_names[i] for i in area_ids],
        "hour_of_day": hour_of_day,
    })

    logger.info(
        f"[SYNTHETIC] Generated {len(df):,} transaksi di {n_areas} area. "
        f"Fraud rate: {is_fraud.mean():.1%}"
    )
    return df
