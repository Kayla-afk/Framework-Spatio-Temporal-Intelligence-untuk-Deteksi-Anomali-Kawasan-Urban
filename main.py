"""
Urban Anomaly Intelligence Framework - Main Pipeline
=====================================================
Framework Spatio-Temporal Intelligence untuk Deteksi Anomali Kawasan Urban
Berbasis Data Mobilitas dan Aktivitas Transaksi Digital

Studi Kasus: Metropolitan Surabaya

Jalankan:
    python main.py                    # mode lengkap (butuh dataset asli)
    python main.py --synthetic        # mode demo dengan data sintetis
    python main.py --synthetic --help # lihat semua opsi

Dataset yang dibutuhkan (jika tidak menggunakan --synthetic):
    Mobilitas:
        GeoLife GPS Dataset (Microsoft Research)
        Download: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
        Ekstrak ke: data/raw/mobility/

    Transaksi:
        IEEE-CIS Fraud Detection Dataset (Kaggle)
        Download: https://www.kaggle.com/competitions/ieee-fraud-detection
        File: train_transaction.csv -> letakkan di data/raw/transactions/

        Alternatif: Credit Card Fraud Dataset
        Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        File: creditcard.csv -> letakkan di data/raw/transactions/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Setup path agar bisa import dari src/
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.mobility_ingestion import (
    load_geolife,
    remap_to_study_area,
    compute_traffic_features,
)
from src.ingestion.transaction_ingestion import (
    load_transaction_dataset,
    assign_spatial_context,
    build_temporal_transaction_features,
    aggregate_transaction_by_area,
)
from src.ingestion.synthetic_generator import (
    generate_synthetic_mobility,
    generate_synthetic_transactions,
)
from src.clustering.hdbscan_clustering import (
    run_hdbscan_clustering,
    compute_cluster_summary,
    extract_baseline_mobility_pattern,
)
from src.anomaly.isolation_forest import (
    run_isolation_forest,
    aggregate_anomaly_zones,
)
from src.transaction.contextualization import (
    compute_transaction_irregularity,
    compute_temporal_transaction_profile,
    identify_transaction_anomaly_hours,
    build_contextual_indicator,
)
from src.integration.urban_intelligence import (
    spatial_join_mobility_transaction,
    compute_urban_anomaly_score,
    identify_priority_monitoring_areas,
    generate_anomaly_correlation_pattern,
)
from src.visualization.map_renderer import (
    create_interactive_anomaly_map,
    create_static_anomaly_figures,
    save_priority_areas_report,
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Urban Anomaly Intelligence Framework — Metropolitan Surabaya",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Jalankan pipeline dengan synthetic data (tidak perlu download dataset)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path ke file konfigurasi (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Level logging (default: INFO)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualisasi (jalankan hanya analisis)",
    )
    return parser.parse_args()


def run_stage_1_mobility_mapping(cfg: dict, use_synthetic: bool) -> tuple:
    """
    Stage 1: Urban Mobility Mapping
    Tujuan: Memahami pola mobilitas kawasan urban
    Output: Mobility clusters, baseline movement pattern, congestion hotspot
    """
    print("\n[Stage 1] Urban Mobility Mapping")
    print("-" * 50)

    area = cfg["study_area"]

    if use_synthetic:
        mobility_df = generate_synthetic_mobility(
            lat_min=area["lat_min"],
            lat_max=area["lat_max"],
            lon_min=area["lon_min"],
            lon_max=area["lon_max"],
        )
    else:
        geolife_cfg = cfg["geolife"]
        raw_dir = os.path.join(ROOT_DIR, cfg["data"]["mobility_raw_dir"])

        mobility_df = load_geolife(
            raw_dir=raw_dir,
            max_trajectories=geolife_cfg["max_trajectories"],
            min_points=geolife_cfg["min_points_per_trajectory"],
        )

        if mobility_df.empty:
            print(
                "\nDataset GeoLife tidak ditemukan. Switching ke synthetic mode.\n"
                "Untuk menggunakan data asli, download dari:\n"
                "  https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/\n"
                "Dan ekstrak ke: data/raw/mobility/\n"
            )
            mobility_df = generate_synthetic_mobility(
                lat_min=area["lat_min"],
                lat_max=area["lat_max"],
                lon_min=area["lon_min"],
                lon_max=area["lon_max"],
            )
        else:
            # Remap koordinat ke bounding box Surabaya
            mobility_df = remap_to_study_area(
                mobility_df,
                lat_min=area["lat_min"],
                lat_max=area["lat_max"],
                lon_min=area["lon_min"],
                lon_max=area["lon_max"],
            )

    # Hitung traffic features
    mobility_df = compute_traffic_features(mobility_df)

    # HDBSCAN clustering
    hdb_cfg = cfg["hdbscan"]
    mobility_df = run_hdbscan_clustering(
        mobility_df,
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        cluster_selection_epsilon=hdb_cfg["cluster_selection_epsilon"],
        metric=hdb_cfg["metric"],
    )

    # Cluster summary dan hotspot identification
    cluster_summary = compute_cluster_summary(
        mobility_df,
        hotspot_percentile=hdb_cfg["hotspot_density_percentile"],
    )

    # Baseline mobility pattern
    baseline = extract_baseline_mobility_pattern(mobility_df, cluster_summary)

    print(f"  GPS points: {len(mobility_df):,}")
    print(f"  Clusters: {len(cluster_summary)}")
    print(f"  Hotspots: {cluster_summary['is_hotspot'].sum() if not cluster_summary.empty else 0}")
    print(f"  Mobility entropy: {baseline.get('mobility_entropy', 0):.3f}")

    return mobility_df, cluster_summary, baseline


def run_stage_2_anomaly_detection(
    mobility_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cfg: dict,
) -> tuple:
    """
    Stage 2: Spatio-Temporal Anomaly Detection
    Tujuan: Deteksi perubahan abnormal pada pola mobilitas
    Output: Anomalous mobility zones, abnormal congestion pattern
    """
    print("\n[Stage 2] Spatio-Temporal Anomaly Detection")
    print("-" * 50)

    if_cfg = cfg["isolation_forest"]

    # Isolation Forest
    mobility_with_anomaly = run_isolation_forest(
        mobility_df,
        cluster_summary=cluster_summary,
        contamination=if_cfg["contamination"],
        n_estimators=if_cfg["n_estimators"],
        random_state=if_cfg["random_state"],
    )

    # Agregasi ke level zona
    anomaly_zones = aggregate_anomaly_zones(
        mobility_with_anomaly,
        grid_resolution=0.01,
    )

    n_anomalies = (mobility_with_anomaly["anomaly_prediction"] == -1).sum()
    n_anomaly_zones = anomaly_zones["is_anomalous_zone"].sum() if not anomaly_zones.empty else 0

    print(f"  Anomaly points: {n_anomalies:,} ({n_anomalies/len(mobility_with_anomaly)*100:.1f}%)")
    print(f"  Anomalous zones: {n_anomaly_zones}")

    return mobility_with_anomaly, anomaly_zones


def run_stage_3_transaction_context(cfg: dict, use_synthetic: bool) -> tuple:
    """
    Stage 3: Transaction Contextualization
    Tujuan: Analisis aktivitas transaksi digital sebagai konteks anomali kawasan
    Output: Contextual urban activity indicator per area
    """
    print("\n[Stage 3] Transaction Contextualization")
    print("-" * 50)

    area = cfg["study_area"]
    tx_cfg = cfg["transaction"]

    if use_synthetic:
        tx_df_raw = generate_synthetic_transactions(
            lat_min=area["lat_min"],
            lat_max=area["lat_max"],
            lon_min=area["lon_min"],
            lon_max=area["lon_max"],
        )
        # synthetic generator sudah include area coords
    else:
        raw_dir = os.path.join(ROOT_DIR, cfg["data"]["transaction_raw_dir"])
        tx_df_raw = load_transaction_dataset(raw_dir)

        if tx_df_raw.empty:
            print(
                "\nDataset transaksi tidak ditemukan. Switching ke synthetic mode.\n"
                "Untuk data asli, download IEEE-CIS atau Credit Card Fraud dari Kaggle.\n"
            )
            tx_df_raw = generate_synthetic_transactions(
                lat_min=area["lat_min"],
                lat_max=area["lat_max"],
                lon_min=area["lon_min"],
                lon_max=area["lon_max"],
            )
        else:
            # Assign spatial context ke dataset asli
            tx_df_raw = assign_spatial_context(
                tx_df_raw,
                lat_min=area["lat_min"],
                lat_max=area["lat_max"],
                lon_min=area["lon_min"],
                lon_max=area["lon_max"],
                n_areas=tx_cfg["n_urban_areas"],
                seed=tx_cfg["random_seed"],
            )
            tx_df_raw = build_temporal_transaction_features(tx_df_raw)

    # Aggregate ke level area
    tx_area_agg = aggregate_transaction_by_area(tx_df_raw)

    # Hitung irregularity
    ta_cfg = cfg["transaction_analysis"]
    tx_area_agg = compute_transaction_irregularity(
        tx_area_agg,
        zscore_threshold=ta_cfg["zscore_threshold"],
    )

    # Temporal profile dan anomaly hours
    temporal_profile = compute_temporal_transaction_profile(tx_df_raw)
    anomaly_hours_df = identify_transaction_anomaly_hours(
        temporal_profile,
        zscore_threshold=ta_cfg["zscore_threshold"],
    )

    # Build contextual indicator
    contextual_indicator = build_contextual_indicator(tx_area_agg, anomaly_hours_df)

    print(f"  Transactions: {len(tx_df_raw):,}")
    print(f"  Areas: {len(contextual_indicator)}")
    print(f"  Flagged areas: {tx_area_agg.get('irregularity_flag', pd.Series()).sum() if not tx_area_agg.empty else 0}")

    return tx_df_raw, contextual_indicator


def run_stage_4_integration(
    anomaly_zones: pd.DataFrame,
    contextual_indicator: pd.DataFrame,
    cfg: dict,
) -> tuple:
    """
    Stage 4: Urban Anomaly Intelligence
    Tujuan: Integrasi mobilitas + transaksi -> kawasan prioritas monitoring
    Output: Urban anomaly map, kawasan prioritas monitoring
    """
    print("\n[Stage 4] Urban Anomaly Intelligence Integration")
    print("-" * 50)

    int_cfg = cfg["integration"]

    # Spatial join
    integrated = spatial_join_mobility_transaction(
        anomaly_zones,
        contextual_indicator,
        max_distance_deg=cfg["integration"]["spatial_join_threshold"],
    )

    # Composite scoring
    scored_zones = compute_urban_anomaly_score(
        integrated,
        weight_mobility=int_cfg["weight_mobility_anomaly"],
        weight_density=int_cfg["weight_spatial_cluster_density"],
        weight_transaction=int_cfg["weight_transaction_irregularity"],
    )

    # Kawasan prioritas
    priority_areas = identify_priority_monitoring_areas(scored_zones, top_n=10)

    # Correlation stats
    correlation_stats = generate_anomaly_correlation_pattern(scored_zones)

    print(f"  Integrated zones: {len(scored_zones)}")
    print(f"  HIGH risk zones: {(scored_zones['risk_level']=='HIGH').sum()}")
    print(f"  Priority areas identified: {len(priority_areas)}")
    print(f"  Mobility-Transaction correlation: r={correlation_stats.get('pearson_r', 0):.3f}")

    return scored_zones, priority_areas, correlation_stats


def main():
    args = parse_args()
    setup_logging(args.log_level)

    print("=" * 60)
    print("Urban Anomaly Intelligence Framework")
    print("Studi Kasus: Metropolitan Surabaya")
    print("=" * 60)

    # Load config
    config_path = os.path.join(ROOT_DIR, args.config)
    cfg = load_config(config_path)

    mode = "SYNTHETIC DATA" if args.synthetic else "REAL DATASET"
    print(f"\nMode: {mode}")
    print(f"Study Area: {cfg['study_area']['name']}")

    start_time = time.time()

    # ---- Stage 1 ----
    mobility_df, cluster_summary, baseline = run_stage_1_mobility_mapping(
        cfg, use_synthetic=args.synthetic
    )

    # ---- Stage 2 ----
    mobility_with_anomaly, anomaly_zones = run_stage_2_anomaly_detection(
        mobility_df, cluster_summary, cfg
    )

    # ---- Stage 3 ----
    tx_df, contextual_indicator = run_stage_3_transaction_context(
        cfg, use_synthetic=args.synthetic
    )

    # ---- Stage 4 ----
    scored_zones, priority_areas, correlation_stats = run_stage_4_integration(
        anomaly_zones, contextual_indicator, cfg
    )

    # ---- Visualisasi ----
    if not args.skip_viz:
        print("\n[Visualization] Generating outputs...")
        print("-" * 50)

        area = cfg["study_area"]
        out_cfg = cfg["output"]

        # Interactive HTML map
        map_path = create_interactive_anomaly_map(
            scored_zones=scored_zones,
            mobility_data=mobility_with_anomaly,
            cluster_summary=cluster_summary,
            priority_areas=priority_areas,
            center_lat=area["reference_lat"],
            center_lon=area["reference_lon"],
            output_path=os.path.join(ROOT_DIR, out_cfg["maps_dir"], "urban_anomaly_map.html"),
        )

        # Static figures
        figure_paths = create_static_anomaly_figures(
            scored_zones=scored_zones,
            mobility_data=mobility_with_anomaly,
            cluster_summary=cluster_summary,
            priority_areas=priority_areas,
            correlation_stats=correlation_stats,
            center_lat=area["reference_lat"],
            center_lon=area["reference_lon"],
            output_dir=os.path.join(ROOT_DIR, out_cfg["figures_dir"]),
            dpi=out_cfg["figure_dpi"],
        )

        # Priority areas report
        report_path = save_priority_areas_report(
            priority_areas=priority_areas,
            correlation_stats=correlation_stats,
            output_path=os.path.join(ROOT_DIR, out_cfg["reports_dir"], "priority_areas_report.html"),
        )

        print(f"\n  Interactive map: {map_path}")
        for fp in figure_paths:
            print(f"  Figure: {fp}")
        print(f"  Report: {report_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline selesai dalam {elapsed:.1f} detik")
    print(f"{'='*60}")

    # Print ringkasan kawasan prioritas
    if not priority_areas.empty:
        print("\nKawasan Prioritas Monitoring (Top 10):")
        print("-" * 50)
        display_cols = ["monitoring_rank", "risk_level", "area_name",
                        "priority_score", "urban_anomaly_score"]
        display_cols = [c for c in display_cols if c in priority_areas.columns]
        print(priority_areas[display_cols].to_string(index=False))

    print(f"\nAnalytic Summary:")
    print(f"  Total zones analyzed  : {correlation_stats.get('total_zones', 0)}")
    print(f"  HIGH risk zones       : {correlation_stats.get('high_risk_zones', 0)}")
    print(f"  Double signal rate    : {correlation_stats.get('double_signal_rate', 0):.1%}")
    print(f"  Spatial concentration : Gini = {correlation_stats.get('spatial_concentration_gini', 0):.3f}")


if __name__ == "__main__":
    main()
