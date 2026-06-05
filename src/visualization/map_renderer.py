"""
Visualization Module
---------------------
Menghasilkan output visual dari Urban Anomaly Intelligence Framework:

1. Interactive HTML map (Folium):
    - Urban anomaly heatmap dengan color gradient
    - Marker untuk kawasan prioritas monitoring
    - Layer toggle: mobility clusters, anomaly zones, transaction context
    - Popup informasi per zona

2. Static figure (Matplotlib/Seaborn):
    - Urban anomaly map untuk laporan/portofolio
    - Congestion anomaly hotspot scatter plot
    - Temporal pattern comparison chart
    - Risk distribution chart

Semua output disimpan ke outputs/maps/ dan outputs/figures/
"""

import logging
import os
from typing import Optional

import folium
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster

logger = logging.getLogger(__name__)

# Warna konsisten untuk risk level
RISK_COLORS = {
    "HIGH": "#d62728",    # merah
    "MEDIUM": "#ff7f0e",  # oranye
    "LOW": "#2ca02c",     # hijau
}

RISK_COLORS_HEX_FOLIUM = {
    "HIGH": "red",
    "MEDIUM": "orange",
    "LOW": "green",
}


def create_interactive_anomaly_map(
    scored_zones: pd.DataFrame,
    mobility_data: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    priority_areas: pd.DataFrame,
    center_lat: float = -7.2575,
    center_lon: float = 112.7521,
    output_path: str = "outputs/maps/urban_anomaly_map.html",
) -> str:
    """
    Buat interactive HTML map menggunakan Folium.

    Layers yang disertakan:
        1. Heatmap kepadatan GPS (mobility base)
        2. Anomaly zone overlay (grid colormap)
        3. Cluster hotspot markers
        4. Priority monitoring markers
        5. Transaction context overlay

    Returns
    -------
    str : path ke file HTML yang dihasilkan
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Inisialisasi map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # --- Layer 1: GPS Mobility Heatmap ---
    if not mobility_data.empty and "latitude" in mobility_data.columns:
        # Sample untuk performa (max 20k points di heatmap)
        sample_size = min(20000, len(mobility_data))
        mob_sample = mobility_data.sample(sample_size, random_state=42)

        heat_data = mob_sample[["latitude", "longitude"]].values.tolist()
        heatmap_layer = folium.FeatureGroup(name="Mobility Density (GPS)", show=True)
        HeatMap(
            heat_data,
            radius=10,
            blur=15,
            max_zoom=14,
            gradient={"0.2": "blue", "0.5": "lime", "0.8": "yellow", "1.0": "red"},
        ).add_to(heatmap_layer)
        heatmap_layer.add_to(m)

    # --- Layer 2: Anomaly Zone Grid ---
    if not scored_zones.empty:
        anomaly_layer = folium.FeatureGroup(name="Anomaly Zones", show=True)

        for _, row in scored_zones.iterrows():
            score = row.get("urban_anomaly_score", 0)
            risk = row.get("risk_level", "LOW")

            if score < 0.15:  # Abaikan zona dengan score sangat rendah
                continue

            # Warna berdasarkan score (gradient merah)
            opacity = float(np.clip(score * 0.7, 0.1, 0.7))
            color = _score_to_color(score)

            # Grid rectangle (0.01 deg x 0.01 deg)
            grid_size = 0.01
            lat = float(row["grid_lat"])
            lon = float(row["grid_lon"])

            popup_text = (
                f"<b>Urban Anomaly Zone</b><br>"
                f"Risk Level: <b style='color:{RISK_COLORS[risk]}'>{risk}</b><br>"
                f"Anomaly Score: {score:.3f}<br>"
                f"Mobility Anomaly Rate: {row.get('anomaly_rate', 0):.1%}<br>"
                f"Avg Anomaly Score: {row.get('avg_anomaly_score', 0):.3f}<br>"
                f"Transaction Context: {row.get('contextual_activity_score', 0):.3f}<br>"
                f"Area: {row.get('area_name', 'Unknown')}"
            )

            folium.Rectangle(
                bounds=[[lat - grid_size/2, lon - grid_size/2],
                        [lat + grid_size/2, lon + grid_size/2]],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                weight=0,
                popup=folium.Popup(popup_text, max_width=300),
            ).add_to(anomaly_layer)

        anomaly_layer.add_to(m)

    # --- Layer 3: HDBSCAN Cluster Hotspots ---
    if not cluster_summary.empty:
        hotspot_layer = folium.FeatureGroup(name="Mobility Hotspots (HDBSCAN)", show=False)

        for _, row in cluster_summary[cluster_summary.get("is_hotspot", False)].iterrows():
            size = float(np.clip(row.get("density_score", 0) * 20 + 5, 5, 25))
            popup_text = (
                f"<b>Mobility Hotspot</b><br>"
                f"Cluster ID: {int(row['cluster_label'])}<br>"
                f"Point Count: {int(row.get('point_count', 0)):,}<br>"
                f"Density Score: {row.get('density_score', 0):.3f}<br>"
                f"Avg Speed: {row.get('avg_speed_kmh', 0):.1f} km/h<br>"
                f"Avg Congestion: {row.get('avg_congestion', 0):.2f}"
            )

            folium.CircleMarker(
                location=[float(row["centroid_lat"]), float(row["centroid_lon"])],
                radius=size,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.6,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(hotspot_layer)

        hotspot_layer.add_to(m)

    # --- Layer 4: Priority Monitoring Areas ---
    if not priority_areas.empty:
        priority_layer = folium.FeatureGroup(name="Priority Monitoring Areas", show=True)

        for _, row in priority_areas.iterrows():
            risk = row.get("risk_level", "HIGH")
            rank = int(row.get("monitoring_rank", 0))
            score = float(row.get("priority_score", 0))

            popup_text = (
                f"<b>Priority Area #{rank}</b><br>"
                f"Risk Level: <b style='color:{RISK_COLORS[risk]}'>{risk}</b><br>"
                f"Priority Score: {score:.3f}<br>"
                f"Urban Anomaly Score: {row.get('urban_anomaly_score', 0):.3f}<br>"
                f"Anomaly Rate: {row.get('anomaly_rate', 0):.1%}<br>"
                f"Contextual Score: {row.get('contextual_activity_score', 0):.3f}<br>"
                f"Area: {row.get('area_name', 'Unknown')}"
            )

            folium.Marker(
                location=[float(row["grid_lat"]), float(row["grid_lon"])],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"#{rank} {risk} RISK",
                icon=folium.Icon(
                    color=RISK_COLORS_HEX_FOLIUM[risk],
                    icon="exclamation-sign",
                    prefix="glyphicon",
                ),
            ).add_to(priority_layer)

        priority_layer.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Title overlay
    title_html = """
    <div style="
        position: fixed;
        top: 10px; left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        background: rgba(255,255,255,0.92);
        padding: 10px 20px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
        text-align: center;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b style="font-size:14px;">Urban Anomaly Intelligence Map</b><br>
        <span style="font-size:11px; color:#666;">Metropolitan Surabaya | Spatio-Temporal Framework</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Legend
    legend_html = _build_legend_html()
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    logger.info(f"Interactive map disimpan: {output_path}")
    return output_path


def create_static_anomaly_figures(
    scored_zones: pd.DataFrame,
    mobility_data: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    priority_areas: pd.DataFrame,
    correlation_stats: dict,
    center_lat: float = -7.2575,
    center_lon: float = 112.7521,
    output_dir: str = "outputs/figures",
    dpi: int = 150,
) -> list:
    """
    Buat static figures untuk laporan dan portofolio.

    Figures yang dihasilkan:
        1. urban_anomaly_map.png — peta utama dengan scatter anomali
        2. congestion_hotspot.png — congestion anomaly distribution
        3. temporal_pattern.png — pola mobilitas per jam
        4. risk_distribution.png — distribusi risk level

    Returns
    -------
    list: path ke semua file yang dihasilkan
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Figure 1: Urban Anomaly Map (main output) ---
    fig1, ax1 = plt.subplots(figsize=(12, 9))

    if not scored_zones.empty:
        # Background: density dari GPS points
        if not mobility_data.empty:
            ax1.hexbin(
                mobility_data["longitude"],
                mobility_data["latitude"],
                gridsize=40,
                cmap="Blues",
                alpha=0.3,
                mincnt=1,
                label="_nolegend_",
            )

        # Scatter anomaly zones
        high = scored_zones[scored_zones["risk_level"] == "HIGH"]
        med = scored_zones[scored_zones["risk_level"] == "MEDIUM"]
        low = scored_zones[scored_zones["risk_level"] == "LOW"]

        for subset, color, label, alpha, size in [
            (low, RISK_COLORS["LOW"], "Low Risk", 0.4, 30),
            (med, RISK_COLORS["MEDIUM"], "Medium Risk", 0.6, 60),
            (high, RISK_COLORS["HIGH"], "High Risk", 0.85, 100),
        ]:
            if not subset.empty:
                sc = ax1.scatter(
                    subset["grid_lon"],
                    subset["grid_lat"],
                    c=subset["urban_anomaly_score"],
                    cmap="RdYlGn_r",
                    vmin=0, vmax=1,
                    s=size,
                    alpha=alpha,
                    label=label,
                    edgecolors="none",
                )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.02)
        cbar.set_label("Urban Anomaly Score", fontsize=10)

        # Priority markers
        if not priority_areas.empty:
            top5 = priority_areas.head(5)
            ax1.scatter(
                top5["grid_lon"],
                top5["grid_lat"],
                s=200,
                c="black",
                marker="*",
                zorder=5,
                label="Top-5 Priority Monitoring",
            )
            for _, row in top5.iterrows():
                ax1.annotate(
                    f"#{int(row['monitoring_rank'])}",
                    xy=(float(row["grid_lon"]), float(row["grid_lat"])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )

    ax1.set_xlim(112.6, 112.85)
    ax1.set_ylim(-7.4, -7.1)
    ax1.set_xlabel("Longitude", fontsize=11)
    ax1.set_ylabel("Latitude", fontsize=11)
    ax1.set_title(
        "Urban Anomaly Intelligence Map\nMetropolitan Surabaya — Spatio-Temporal Framework",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Statistik ringkas
    if not scored_zones.empty:
        stats_text = (
            f"Total Zones: {len(scored_zones)}\n"
            f"HIGH Risk: {(scored_zones['risk_level']=='HIGH').sum()}\n"
            f"MEDIUM Risk: {(scored_zones['risk_level']=='MEDIUM').sum()}\n"
            f"LOW Risk: {(scored_zones['risk_level']=='LOW').sum()}"
        )
        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig1.tight_layout()
    path1 = os.path.join(output_dir, "urban_anomaly_map.png")
    fig1.savefig(path1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    output_files.append(path1)
    logger.info(f"Saved: {path1}")

    # --- Figure 2: Congestion Anomaly Hotspot ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    if not mobility_data.empty and "congestion_index" in mobility_data.columns:
        # Kiri: Congestion distribution
        ax = axes2[0]
        if "anomaly_prediction" in mobility_data.columns:
            normal = mobility_data[mobility_data["anomaly_prediction"] == 1]["congestion_index"]
            anomaly = mobility_data[mobility_data["anomaly_prediction"] == -1]["congestion_index"]
            ax.hist(normal, bins=40, alpha=0.6, color="#2ca02c", label="Normal", density=True)
            ax.hist(anomaly, bins=40, alpha=0.6, color="#d62728", label="Anomaly", density=True)
            ax.set_xlabel("Congestion Index", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title("Congestion Index Distribution\nNormal vs Anomaly Points", fontsize=12)
            ax.legend(fontsize=10)
        else:
            ax.hist(mobility_data["congestion_index"], bins=40, color="#1f77b4", alpha=0.7)
            ax.set_xlabel("Congestion Index")
            ax.set_title("Congestion Index Distribution")

        # Kanan: Spatial congestion hotspot
        ax2 = axes2[1]
        if "anomaly_prediction" in mobility_data.columns:
            anom_mob = mobility_data[mobility_data["anomaly_prediction"] == -1]
            if not anom_mob.empty:
                ax2.hexbin(
                    anom_mob["longitude"],
                    anom_mob["latitude"],
                    C=anom_mob["congestion_index"],
                    gridsize=30,
                    cmap="hot_r",
                    reduce_C_function=np.mean,
                )
                sm2 = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(0, 1))
                sm2.set_array([])
                cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.7)
                cbar2.set_label("Avg Congestion Index", fontsize=9)
        ax2.set_xlim(112.6, 112.85)
        ax2.set_ylim(-7.4, -7.1)
        ax2.set_xlabel("Longitude", fontsize=11)
        ax2.set_ylabel("Latitude", fontsize=11)
        ax2.set_title("Congestion Anomaly Hotspot\nSpatial Distribution", fontsize=12)
    else:
        for ax in axes2:
            ax.text(0.5, 0.5, "Data tidak tersedia", ha="center", va="center", transform=ax.transAxes)

    fig2.suptitle("Congestion Anomaly Analysis — Metropolitan Surabaya", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "congestion_hotspot.png")
    fig2.savefig(path2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    output_files.append(path2)
    logger.info(f"Saved: {path2}")

    # --- Figure 3: Temporal Mobility Pattern ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    if not mobility_data.empty and "hour" in mobility_data.columns:
        ax = axes3[0]
        if "anomaly_prediction" in mobility_data.columns:
            hourly_normal = mobility_data[mobility_data["anomaly_prediction"] == 1].groupby("hour").size()
            hourly_anomaly = mobility_data[mobility_data["anomaly_prediction"] == -1].groupby("hour").size()
            hours = list(range(24))
            ax.bar(hours, [hourly_normal.get(h, 0) for h in hours], alpha=0.7, color="#2ca02c", label="Normal")
            ax.bar(hours, [hourly_anomaly.get(h, 0) for h in hours], alpha=0.7, color="#d62728", label="Anomaly", bottom=[hourly_normal.get(h, 0) for h in hours])
        else:
            hourly = mobility_data.groupby("hour").size()
            ax.bar(hourly.index, hourly.values, color="#1f77b4", alpha=0.8)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("GPS Point Count", fontsize=11)
        ax.set_title("Mobility Volume by Hour\nNormal vs Anomaly", fontsize=12)
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=10)

        # Kanan: risk level by hour (jika ada scored_zones dengan jam)
        ax2 = axes3[1]
        if not scored_zones.empty and "urban_anomaly_score" in scored_zones.columns:
            risk_counts = scored_zones["risk_level"].value_counts()
            colors = [RISK_COLORS.get(r, "gray") for r in risk_counts.index]
            ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.85, edgecolor="white")
            ax2.set_xlabel("Risk Level", fontsize=11)
            ax2.set_ylabel("Number of Zones", fontsize=11)
            ax2.set_title("Risk Level Distribution\nAcross Urban Zones", fontsize=12)
            for i, (cat, val) in enumerate(zip(risk_counts.index, risk_counts.values)):
                ax2.text(i, val + 0.5, str(val), ha="center", fontsize=11, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "Data tidak tersedia", ha="center", va="center", transform=ax2.transAxes)

    fig3.suptitle("Temporal Mobility Pattern & Risk Distribution — Metropolitan Surabaya", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    path3 = os.path.join(output_dir, "temporal_pattern.png")
    fig3.savefig(path3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    output_files.append(path3)
    logger.info(f"Saved: {path3}")

    # --- Figure 4: Anomaly Correlation Scatter ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))

    if not scored_zones.empty and "avg_anomaly_score" in scored_zones.columns and "contextual_activity_score" in scored_zones.columns:
        colors = [RISK_COLORS[r] for r in scored_zones["risk_level"]]
        ax4.scatter(
            scored_zones["avg_anomaly_score"],
            scored_zones["contextual_activity_score"],
            c=colors,
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

        # Quadrant lines
        ax4.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax4.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax4.text(0.75, 0.75, "Double\nAnomaly", ha="center", va="center",
                 fontsize=9, color="darkred", style="italic",
                 transform=ax4.transAxes)

        # Legend
        patches = [mpatches.Patch(color=c, label=l) for l, c in RISK_COLORS.items()]
        ax4.legend(handles=patches, loc="lower right", fontsize=9)

        r = correlation_stats.get("pearson_r", 0)
        ax4.set_title(
            f"Mobility vs Transaction Anomaly Correlation\nr = {r:.3f}",
            fontsize=12, fontweight="bold",
        )
        ax4.set_xlabel("Mobility Anomaly Score (Isolation Forest)", fontsize=11)
        ax4.set_ylabel("Transaction Contextual Score", fontsize=11)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

    fig4.tight_layout()
    path4 = os.path.join(output_dir, "anomaly_correlation.png")
    fig4.savefig(path4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    output_files.append(path4)
    logger.info(f"Saved: {path4}")

    return output_files


def save_priority_areas_report(
    priority_areas: pd.DataFrame,
    correlation_stats: dict,
    output_path: str = "outputs/reports/priority_areas_report.html",
) -> str:
    """
    Simpan laporan kawasan prioritas monitoring dalam format HTML.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if priority_areas.empty:
        return ""

    rows_html = ""
    for _, row in priority_areas.iterrows():
        risk = row.get("risk_level", "LOW")
        color = RISK_COLORS[risk]
        rows_html += f"""
        <tr>
            <td style="text-align:center; font-weight:bold;">#{int(row.get('monitoring_rank', 0))}</td>
            <td><span style="color:{color}; font-weight:bold;">{risk}</span></td>
            <td>{row.get('area_name', 'Unknown')}</td>
            <td>{float(row.get('priority_score', 0)):.3f}</td>
            <td>{float(row.get('urban_anomaly_score', 0)):.3f}</td>
            <td>{float(row.get('anomaly_rate', 0)):.1%}</td>
            <td>{float(row.get('contextual_activity_score', 0)):.3f}</td>
            <td>{float(row.get('grid_lat', 0)):.4f}, {float(row.get('grid_lon', 0)):.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Urban Anomaly Framework — Priority Monitoring Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left; font-size: 13px; }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
    tr:hover {{ background: #f8f9fa; }}
    .stat-box {{ display: inline-block; background: #f0f4f8; border-radius: 8px; padding: 14px 22px; margin: 8px; text-align: center; }}
    .stat-num {{ font-size: 26px; font-weight: bold; color: #2c3e50; }}
    .stat-label {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>
<h1>Urban Anomaly Intelligence Framework</h1>
<p>Studi Kasus: Metropolitan Surabaya | Framework: Spatio-Temporal Anomaly Detection</p>

<h2>Summary Statistics</h2>
<div>
    <div class="stat-box">
        <div class="stat-num">{correlation_stats.get('total_zones', 0)}</div>
        <div class="stat-label">Total Zones Analyzed</div>
    </div>
    <div class="stat-box">
        <div class="stat-num" style="color:#d62728">{correlation_stats.get('high_risk_zones', 0)}</div>
        <div class="stat-label">HIGH Risk Zones</div>
    </div>
    <div class="stat-box">
        <div class="stat-num" style="color:#ff7f0e">{correlation_stats.get('medium_risk_zones', 0)}</div>
        <div class="stat-label">MEDIUM Risk Zones</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">{correlation_stats.get('pearson_r', 0):.3f}</div>
        <div class="stat-label">Mobility-Transaction Correlation (r)</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">{correlation_stats.get('double_signal_rate', 0):.1%}</div>
        <div class="stat-label">Double Signal Rate</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">{correlation_stats.get('spatial_concentration_gini', 0):.3f}</div>
        <div class="stat-label">Spatial Concentration (Gini)</div>
    </div>
</div>

<h2>Kawasan Prioritas Monitoring</h2>
<table>
    <thead>
        <tr>
            <th>Rank</th><th>Risk Level</th><th>Area</th><th>Priority Score</th>
            <th>Anomaly Score</th><th>Anomaly Rate</th><th>Context Score</th><th>Koordinat</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>

<br><p style="color:#999; font-size:12px;">
Generated by Urban Anomaly Intelligence Framework |
Methods: HDBSCAN (Stage 1), Isolation Forest (Stage 2), Transaction Contextualization (Stage 3), Composite Scoring (Stage 4)
</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report disimpan: {output_path}")
    return output_path


def _score_to_color(score: float) -> str:
    """Konversi anomaly score (0-1) ke hex color."""
    if score >= 0.65:
        return "#d62728"
    elif score >= 0.45:
        return "#ff7f0e"
    elif score >= 0.25:
        return "#ffdd57"
    else:
        return "#2ca02c"


def _build_legend_html() -> str:
    return """
    <div style="
        position: fixed;
        bottom: 30px; right: 15px;
        z-index: 1000;
        background: rgba(255,255,255,0.95);
        padding: 12px 16px;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        min-width: 160px;
    ">
        <b>Risk Level</b><br>
        <span style="color:#d62728;">&#9632;</span> HIGH (&ge; 0.65)<br>
        <span style="color:#ff7f0e;">&#9632;</span> MEDIUM (0.35 - 0.65)<br>
        <span style="color:#2ca02c;">&#9632;</span> LOW (&lt; 0.35)<br>
        <hr style="margin:6px 0;">
        <b>&#9733;</b> Priority Monitoring Area<br>
        <span style="color:#1f77b4;">&#9679;</span> Mobility Hotspot (HDBSCAN)
    </div>
    """
