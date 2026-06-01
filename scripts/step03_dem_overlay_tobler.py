#!/usr/bin/env python3
"""
Step 03: DEM Overlay + Tobler's Terrain Impedance
===================================================
Overlays DEM elevation data onto the merged pedestrian network and applies
Tobler's hiking function to compute 3D terrain-adjusted walking times.

Key outputs per segment:
  - cost_seconds_2d            (flat walking time at assigned speed)
  - walk_time_3d_AB_sec        (A→B direction, Tobler)
  - walk_time_3d_BA_sec        (B→A direction, Tobler reversed)
  - walk_time_3d_avg_sec       (round-trip average)
  - time_ratio_3d_vs_2d        (terrain penalty ratio)

Tobler's Function (1993):
    W = 6.0 * exp(-3.5 * |S + 0.05|)
    where W = speed in km/h, S = slope tangent (rise/run)

Additional penalties:
  - Steps: 0.45x multiplier on Tobler speed
  - Footways: 0.88x multiplier on Tobler speed

Inputs:
  - merged_pedestrian_network.gpkg (from Step 02)
  - DEM.tif (GeoTIFF elevation raster)

Outputs:
  - walkable_network.gpkg (with 2D/3D time fields)
  - walkable_network_study_area.gpkg (clipped)
  - terrain_analysis_figures (PNG)
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def run(config):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _utils import sample_elevation_batch, compute_tobler_times

    output_dir = Path(config["output_dir"]) / "Step03_Tobler"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    network_path = Path(config["output_dir"]) / "Step02_Network" / "merged_pedestrian_network.gpkg"
    dem_path = Path(config["dem_path"])
    study_path = Path(config["study_area_path"])
    tobler_cfg = config.get("tobler", {})

    print(f"  Network: {network_path}")
    print(f"  DEM: {dem_path}")
    print(f"  Output: {output_dir}")

    target_crs = config["crs"]
    study_area = gpd.read_file(study_path).to_crs(target_crs)
    edges = gpd.read_file(network_path).to_crs(target_crs)

    # ============================================================
    # 1. DEM sample: elevation at start/end of every segment
    # ============================================================
    print("  Sampling DEM...")
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)
        dem_transform = dem.transform
        dem_nodata = dem.nodata

        # Ensure CRS match
        if edges.crs.to_string() != dem.crs.to_string():
            print(f"    Reprojecting: {edges.crs} → {dem.crs}")
            edges = edges.to_crs(dem.crs)

        e_start, e_end = sample_elevation_batch(edges.geometry, dem_data, dem_transform, dem_nodata)

    edges["elev_start"] = e_start
    edges["elev_end"] = e_end

    n_valid = (~np.isnan(e_start) & ~np.isnan(e_end)).sum()
    print(f"    Valid elevations: {n_valid}/{len(edges)} ({n_valid/len(edges)*100:.1f}%)")

    # Print stats
    valid_mask = (~np.isnan(e_start) & ~np.isnan(e_end))
    elev_diff = np.abs(e_end[valid_mask] - e_start[valid_mask])
    print(f"    Elevation range: {np.nanmin(np.fmin(e_start[valid_mask], e_end[valid_mask])):.0f}–"
          f"{np.nanmax(np.fmax(e_start[valid_mask], e_end[valid_mask])):.0f} m")
    slope_deg = np.degrees(np.arctan(elev_diff / edges.loc[valid_mask, "length_m"].values))
    print(f"    Slope: mean={np.mean(slope_deg):.1f}°, median={np.median(slope_deg):.1f}°, "
          f">15°={(slope_deg>15).sum()} segments")

    # ============================================================
    # 2. Apply Tobler's function
    # ============================================================
    print("  Computing Tobler impedance...")
    edges = compute_tobler_times(
        edges,
        length_col="length_m",
        stair_penalty=tobler_cfg.get("stair_penalty", 0.45),
        footway_factor=tobler_cfg.get("footway_factor", 0.88),
        slope_clamp=tobler_cfg.get("slope_clamp", 0.6),
    )

    # Stats
    walkable = edges[edges["is_walkable"] & np.isfinite(edges["walk_time_3d_avg_sec"])]
    ratios = walkable["time_ratio_3d_vs_2d"].dropna()
    print(f"    Mean 3D/2D ratio: {ratios.mean():.3f}")
    print(f"    Ratio > 1.5: {(ratios > 1.5).sum()}, > 2.0: {(ratios > 2.0).sum()}, "
          f"> 3.0: {(ratios > 3.0).sum()}")

    # ============================================================
    # 3. Export
    # ============================================================
    edges.to_file(output_dir / "walkable_network.gpkg", driver="GPKG")

    # Study area subset
    sa_buf = study_area.buffer(300)
    edges_sa = gpd.clip(edges, sa_buf)
    edges_sa.to_file(output_dir / "walkable_network_study_area.gpkg", driver="GPKG")

    # ============================================================
    # 4. Visualisation
    # ============================================================
    print("  Generating terrain figures...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Slope distribution
    ax = axes[0]
    degs = np.degrees(np.arctan(np.abs(edges["slope_tangent"].dropna())))
    ax.hist(degs.clip(0, 30), bins=40, color="#534AB7", alpha=0.7, edgecolor="white")
    ax.axvline(degs.median(), color="#D85A30", ls="--", lw=1.5, label=f'Median: {degs.median():.1f}°')
    ax.set(xlabel="Absolute slope (degrees)", ylabel="Segments", title="Slope Distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # 3D/2D ratio
    ax = axes[1]
    r = ratios.clip(0, 5)
    ax.hist(r, bins=50, color="#1D9E75", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="#D85A30", ls="--", lw=1.5, label="Flat (1.0)")
    ax.axvline(ratios.mean(), color="#EF9F27", ls="--", lw=1.5, label=f'Mean: {ratios.mean():.2f}')
    ax.set(xlabel="3D/2D time ratio", ylabel="Segments", title="Terrain Impedance Ratio")
    ax.legend(); ax.grid(alpha=0.3)

    # Ratio by type
    ax = axes[2]
    types = ["vehicle_road", "footway", "steps"]
    type_data = [edges.loc[edges["source"] == t, "time_ratio_3d_vs_2d"].dropna().clip(0, 5) for t in types]
    bp = ax.boxplot(type_data, labels=["Vehicle\nRoad", "Footway", "Steps"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#534AB7", "#1D9E75", "#EF9F27"]):
        patch.set_facecolor(color)
    ax.set(ylabel="3D/2D time ratio", title="Impedance Ratio by Network Type")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "Fig_terrain_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Exported: walkable_network.gpkg ({len(edges):,} segments)")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step03_dem_overlay_tobler.py config.json")
