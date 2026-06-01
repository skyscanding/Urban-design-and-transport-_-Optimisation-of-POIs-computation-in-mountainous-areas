#!/usr/bin/env python3
"""
Step 02: Merge Vehicle Roads + Footways + Steps into Unified Pedestrian Network
================================================================================
Merges three road network sources into a single GeoDataFrame with harmonised schema:
  1. Vehicle roads (All_roadsr.gpkg): standard pedestrian walkways on roads
  2. Footway lines (OSM highway=footway/path)
  3. Steps lines (OSM highway=steps)

Assigns base walking speeds by type and flags non-walkable segments.

Inputs:
  - Vehicle road shapefile/gpkg
  - Footway lines gpkg
  - Steps lines gpkg
  - Study area boundary

Outputs:
  - merged_pedestrian_network.gpkg (all segments, unfiltered)
  - network_summary.csv (per-type statistics)
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiLineString, LineString

warnings.filterwarnings("ignore")


def run(config):
    output_dir = Path(config["output_dir"]) / "Step02_Network"
    output_dir.mkdir(parents=True, exist_ok=True)

    roads_path = Path(config["roads_path"])
    footway_path = Path(config.get("footway_lines_path", ""))
    steps_path = Path(config.get("steps_lines_path", ""))
    study_path = Path(config["study_area_path"])
    net_cfg = config.get("network", {})
    target_crs = config["crs"]

    print(f"  Roads: {roads_path}")
    print(f"  Footways: {footway_path}")
    print(f"  Steps: {steps_path}")
    print(f"  Output: {output_dir}")

    study_area = gpd.read_file(study_path).to_crs(target_crs)

    # ============================================================
    # 1. Load and harmonise each source
    # ============================================================
    def harmonise(gdf, source_type):
        """Standardise columns for merging."""
        df = gdf.copy()
        df["source"] = source_type

        # Ensure consistent column naming
        col_map = {}
        if "osm_id" not in df.columns and "full_id" in df.columns:
            col_map["full_id"] = "osm_id"
        if "highway" not in df.columns and "fclass" in df.columns:
            col_map["fclass"] = "road_class"
        elif "highway" in df.columns:
            col_map["highway"] = "road_class"
        if "name" not in df.columns and "name:en" in df.columns:
            col_map["name:en"] = "name"

        df = df.rename(columns={k: v for k, v in col_map.items() if v not in df.columns})

        for col in ["osm_id", "road_class", "name", "bridge", "tunnel", "surface", "incline"]:
            if col not in df.columns:
                df[col] = None

        # MultiLineString → explode
        df = df.explode(index_parts=False).reset_index(drop=True)

        # Drop invalid
        df = df[df.geometry.notna() & ~df.geometry.is_empty]
        df = df[df.geometry.apply(lambda g: isinstance(g, (LineString,)) and g.length > 0.1)]

        # Length
        df["length_m"] = df.geometry.length

        print(f"    {source_type}: {len(df)} segments, {df['length_m'].sum()/1000:.1f} km")
        return df

    roads = harmonise(gpd.read_file(roads_path).to_crs(target_crs), "vehicle_road")

    segments = [roads]
    if Path(footway_path).exists():
        fw = harmonise(gpd.read_file(footway_path).to_crs(target_crs), "footway")
        segments.append(fw)
    if Path(steps_path).exists():
        st = harmonise(gpd.read_file(steps_path).to_crs(target_crs), "steps")
        segments.append(st)

    merged = pd.concat(segments, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)

    # ============================================================
    # 2. Assign walking speeds
    # ============================================================
    speeds = net_cfg.get("walk_speeds_kmh", {
        "vehicle_road": 4.5, "footway": 4.0, "steps": 1.8
    })
    non_walkable = net_cfg.get("non_walkable_classes", ["motorway", "trunk", "trunk_link"])

    speed_arr = np.where(merged["source"] == "footway", speeds["footway"],
                         np.where(merged["source"] == "steps", speeds["steps"],
                                  speeds["vehicle_road"]))

    # Flag non-walkable
    is_non_walkable = merged["road_class"].isin(non_walkable)
    speed_arr[is_non_walkable] = 0.0

    merged["walk_speed_kmh"] = speed_arr
    merged["is_walkable"] = speed_arr > 0

    n_non = is_non_walkable.sum()
    print(f"  Non-walkable segments: {n_non} (motorway/trunk)")

    # ============================================================
    # 3. Statistics
    # ============================================================
    print(f"\n  Network Summary:")
    for src in merged["source"].unique():
        sub = merged[merged["source"] == src]
        print(f"    {src:15s}: {len(sub):5d} segments, "
              f"{sub['length_m'].sum()/1000:7.1f} km, "
              f"speed={speeds.get(src, '?'):.1f} km/h")

    # ============================================================
    # 4. Export
    # ============================================================
    merged.to_file(output_dir / "merged_pedestrian_network.gpkg", driver="GPKG")

    summary = []
    for src in merged["source"].unique():
        sub = merged[merged["source"] == src]
        summary.append({
            "source": src,
            "segments": len(sub),
            "length_km": sub["length_m"].sum() / 1000,
            "speed_kmh": speeds.get(src, 0),
            "pct_walkable": sub["is_walkable"].mean() * 100,
        })

    pd.DataFrame(summary).to_csv(output_dir / "network_summary.csv", index=False)
    print(f"  ✓ Exported: merged_pedestrian_network.gpkg ({len(merged):,} segments)")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step02_merge_network.py config.json")
