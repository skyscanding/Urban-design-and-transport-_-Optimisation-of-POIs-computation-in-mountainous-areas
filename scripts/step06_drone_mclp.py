#!/usr/bin/env python3
"""
Step 06: Drone Landing Pad Location Optimisation via MCLP
===========================================================
Maximal Covering Location Problem for rooftop drone delivery pads.

Approach:
  - Candidate sites: building rooftops filtered by area, height,
    elevation validity, and LUCC land-use type (excludes pure residential)
  - Hub: user-specified WGS84 location (e.g. waterfront commercial strip)
  - Distance: 3D Euclidean (straight-line flight) via scipy.spatial.distance.cdist
  - Coverage: residents walk up to coverage_radius m to nearest pad
  - Optimisation: MCLP via PuLP/CBC: maximise population covered by p pads

Inputs:
  - HK_Buildings.gpkg (with Elevation/height field)
  - Studyrange.shp (study area boundary)
  - DEM.tif (ground elevation raster)
  - HK_population.gpkg (demand)
  - LUCC raster + code table (land-use classification, optional)

Outputs:
  - drone_MCLP_summary.csv
  - drone_stations_p{N}.gpkg
  - distribution_hub.gpkg
  - Drone MCLP figures
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import pulp

warnings.filterwarnings("ignore")


def run(config):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _utils import sample_dem

    output_dir = Path(config["output_dir"]) / "Step06_Drone"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    drone_cfg = config.get("drone_mclp", {})
    target_crs = config["crs"]

    buildings_path = Path(config["buildings_path"])
    study_path = Path(config["study_area_path"])
    dem_path = Path(config["dem_path"])
    pop_path = Path(config["pop_path"])
    lucc_path = Path(config.get("lucc_raster_path", ""))
    code_table_path = Path(config.get("lucc_code_table_path", ""))

    # Parameters
    hub_wgs84 = drone_cfg.get("hub_wgs84", [114.14154, 22.28780])
    drone_speed_ms = drone_cfg.get("drone_speed_ms", 10.0)
    pad_footprint = drone_cfg.get("pad_footprint_m2", 15.0)
    walk_radius = drone_cfg.get("walking_coverage_radius_m", 200)
    flight_radius = drone_cfg.get("service_radius_flight_m", 2000)
    detour = drone_cfg.get("detour_factor", 1.3)
    min_area = drone_cfg.get("min_roof_area_m2", 50.0)
    min_h = drone_cfg.get("min_building_height_m", 3.0)
    max_h = drone_cfg.get("max_building_height_m", 100.0)
    p_vals = drone_cfg.get("p_values", [3, 5, 7, 8, 10])

    print(f"  Hub (WGS84): {hub_wgs84}")
    print(f"  Walk radius: {walk_radius}m, Flight radius: {flight_radius}m")
    print(f"  Output: {output_dir}")

    # ============================================================
    # 1. Load & Clip Buildings
    # ============================================================
    study_area = gpd.read_file(study_path).to_crs(target_crs)
    study_bounds = study_area.unary_union

    bldg = gpd.read_file(buildings_path).to_crs(target_crs)
    bldg = gpd.clip(bldg, study_area).copy()
    bldg["roof_area_m2"] = bldg.geometry.area
    bldg["centroid"] = bldg.geometry.centroid
    print(f"    Buildings: {len(bldg)}")

    # ============================================================
    # 2. DEM Sampling
    # ============================================================
    print("  Sampling DEM for building ground elevations...")
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_transform = dem_src.transform
        dem_nodata = dem_src.nodata

        bldg["ground_elev_m"] = [
            sample_dem(c.x, c.y, dem_data, dem_transform, dem_nodata)
            for c in bldg["centroid"]
        ]

    # Building height
    elev_field = config.get("field_mapping", {}).get("building_elevation", "Elevation")
    bldg["rooftop_elev_m"] = pd.to_numeric(bldg[elev_field], errors="coerce")
    bldg["building_height_m"] = bldg["rooftop_elev_m"] - bldg["ground_elev_m"]

    valid = bldg.dropna(subset=["rooftop_elev_m", "ground_elev_m", "building_height_m"])
    print(f"    Valid elevations: {len(valid)}/{len(bldg)}")
    print(f"    Height range: {valid['building_height_m'].min():.1f}–"
          f"{valid['building_height_m'].max():.1f} m")

    # ============================================================
    # 3. LUCC Classification (optional)
    # ============================================================
    if lucc_path.exists() and code_table_path.exists():
        print("  Classifying buildings by LUCC land use...")
        # Load code table with encoding fallback
        for enc in ["utf-8", "gb18030", "big5", "gbk"]:
            try:
                code_table = pd.read_csv(code_table_path, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            code_table = pd.read_csv(code_table_path, encoding="utf-8", errors="replace")

        # Find code and name columns
        code_col = [c for c in code_table.columns if "code" in c.lower() or "value" in c.lower()]
        name_col = [c for c in code_table.columns if "name" in c.lower() or "class" in c.lower()
                    or "desc" in c.lower()]
        if code_col and name_col:
            code_map = dict(zip(code_table[code_col[0]], code_table[name_col[0]]))
        else:
            code_map = dict(zip(code_table.iloc[:, 0], code_table.iloc[:, 1]))

        # Sample LUCC
        with rasterio.open(lucc_path) as lucc_src:
            lucc_data = lucc_src.read(1)
            lucc_transform = lucc_src.transform
            bldg_wgs84 = bldg.set_geometry("centroid").to_crs(lucc_src.crs).geometry

            lucc_codes = []
            for pt in bldg_wgs84:
                try:
                    r, c = rowcol(lucc_transform, pt.x, pt.y)
                    r, c = int(np.clip(r, 0, lucc_data.shape[0]-1)), int(np.clip(c, 0, lucc_data.shape[1]-1))
                    lucc_codes.append(int(lucc_data[r, c]))
                except Exception:
                    lucc_codes.append(0)
            bldg["lucc_type"] = [code_map.get(c, "Unknown") for c in lucc_codes]

        # Filter unsuitable types
        unsuitable_kw = drone_cfg.get("unsuitable_landuse_keywords", ["residential", "housing"])
        suitable_kw = drone_cfg.get("suitable_landuse_keywords", ["commercial", "institutional"])

        def is_suitable(s):
            s = str(s).lower()
            for kw in unsuitable_kw:
                if kw in s:
                    return False
            for kw in suitable_kw:
                if kw in s:
                    return True
            return True  # permissive default

        bldg["drone_suitable"] = bldg["lucc_type"].apply(is_suitable)
        print(f"    Drone-suitable: {bldg['drone_suitable'].sum()}/{len(bldg)}")
    else:
        bldg["drone_suitable"] = True
        bldg["lucc_type"] = "Unknown"

    # ============================================================
    # 4. Candidate Filtering
    # ============================================================
    candidates = bldg[
        (bldg["roof_area_m2"] >= min_area) &
        (bldg["building_height_m"] >= min_h) &
        (bldg["building_height_m"] <= max_h) &
        bldg["rooftop_elev_m"].notna() &
        bldg["ground_elev_m"].notna() &
        bldg["drone_suitable"]
    ].copy().reset_index(drop=True)
    print(f"    Candidates: {len(candidates)} rooftops")

    # ============================================================
    # 5. Hub Location
    # ============================================================
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    hub_x, hub_y = transformer.transform(hub_wgs84[0], hub_wgs84[1])

    with rasterio.open(dem_path) as dem_src:
        hub_ground = sample_dem(hub_x, hub_y, dem_src.read(1), dem_src.transform, dem_src.nodata)

    # Fallback search for valid DEM
    if np.isnan(hub_ground):
        for dx in range(-50, 51, 10):
            for dy in range(-50, 51, 10):
                val = sample_dem(hub_x+dx, hub_y+dy, dem_src.read(1), dem_src.transform, dem_src.nodata)
                if not np.isnan(val):
                    hub_ground = val; hub_x += dx; hub_y += dy; break
            if not np.isnan(hub_ground):
                break
    if np.isnan(hub_ground):
        hub_ground = 5.0

    hub_z = hub_ground + 3.0
    print(f"    Hub: ({hub_x:.0f}, {hub_y:.0f}), ground={hub_ground:.1f}m, launch={hub_z:.1f}m")

    # ============================================================
    # 6. 3D Flight Distances
    # ============================================================
    hub_3d = np.array([[hub_x, hub_y, hub_z]])
    cand_3d = np.column_stack([
        candidates["centroid"].x.values,
        candidates["centroid"].y.values,
        candidates["rooftop_elev_m"].values,
    ])
    candidates["flight_dist_3d_m"] = cdist(hub_3d, cand_3d, "euclidean").flatten()
    candidates["flight_time_min"] = candidates["flight_dist_3d_m"] / drone_speed_ms / 60.0

    cand_ok = candidates[candidates["flight_dist_3d_m"] <= flight_radius].copy().reset_index(drop=True)
    print(f"    In range: {len(cand_ok)} (within {flight_radius}m)")
    if len(cand_ok) == 0:
        print("    [ERROR] No candidates in range!")
        return False

    # ============================================================
    # 7. Demand Points
    # ============================================================
    pop = gpd.read_file(pop_path).to_crs(target_crs)
    pop_clipped = gpd.overlay(pop, study_area[["geometry"]], how="intersection")
    pop_field = config.get("field_mapping", {}).get("population_density", "Averag_pop")
    pop_clipped["area_sqkm"] = pop_clipped.geometry.area / 1e6
    pop_clipped["pop_count"] = pop_clipped[pop_field] * pop_clipped["area_sqkm"]
    pop_clipped = pop_clipped[pop_clipped["pop_count"] > 0].copy()

    demand = gpd.GeoDataFrame(
        {"pop_weight": pop_clipped["pop_count"].values},
        geometry=pop_clipped.geometry.centroid.values, crs=target_crs
    ).reset_index(drop=True)
    print(f"    Demand: {len(demand)} points, pop={demand['pop_weight'].sum():.0f}")

    # ============================================================
    # 8. Coverage Matrix + MCLP
    # ============================================================
    d_xy = np.column_stack([demand.geometry.x, demand.geometry.y])
    c_xy = np.column_stack([cand_ok["centroid"].x, cand_ok["centroid"].y])
    walk_dist = cdist(d_xy, c_xy, "euclidean") * detour
    cov_matrix = (walk_dist <= walk_radius).astype(int)
    n_d, n_c = cov_matrix.shape
    coverable = (cov_matrix.sum(axis=1) > 0).sum()
    print(f"    Coverage matrix: {n_d}×{n_c}, coverable: {coverable}/{n_d} ({coverable/n_d:.1%})")

    w = demand["pop_weight"].values
    results = {}

    for p in p_vals:
        if p > n_c:
            print(f"    p={p} > candidates={n_c}, skip")
            continue

        prob = pulp.LpProblem(f"MCLP_p{p}", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", range(n_c), cat="Binary")
        z = pulp.LpVariable.dicts("z", range(n_d), cat="Binary")
        prob += pulp.lpSum(w[i] * z[i] for i in range(n_d))
        prob += pulp.lpSum(y[j] for j in range(n_c)) == p

        for i in range(n_d):
            cov_j = [j for j in range(n_c) if cov_matrix[i, j]]
            prob += z[i] <= (pulp.lpSum(y[j] for j in cov_j) if cov_j else 0)

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
        sel = [j for j in range(n_c) if y[j].varValue and y[j].varValue > 0.5]
        cov_ids = [i for i in range(n_d) if z[i].varValue and z[i].varValue > 0.5]
        cov_pop = sum(w[i] for i in cov_ids)
        pct = cov_pop / w.sum() * 100
        sdf = cand_ok.iloc[sel]

        results[p] = {
            "status": pulp.LpStatus[prob.status], "selected": sel,
            "covered_pop": cov_pop, "pct_coverage": pct,
            "mean_flight": sdf["flight_time_min"].mean(),
            "max_flight": sdf["flight_time_min"].max(),
            "stations": sdf,
        }
        print(f"    p={p}: {pulp.LpStatus[prob.status]} | Coverage={pct:.1f}% | "
              f"Flight: {sdf['flight_time_min'].mean():.1f}min avg")

    # ============================================================
    # 9. Export
    # ============================================================
    for p in p_vals:
        if p in results:
            sel = cand_ok.iloc[results[p]["selected"]]
            gpd.GeoDataFrame(
                {"station_id": [f"S{i+1}" for i in range(len(sel))], "p": p},
                geometry=[Point(r.centroid.x, r.centroid.y) for _, r in sel.iterrows()],
                crs=target_crs
            ).to_file(output_dir / f"drone_stations_p{p}.gpkg", driver="GPKG")

    gpd.GeoDataFrame(
        {"name": ["Distribution Hub"], "elev": [hub_z]},
        geometry=[Point(hub_x, hub_y)], crs=target_crs
    ).to_file(output_dir / "distribution_hub.gpkg", driver="GPKG")

    # Summary
    pd.DataFrame([{
        "p": p, "status": r["status"], "coverage_pct": f"{r['pct_coverage']:.1f}",
        "mean_flight_min": f"{r['mean_flight']:.2f}",
        "max_flight_min": f"{r['max_flight']:.2f}",
    } for p, r in results.items()]).to_csv(output_dir / "drone_MCLP_summary.csv", index=False)

    # ============================================================
    # 10. Figures
    # ============================================================
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        valid_p = sorted(results.keys())
        ax1.plot(valid_p, [results[p]["pct_coverage"] for p in valid_p],
                 "o-", color="#534AB7", lw=2, ms=8)
        ax1.set(xlabel="Drone stations (p)", ylabel="Coverage (%)")
        ax1.set_title("Coverage vs Station Count", fontweight="bold")
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 100)

        ax2.plot(valid_p, [results[p]["mean_flight"] for p in valid_p],
                 "s-", color="#D85A30", lw=2, ms=8)
        ax2.set(xlabel="Drone stations (p)", ylabel="Mean flight (min)")
        ax2.set_title("Flight Time from Hub", fontweight="bold")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "Fig_drone_coverage_vs_p.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Figures failed: {e}")

    print(f"  ✓ Exported: drone_stations_p{N}.gpkg for p={p_vals}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step06_drone_mclp.py config.json")
