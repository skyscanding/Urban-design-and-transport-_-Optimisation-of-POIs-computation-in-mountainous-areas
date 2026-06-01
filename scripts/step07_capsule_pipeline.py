#!/usr/bin/env python3
"""
Step 07: Capsule Cargo Pipeline Routing: SPT vs Steiner Tree
================================================================
Uses P-median optimised terminal locations to design a capsule cargo
pipeline network through the OSM road graph. Two routing methods are
compared:

  1. SPT (Shortest Path Tree): Union of shortest paths from hub to each terminal
  2. Steiner Tree: Minimum-weight subtree connecting all required nodes
     (networkx.algorithms.approximation.steiner_tree)

Coverage analysis uses walking buffers around the pipeline alignment.

Inputs:
  - stations_3D_p10.gpkg (from Step 04 P-median, or lastmile_stations.shp)
  - OSM_NetworkX.gpkg (road graph lines)
  - HK_population.gpkg (demand)
  - Studyrange.shp

Outputs:
  - pipeline_SPT.gpkg, pipeline_Steiner.gpkg
  - pipeline_nodes.gpkg (hub + terminals)
  - pipeline_comparison.csv
  - Pipeline routing figures
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from pyproj import Transformer

warnings.filterwarnings("ignore")


def run(config):
    output_dir = Path(config["output_dir"]) / "Step07_Capsule"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    capsule_cfg = config.get("capsule_pipeline", {})
    target_crs = config["crs"]

    # Find terminal stations (try P-median output first, fallback to lastmile)
    stations_sources = [
        Path(config["output_dir"]) / "Step04_PMedian" / "stations_3D_p10.gpkg",
        Path(config["output_dir"]) / "Step01_POI" / "lastmile_stations.shp",
    ]
    osm_path = Path(config.get("roads_path", "")).parent / "Road_supplement" / "OSM_NetworkX.gpkg"
    study_path = Path(config["study_area_path"])
    pop_path = Path(config["pop_path"])

    # Hub WGS84
    hub_wgs84 = capsule_cfg.get("hub_wgs84", [114.14154, 22.28780])
    capsule_speed = capsule_cfg.get("capsule_speed_ms", 2.0)
    walk_radii = capsule_cfg.get("walk_radii", [200, 300, 400])
    detour = capsule_cfg.get("detour_factor", 1.3)
    snap_tol = config.get("network", {}).get("snap_tolerance", 15.0)

    # ============================================================
    # 1. Load Terminal Stations
    # ============================================================
    stations_gdf = None
    for sp in stations_sources:
        if sp.exists():
            stations_gdf = gpd.read_file(sp).to_crs(target_crs)
            print(f"    Loaded terminals: {sp} ({len(stations_gdf)} stations)")
            break
    if stations_gdf is None:
        print("    [ERROR] No station file found!")
        return False

    terminal_coords = [(r.geometry.x, r.geometry.y) for _, r in stations_gdf.iterrows()]

    # ============================================================
    # 2. Build OSM Pipeline Graph
    # ============================================================
    print("  Building OSM pipeline graph...")
    if not osm_path.exists():
        # Fallback: use the merged pedestrian network directly
        osm_path = Path(config["output_dir"]) / "Step03_Tobler" / "walkable_network.gpkg"
        print(f"    OSM_NetworkX not found, using: {osm_path}")

    osm_edges = gpd.read_file(osm_path).to_crs(target_crs)

    # Build undirected graph with coordinate snap
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_node(x, y):
        key = (round(x / snap_tol) * snap_tol, round(y / snap_tol) * snap_tol)
        if key not in coord_to_node:
            coord_to_node[key] = node_counter[0]
            node_coords[node_counter[0]] = key
            node_counter[0] += 1
        return coord_to_node[key]

    G = nx.Graph()
    for _, row in osm_edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            u, v = get_node(*coords[0][:2]), get_node(*coords[-1][:2])
            if u == v:
                continue
            w = line.length
            if not G.has_edge(u, v) or w < G[u][v].get("weight", np.inf):
                G.add_edge(u, v, weight=w, geometry=line)

    # Largest component
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main_comp = comps[0]
    G_pipe = G.subgraph(main_comp).copy()
    print(f"    Graph: {G_pipe.number_of_nodes()} nodes, {G_pipe.number_of_edges()} edges "
          f"({len(comps)} components)")

    # Spatial index
    main_ids = sorted(main_comp)
    main_xy = np.array([node_coords[n] for n in main_ids])
    tree = cKDTree(main_xy)

    # ============================================================
    # 3. Snap Hub and Terminals
    # ============================================================
    tfm = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    hub_x, hub_y = tfm.transform(*hub_wgs84)
    d_h, i_h = tree.query([hub_x, hub_y])
    hub_node = main_ids[i_h]
    hub_xy = node_coords[hub_node]
    print(f"    Hub: ({hub_x:.0f},{hub_y:.0f}) → node {hub_node} (snap={d_h:.0f}m)")

    term_nodes = []
    for tx, ty in terminal_coords:
        d_t, i_t = tree.query([tx, ty])
        nid = main_ids[i_t]
        if nid not in main_comp:
            print(f"    T: ({tx:.0f},{ty:.0f}): not in main component!")
            continue
        if nid != hub_node and nid not in term_nodes:
            term_nodes.append(nid)

    print(f"    Terminals (unique, excl hub): {len(term_nodes)}")
    required = [hub_node] + term_nodes

    # ============================================================
    # 4. SPT (Shortest Path Tree)
    # ============================================================
    print("  Computing Shortest Path Tree...")
    spt_edges = set()
    spt_times = {}

    for tn in term_nodes:
        try:
            path = nx.shortest_path(G_pipe, hub_node, tn, weight="weight")
            plen = nx.shortest_path_length(G_pipe, hub_node, tn, weight="weight")
            spt_times[tn] = plen / capsule_speed / 60.0
            for k in range(len(path) - 1):
                spt_edges.add(tuple(sorted([path[k], path[k+1]])))
        except nx.NetworkXNoPath:
            print(f"    ⚠ No path hub → terminal")

    spt_len = sum(G_pipe[u][v]["weight"] for u, v in spt_edges)
    print(f"    SPT: {len(spt_edges)} edges, {spt_len:.0f}m ({spt_len/1000:.2f} km)")

    # ============================================================
    # 5. Steiner Tree
    # ============================================================
    print("  Computing Steiner Tree...")
    has_steiner = False
    steiner_edges = set()
    steiner_len = 0
    steiner_times = {}

    try:
        st = nx.approximation.steiner_tree(G_pipe, required, weight="weight")
        steiner_edges = set(tuple(sorted(e[:2])) for e in st.edges())
        steiner_len = sum(G_pipe[u][v]["weight"] for u, v in steiner_edges)

        for tn in term_nodes:
            try:
                pl = nx.shortest_path_length(st, hub_node, tn, weight="weight")
                steiner_times[tn] = pl / capsule_speed / 60.0
            except Exception:
                steiner_times[tn] = np.nan

        has_steiner = True
        print(f"    Steiner: {len(steiner_edges)} edges, {steiner_len:.0f}m "
              f"({steiner_len/1000:.2f} km)")
        print(f"    Length savings: {(1 - steiner_len/spt_len)*100:+.1f}%")
    except Exception as e:
        print(f"    ⚠ Steiner failed: {e}")
        steiner_edges = spt_edges
        steiner_len = spt_len
        steiner_times = spt_times

    # ============================================================
    # 6. Export Geometries
    # ============================================================
    def to_gdf(edge_set):
        geoms, lens = [], []
        for u, v in edge_set:
            data = G_pipe[u][v]
            geoms.append(data.get("geometry", LineString([node_coords[u], node_coords[v]])))
            lens.append(data.get("weight", 0))
        return gpd.GeoDataFrame({"length_m": lens}, geometry=geoms, crs=target_crs)

    spt_gdf = to_gdf(spt_edges)
    spt_geom = unary_union(spt_gdf.geometry)
    spt_gdf.to_file(output_dir / "pipeline_SPT.gpkg", driver="GPKG")

    if has_steiner:
        st_gdf = to_gdf(steiner_edges)
        st_geom = unary_union(st_gdf.geometry)
        st_gdf.to_file(output_dir / "pipeline_Steiner.gpkg", driver="GPKG")
    else:
        st_gdf, st_geom = spt_gdf, spt_geom

    # Nodes
    pts = [{"label": "Hub", "type": "hub"}] + [
        {"label": f"T{i+1}", "type": "terminal"} for i in range(len(term_nodes))]
    pts_geom = [Point(hub_xy)] + [Point(node_coords[n]) for n in term_nodes]
    gpd.GeoDataFrame(pts, geometry=pts_geom, crs=target_crs).to_file(
        output_dir / "pipeline_nodes.gpkg", driver="GPKG")

    # ============================================================
    # 7. Coverage Analysis
    # ============================================================
    print("  Computing walk coverage...")
    pop = gpd.read_file(pop_path).to_crs(target_crs)
    study_area = gpd.read_file(study_path).to_crs(target_crs)
    pop_clipped = gpd.overlay(pop, study_area[["geometry"]], how="intersection")
    pop_field = config.get("field_mapping", {}).get("population_density", "Averag_pop")
    pop_clipped["area_sqkm"] = pop_clipped.geometry.area / 1e6
    pop_clipped["pop_count"] = pop_clipped[pop_field] * pop_clipped["area_sqkm"]
    pop_clipped = pop_clipped[pop_clipped["pop_count"] > 0].copy()
    total_pop = pop_clipped["pop_count"].sum()

    demand_pts = gpd.GeoDataFrame(
        {"pop": pop_clipped["pop_count"].values},
        geometry=pop_clipped.geometry.centroid.values, crs=target_crs)

    cov = {}
    for r in walk_radii:
        eff = r / detour
        spt_cov = demand_pts[demand_pts.geometry.within(spt_geom.buffer(eff))]["pop"].sum()
        st_cov = demand_pts[demand_pts.geometry.within(st_geom.buffer(eff))]["pop"].sum()
        cov[r] = {"spt": spt_cov / total_pop * 100, "steiner": st_cov / total_pop * 100}
        print(f"    {r}m walk: SPT={cov[r]['spt']:.1f}%, Steiner={cov[r]['steiner']:.1f}%")

    # ============================================================
    # 8. Comparison CSV
    # ============================================================
    spt_mean = np.mean(list(spt_times.values())) if spt_times else 0
    spt_max = max(spt_times.values()) if spt_times else 0
    st_mean = np.nanmean(list(steiner_times.values())) if steiner_times else 0
    st_max = np.nanmax(list(steiner_times.values())) if steiner_times else 0

    pd.DataFrame([
        {
            "method": "SPT", "length_m": spt_len, "edges": len(spt_edges),
            "mean_travel_min": spt_mean, "max_travel_min": spt_max,
            **{f"cov_{r}m": f"{cov[r]['spt']:.1f}%" for r in walk_radii},
        },
        {
            "method": "Steiner", "length_m": steiner_len, "edges": len(steiner_edges),
            "mean_travel_min": st_mean, "max_travel_min": st_max,
            **{f"cov_{r}m": f"{cov[r]['steiner']:.1f}%" for r in walk_radii},
        },
    ]).to_csv(output_dir / "pipeline_comparison.csv", index=False)

    best = "Steiner" if has_steiner and steiner_len < spt_len else "SPT"
    savings = (1 - steiner_len / spt_len) * 100 if spt_len > 0 and has_steiner else 0
    print(f"    Best method: {best} | Savings: {savings:+.1f}%")

    # ============================================================
    # 9. Figures
    # ============================================================
    try:
        import matplotlib.pyplot as plt
        buildings_path = Path(config["buildings_path"])
        if buildings_path.exists():
            buildings = gpd.clip(gpd.read_file(buildings_path).to_crs(target_crs),
                                 study_area)

        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        for ax_i, (ax, name, egdf, rgeom, times, tlen) in enumerate([
            (axes[0], "Shortest Path Tree", spt_gdf, spt_geom, spt_times, spt_len),
            (axes[1], "Steiner Tree", st_gdf, st_geom, steiner_times, steiner_len),
        ]):
            study_area.boundary.plot(ax=ax, color="black", lw=1)
            if buildings_path.exists():
                buildings.plot(ax=ax, color="#E8E6E0", edgecolor="#C0BEB5", lw=0.1, alpha=0.4)

            # Coverage buffer
            gpd.GeoDataFrame(geometry=[rgeom.buffer(200/detour)], crs=target_crs).plot(
                ax=ax, color="#534AB7", alpha=0.06, edgecolor="#534AB7", lw=0.5, ls="--")

            # Pipeline
            egdf.plot(ax=ax, color="#534AB7", lw=2, alpha=0.9)

            # Hub
            ax.plot(hub_xy[0], hub_xy[1], "*", color="#D85A30", ms=16, mec="white", mew=1.2, zorder=11)
            ax.annotate("Hub", hub_xy, xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color="#D85A30", fontweight="bold")

            # Terminals
            for i, nid in enumerate(term_nodes):
                xy = node_coords[nid]
                tt = times.get(nid, np.nan)
                ax.plot(xy[0], xy[1], "^", color="#1D9E75", ms=10, mec="white", mew=1, zorder=10)
                ax.annotate(f"T{i+1}\n{tt:.1f}min", xy, xytext=(5, 5),
                            textcoords="offset points", fontsize=6, color="#085041",
                            fontweight="bold")

            ck = "spt" if ax_i == 0 else "steiner"
            ax.set_title(f"{name}\n{tlen/1000:.2f} km | 200m cov: {cov[200][ck]:.1f}%",
                         fontsize=12, fontweight="bold")
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(figures_dir / "Fig_pipeline_SPT_vs_Steiner.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Figures failed: {e}")

    print(f"  ✓ Exported: pipeline_SPT.gpkg, pipeline_Steiner.gpkg, comparison CSV")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step07_capsule_pipeline.py config.json")
