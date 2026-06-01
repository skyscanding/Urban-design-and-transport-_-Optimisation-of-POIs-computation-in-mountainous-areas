"""
_utils.py: Shared utility functions for the Mountainous Terrain Optimisation Pipeline.

Provides:
  - DEM sampling (rasterio-based, with nodata handling)
  - Coordinate snapping for NetworkX graph construction
  - Graph building helpers
  - P-median evaluation metrics
  - Directional cost computation from Tobler's function
  - Spatial index helpers (cKDTree)

Dependencies: numpy, rasterio, networkx, scipy, shapely, geopandas
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ============================================================
# DEM Sampling
# ============================================================

def sample_dem(x, y, dem_data, dem_transform, dem_nodata, dem_nodata_threshold=-9000):
    """Sample DEM elevation at (x, y) in raster CRS. Returns np.nan on failure."""
    from rasterio.transform import rowcol
    try:
        r, c = rowcol(dem_transform, x, y)
        r = int(np.clip(r, 0, dem_data.shape[0] - 1))
        c = int(np.clip(c, 0, dem_data.shape[1] - 1))
        val = float(dem_data[r, c])
        if (dem_nodata is not None and val == dem_nodata):
            return np.nan
        if val < dem_nodata_threshold or val > 1000:
            return np.nan
        return val
    except Exception:
        return np.nan


def sample_elevation_batch(geometries, dem_data, dem_transform, dem_nodata):
    """Sample start and end elevations for a batch of LineString geometries."""
    from rasterio.transform import rowcol
    n = len(geometries)
    elev_start = np.full(n, np.nan)
    elev_end = np.full(n, np.nan)

    for i, geom in enumerate(geometries):
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        for xy, arr in [(coords[0][:2], elev_start), (coords[-1][:2], elev_end)]:
            try:
                r, c = rowcol(dem_transform, xy[0], xy[1])
                r, c = int(np.clip(r, 0, dem_data.shape[0] - 1)), int(np.clip(c, 0, dem_data.shape[1] - 1))
                val = float(dem_data[r, c])
                if dem_nodata is not None and val == dem_nodata:
                    continue
                if -100 <= val <= 1000:
                    arr[i] = val
            except Exception:
                pass

    return elev_start, elev_end


# ============================================================
# Tobler's Hiking Function
# ============================================================

def tobler_speed_kmh(slope_tangent):
    """Tobler's hiking function (1993): speed in km/h given slope tangent (rise/run)."""
    return 6.0 * np.exp(-3.5 * np.abs(slope_tangent + 0.05))


def compute_tobler_times(edges_gdf, length_col="Shape_Leng", stair_penalty=0.45,
                         footway_factor=0.88, slope_clamp=0.6,
                         speed_road=4.5, speed_footway=4.0, speed_steps=1.8):
    """
    Compute 2D and 3D walking times for all segments.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        Must have: geometry (LineString), source (vehicle_road/footway/steps),
        elev_start, elev_end, road_class.
    length_col : str
        Column with segment length in metres.
    stair_penalty : float
        Multiplier on Tobler speed for steps (0.45 = 45% of equivalent-slope walking).
    footway_factor : float
        Multiplier on Tobler speed for footways.
    slope_clamp : float
        Clamp slope tangent to [-slope_clamp, +slope_clamp] to suppress DEM artifacts.
    speed_road, speed_footway, speed_steps : float
        Flat walking speeds in km/h for each network type.

    Returns
    -------
    edges_gdf with added columns: cost_seconds_2d, walk_time_3d_AB_sec,
    walk_time_3d_BA_sec, walk_time_3d_avg_sec, time_ratio_3d_vs_2d, etc.
    """
    edges = edges_gdf.copy()
    n = len(edges)

    # Flat speeds
    speeds_kmh = np.where(edges["source"] == "footway", speed_footway,
                          np.where(edges["source"] == "steps", speed_steps, speed_road))

    # 2D walking time
    edges["walk_speed_kmh_2d"] = speeds_kmh
    edges["cost_seconds_2d"] = edges[length_col] / (speeds_kmh / 3.6)

    # Non-walkable
    non_walkable = edges.get("road_class", pd.Series([""]*n)).isin(["motorway", "trunk", "trunk_link"])
    edges.loc[non_walkable, ["walk_speed_kmh_2d", "cost_seconds_2d"]] = 0

    # Slope computation
    edges["elev_diff"] = edges.get("elev_end", 0) - edges.get("elev_start", 0)
    edges["slope_tangent"] = (edges["elev_diff"] / edges[length_col]).fillna(0)
    edges["slope_tangent_clamped"] = edges["slope_tangent"].clip(-slope_clamp, slope_clamp)
    edges["slope_deg"] = np.degrees(np.arctan(edges["slope_tangent"]))
    edges["slope_deg_clamped"] = np.degrees(np.arctan(edges["slope_tangent_clamped"]))

    # Tobler directional speeds
    speed_ab = tobler_speed_kmh(edges["slope_tangent_clamped"].values)
    speed_ba = tobler_speed_kmh(-edges["slope_tangent_clamped"].values)

    # Type penalties
    is_steps = (edges["source"] == "steps").values
    is_footway = (edges["source"] == "footway").values
    speed_ab[is_steps] *= stair_penalty
    speed_ba[is_steps] *= stair_penalty
    speed_ab[is_footway] *= footway_factor
    speed_ba[is_footway] *= footway_factor

    # 3D times
    edges["walk_time_3d_AB_sec"] = np.where(
        speed_ab > 0, edges[length_col] / (speed_ab / 3.6), np.inf)
    edges["walk_time_3d_BA_sec"] = np.where(
        speed_ba > 0, edges[length_col] / (speed_ba / 3.6), np.inf)
    edges["walk_time_3d_avg_sec"] = (edges["walk_time_3d_AB_sec"] + edges["walk_time_3d_BA_sec"]) / 2
    edges["walk_time_3d_max_sec"] = np.maximum(edges["walk_time_3d_AB_sec"], edges["walk_time_3d_BA_sec"])

    edges.loc[non_walkable, ["walk_time_3d_AB_sec", "walk_time_3d_BA_sec",
                              "walk_time_3d_avg_sec", "walk_time_3d_max_sec"]] = np.inf

    # Ratio
    valid = (edges["cost_seconds_2d"] > 0) & np.isfinite(edges["walk_time_3d_avg_sec"])
    edges["time_ratio_3d_vs_2d"] = np.where(
        valid, edges["walk_time_3d_avg_sec"] / edges["cost_seconds_2d"], 1.0)

    return edges


# ============================================================
# Graph Construction
# ============================================================

def build_graph_from_edges(edges_gdf, snap_tolerance=5.0, weight_col_2d="cost_seconds_2d",
                           weight_col_3d="walk_time_3d_avg_sec"):
    """
    Build dual (2D/3D) NetworkX graphs from a merged edge GeoDataFrame.

    Nodes are created at line endpoints and deduplicated by snapping within
    snap_tolerance metres.

    Returns (G_2d, G_3d, node_coords) where node_coords maps node_id -> (x, y).
    """
    coord_to_node = {}
    node_coords = {}
    node_counter = [0]

    def get_node(x, y):
        key = (round(x / snap_tolerance) * snap_tolerance,
               round(y / snap_tolerance) * snap_tolerance)
        if key not in coord_to_node:
            coord_to_node[key] = node_counter[0]
            node_coords[node_counter[0]] = key
            node_counter[0] += 1
        return coord_to_node[key]

    G_2d = nx.Graph()
    G_3d = nx.Graph()

    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        u = get_node(*coords[0][:2])
        v = get_node(*coords[-1][:2])
        if u == v:
            continue

        w2d = row.get(weight_col_2d, np.inf)
        if not np.isfinite(w2d) or w2d <= 0:
            continue
        w3d = row.get(weight_col_3d, w2d)
        if not np.isfinite(w3d) or w3d <= 0:
            w3d = w2d

        if not G_2d.has_edge(u, v) or w2d < G_2d[u][v]["weight"]:
            G_2d.add_edge(u, v, weight=w2d)
            G_3d.add_edge(u, v, weight=w3d)

    return G_2d, G_3d, node_coords


def largest_connected_component(G):
    """Return the largest connected component of G."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy() if comps else G


# ============================================================
# Candidate & Demand Node Selection
# ============================================================

def snap_points_to_graph(points_gdf, G, node_coords, max_dist=500.0):
    """
    Snap point geometries to nearest graph nodes within max_dist.

    Returns dict: {node_id: [{'weight': ..., 'original_geom': ...}]}
    """
    node_ids = sorted(G.nodes())
    node_xy = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(node_xy)
    query_xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])

    dists, idxs = tree.query(query_xy)

    snapped = {}
    for i, (d, ni) in enumerate(zip(dists, idxs)):
        if d > max_dist:
            continue
        nid = node_ids[ni]
        if nid not in snapped:
            snapped[nid] = []
        snapped[nid].append({"weight": points_gdf.iloc[i].get("weight", 1.0)})

    return snapped


def get_intersection_nodes(G, min_degree=3):
    """Return nodes with degree >= min_degree."""
    return [n for n in G.nodes() if G.degree(n) >= min_degree]


# ============================================================
# Cost Matrix
# ============================================================

def build_cost_matrix(G, candidate_nodes, demand_nodes, unreachable_penalty=9999.0):
    """
    Build |demand| x |candidates| shortest-path cost matrix via Dijkstra.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph with 'weight' edge attribute.
    candidate_nodes : list of int
        Candidate facility node IDs.
    demand_nodes : list of int
        Demand node IDs.
    unreachable_penalty : float
        Cost for unreachable demand-candidate pairs.

    Returns
    -------
    cost_mat : np.ndarray of shape (len(demand_nodes), len(candidate_nodes))
    """
    n_d = len(demand_nodes)
    n_c = len(candidate_nodes)
    cost_mat = np.full((n_d, n_c), unreachable_penalty)

    for j, cand in enumerate(candidate_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, cand, weight="weight")
        for i, dn in enumerate(demand_nodes):
            if dn in lengths:
                cost_mat[i, j] = lengths[dn]

    return cost_mat


# ============================================================
# P-Median Evaluation
# ============================================================

def evaluate_pmedian(cost_mat, weights, selected_indices, thresholds_min=None):
    """
    Evaluate a set of selected facilities.

    Parameters
    ----------
    cost_mat : np.ndarray (n_demand x n_candidates)
        Full cost matrix.
    weights : array-like of length n_demand
        Population weights.
    selected_indices : list of int
        Indices of selected facilities (columns in cost_mat).
    thresholds_min : list of float, optional
        Coverage thresholds in minutes. Default: [3, 5, 7, 10, 15].

    Returns
    -------
    dict with keys: pw_mean_sec, pw_mean_min, max_sec, max_min,
    cov_{threshold}min (percentage), total_pop
    """
    if thresholds_min is None:
        thresholds_min = [3, 5, 7, 10, 15]

    w = np.asarray(weights, dtype=float)
    min_costs = cost_mat[:, selected_indices].min(axis=1)

    pw_mean = np.average(min_costs, weights=w)
    result = {
        "pw_mean_sec": pw_mean,
        "pw_mean_min": pw_mean / 60.0,
        "max_sec": min_costs.max(),
        "max_min": min_costs.max() / 60.0,
        "total_pop": w.sum()
    }

    for thr_min in thresholds_min:
        thr_sec = thr_min * 60
        cov_pop = w[min_costs <= thr_sec].sum()
        result[f"cov_{thr_min}min"] = cov_pop / w.sum() * 100

    return result


# ============================================================
# Curve Fitting for Diminishing Returns
# ============================================================

def fit_power_law(p_values, travel_times):
    """Fit T(p) = a * p^(-b). Returns (a, b, r_squared)."""
    from scipy.optimize import curve_fit
    p = np.array(p_values, dtype=float)
    t = np.array(travel_times, dtype=float)

    def model(p, a, b):
        return a * p ** (-b)

    popt, _ = curve_fit(model, p, t, p0=[t[0] * p[0] ** 0.5, 0.5], maxfev=10000)
    t_pred = model(p, *popt)
    ss_res = np.sum((t - t_pred) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], r2


def fit_saturating_exponential(p_values, coverage_pct, cmax_bounds=(50, 100)):
    """Fit C(p) = Cmax * (1 - exp(-k * p)). Returns (Cmax, k, r_squared)."""
    from scipy.optimize import curve_fit
    p = np.array(p_values, dtype=float)
    c = np.array(coverage_pct, dtype=float)

    def model(p, Cmax, k):
        return Cmax * (1 - np.exp(-k * p))

    popt, _ = curve_fit(model, p, c, p0=[90, 0.15],
                        bounds=([cmax_bounds[0], 0.001], [cmax_bounds[1], 2.0]),
                        maxfev=10000)
    c_pred = model(p, *popt)
    ss_res = np.sum((c - c_pred) ** 2)
    ss_tot = np.sum((c - c.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], r2


# ============================================================
# Optimal-P Detection Methods
# ============================================================

def find_optimal_p_kneedle(p_values, metric_values, sensitivity=1.0):
    """Kneedle algorithm: find point of maximum curvature."""
    x = np.array(p_values, dtype=float)
    y = np.array(metric_values, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    diff = y_norm - x_norm
    idx = np.argmax(diff * sensitivity)
    return p_values[idx]


def find_optimal_p_marginal(p_values, mean_times, threshold_min=0.1):
    """Find first p where marginal benefit drops below threshold_min."""
    marginal = -np.diff(mean_times)  # reduction in minutes
    for i, m in enumerate(marginal):
        if m < threshold_min:
            return p_values[i + 1]  # return the p where it dropped
    return p_values[-1]


def find_optimal_p_consensus(p_values, metric_values, mean_times=None):
    """Run multiple methods and return the median consensus."""
    methods = {}
    methods["kneedle"] = find_optimal_p_kneedle(p_values, metric_values)
    if mean_times is not None:
        methods["marginal"] = find_optimal_p_marginal(p_values, mean_times)
    vals = list(methods.values())
    return int(np.median(vals)), methods


# ============================================================
# Spatial Helpers
# ============================================================

def build_spatial_index(node_coords):
    """Build cKDTree spatial index for graph nodes."""
    node_ids = sorted(node_coords.keys())
    node_xy = np.array([node_coords[n] for n in node_ids])
    return cKDTree(node_xy), node_ids, node_xy


def clip_to_largest_component(G):
    """Return subgraph of largest connected component (alias)."""
    return largest_connected_component(G)


import pandas as pd  # noqa: E402 (used in compute_tobler_times fallback paths)
