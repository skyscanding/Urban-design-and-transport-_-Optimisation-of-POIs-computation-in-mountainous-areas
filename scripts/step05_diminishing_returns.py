#!/usr/bin/env python3
"""
Step 05: Diminishing Returns Analysis (p = 3 to 20)
=====================================================
Solves P-median for p = 3 through 20 under both 2D and 3D scenarios,
then fits functional forms and detects the optimal station count.

Curve Fitting:
  - Mean travel time: T(p) = a × p^(-b)    (power law)
  - 5-min coverage:   C(p) = Cmax × (1 − e^(-k×p))  (saturating exponential)

Optimal-p Detection Methods:
  1. Kneedle Algorithm: max curvature on normalised curve
  2. Marginal Analysis: first p where dT/dp < 0.1 min
  3. L-Method: best two-segment linear fit
  → Consensus: median of methods

Inputs:
  - walkable_network.gpkg (from Step 03)
  - lastmile_stations.shp (from Step 01)
  - HK_population.gpkg

Outputs:
  - diminishing_returns_full.csv (all p results)
  - diminishing_returns_data.csv (summary)
  - optimal_p_methods.csv (consensus result)
  - diminishing returns figures
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import Point
import pulp

warnings.filterwarnings("ignore")


def run(config):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _utils import (
        build_graph_from_edges, largest_connected_component,
        build_cost_matrix, evaluate_pmedian,
        fit_power_law, fit_saturating_exponential,
        find_optimal_p_consensus,
    )

    output_dir = Path(config["output_dir"]) / "Step05_Diminishing"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    network_path = Path(config["output_dir"]) / "Step03_Tobler" / "walkable_network.gpkg"
    stations_path = Path(config["output_dir"]) / "Step01_POI" / "lastmile_stations.shp"
    study_path = Path(config["study_area_path"])
    pop_path = Path(config["pop_path"])
    dr_cfg = config.get("diminishing_returns", {})
    pm_cfg = config.get("p_median", {})
    net_cfg = config.get("network", {})

    target_crs = config["crs"]
    p_range = dr_cfg.get("p_range", list(range(3, 21)))
    cov_thresholds = dr_cfg.get("coverage_thresholds_min", [3, 5, 7, 10, 15])
    marginal_thr = dr_cfg.get("marginal_threshold_min", 0.1)
    time_limit = pm_cfg.get("solver_time_limit", 300)

    print(f"  P values: {p_range}")
    print(f"  Output: {output_dir}")

    # ============================================================
    # 1. Build graphs (same as Step 04)
    # ============================================================
    print("  Building graphs...")
    edges = gpd.read_file(network_path).to_crs(target_crs)
    G_2d, G_3d, node_coords = build_graph_from_edges(
        edges, snap_tolerance=net_cfg.get("snap_tolerance", 5.0))

    G_2d = largest_connected_component(G_2d)
    main_nodes = set(G_2d.nodes())
    G_3d = G_3d.subgraph(main_nodes).copy()
    print(f"    Graph: {G_2d.number_of_nodes()} nodes, {G_2d.number_of_edges()} edges")

    # Candidates
    study_area = gpd.read_file(study_path).to_crs(target_crs)
    study_bounds = study_area.unary_union
    node_ids = sorted(G_2d.nodes())
    node_xy = np.array([node_coords[n] for n in node_ids])
    tree = cKDTree(node_xy)

    stations = gpd.read_file(stations_path).to_crs(target_crs)
    existing_nodes = set()
    for _, st in stations.iterrows():
        d, idx = tree.query([st.geometry.x, st.geometry.y])
        if d < 500:
            existing_nodes.add(node_ids[idx])

    intersection_nodes = {
        n for n in G_2d.nodes()
        if G_2d.degree(n) >= pm_cfg.get("min_intersection_degree", 3)
        and Point(node_coords[n]).within(study_bounds)
    }
    candidate_nodes = list(existing_nodes | intersection_nodes)

    # Demand
    pop = gpd.read_file(pop_path).to_crs(target_crs)
    pop_clipped = gpd.overlay(pop, study_area[["geometry"]], how="intersection")
    pop_field = config.get("field_mapping", {}).get("population_density", "Averag_pop")
    pop_clipped["area_sqkm"] = pop_clipped.geometry.area / 1e6
    pop_clipped["pop_count"] = pop_clipped[pop_field] * pop_clipped["area_sqkm"]
    pop_clipped = pop_clipped[pop_clipped["pop_count"] > 0].copy()

    demand_weights = {}
    for _, row in pop_clipped.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        d, idx = tree.query([cx, cy])
        if d < net_cfg.get("demand_snap_max_dist", 500.0):
            nid = node_ids[idx]
            demand_weights[nid] = demand_weights.get(nid, 0) + row["pop_count"]

    demand_nodes = sorted(demand_weights.keys())
    demand_w_array = np.array([demand_weights[n] for n in demand_nodes], dtype=float)

    # Cost matrices
    cost_2d = build_cost_matrix(G_2d, candidate_nodes, demand_nodes)
    cost_3d = build_cost_matrix(G_3d, candidate_nodes, demand_nodes)

    print(f"    {len(candidate_nodes)} candidates, {len(demand_nodes)} demand, "
          f"pop={demand_w_array.sum():,.0f}")

    # ============================================================
    # 2. Solve for each p
    # ============================================================
    print(f"\n  Solving for p = {p_range[0]}..{p_range[-1]}...")
    all_results = []

    for p in p_range:
        if p > len(candidate_nodes):
            break
        n_d, n_c = cost_2d.shape

        # Solve 2D
        prob_2d = pulp.LpProblem(f"PM_2D_p{p}", pulp.LpMinimize)
        y2 = [pulp.LpVariable(f"y2_{j}", cat="Binary") for j in range(n_c)]
        x2 = [[pulp.LpVariable(f"x2_{i}_{j}", cat="Binary") for j in range(n_c)] for i in range(n_d)]
        prob_2d += pulp.lpSum(demand_w_array[i] * cost_2d[i, j] * x2[i][j]
                              for i in range(n_d) for j in range(n_c))
        for i in range(n_d):
            prob_2d += pulp.lpSum(x2[i][j] for j in range(n_c)) == 1
        for i in range(n_d):
            for j in range(n_c):
                prob_2d += x2[i][j] <= y2[j]
        prob_2d += pulp.lpSum(y2) == p
        prob_2d.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        sel_2d = [j for j in range(n_c) if pulp.value(y2[j]) > 0.5]

        # Solve 3D
        prob_3d = pulp.LpProblem(f"PM_3D_p{p}", pulp.LpMinimize)
        y3 = [pulp.LpVariable(f"y3_{j}", cat="Binary") for j in range(n_c)]
        x3 = [[pulp.LpVariable(f"x3_{i}_{j}", cat="Binary") for j in range(n_c)] for i in range(n_d)]
        prob_3d += pulp.lpSum(demand_w_array[i] * cost_3d[i, j] * x3[i][j]
                              for i in range(n_d) for j in range(n_c))
        for i in range(n_d):
            prob_3d += pulp.lpSum(x3[i][j] for j in range(n_c)) == 1
        for i in range(n_d):
            for j in range(n_c):
                prob_3d += x3[i][j] <= y3[j]
        prob_3d += pulp.lpSum(y3) == p
        prob_3d.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        sel_3d = [j for j in range(n_c) if pulp.value(y3[j]) > 0.5]

        # Evaluate
        ev_2d = evaluate_pmedian(cost_2d, demand_w_array, sel_2d, cov_thresholds)
        ev_3d = evaluate_pmedian(cost_3d, demand_w_array, sel_3d, cov_thresholds)
        ev_cross = evaluate_pmedian(cost_3d, demand_w_array, sel_2d, cov_thresholds)

        row = {
            "p": p,
            "mean_2d_min": ev_2d["pw_mean_min"], "mean_3d_min": ev_3d["pw_mean_min"],
            "mean_2d_under_3d_min": ev_cross["pw_mean_min"],
            "max_2d_min": ev_2d["max_min"], "max_3d_min": ev_3d["max_min"],
            "obj_2d": pulp.value(prob_2d.objective), "obj_3d": pulp.value(prob_3d.objective),
            "n_common_2d_3d": len(set(sel_2d) & set(sel_3d)),
            "retained_2d": sum(1 for j in sel_2d if candidate_nodes[j] in existing_nodes),
            "retained_3d": sum(1 for j in sel_3d if candidate_nodes[j] in existing_nodes),
        }
        for thr in cov_thresholds:
            row[f"cov{thr}_2d"] = ev_2d[f"cov_{thr}min"]
            row[f"cov{thr}_3d"] = ev_3d[f"cov_{thr}min"]
            row[f"cov{thr}_cross"] = ev_cross[f"cov_{thr}min"]

        all_results.append(row)
        print(f"    p={p:2d}: 2D={ev_2d['pw_mean_min']:.1f}min, 3D={ev_3d['pw_mean_min']:.1f}min, "
              f"cross={ev_cross['pw_mean_min']:.1f}min | "
              f"cov5={ev_3d['cov_5min']:.0f}%, common={row['n_common_2d_3d']}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "diminishing_returns_full.csv", index=False)

    # ============================================================
    # 3. Curve Fitting
    # ============================================================
    print("\n  Curve fitting...")
    p_arr = df["p"].values
    t2, t3 = df["mean_2d_min"].values, df["mean_3d_min"].values
    c2, c3 = df["cov5_2d"].values, df["cov5_3d"].values

    a2, b2, r2_t2 = fit_power_law(p_arr, t2)
    a3, b3, r2_t3 = fit_power_law(p_arr, t3)
    print(f"    Travel time power-law:")
    print(f"      2D: T(p) = {a2:.1f} × p^(-{b2:.3f})  R²={r2_t2:.4f}")
    print(f"      3D: T(p) = {a3:.1f} × p^(-{b3:.3f})  R²={r2_t3:.4f}")

    c_max2, k2, r2_c2 = fit_saturating_exponential(p_arr, c2)
    c_max3, k3, r2_c3 = fit_saturating_exponential(p_arr, c3)
    print(f"    5-min coverage saturating exponential:")
    print(f"      2D: C(p) = {c_max2:.1f}% × (1 − e^(-{k3:.4f}×p))  R²={r2_c2:.4f}")
    print(f"      3D: C(p) = {c_max3:.1f}% × (1 − e^(-{k3:.4f}×p))  R²={r2_c3:.4f}")

    # Half-saturation
    p_half_2d = -np.log(0.5) / k2
    p_half_3d = -np.log(0.5) / k3
    print(f"    Half-saturation: 2D={p_half_2d:.1f}, 3D={p_half_3d:.1f} stations")

    # ============================================================
    # 4. Optimal-P Detection
    # ============================================================
    opt_p, methods = find_optimal_p_consensus(p_arr, c3, t3)
    print(f"\n  Optimal p (consensus): {opt_p}")
    print(f"    Methods: {methods}")

    opt_df = pd.DataFrame([{"method": k, "optimal_p": v} for k, v in methods.items()])
    opt_df.to_csv(output_dir / "optimal_p_methods.csv", index=False)

    # Marginal
    marginal = -np.diff(t3)
    m_df = pd.DataFrame({
        "p_from": p_arr[:-1], "p_to": p_arr[1:],
        "marginal_reduction_min": marginal,
        "below_threshold": marginal < marginal_thr,
    })
    m_df.to_csv(output_dir / "marginal_analysis.csv", index=False)

    # ============================================================
    # 5. Summary
    # ============================================================
    summary = {
        "optimal_p_consensus": opt_p,
        "power_law_a_2d": a2, "power_law_b_2d": b2,
        "power_law_a_3d": a3, "power_law_b_3d": b3,
        "exp_cmax_2d": c_max2, "exp_k_2d": k2,
        "exp_cmax_3d": c_max3, "exp_k_3d": k3,
        "half_saturation_2d": p_half_2d, "half_saturation_3d": p_half_3d,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "diminishing_returns_data.csv", index=False)

    # ============================================================
    # 6. Publication Figures
    # ============================================================
    print("  Generating diminishing returns figures...")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: Travel time power-law
        ax = axes[0, 0]
        x_fit = np.linspace(p_arr[0], p_arr[-1], 100)
        ax.scatter(p_arr, t2, c="#534AB7", s=60, zorder=5, label="2D")
        ax.scatter(p_arr, t3, c="#D85A30", s=60, zorder=5, label="3D")
        ax.plot(x_fit, a2 * x_fit**(-b2), "--", c="#534AB7", lw=2)
        ax.plot(x_fit, a3 * x_fit**(-b3), "--", c="#D85A30", lw=2)
        ax.axvline(opt_p, color="#1D9E75", ls=":", lw=1.5, label=f"Optimal p={opt_p}")
        ax.set(xlabel="Stations (p)", ylabel="Mean travel time (min)",
               title=f"A. Travel Time: T(p)=a·p^(-b)\n2D: a={a2:.1f}, b={b2:.3f} | 3D: a={a3:.1f}, b={b3:.3f}")
        ax.legend(); ax.grid(alpha=0.3)

        # Panel B: Coverage exponential
        ax = axes[0, 1]
        ax.scatter(p_arr, c2, c="#534AB7", s=60, zorder=5)
        ax.scatter(p_arr, c3, c="#D85A30", s=60, zorder=5)
        ax.plot(x_fit, c_max2 * (1 - np.exp(-k2 * x_fit)), "--", c="#534AB7", lw=2)
        ax.plot(x_fit, c_max3 * (1 - np.exp(-k3 * x_fit)), "--", c="#D85A30", lw=2)
        ax.axvline(opt_p, color="#1D9E75", ls=":", lw=1.5)
        ax.set(xlabel="Stations (p)", ylabel="5-min coverage (%)",
               title=f"B. Coverage: C(p)=Cmax·(1-e^(-kp))\n2D: Cmax={c_max2:.0f}%, k={k2:.4f} | 3D: Cmax={c_max3:.0f}%, k={k3:.4f}")
        ax.grid(alpha=0.3)

        # Panel C: Marginal benefit
        ax = axes[1, 0]
        ax.bar(p_arr[:-1], marginal, width=0.6, color="#534AB7", alpha=0.7)
        ax.axhline(marginal_thr, color="#D85A30", ls="--", lw=1.5,
                   label=f"Threshold: {marginal_thr} min")
        ax.set(xlabel="p (additional station)", ylabel="Reduction in mean time (min)",
               title="C. Marginal Benefit of Adding Stations")
        ax.legend(); ax.grid(alpha=0.3, axis="y")

        # Panel D: Cross-evaluation gap
        ax = axes[1, 1]
        ax.plot(p_arr, t3, "o-", c="#D85A30", lw=2, ms=6, label="3D optimised → 3D cost")
        ax.plot(p_arr, df["mean_2d_under_3d_min"].values, "s--", c="#534AB7", lw=2, ms=6,
                label="2D optimised → 3D cost")
        ax.fill_between(p_arr, df["mean_2d_under_3d_min"].values, t3,
                        color="#EF9F27", alpha=0.25, label="Suboptimality gap")
        ax.set(xlabel="Stations (p)", ylabel="Mean travel time under 3D (min)",
               title="D. Cost of Ignoring Terrain")
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / "Fig_diminishing_returns.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"    [WARN] Figures failed: {e}")

    print(f"  ✓ Exported: diminishing_returns_full.csv ({len(df)} rows, {len(df.columns)} cols)")
    print(f"    Optimal station count: p ≈ {opt_p}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step05_diminishing_returns.py config.json")
