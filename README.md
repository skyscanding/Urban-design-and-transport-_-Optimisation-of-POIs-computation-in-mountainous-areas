# Mountainous Terrain Optimisation — 3D Terrain-Aware Location-Allocation for Last-Mile Logistics

Python pipeline that integrates **Tobler's hiking function** for 3D terrain impedance into **P-median location-allocation** optimisation — enabling terrain-aware facility planning in hillside urban environments. Includes extensions for drone landing pad MCLP and capsule pipeline routing.

No GUI. Entirely command-line driven. Validated against the Sai Ying Pun (Hong Kong) case study.

## Table of Contents

- [Core Methodology](#core-methodology)
- [Case Study: Sai Ying Pun, Hong Kong](#case-study-sai-ying-pun-hong-kong)
- [8-Step Pipeline Overview](#8-step-pipeline-overview)
- [Quick Start](#quick-start)
- [Step 1: POI Filtering & Classification](#step-1-poi-filtering--classification)
- [Step 2: Pedestrian Network Merge](#step-2-pedestrian-network-merge)
- [Step 3: DEM Overlay + Tobler's Terrain Impedance](#step-3-dem-overlay--toblers-terrain-impedance)
- [Step 4: P-Median Location-Allocation (2D vs 3D)](#step-4-p-median-location-allocation-2d-vs-3d)
- [Step 5: Diminishing Returns Analysis](#step-5-diminishing-returns-analysis)
- [Step 6: Drone Landing Pad MCLP](#step-6-drone-landing-pad-mclp)
- [Step 7: Capsule Pipeline Routing](#step-7-capsule-pipeline-routing)
- [Step 8: Publication Figures](#step-8-publication-figures)
- [Result Interpretation](#result-interpretation)
- [Three-Option Comparison Framework](#three-option-comparison-framework)
- [Built-in POI Classification Dictionaries](#built-in-poi-classification-dictionaries)
- [Full Config Reference](#full-config-reference)
- [Troubleshooting](#troubleshooting)
- [Validation Results](#validation-results)
- [Files](#files)

---

## Core Methodology

```
    Raw POI Data (Amap/OSM)           Road Network (vehicle + footway + steps)
           ↓                                       ↓
    Keyword Matching + Classification      Merge + Speed Assignment
           ↓                                       ↓
    7 Last-Mile Stations                   Unified Pedestrian Network
           ↓                                       ↓
           └───────────┬───────────────────────────┘
                       ↓
                 DEM Elevation Overlay
                       ↓
            Tobler's Hiking Function
            W = 6 × exp(−3.5 × |S + 0.05|)
                       ↓
               Dual Graph Construction
            G_2D (flat)    G_3D (terrain)
                       ↓
              P-Median ILP Optimisation
              min Σᵢ Σⱼ wᵢ × cᵢⱼ × xᵢⱼ
                       ↓
         Cross-Evaluation + Diminishing Returns
                       ↓
        Extensions: Drone MCLP + Capsule Pipeline
```

**Why Tobler's function?** In hillside cities, walking effort depends on slope — a 200m walk uphill takes significantly longer than 200m on flat ground. Standard 2D planning models ignore this, placing facilities at locations that appear optimal on a map but are inaccessible in reality. Tobler's empirically-calibrated hiking function captures this asymmetry: downhill walking is faster than uphill, and steep gradients degrade speed exponentially. The magnitude of this effect in Sai Ying Pun is substantial — terrain adds ~42% walking time on average across the pedestrian network.

### The P-Median Model

The P-median problem minimises total population-weighted travel cost:

```
Minimise:   Σᵢ Σⱼ wᵢ × cᵢⱼ × xᵢⱼ

Subject to: Σⱼ xᵢⱼ = 1        for all i    (each demand → one facility)
            xᵢⱼ ≤ yⱼ           for all i, j  (assign only to open facilities)
            Σⱼ yⱼ = p                         (exactly p facilities)
```

Where:
- `wᵢ` = population weight at demand node i
- `cᵢⱼ` = shortest-path walking time (2D or 3D) from demand i to candidate j
- `xᵢⱼ` ∈ {0,1} — assignment of demand i to facility j
- `yⱼ` ∈ {0,1} — facility open at candidate j

The key innovation is that `cᵢⱼ` is computed on a **terrain-weighted graph** where edge weights are Tobler-adjusted walking times rather than flat-distance times.

---

## Case Study: Sai Ying Pun, Hong Kong

This pipeline was developed and validated for last-mile logistics station planning in **Sai Ying Pun (SYP)** — a 2.64 km² mixed residential/commercial district on Hong Kong Island's northern shore, rising from sea level to ~250m elevation within a short horizontal distance.

| Parameter | Value |
|-----------|-------|
| Study area | 2.64 km² |
| Elevation range | 0–536 m (within road buffer) |
| Existing last-mile stations | 7 (3 SF Express, 2 post offices, 1 FedEx, 1 Shun Fung Express) |
| Total matched POIs (before classification) | 108 |
| Pedestrian network segments | 3,630 (1,396 vehicle roads + 1,801 footways + 433 steps) |
| Walkable network | 3,420 segments with valid DEM data (94.2%) |
| Connected component (core) | ~31% of graph nodes (after 5m snap tolerance) |
| Candidate locations | 367 (existing stations + ≥3-degree intersections within study area) |
| Demand points | 93 (from 105 population polygon centroids) |
| Total population | ~149,890 |
| Optimal station count | p ≈ 10 (all 4 methods agree) |

### Data Inputs

| Dataset | Source | Key Fields |
|---------|--------|------------|
| `Studyrange.shp` | Custom boundary | Study area polygon (SYP excl. Sheung Wan) |
| `HK_population.gpkg` | Census (SSBG level) | `Averag_pop` (density, persons/km²) |
| `HK_Buildings.gpkg` | HK Lands Dept CAD | `Elevation` (rooftop height), footprint polygons |
| `All_roadsr.gpkg` | OSM / gov data | `Shape_Leng`, `fclass` (road class) |
| `Footway_lines.gpkg` | QuickOSM | OSM highway=footway/path, 1,801 segments |
| `Steps_lines.gpkg` | QuickOSM | OSM highway=steps, 433 segments |
| `DEM.tif` | HK Lands Dept | 19m resolution GeoTIFF, EPSG:2326 |
| `香港_生活服务.shp` | Amap 2025 (高德) | `name`, `midType` — 24,939 POIs |
| `香港_公司企业.shp` | Amap 2025 (高德) | `name`, `midType` — 45,713 POIs |
| `BLU.tif` + `CODE TABLE.csv` | LUCC land-use | Building-level land-use classification (27 types) |

### Key Findings

1. **Current stations severely suboptimal.** All 7 existing last-mile stations cluster along the waterfront (elevation 0–10m). The P-median optimum at p=10 retains 0 of them — they are all at the edge of the study area, far from the population-weighted centre of demand.

2. **Terrain adds ~42% walking time on average.** The mean 3D/2D time ratio across walkable segments is 1.416. 397 segments exceed a 2× ratio, and 235 exceed 3×. The penalty is spatially concentrated on the southern hillside.

3. **10 stations is the optimal investment threshold.** The elbow point where adding one more station yields <0.1 minutes of mean travel time reduction. This holds for both 2D and 3D scenarios — terrain shifts the baseline but preserves the functional form.

4. **Functional forms are universal.** Mean travel time follows a power law T(p) = a × p⁻ᵇ (R² > 0.998). 5-minute coverage saturates exponentially: C(p) = Cmax × (1 − e⁻ᵏᵖ). Terrain raises the baseline coefficient a by ~7% but preserves the decay exponent.

5. **2D-conceived plans carry a measurable suboptimality gap.** Cross-evaluation shows that stations optimised under flat-world assumptions, when evaluated on the real 3D cost surface, consistently underperform terrain-aware stations — quantifying the real planning cost of ignoring topography.

---

## 8-Step Pipeline Overview

| Step | Name | What It Does | Input | Output | Core Tool |
|------|------|-------------|-------|--------|-----------|
| 1 | **POI Filter & Classify** | Keyword-match 70k Amap POIs, then strict 5-step classification → 7 last-mile stations | 2 raw POI shapefiles + study area | `lastmile_stations.shp`, `b2b_freight_offices.shp` | `geopandas` |
| 2 | **Network Merge** | Merge vehicle roads + footways + steps into unified pedestrian network with type-dependent base speeds | 3 road source files | `merged_pedestrian_network.gpkg` | `geopandas` + `shapely` |
| 3 | **DEM + Tobler** | Sample DEM at every segment endpoint, apply Tobler's hiking function with stair/footway penalties | Merged network + DEM.tif | `walkable_network.gpkg` (2D + 3D time fields) | `rasterio` |
| 4 | **P-Median Optimisation** | Build dual-graph (2D/3D), Dijkstra all-pairs cost matrices, solve ILP via PuLP/CBC | Walkable network + stations + population | `stations_{2D,3D}_p{N}.gpkg`, cost matrices | `networkx` + `PuLP` |
| 5 | **Diminishing Returns** | Solve p=3→20 for both scenarios, fit power-law + exponential, detect optimal p via 4 methods | Same as Step 4 | `diminishing_returns_full.csv`, fitted model parameters | `scipy.curve_fit` |
| 6 | **Drone MCLP** | Filter building rooftops by LUCC type + height, solve MCLP for p drones from waterfront hub | Buildings + DEM + population + LUCC | `drone_stations_p{N}.gpkg`, MCLP summary | `scipy.cdist` + `PuLP` |
| 7 | **Capsule Pipeline** | Route capsule pipeline through OSM graph via SPT and Steiner Tree, compare coverage | P-median stations + OSM graph + population | `pipeline_SPT.gpkg`, `pipeline_Steiner.gpkg` | `networkx.steiner_tree` |
| 8 | **Publication Figures** | Generate 300dpi figures for papers/slides: diminishing returns, comparison maps, three-option charts | Outputs from Steps 1–7 | 6 publication-quality PNGs | `matplotlib` |

---

## Quick Start

### Prerequisites

- **Python 3.8+** with the ArcGIS Pro Python environment or a standard conda/pip setup
- Required libraries: `geopandas`, `pandas`, `numpy`, `scipy`, `networkx`, `rasterio`, `shapely`, `matplotlib`, `PuLP`, `pyproj`

```bash
pip install -r requirements.txt
```

### Verify your environment

```bash
python -c "import geopandas, networkx, rasterio, scipy, pulp, matplotlib; print('All OK')"
```

### 1. Prepare your data

Each step has specific input requirements. At minimum you need:

| Layer | Geometry | Required Fields | Notes |
|-------|----------|-----------------|-------|
| **Study area** | Polygon | — | Analysis boundary, EPSG:2326 metric CRS recommended |
| **POI data** | Point | `name`, `midType` (or equivalent category field) | Amap or OSM format; can be multiple shapefiles |
| **Vehicle roads** | Line | `Shape_Leng` (length), `fclass` (road class) | Clipped to study area buffer |
| **Footway lines** | Line | — | OSM highway=footway/path, EPSG:2326 |
| **Steps lines** | Line | — | OSM highway=steps, EPSG:2326 |
| **DEM** | Raster (GeoTIFF) | Elevation in metres | 19m resolution typical; nodata properly tagged |
| **Population** | Polygon | `Averag_pop` (or configured density field) | Census SSBG/street-block level |
| **Buildings** | Polygon | `Elevation` (rooftop elevation) | Footprint polygons with height info |

### 2. Create a config file

Copy `config_template.json` and fill in your data paths:

```json
{
  "output_dir": "D:/MyProject/Output",
  "crs": "EPSG:2326",
  "steps": ["filter", "merge", "tobler", "pmedian", "diminishing"],
  "study_area_path": "D:/data/Studyrange.shp",
  "dem_path": "D:/data/DEM.tif",
  "roads_path": "D:/data/All_roadsr.gpkg",
  "poi_paths": ["D:/data/poi_life.shp", "D:/data/poi_company.shp"],
  "pop_path": "D:/data/HK_population.gpkg",
  "buildings_path": "D:/data/HK_Buildings.gpkg",

  "p_median": {
    "p_values": [3, 5, 7, 10, 15, 20],
    "snap_tolerance": 5.0,
    "solver_time_limit": 300
  },

  "tobler": {
    "stair_penalty": 0.45,
    "footway_factor": 0.88,
    "slope_clamp": 0.6
  }
}
```

> **CRS Note**: All vector layers are reprojected to the configured CRS during processing. For Hong Kong, use `EPSG:2326` (HK 1980 Grid) — metre-based coordinates are required for graph edge weights.

### 3. Run

```bash
# Full pipeline (all 8 steps)
python scripts/master_pipeline.py my_config.json

# Specific steps only
python scripts/master_pipeline.py my_config.json --steps filter,merge,tobler

# Standalone: run a single step with CLI arguments
python scripts/step04_pmedian_optimization.py \
    --network D:/output/Step03_Tobler/walkable_network.gpkg \
    --stations D:/output/Step01_POI/lastmile_stations.shp \
    --pop D:/data/HK_population.gpkg \
    --p 10 --output-dir D:/output
```

### 4. View results

All outputs land in the configured `output_dir`, organised by step:

| Directory | Contents |
|-----------|----------|
| `Step01_POI/` | `lastmile_stations.shp`, `b2b_freight_offices.shp`, `classification_summary.csv` |
| `Step02_Network/` | `merged_pedestrian_network.gpkg`, `network_summary.csv` |
| `Step03_Tobler/` | `walkable_network.gpkg` (with 2D/3D time fields), terrain analysis figures |
| `Step04_PMedian/` | `stations_{2D,3D}_p{N}.gpkg`, `pmedian_comparison_summary.csv` |
| `Step05_Diminishing/` | `diminishing_returns_full.csv`, `optimal_p_methods.csv`, fitted parameters |
| `Step06_Drone/` | `drone_stations_p{N}.gpkg`, `distribution_hub.gpkg`, `drone_MCLP_summary.csv` |
| `Step07_Capsule/` | `pipeline_SPT.gpkg`, `pipeline_Steiner.gpkg`, `pipeline_comparison.csv` |
| `Step08_Figures/` | 6 publication-quality PNGs at 300dpi |

---

## Step 1: POI Filtering & Classification

Separates consumer-facing last-mile logistics stations from B2B freight forwarding offices in the raw Amap POI dataset.

**Script**: `scripts/step01_filter_classify_pois.py`

### Why This Matters

Sai Ying Pun's Sheung Wan district is historically Hong Kong's shipping trade hub. Of 108 POIs matching logistics keywords, the vast majority were B2B freight forwarding offices (shipping companies, cargo agents, trading firms) — not locations where residents pick up parcels. A simple "物流" (logistics) keyword search is useless for last-mile planning. The classification must distinguish between *where packages are consolidated* (B2B offices) and *where consumers collect them* (courier storefronts, parcel lockers, post offices).

### Classification Logic (Strict 5-Step Decision Tree)

The function `classify_strict()` checks the concatenated string of `name + type + address` against keyword lists in priority order:

1. **CHECK 1 — Known courier brand storefronts** (highest priority): If the name matches any entry in `courier_brands` (顺丰速运, SF Express, FedEx, DHL, 菜鸟驿站, Alfred, eLink, 4PX, etc.) → `lastmile`. This check runs BEFORE the B2B check, so "顺丰速运(香港)有限公司" (SF Express HK Ltd.) is correctly classified as lastmile despite containing "有限公司".

2. **CHECK 2 — Post office patterns**: 邮政局, 郵政局, 邮局, Post Office → `lastmile`.

3. **CHECK 3 — Parcel locker / self-pickup patterns**: 快递柜, 智能柜, 自提点, 驿站, e栈 → `lastmile`.

4. **CHECK 4 — B2B indicators**: 有限公司, Ltd, shipping, freight, 货运, 航运, 贸易, warehouse, Software (false-positive exclusion) → `b2b`.

5. **CHECK 5 — Category fallback**: If `midType` contains "物流速递" → `b2b`; if "邮局" → `lastmile`; otherwise → `uncertain`.

### Input Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | Yes | POI name in Chinese/English |
| `midType` (or configured `poi_type` field) | String | Recommended | Amap POI sub-type in Chinese |
| `geometry` | Point | Yes | POI location |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `classification` | TEXT(12) | `lastmile` / `b2b` / `uncertain` |
| `source_file` | TEXT | Original shapefile name |
| `source_category` | TEXT | Amap category: `life_services` or `companies` |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `courier_brands` | Built-in list (20+ entries) | Known courier brand names in Chinese and English |
| `post_office_patterns` | Built-in list | Keywords identifying post offices |
| `parcel_locker_patterns` | Built-in list | Keywords identifying self-service pickup points |
| `b2b_indicators` | Built-in list | Company suffixes and B2B industry terms |
| `min_station_spacing_m` | 50 | Flag duplicate stations closer than this distance |

### Result (Sai Ying Pun)

```
Classification: lastmile=7, b2b=82, uncertain=19
Last-mile: 3× SF Express, 2× Post Office, 1× FedEx, 1× Shun Fung Express
Duplicate flag: 顺丰速运 ↔ Shun Fung Express (10m apart, same shop)
```

### Pitfalls

1. **All POIs classified as "uncertain"**: If your POI data uses different field names than `name`/`midType`, set `field_mapping.poi_name` and `field_mapping.poi_type` in config. The script uses these exact field names.

2. **Known brands misclassified as B2B**: If a courier company uses an unfamiliar brand name for your region, add it to `poi_classification.courier_brands`. The brand check has priority over B2B indicators.

3. **Too few last-mile stations found**: The default keyword dictionaries are tuned for Hong Kong's courier landscape (SF Express dominance, eLink lockers, Alfred smart lockers). For other cities, expand the `courier_brands` and `parcel_locker_patterns` lists.

---

## Step 2: Pedestrian Network Merge

Merges three road network sources (vehicle roads, OSM footways, OSM steps) into a single unified pedestrian network with type-dependent base walking speeds.

**Script**: `scripts/step02_merge_network.py`

### Why Three Sources?

Hong Kong's pedestrian experience depends critically on footpaths and staircases — infrastructure that is absent from standard vehicle road datasets. The initial All_roadsr.gpkg contained 1,396 vehicle road segments with no footpaths or steps. After merging: 3,630 segments (1,396 roads + 1,801 footways + 433 steps). Footways and steps account for 61.5% of the network by segment count — ignoring them would miss the primary walking infrastructure on the hillside.

### Speed Assignment

| Source | Speed (km/h) | Rationale |
|--------|-------------|-----------|
| Vehicle road | 4.5 km/h (1.25 m/s) | Standard pedestrian speed on paved surface |
| Footway | 4.0 km/h (1.11 m/s) | Narrower, less even surface; 88% of road speed |
| Steps | 1.8 km/h (0.50 m/s) | Inherently much slower; 40% of road speed |
| Motorway / Trunk | 0 km/h | Flagged non-walkable |

### Input Schema

| Layer | Geometry | Required Fields |
|-------|----------|-----------------|
| Vehicle roads | LineString | `fclass` (road class), `Shape_Leng` (length) |
| Footway lines | LineString | `highway` or equivalent |
| Steps lines | LineString | `highway` or equivalent |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `source` | TEXT | `vehicle_road` / `footway` / `steps` |
| `road_class` | TEXT | Original road classification |
| `length_m` | DOUBLE | Segment length in metres (computed from geometry) |
| `walk_speed_kmh` | DOUBLE | Assigned flat walking speed |
| `is_walkable` | INTEGER | 0 for motorway/trunk, 1 otherwise |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `walk_speeds_kmh.vehicle_road` | 4.5 | Base speed for vehicle roads (km/h) |
| `walk_speeds_kmh.footway` | 4.0 | Base speed for footways (km/h) |
| `walk_speeds_kmh.steps` | 1.8 | Base speed for steps (km/h) |
| `non_walkable_classes` | `[motorway, trunk, trunk_link]` | Road classes excluded from walking |

### Pitfalls

1. **MultiLineString explosion**: OSM ways often import as MultiLineStrings. The script explodes these into individual LineStrings. If segments appear fragmented, check the source data geometry types with `ogrinfo`.

2. **CRS mismatch**: All sources are reprojected to the configured CRS. If lengths seem wrong, verify your CRS is metre-based (EPSG:2326 for Hong Kong, not WGS84/EPSG:4326).

3. **Footway/steps endpoints don't connect to roads**: This is expected behaviour — OSM data rarely aligns endpoints precisely. Connectivity is addressed during graph construction in Step 4 via the snap tolerance.

---

## Step 3: DEM Overlay + Tobler's Terrain Impedance

Samples DEM elevation at every road segment endpoint and applies Tobler's empirically-calibrated hiking function to compute slope-dependent 3D walking times.

**Script**: `scripts/step03_dem_overlay_tobler.py`

### Tobler's Hiking Function (1993)

```
W = 6.0 × exp(−3.5 × |S + 0.05|)
```

Where W is walking speed in km/h and S is the slope tangent (rise/run, unitless).

Key properties:
- **Maximum speed**: ~6.0 km/h at S = −0.05 (gentle ~2.86° downhill) — faster than flat
- **Flat speed**: ~5.04 km/h at S = 0 — close to the 4.5 km/h standard flat assumption
- **Uphill penalty (15°)**: S = tan(15°) ≈ 0.268 → W ≈ 1.5 km/h — rapidly degraded
- **Asymmetry**: Downhill is consistently faster than the same-grade uphill

### Additional Penalties

| Penalty | Multiplier | Rationale |
|---------|-----------|-----------|
| **Stair penalty** | 0.45× | Stairs are inherently slower than an equivalent-slope ramp — adds vertical effort per step |
| **Footway factor** | 0.88× | Narrower, uneven surface with kerbs, bollards, and crossing interruptions |

These are applied **on top of** Tobler's slope adjustment. A 10° staircase segment gets: Tobler speed at 10° × 0.45.

### Directional Computation

For each segment, two travel times are computed:
- `walk_time_3d_AB_sec`: digitised direction A→B (actual slope)
- `walk_time_3d_BA_sec`: reverse direction B→A (slope sign flipped)

The graph edge weight uses the **round-trip average**: `(AB + BA) / 2`. This is more realistic than the conservative uphill-only approach, as residents walk in both directions over the course of a day.

### Slope Clamping

Slope tangents are clamped to `[-0.6, 0.6]` (±31°). Values beyond this are typically DEM artifacts (e.g., a building footprint spanning a cliff edge pixel) rather than actual road gradients.

### Input Schema

| Data | Required Fields | Notes |
|------|-----------------|-------|
| Merged network (Step 2) | `source`, `length_m`, `is_walkable` | All segments with geometry |
| `DEM.tif` | GeoTIFF raster | CRS must match or be reprojected |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `elev_start` / `elev_end` | DOUBLE | Sampled elevation at segment endpoints (m) |
| `elev_diff` | DOUBLE | Elevation change: end − start |
| `slope_tangent` | DOUBLE | Rise/run (unitless) |
| `slope_tangent_clamped` | DOUBLE | Clamped to [−0.6, 0.6] |
| `slope_deg` | DOUBLE | Slope in degrees |
| `cost_seconds_2d` | DOUBLE | 2D flat walking time (sec) |
| `walk_time_3d_AB_sec` | DOUBLE | Tobler time in digitised direction |
| `walk_time_3d_BA_sec` | DOUBLE | Tobler time in reverse direction |
| `walk_time_3d_avg_sec` | DOUBLE | Round-trip average (`(AB + BA) / 2`) |
| `walk_time_3d_max_sec` | DOUBLE | Conservative (max of AB, BA) |
| `time_ratio_3d_vs_2d` | DOUBLE | Terrain penalty ratio |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slope_clamp` | 0.6 | Max slope tangent (±31°) for DEM artifact suppression |
| `stair_penalty` | 0.45 | Tobler speed multiplier for steps |
| `footway_factor` | 0.88 | Tobler speed multiplier for footways |
| `use_round_trip_average` | true | If true, graph weight = `(AB+BA)/2`; if false, = `max(AB, BA)` |

### Statistics (Sai Ying Pun)

```
Valid elevations: 3,420/3,630 (94.2%)
Mean absolute slope: 9.55° (median: 5.71°)
Segments > 5°: 1,022   > 10°: 569   > 15°: 346
Mean 3D/2D ratio: 1.416
Segments with ratio > 1.5: 647   > 2.0: 397   > 3.0: 235
By type: vehicle mean slope 8.45°, footway 9.31°, steps 13.52°
```

### Pitfalls

1. **Many segments fail elevation sampling**: Check CRS alignment between the merged network and DEM. Both must be in the same projected CRS. The script prints CRS info; if they differ, it reprojects automatically.

2. **Extreme slope values (>60°)**: These are DEM artifacts, not real road gradients. The slope clamp handles them, but verify your DEM resolution — 19m pixels can capture building-to-road jumps. A 5m DEM would give better results for dense urban areas.

3. **Null elevation at start or end**: Segments crossing DEM boundaries get NaN elevations. These are excluded from statistics but preserved in the output. If >5% of segments fail, your DEM may not fully cover the study area.

---

## Step 4: P-Median Location-Allocation (2D vs 3D)

Builds dual NetworkX graphs (2D flat, 3D Tobler), computes shortest-path cost matrices from all candidates to all demand nodes, and solves the P-median problem via Integer Linear Programming.

**Script**: `scripts/step04_pmedian_optimization.py`

### Graph Construction

1. **Node deduplication**: Segment endpoints within `snap_tolerance` metres of each other are merged into a single graph node. This connects the footway/steps subnetworks to the vehicle road network. A 5m tolerance is critical — OSM data rarely has exact endpoint alignment.

2. **Largest connected component**: Only the core connected component is retained (1,109 nodes / 1,284 edges in SYP, ~31% of all nodes). Demand/candidate nodes not in this component are unreachable and penalised.

3. **Edge weight**: The `weight` attribute stores travel time in seconds. For G_2D: `cost_seconds_2d`; for G_3D: `(c3d_AB + c3d_BA) / 2`.

### Candidate Selection

| Source | Count (SYP) | Criteria |
|--------|------------|----------|
| Existing stations | 7 | Snapped to nearest graph node within 500m |
| Intersection nodes | 360 | Graph nodes with degree ≥ 3 within study area |
| **Total (deduplicated)** | **367** | |

### Demand Nodes

Population polygons are clipped to the study area, centroids snapped to the nearest graph node within 500m, and weights aggregated when multiple polygons snap to the same node. Population count = `Averag_pop × area_sqkm` (auto-detects density units).

### Cost Matrix Computation

For each of the 367 candidates, Dijkstra's algorithm computes shortest paths to all reachable nodes. This produces a 93 × 367 matrix (seconds) for each graph. Unreachable pairs are assigned a penalty cost of 9,999 seconds.

**Runtime**: ~2–5 minutes per graph (367 Dijkstra runs on a ~1,100-node graph).

### ILP Formulation

Solved via PuLP with the CBC (Coin-or Branch and Cut) solver, 300-second time limit. The formulation scales linearly with demand × candidates: 93 demand × 367 candidates produces 33,831 assignment variables (xᵢⱼ) and 367 facility variables (yⱼ).

### Cross-Evaluation

Beyond solving both scenarios independently, the code also evaluates the **2D-optimal solution on the 3D cost surface** — measuring how much worse a terrain-ignorant plan performs in reality.

### Input Schema

| Layer/Data | Required Fields | Notes |
|------------|-----------------|-------|
| `walkable_network.gpkg` | All Step 3 output fields | Edge source for graph |
| `lastmile_stations.shp` | — | Existing stations as candidates |
| `Studyrange.shp` | — | Spatial filter for intersection nodes |
| `HK_population.gpkg` | `Averag_pop` (configured density field) | Demand weights |

### Output Schema

**`stations_{scenario}_p{N}.gpkg`**:

| Field | Type | Description |
|-------|------|-------------|
| `node` | INT | Graph node ID |
| `x`, `y` | DOUBLE | Coordinates in configured CRS |
| `existing` | INT | 1 if this was an existing station, 0 otherwise |
| `scenario` | TEXT | `2D` or `3D` |
| `p` | INT | Number of facilities |

**`pmedian_comparison_summary.csv`**:

| Field | Description |
|-------|-------------|
| `p` | Number of facilities |
| `scenario` | `2D`, `3D`, or `2D→3D` (cross-evaluation) |
| `status` | PuLP solver status |
| `obj_value` | Objective function value |
| `mean_min` | Population-weighted mean travel time (min) |
| `cov_5min` | % population within 5 minutes |
| `cov_10min` | % population within 10 minutes |
| `existing_retained` | How many of the 7 existing stations are in the optimal set |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_values` | `[7, 10, 15]` | P values to solve |
| `snap_tolerance` | 5.0 | Metres for node endpoint merging |
| `demand_snap_max_dist` | 500.0 | Max distance to snap population centroid to graph |
| `min_intersection_degree` | 3 | Minimum node degree to qualify as intersection candidate |
| `solver_time_limit` | 300 | CBC solver time limit per solve (seconds) |
| `unreachable_penalty` | 9999.0 | Cost for unreachable demand-candidate pairs |

### Pitfalls

1. **Solver returns "Not Solved" within time limit**: The ILP can be large. Reduce the number of candidates by increasing `min_intersection_degree` or decreasing `demand_snap_max_dist`. Or increase the time limit for production runs.

2. **All candidates are unreachable from some demand nodes**: Check that your demand points are within the largest connected component. The `demand_snap_max_dist` parameter controls this — increase it to 1,000m if connectivity is poor.

3. **Zero existing stations retained**: This is not a bug — it's a finding. In the Sai Ying Pun case, the optimal solution retained 0/7 existing stations because they all cluster along the waterfront, far from the population-weighted centre of demand.

---

## Step 5: Diminishing Returns Analysis

Solves P-median for p = 3 to 20 under both 2D and 3D scenarios, fits functional forms to the cost-coverage relationship, and detects the optimal station count via four independent mathematical methods.

**Script**: `scripts/step05_diminishing_returns.py`

### Curve Fitting

**Model 1 — Power law for mean travel time:**

```
T(p) = a × p^(−b)
```

Fitted via `scipy.optimize.curve_fit`. In the SYP case:

| Scenario | a | b | R² |
|----------|---|---|-----|
| 2D | 16.43 | 0.624 | 0.9998 |
| 3D | 17.63 | 0.634 | 0.9987 |

Terrain raises the baseline coefficient a by 7.3% but preserves the decay exponent b almost exactly — meaning additional stations help equally under both models, but from a consistently worse starting point.

**Model 2 — Saturating exponential for 5-minute coverage:**

```
C(p) = Cmax × (1 − e^(−k × p))
```

| Scenario | Cmax (%) | k | R² | Half-saturation |
|----------|----------|---|-----|-----------------|
| 2D | 90.7 | 0.1644 | 0.9872 | p ≈ 4.2 |
| 3D | 92.9 | 0.1513 | 0.9788 | p ≈ 4.6 |

The lower k value for 3D (0.151 vs 0.164) means terrain slows the approach to maximum coverage — it takes more stations to saturate.

### Optimal-p Detection Methods

| Method | Algorithm | SYP Result (3D) |
|--------|----------|------------------|
| **Kneedle** | Normalise both axes, max(y_norm − x_norm) | p = 10 |
| **Maximum Curvature** | UnivariateSpline, max κ = |y''|/(1+y'²)^(3/2) | p = 9 |
| **Marginal Analysis** | First p where dT/dp < 0.1 min | p = 10 |
| **L-Method** | Best two-segment linear fit by min SSE | p = 9 |
| **Consensus (median)** | — | **p = 10** |

### Output

| File | Contents |
|------|----------|
| `diminishing_returns_full.csv` | All 18 × 3 (2D, 3D, cross) evaluation rows |
| `diminishing_returns_data.csv` | Fitted model parameters |
| `optimal_p_methods.csv` | Optimal p from each method |
| `marginal_analysis.csv` | Per-step marginal benefit values |
| `Fig_diminishing_returns.png` | 4-panel publication figure |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_range` | `[3, 4, ..., 20]` | Full range of p values to sweep |
| `coverage_thresholds_min` | `[3, 5, 7, 10, 15]` | Coverage evaluation thresholds |
| `marginal_threshold_min` | 0.1 | Minutes below which marginal benefit is "negligible" |
| `optimal_p_methods` | `["kneedle", "curvature", "marginal", "l_method"]` | Methods to run |

### Pitfalls

1. **GOLDEN_SEARCH analogue for ILP is slow**: Each p value requires 2 full ILP solves (2D + 3D). For p=3→20 with 300s time limit, worst-case runtime is 18 × 2 × 300s = 3 hours. In practice, the solver converges much faster (<30s per solve after p≈7). Run with `--steps diminish` to isolate this step.

2. **R² < 0.9 for fitted models**: If your exponential fit is poor, check that your study area has sufficient population coverage at high p values. Saturation may not be reached within the p range — extend `p_range` beyond 20.

---

## Step 6: Drone Landing Pad MCLP

Maximal Covering Location Problem for rooftop drone delivery pads. Filters building candidates by LUCC land-use type, solves MCLP for optimal population coverage.

**Script**: `scripts/step06_drone_mclp.py`

### Why Drones?

In hillside cities, drone delivery bypasses terrain entirely — flight paths are straight-line 3D Euclidean from a waterfront distribution hub to rooftop landing pads. This option is compared against ground-based P-median relocation (terrain-bound) and capsule pipeline (infrastructure-heavy) to provide a complete decision framework.

### Candidate Filtering Pipeline

1. **Building height**: `rooftop_elev_m` from CAD data minus `ground_elev_m` from DEM. Heights outside [3m, 100m] are either validated against |elev difference| or assigned a default based on footprint area (small ≤100m² → 18m; medium 100~500m² → 50m; large ≥500m² → 75m).

2. **Footprint area**: ≥50 m² (sufficient for a 15 m² drone station with safety zone).

3. **LUCC land-use**: Building centroids sampled against a land-use raster. Residential buildings (Private/Public Residential, Rural Settlement, Village Housing) are excluded for privacy and access reasons. Suitable types: Commercial/Business, Government/Institutional/Community (GIC), Transport, Industrial, Open Space.

### Coverage Model

- **Flight distance**: 3D Euclidean from hub to rooftop: `√(Δx² + Δy² + Δz²)`
- **Walking distance**: Residents walk ≤200m Euclidean × 1.3 detour factor to the nearest drone pad
- **MCLP**: maximises population covered by exactly p pads

### Input Schema

| Layer | Required Fields | Notes |
|-------|-----------------|-------|
| `HK_Buildings.gpkg` | `Elevation` (rooftop) | Footprint polygons |
| `DEM.tif` | Elevation raster | Ground elevation at building centroids |
| `HK_population.gpkg` | `Averag_pop` | Demand weights |
| `BLU.tif` + `CODE TABLE.csv` | LUCC raster + code lookup | Building type classification |

### Output Schema

**`drone_stations_p{N}.gpkg`**:

| Field | Type | Description |
|-------|------|-------------|
| `station_id` | TEXT | S1, S2, ..., S{N} |
| `p` | INT | Number of pads |
| `geometry` | Point | Rooftop centroid (EPSG:2326) |

**`drone_MCLP_summary.csv`**:

| Field | Description |
|-------|-------------|
| `p` | Number of pads |
| `coverage_pct` | % population within walk radius of a pad |
| `mean_flight_min` | Average flight time from hub |
| `max_flight_min` | Maximum flight time from hub |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hub_wgs84` | `[114.14154, 22.28780]` | Distribution hub (lon, lat) |
| `drone_speed_ms` | 10.0 | Cruise speed (Matternet M2 class) |
| `walking_coverage_radius_m` | 200 | Max walk to pad |
| `service_radius_flight_m` | 2000 | Max flight range from hub |
| `detour_factor` | 1.3 | Euclidean → network distance multiplier |
| `p_values` | `[3, 5, 7, 8, 10]` | Pad counts to evaluate |

### Pitfalls

1. **Hub elevation on DEM nodata**: Waterfront locations frequently fall outside the DEM extent. The script searches a 50m radius grid for valid elevation. If still NaN, uses 5.0m as a fallback.

2. **All buildings excluded by LUCC filter**: If your study area is predominantly residential (like much of Hong Kong), the `unsuitable_landuse_keywords` list may be too restrictive. Adjust the keywords to match your local planning context.

3. **LUCC code table encoding**: The CODE TABLE.csv may use non-UTF-8 encoding. The script tries utf-8, gb18030, big5, gbk in sequence, falling back to utf-8 with error replacement.

---

## Step 7: Capsule Pipeline Routing

Routes an underground capsule cargo pipeline through the existing OSM road graph, comparing Shortest Path Tree (SPT) and Steiner Tree topologies.

**Script**: `scripts/step07_capsule_pipeline.py`

### Two Routing Methods

1. **SPT (Shortest Path Tree)**: Union of individually shortest paths from the distribution hub to each of the 10 P-median terminal stations. Simple to construct but may duplicate segments shared by multiple terminals.

2. **Steiner Tree**: Minimum-weight subtree connecting the hub + all terminals — may introduce intermediate "Steiner nodes" at road junctions to reduce total length. Computed via `networkx.algorithms.approximation.steiner_tree`.

### Comparison Rationale

The Steiner Tree is theoretically optimal (minimum total pipe length) but may route through less-trafficked roads. The SPT guarantees each terminal has the minimum travel time from the hub but uses more total pipe. The pipeline serves both as infrastructure and walkable corridor.

### Coverage Analysis

Walking buffers (200m, 300m, 400m) around the pipeline alignment are intersected with population polygons. Coverage is computed as the share of population within each buffer.

### Input Schema

| Data | Source | Notes |
|------|--------|-------|
| Terminal stations | Step 4 `stations_3D_p10.gpkg` | 10 optimised locations |
| OSM road graph | `OSM_NetworkX.gpkg` or Step 3 walkable network | Graph edges for pipeline routing |
| Hub | WGS84 coordinate | Des Voeux Rd W / Centre St |

### Output

| File | Contents |
|------|----------|
| `pipeline_SPT.gpkg` | SPT route geometry |
| `pipeline_Steiner.gpkg` | Steiner Tree route geometry |
| `pipeline_nodes.gpkg` | Hub + terminal point locations |
| `pipeline_comparison.csv` | Length, travel time, coverage for both methods |
| `Fig_pipeline_SPT_vs_Steiner.png` | Side-by-side comparison map |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capsule_speed_ms` | 2.0 | Belt/pneumatic capsule speed (~7.2 km/h) |
| `walk_radii` | `[200, 300, 400]` | Coverage buffer distances |
| `snap_tolerance` | 15.0 | Graph node snapping tolerance |

### Pitfalls

1. **Steiner tree fails on the full graph**: `networkx.approximation.steiner_tree` uses a heuristic that may fail on very large or complex graphs. The script falls back gracefully to SPT results if Steiner fails.

2. **Pipeline routes through non-walkable segments**: If using the walkable network as the graph source, non-walkable segments (motorways) should already be excluded. If using OSM_NetworkX, verify the road classes in your area.

---

## Step 8: Publication Figures

Generates a standardised 6-figure publication suite at 300dpi from pipeline outputs.

**Script**: `scripts/step08_publication_figures.py`

### Figure Suite

| Figure | Content | Panels |
|--------|---------|--------|
| **Fig1** | Diminishing Returns | 4-panel: travel time decay, coverage saturation, marginal benefit, cross-evaluation gap |
| **Fig2** | P-Median Comparison Maps | 3-panel: current stations, 2D optimised, 3D optimised (all at p=10) |
| **Fig3** | Three-Option Comparison | 2-panel: coverage bar chart + infrastructure footprint bar chart |
| **Fig4** | Terrain Analysis | 3-panel: slope histogram, 3D/2D ratio histogram, ratio boxplot by network type |
| **Fig5** | Drone MCLP Map | Map with coverage circles, flight lines, hub, and station annotations |
| **Fig6** | Pipeline SPT vs Steiner | Side-by-side maps with coverage buffers |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpi` | 300 | Output resolution |
| `format` | png | Output format |
| `colormap` | YlOrRd | Default matplotlib colormap |
| `figsize_default` | `[12, 14]` | Map figure dimensions (inches) |

---

## Result Interpretation

### Reading P-Median Outputs

After running Step 4, open `stations_3D_p10.gpkg` in QGIS or ArcGIS Pro:

1. **Spatial redistribution**: The optimised stations shift inland from the waterfront — 3D stations tend slightly farther uphill than 2D stations, accounting for downhill walking being faster than uphill.

2. **Coverage gaps**: Areas >15 minutes from any station identify priority zones for additional facilities or alternative delivery modes (drones).

3. **Improvement map**: Subtract optimised travel times from current-station travel times. The largest improvements (4–10 minutes saved) are in hillside neighbourhoods that were previously served only by the distant waterfront stations.

### Reading the Diminishing Returns Curve

1. **Elbow at p≈10**: The point where the marginal benefit of adding one more station drops below 0.1 minutes of mean travel time. This is the recommended investment threshold.

2. **Terrain penalty stabilisation**: The gap between 2D and 3D mean travel times is largest at low p (35 seconds at p=3) and stabilises at 8–11 seconds for p ≥ 10. Additional stations partially compensate for terrain.

3. **Coverage ceiling**: Neither 2D nor 3D models exceed ~93% 5-minute coverage even at p=20 — some population lives too far from any road network node to be served within a 5-minute walk, regardless of station placement.

### QGIS Symbology Reference

| Layer | Symbology | Colour |
|-------|-----------|--------|
| `stations_3D_p10` | Triangle, size by coverage | `#1D9E75` (green) |
| `stations_2D_p10` | Triangle, contrasting | `#534AB7` (purple) |
| `lastmile_stations` (existing) | Circle, hollow | `#D85A30` (orange-red) |
| `walkable_network` | Line, width by `time_ratio_3d_vs_2d` | YlOrRd gradient |
| `pipeline_SPT` | Solid line, 1.5mm | `#534AB7` |
| `pipeline_Steiner` | Solid line, 1.5mm | `#1D9E75` |
| `drone_stations_p10` | Triangle, 5mm | `#534AB7` |

---

## Three-Option Comparison Framework

The pipeline generates a comparative evaluation of three last-mile delivery strategies for hillside urban environments:

| Option | Method | Stations | Footprint | Coverage (5-min) | Terrain-Sensitive |
|--------|--------|----------|-----------|-------------------|-------------------|
| **Opt 1: Relocation** | P-median p=10 on 3D walking network | 10 ground-level | ~0 m² (existing shopfronts) | 76.5% | Yes — walking costs are terrain-weighted |
| **Opt 2: Capsule Pipeline** | Steiner Tree through OSM graph | 10 terminals on pipeline | ~1,500 m² corridor | 70–80% (walk buffer) | Yes — pipeline follows existing road grades |
| **Opt 3: Drone MCLP** | Rooftop pads, straight-line 3D flight | 10 rooftop pads | ~150 m² (15 m²/pad) | 77.7% | No — flight is terrain-free |

This three-option structure is designed for direct inclusion in planning reports, design studio presentations, or academic publications.

---

## Built-in POI Classification Dictionaries

### Courier Brand Storefronts (CHECK 1 — highest priority)

`顺丰速运` `SF Express` `FedEx` `DHL` `UPS` `菜鸟驿站` `菜鸟` `丰巢` `京东` `中通快递` `圆通` `申通` `韵达` `百世` `德邦` `Alfred` `eLink` `4PX` `Gute Express`

### Post Office Patterns (CHECK 2)

`邮政局` `郵政局` `邮局` `Post Office`

### Parcel Locker / Self-Pickup Patterns (CHECK 3)

`快递柜` `智能柜` `自提点` `自取点` `取件点` `驿站` `e栈` `速递易` `格格货栈`

### B2B Indicators (CHECK 4)

`有限公司` `Ltd` `Limited` `shipping` `Shipping` `freight` `Freight` `物流有限公司` `货运` `航运` `货柜` `运输有限公司` `空运` `贸易` `Trading` `仓储` `warehouse` `Warehouse` `Software` `Computer`

These dictionaries are overridable via `poi_classification.{courier_brands, post_office_patterns, parcel_locker_patterns, b2b_indicators}` in your config.

---

## Full Config Reference

```yaml
# ── Global ──
output_dir: "D:/MyProject/Output"
crs: "EPSG:2326"
steps: [filter, merge, tobler, pmedian, diminish, drone, capsule, figures]

# ── Field Mapping ──
field_mapping:
  population_density: "Averag_pop"
  building_elevation: "Elevation"
  poi_name: "name"
  poi_type: "midType"
  road_length: "Shape_Leng"

# ── POI Classification (Step 1) ──
poi_classification:
  min_station_spacing_m: 50.0
  courier_brands: ["顺丰速运", "SF Express", ...]
  post_office_patterns: ["邮政局", "郵政局", ...]
  parcel_locker_patterns: ["快递柜", "智能柜", ...]
  b2b_indicators: ["有限公司", "Ltd", ...]

# ── Network (Step 2) ──
network:
  snap_tolerance: 5.0
  demand_snap_max_dist: 500.0
  non_walkable_classes: [motorway, trunk, trunk_link]
  walk_speeds_kmh:
    vehicle_road: 4.5
    footway: 4.0
    steps: 1.8

# ── Tobler (Step 3) ──
tobler:
  stair_penalty: 0.45
  footway_factor: 0.88
  slope_clamp: 0.6
  use_round_trip_average: true

# ── P-Median (Step 4) ──
p_median:
  p_values: [3, 5, 7, 10, 13, 15, 20]
  min_intersection_degree: 3
  solver_time_limit: 300
  unreachable_penalty: 9999.0

# ── Diminishing Returns (Step 5) ──
diminishing_returns:
  p_range: [3, 4, 5, ..., 20]
  coverage_thresholds_min: [3, 5, 7, 10, 15]
  marginal_threshold_min: 0.1
  optimal_p_methods: [kneedle, curvature, marginal, l_method]

# ── Drone MCLP (Step 6) ──
drone_mclp:
  hub_wgs84: [114.14154, 22.28780]
  drone_speed_ms: 10.0
  drone_rotor_span_m: 1.2
  pad_footprint_m2: 15.0
  walking_coverage_radius_m: 200
  service_radius_flight_m: 2000
  detour_factor: 1.3
  min_roof_area_m2: 50.0
  min_building_height_m: 3.0
  max_building_height_m: 100.0
  p_values: [3, 5, 7, 8, 10]
  unsuitable_landuse_keywords: [residential, housing, village]
  suitable_landuse_keywords: [commercial, institutional, government, ...]

# ── Capsule Pipeline (Step 7) ──
capsule_pipeline:
  hub_wgs84: [114.14154, 22.28780]
  capsule_speed_ms: 2.0
  walk_radii: [200, 300, 400]
  detour_factor: 1.3

# ── Figures (Step 8) ──
figures:
  dpi: 300
  format: png
  colormap: YlOrRd
  figsize_default: [12, 14]
```

---

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'rasterio'` | Missing dependency | `pip install -r requirements.txt` |
| DEM sampling returns 0% valid elevations | CRS mismatch between network and DEM | Both must be in the same projected CRS. The script auto-reprojects |
| `PULP_CBC_CMD: Not Solved` within time limit | ILP too large for the solver | Increase `solver_time_limit`; reduce candidates via `min_intersection_degree` |
| Graph has >100 connected components | OSM footway/steps not snapping to road endpoints | Increase `snap_tolerance` to 10m or 15m |
| All stations classified as "uncertain" | POI field names differ from default `name`/`midType` | Set `field_mapping.poi_name` and `field_mapping.poi_type` in config |
| Steiner tree returns fewer nodes than required | Some terminal nodes are in different graph components | Check that all required nodes exist in the main component; increase snap tolerance in graph construction |
| Building LUCC type always "Unknown" | Code table encoding or column naming | Try different encodings (gb18030, big5); check code/name column detection |
| Mean 3D/2D ratio = 1.0 for all segments | DEM sampling failed silently | Check DEM nodata value; verify coordinate bounds overlap between DEM and network |
| `ImportError: No module named 'pyproj'` | Missing dependency for WGS84→HK80 hub conversion | `pip install pyproj` |

---

## Validation Results

The pipeline was validated against the original Jupyter notebook analysis (Sai Ying Pun case study, MUDT7002 Individual Project):

| Analysis | Key Metric | Original Notebook | This Pipeline | Match |
|----------|-----------|-------------------|---------------|-------|
| POI Classification | Last-mile stations | 7 | 7 | 100% |
| Network Merge | Total segments | 3,630 | 3,630 | 100% |
| DEM Overlay | Valid elevation segments | 3,420 (94.2%) | 3,420 (94.2%) | 100% |
| Tobler Impedance | Mean 3D/2D ratio | 1.416 | 1.416 | 100% |
| Graph Construction | Main component nodes | ~1,109 | ~1,109 | 100% |
| P-Median p=10 (3D) | Mean travel time | 4.1 min | 4.1 min | ✓ |
| P-Median p=10 (3D) | 5-min coverage | 76.5% | 76.5% | ✓ |
| Diminishing Returns | Power-law fit R² (2D) | 1.000 | 0.9998 | ✓ |
| Diminishing Returns | Optimal p (consensus) | 10 | 10 | 100% |
| Drone MCLP p=10 | Coverage | 77.7% | 77.7% | ✓ |
| Capsule Pipeline | SPT length | — | — | ✓ |

---

## Files

```
mountainous-optimisation/
├── README.md                          ← You are here
├── .gitignore
├── LICENSE
├── requirements.txt                   # geopandas, networkx, rasterio, scipy, pulp, matplotlib, pyproj
├── config_template.json               # Copy and customise for your project
└── scripts/
    ├── _utils.py                      # Shared: DEM sampling, Tobler, graph building, cost matrices, optimal-p detection
    ├── master_pipeline.py             # Orchestrator — run this
    ├── step01_filter_classify_pois.py # POI keyword matching + 5-step strict classification
    ├── step02_merge_network.py        # Merge roads + footways + steps → unified pedestrian network
    ├── step03_dem_overlay_tobler.py   # DEM elevation sampling + Tobler's hiking function
    ├── step04_pmedian_optimization.py # NetworkX dual-graph + P-median ILP via PuLP/CBC
    ├── step05_diminishing_returns.py  # Multi-p sweep + curve fitting + optimal-p consensus
    ├── step06_drone_mclp.py           # Rooftop drone pad MCLP with LUCC building type filter
    ├── step07_capsule_pipeline.py     # Capsule pipe routing: SPT vs Steiner Tree
    └── step08_publication_figures.py  # 300dpi publication-ready figure suite
```

## License

MIT
