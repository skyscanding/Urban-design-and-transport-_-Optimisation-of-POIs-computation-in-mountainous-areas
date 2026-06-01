# Mountainous Terrain Optimisation Toolkit

3D terrain-aware location-allocation optimisation for urban logistics facility planning — developed and validated on the **Sai Ying Pun, Hong Kong** case study.

## What's Inside

| Tool | Description | Runtime | License |
|------|-------------|---------|---------|
| [**P-Median Pipeline**](README.md#p-median-pipeline) | 8-step Python pipeline: POI filtering → network merge → DEM+Tobler impedance → P-median ILP → diminishing returns → drone MCLP → capsule routing → figures | Python 3.x (free) | MIT |

## P-Median Pipeline

A comprehensive **Python 3.x** pipeline for terrain-aware location-allocation optimisation of last-mile logistics stations in hillside urban environments. Uses Tobler's hiking function for 3D walking impedance and Integer Linear Programming (PuLP/CBC) for P-median facility location.

**What you can do:**
- Filter and classify logistics POIs from raw Amap/OSM data (courier storefronts vs B2B freight offices)
- Merge vehicle roads + footways + steps into a unified pedestrian network
- Overlay DEM elevation data on road segments
- Apply Tobler's hiking function for slope-dependent walking speeds
- Solve P-median location-allocation with 2D (flat) vs 3D (terrain) walking costs
- Analyse diminishing returns across p = 3 to 20 stations
- Cross-evaluate: quantify the cost of ignoring terrain in facility planning
- Extension: drone landing pad MCLP optimisation with LUCC building classification
- Extension: capsule pipeline routing via SPT and Steiner Tree algorithms
- Generate publication-quality figures (maps, histograms, curves, coverage tables)

Includes 8 standalone Python scripts, fully configurable via JSON, with the Sai Ying Pun case study as a worked example.

📖 **[Read the full documentation →](README.md#pipeline-documentation)**

## Case Study: Sai Ying Pun, Hong Kong

The pipeline was developed and validated for last-mile logistics station planning in **Sai Ying Pun** — a 2.64 km² mixed residential/commercial district on Hong Kong Island with extreme topographic variation.

| Parameter | Value |
|-----------|-------|
| Study area | 2.64 km² |
| Elevation range | 0–536 m |
| Road segments (merged) | 3,630 (vehicle + footways + steps) |
| Existing last-mile stations | 7 (SF Express, FedEx, Post Offices) |
| Candidate locations | 367 (existing stations + ≥3-degree intersections) |
| Demand points | 93 (from 105 population polygons) |
| Total population | ~149,890 |
| Optimal station count | p ≈ 10 |

### Key Findings

- **Current stations severely suboptimal.** All 7 existing stations cluster along the waterfront; the P-median optimum retains 0 of them.
- **Terrain matters.** Mean 3D/2D walking time ratio is 1.416 — terrain adds ~42% travel time on steep segments. 397 segments exceed a 2× 3D/2D ratio.
- **10 stations is the investment threshold.** Marginal benefit drops below 0.1 min/station beyond p=10.
- **Functional forms are universal.** Mean travel time follows T(p) = a × p⁻ᵇ; coverage follows C(p) = Cmax × (1 − e⁻ᵏᵖ). Terrain shifts parameters but preserves the form.
- **2D planning has measurable cost.** When 2D-optimised stations face 3D terrain reality, residents experience longer travel times — the "suboptimality gap."

### Three Planning Options Compared

| Option | Method | Stations | Footprint | Coverage (5-min) |
|--------|--------|----------|-----------|-------------------|
| **Opt 1: Relocation** | P-median p=10, terrain-bound walking | 10 | ~0 m² new infra | 76.5% |
| **Opt 2: Capsule Pipeline** | Steiner Tree, Centre St corridor | 10 terminals | ~1,500 m² | 70-80% |
| **Opt 3: Drone MCLP** | Rooftop pads, terrain-free flight | 10 pads | ~150 m² | 77.7% |

---

## Repository Structure

```
mountainous-optimisation/
├── README.md                       ← You are here
├── .gitignore
├── LICENSE
├── requirements.txt
├── config_template.json
└── scripts/
    ├── _utils.py                    ← Shared: DEM, snapping, graph, evaluation
    ├── master_pipeline.py           ← Orchestrator — run this
    ├── step01_filter_classify_pois.py   ← POI keyword matching + classification
    ├── step02_merge_network.py          ← Merge roads + footways + steps
    ├── step03_dem_overlay_tobler.py     ← DEM sampling + Tobler's function
    ├── step04_pmedian_optimization.py   ← NetworkX graph + P-median ILP
    ├── step05_diminishing_returns.py    ← Multi-p analysis + curve fitting
    ├── step06_drone_mclp.py             ← Drone pad MCLP with LUCC filtering
    ├── step07_capsule_pipeline.py       ← Capsule pipe SPT vs Steiner Tree
    └── step08_publication_figures.py    ← Publication-quality figures
```

## Quick Start

### Prerequisites

- **Python 3.8+** with `geopandas`, `networkx`, `rasterio`, `scipy`, `shapely`, `pulp`, `matplotlib`
- Optional: `contextily` for basemap tiles

```bash
pip install -r requirements.txt
```

### Verify your environment

```bash
python -c "import geopandas, networkx, rasterio, scipy, pulp, matplotlib; print('All OK')"
```

### 1. Prepare your data

At minimum you need:

| Layer | Geometry | Required Fields | Notes |
|-------|----------|-----------------|-------|
| **Study area** | Polygon | — | Your analysis boundary |
| **POI data** | Point | `name`, `type` or `midType` | Amap/OSM format |
| **Road network** | Line | Length | Vehicle roads |
| **Footway lines** | Line | — | OSM highway=footway |
| **Steps lines** | Line | — | OSM highway=steps |
| **DEM** | Raster | — | GeoTIFF, elevation in metres |
| **Population** | Polygon | Population density (e.g. `Averag_pop`) | Census units |
| **Buildings** | Polygon | Elevation/height | Footprint polygons |

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
  "footway_lines_path": "D:/data/Footway_lines.gpkg",
  "steps_lines_path": "D:/data/Steps_lines.gpkg",
  "poi_paths": ["D:/data/poi_life_services.shp", "D:/data/poi_companies.shp"],
  "pop_path": "D:/data/HK_population.gpkg",
  "buildings_path": "D:/data/HK_Buildings.gpkg",
  "p_median": {
    "p_values": [3, 5, 7, 10, 15, 20],
    "snap_tolerance": 5.0,
    "demand_snap_max_dist": 500.0,
    "solver_time_limit": 300
  },
  "tobler": {
    "stair_penalty": 0.45,
    "footway_factor": 0.88,
    "slope_clamp": 0.6,
    "speed_flat_road": 4.5,
    "speed_flat_footway": 4.0,
    "speed_flat_steps": 1.8
  }
}
```

### 3. Run

```bash
# Full pipeline
python scripts/master_pipeline.py my_config.json

# Specific steps only
python scripts/master_pipeline.py my_config.json --steps filter,merge,tobler

# Standalone: run a single step with CLI arguments
python scripts/step04_pmedian_optimization.py \
    --network D:/output/walkable_network.gpkg \
    --stations D:/output/lastmile_stations.shp \
    --pop D:/data/HK_population.gpkg \
    --p 10 --output-dir D:/output
```

---

# 中文说明

面向山地城市环境的 3D 地形感知选址-分配优化工具包 —— 基于 **香港西营盘** 案例开发与验证。

## 包含工具

| 工具 | 说明 | 运行环境 | 协议 |
|------|------|---------|------|
| **P-Median 优化管道** | 8 步 Python 自动化管道：POI 筛选 → 网络合并 → DEM+Tobler 地形阻抗 → P-median 整数线性规划 → 边际收益分析 → 无人机 MCLP → 胶囊管道布线 → 论文图表 | Python 3.x（免费） | MIT |

## 核心能力

- 从原始高德地图/OSM POI 数据中筛选并分类物流站点（快递门店 vs B2B 货运办公室）
- 合并机动车道 + 步行道 + 阶梯为统一行人网络
- DEM 高程叠加与 Tobler 远足函数坡度感知步行速度计算
- 2D（平地）vs 3D（地形）双场景 P-median 求解
- p=3~20 边际收益分析与最优站点数检测
- 交叉评估：量化"忽略地形"对设施规划的代价
- 扩展：基于 LUCC 用地分类的无人机屋顶起降点 MCLP 优化
- 扩展：胶囊管道 SPT 与 Steiner Tree 布线对比
- 论文级图表输出（地图、柱状图、曲线、覆盖表格）

含 8 个独立 Python 脚本，通过 JSON 配置文件全参数化控制，以西营盘案例为完整示例。

📖 **[阅读完整文档 →](README.md#pipeline-documentation)**

## 案例研究：香港西营盘

本管道为香港岛西营盘（2.64 km² 高密度混合用途区）末端物流站点规划开发与验证。

- **数据**：3,630 条道路分段、367 个候选位置、93 个需求点、约 149,890 人口
- **高程范围**：0–536m
- **关键发现**：现有 7 个站点全部分布在滨海低地，P-median 最优解保留 0 个；地形使步行时间平均增加 42%；p≈10 为最优投资阈值；2D 规划假设会导致可衡量的服务水平下降

## 仓库结构

与上方英文部分相同。

## 快速开始

与上方英文部分相同。

## License

MIT
