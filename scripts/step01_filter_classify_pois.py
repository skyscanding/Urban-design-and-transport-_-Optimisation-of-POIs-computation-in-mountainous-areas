#!/usr/bin/env python3
"""
Step 01: Filter and Classify Logistics POIs
============================================
Loads raw Amap/OSM POI data, performs keyword-based matching and strict
classification to separate last-mile consumer stations from B2B freight offices.

Inputs:
  - POI shapefiles (name, type/midType fields)
  - Study area boundary

Outputs:
  - lastmile_stations.shp/geojson (consumer-facing pickup points)
  - b2b_freight_offices.shp (commercial freight offices)
  - logistics_stations.geojson (all matched, with classification field)

Methodology:
  1. Broad keyword matching on name + type fields
  2. Strict 5-step classification with brand-first priority:
     CHECK 1: Known courier brands → lastmile
     CHECK 2: Post office patterns → lastmile
     CHECK 3: Parcel locker patterns → lastmile
     CHECK 4: B2B indicators (Ltd, shipping, etc.) → b2b
     CHECK 5: Category fallback
  3. Duplicate detection (within 50m)
  4. Population-weighted coverage analysis (2D Euclidean baseline)
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

warnings.filterwarnings("ignore")


def run(config):
    """Main entry point. Called by master_pipeline.py with config dict."""
    output_dir = Path(config["output_dir"]) / "Step01_POI"
    output_dir.mkdir(parents=True, exist_ok=True)

    study_path = Path(config["study_area_path"])
    poi_paths = [Path(p) for p in config.get("poi_paths", [])]
    poi_cfg = config.get("poi_classification", {})
    field_map = config.get("field_mapping", {})

    print(f"  Study area: {study_path}")
    print(f"  POI files: {len(poi_paths)}")
    print(f"  Output: {output_dir}")

    # ============================================================
    # 1. Load study area
    # ============================================================
    target_crs = config["crs"]
    study_area = gpd.read_file(study_path).to_crs(target_crs)

    # ============================================================
    # 2. Load and match POIs
    # ============================================================
    name_field = field_map.get("poi_name", "name")
    type_field = field_map.get("poi_type", "midType")

    logistics_name_kw = [
        "快递", "物流", "邮政", "包裹", "速递", "货运", "配送",
        "courier", "express", "logistics", "parcel", "delivery",
        "post", "shipping", "freight", "locker", "丰巢", "菜鸟",
        "顺丰", "中通", "圆通", "申通", "韵达", "百世", "德邦",
        "DHL", "FedEx", "UPS", "TNT", "SF Express", "Alfred",
        "eLink", "4PX", "Gute", "驿站", "自提", "快递柜", "联运",
        "邮局", "邮政局", "郵政局", "电子换领站", "e栈",
    ]

    logistics_type_kw = [
        "快递", "物流", "邮政", "courier", "express", "logistics",
        "物流速递", "邮局",
    ]

    def match_keywords(text, keywords):
        if pd.isna(text) or not isinstance(text, str):
            return False
        t = text.lower()
        return any(kw.lower() in t for kw in keywords)

    all_matched = []
    for p in poi_paths:
        if not p.exists():
            print(f"  [WARN] POI file not found: {p}")
            continue
        gdf = gpd.read_file(p).to_crs(target_crs)

        # Build match mask
        mask = pd.Series(False, index=gdf.index)
        if name_field in gdf.columns:
            mask |= gdf[name_field].apply(lambda x: match_keywords(x, logistics_name_kw))
        for col in gdf.columns:
            if "type" in col.lower() or "类" in col or "cat" in col.lower():
                mask |= gdf[col].apply(lambda x: match_keywords(x, logistics_type_kw))

        matched = gdf[mask].copy()
        matched["source_file"] = p.name
        matched["source_category"] = "life_services" if "生活服务" in p.name else "companies"
        all_matched.append(matched)
        print(f"    {p.name}: {len(gdf)} total → {len(matched)} matched")

    if not all_matched:
        print("  [ERROR] No POIs matched!")
        return False

    matched_all = pd.concat(all_matched, ignore_index=True)
    print(f"  Total matched (before clip): {len(matched_all)}")

    # Clip to study area
    matched_all = gpd.sjoin(matched_all, study_area[["geometry"]], predicate="within")
    print(f"  After clip to study area: {len(matched_all)}")

    # ============================================================
    # 3. Strict Classification
    # ============================================================
    courier_brands = poi_cfg.get("courier_brands", ["顺丰速运", "SF Express", "FedEx", "DHL"])
    post_patterns = poi_cfg.get("post_office_patterns", ["邮政局", "郵政局", "邮局", "Post Office"])
    locker_patterns = poi_cfg.get("parcel_locker_patterns", ["快递柜", "智能柜", "自提点", "驿站", "e栈"])
    b2b_indicators = poi_cfg.get("b2b_indicators", ["有限公司", "Ltd", "Limited", "shipping", "freight"])

    def classify_strict(row):
        text = " ".join([
            str(row.get(name_field, "")),
            str(row.get(type_field, "")),
            str(row.get("address", "")),
        ]).lower()

        # CHECK 1: Known courier brands
        for brand in courier_brands:
            if brand.lower() in text:
                return "lastmile"

        # CHECK 2: Post office
        for pat in post_patterns:
            if pat.lower() in text:
                return "lastmile"

        # CHECK 3: Parcel locker
        for pat in locker_patterns:
            if pat.lower() in text:
                return "lastmile"

        # CHECK 4: B2B indicators
        for ind in b2b_indicators:
            if ind.lower() in text:
                return "b2b"

        # CHECK 5: Category fallback
        type_val = str(row.get(type_field, "")).lower()
        if "物流速递" in type_val or "物流" in type_val:
            return "b2b"
        if "邮局" in type_val:
            return "lastmile"

        return "uncertain"

    matched_all["classification"] = matched_all.apply(classify_strict, axis=1)

    # Duplicate detection for lastmile
    lastmile = matched_all[matched_all["classification"] == "lastmile"].copy()
    if len(lastmile) > 1:
        coords = np.column_stack([lastmile.geometry.x, lastmile.geometry.y])
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(coords))
        min_spacing = poi_cfg.get("min_station_spacing_m", 50.0)
        dups = []
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                if dists[i, j] < min_spacing:
                    dups.append((lastmile.index[i], lastmile.index[j]))
        if dups:
            print(f"  Duplicate pairs (<{min_spacing}m): {len(dups)}")
            for a, b in dups:
                print(f"    {matched_all.loc[a, name_field]} ↔ {matched_all.loc[b, name_field]}")

    b2b = matched_all[matched_all["classification"] == "b2b"].copy()
    uncertain = matched_all[matched_all["classification"] == "uncertain"].copy()

    print(f"  Classification: lastmile={len(lastmile)}, b2b={len(b2b)}, uncertain={len(uncertain)}")

    # ============================================================
    # 4. Export
    # ============================================================
    lastmile.to_file(output_dir / "lastmile_stations.shp", driver="ESRI Shapefile", encoding="utf-8")
    lastmile.to_file(output_dir / "lastmile_stations.geojson", driver="GeoJSON")

    if len(b2b) > 0:
        b2b.to_file(output_dir / "b2b_freight_offices.shp", driver="ESRI Shapefile", encoding="utf-8")
    if len(uncertain) > 0:
        uncertain.to_file(output_dir / "uncertain_pois.shp", driver="ESRI Shapefile", encoding="utf-8")

    matched_all.to_file(output_dir / "logistics_stations.geojson", driver="GeoJSON")

    # Summary
    summary = {
        "total_matched": len(matched_all),
        "lastmile": len(lastmile),
        "b2b": len(b2b),
        "uncertain": len(uncertain),
        "lastmile_names": lastmile[name_field].tolist() if name_field in lastmile.columns else [],
    }
    pd.DataFrame([summary]).to_csv(output_dir / "classification_summary.csv", index=False)

    print(f"  ✓ Exported: {len(lastmile)} last-mile, {len(b2b)} B2B, {len(uncertain)} uncertain")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
        run(cfg)
    else:
        print("Usage: python step01_filter_classify_pois.py config.json")
