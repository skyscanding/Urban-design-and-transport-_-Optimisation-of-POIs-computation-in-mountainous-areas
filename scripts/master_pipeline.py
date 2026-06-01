#!/usr/bin/env python3
"""
master_pipeline.py: Orchestrator for the Mountainous Terrain Optimisation Pipeline.

Usage:
    python master_pipeline.py config.json
    python master_pipeline.py config.json --steps filter,merge,tobler
    python master_pipeline.py config.json --steps pmedian,drone,capsule

Steps:
    filter   : Step 01: Filter and classify logistics POIs
    merge    : Step 02: Merge vehicle roads + footways + steps
    tobler   : Step 03: DEM overlay + Tobler's terrain impedance
    pmedian  : Step 04: P-median location-allocation (2D vs 3D)
    diminish : Step 05: Diminishing returns analysis (p=3-20)
    drone    : Step 06: Drone landing pad MCLP optimisation
    capsule  : Step 07: Capsule pipeline routing (SPT vs Steiner)
    figures  : Step 08: Publication-quality figures
"""

import json
import sys
import argparse
import time
from pathlib import Path


STEP_MODULES = {
    "filter":   "step01_filter_classify_pois",
    "merge":    "step02_merge_network",
    "tobler":   "step03_dem_overlay_tobler",
    "pmedian":  "step04_pmedian_optimization",
    "diminish": "step05_diminishing_returns",
    "drone":    "step06_drone_mclp",
    "capsule":  "step07_capsule_pipeline",
    "figures":  "step08_publication_figures",
}

STEP_NAMES = {
    "filter":   "Step 01: POI Filtering & Classification",
    "merge":    "Step 02: Network Merge (roads + footways + steps)",
    "tobler":   "Step 03: DEM Overlay + Tobler's Terrain Impedance",
    "pmedian":  "Step 04: P-Median Location-Allocation Optimisation",
    "diminish": "Step 05: Diminishing Returns Analysis",
    "drone":    "Step 06: Drone Landing Pad MCLP",
    "capsule":  "Step 07: Capsule Pipeline Routing",
    "figures":  "Step 08: Publication Figures",
}


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_step(step_key, config):
    """Dynamically import and run a step module."""
    module_name = STEP_MODULES[step_key]
    try:
        mod = __import__(f"scripts.{module_name}", fromlist=["run"])
        if hasattr(mod, "run"):
            return mod.run(config)
        else:
            print(f"  [WARN] {module_name}.py has no run(config) function: skipping")
            return True
    except ImportError as e:
        print(f"  [ERROR] Cannot import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {module_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Mountainous Terrain Optimisation Pipeline")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--steps", help="Comma-separated step keys (default: all in config)", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine steps to run
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    else:
        steps = config.get("steps", list(STEP_MODULES.keys()))

    # Validate
    unknown = [s for s in steps if s not in STEP_MODULES]
    if unknown:
        print(f"Unknown steps: {unknown}")
        print(f"Available: {list(STEP_MODULES.keys())}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(config.get("output_dir", "./Output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MOUNTAINOUS TERRAIN OPTIMISATION PIPELINE")
    print("=" * 70)
    print(f"Steps: {steps}")
    print(f"Output: {output_dir}")
    print()

    start_all = time.time()
    results = {}

    for step_key in steps:
        print(f"\n{'─' * 70}")
        print(f"  {STEP_NAMES[step_key]}")
        print(f"{'─' * 70}")

        t0 = time.time()
        ok = run_step(step_key, config)
        elapsed = time.time() - t0

        results[step_key] = ok
        if ok:
            print(f"  ✓ Done in {elapsed:.1f}s")
        else:
            print(f"  ✗ FAILED after {elapsed:.1f}s")
            if config.get("stop_on_error", False):
                print("  Stopping (stop_on_error=true)")
                break

    # Summary
    total = time.time() - start_all
    print(f"\n{'=' * 70}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    for step_key in steps:
        status = "✓ PASS" if results.get(step_key) else "✗ FAIL"
        print(f"  {STEP_NAMES[step_key]:<55s} {status}")
    print(f"{'─' * 70}")
    print(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Outputs → {output_dir}")
    print(f"{'=' * 70}")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
