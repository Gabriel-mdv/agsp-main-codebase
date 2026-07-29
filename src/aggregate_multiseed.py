#!/usr/bin/env python3
"""
aggregate_multiseed.py
──────────────────────
Aggregates multi-seed Pareto results and computes mean ± std for Table 3.

PURPOSE
───────
After running run_multiseed.sh (3 seeds × 2 ratios = 6 jobs), this script:
  1. Loads all results JSON files from ocean storage
  2. Computes mean ± std for Epoch-1 loss and final loss at each (ratio, strategy)
  3. Prints an updated Table 3 with statistical confidence
  4. Saves to outputs/phase2/pareto_table_multiseed.json

USAGE
─────
  python aggregate_multiseed.py

  # Custom ocean output directory
  python aggregate_multiseed.py --ocean_dir /ocean/projects/cis250019p/bgodwin/outputs

  # Specific ratios and seeds
  python aggregate_multiseed.py --ratios 0.20 0.60 --seeds 42 123 7
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

OCEAN_OUT = "/ocean/projects/cis250019p/bgodwin/outputs"
SEEDS     = [42, 123, 7]
RATIOS    = [0.20, 0.60]


def load_pareto_results(ocean_dir: str, ratios: List[float], seeds: List[int]) -> Dict:
    """
    Load all pareto_data_seed{seed}.json files and organise by (ratio, strategy).

    Returns nested dict: results[ratio][strategy] = list of result dicts
    """
    results = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        pareto_path = os.path.join(ocean_dir, f"pareto_data_seed{seed}.json")
        if not os.path.exists(pareto_path):
            print(f"WARNING: {pareto_path} not found — seed {seed} may not have finished.")
            continue

        with open(pareto_path) as f:
            pareto = json.load(f)

        for ratio_str, ratio_data in pareto.items():
            ratio = float(ratio_str)
            if ratio not in ratios:
                continue
            for strategy, res in ratio_data.items():
                res["seed"] = seed
                results[ratio][strategy].append(res)

    return results


def compute_stats(values: List[float]) -> Dict:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "n":    len(arr),
    }


def format_cell(stats: Dict) -> str:
    """Format mean ± std for table display."""
    if stats["mean"] is None:
        return "—"
    if stats["n"] == 1:
        return f"{stats['mean']:.4f} (single run)"
    return f"{stats['mean']:.4f} ± {stats['std']:.4f}"


def main(args):
    print("AGSP Multi-Seed Pareto Aggregation")
    print("=" * 60)

    results = load_pareto_results(args.ocean_dir, args.ratios, args.seeds)

    if not results:
        print("No results found. Check that all jobs completed and paths are correct.")
        return

    # Compute statistics
    summary = {}
    for ratio in sorted(results.keys()):
        summary[ratio] = {}
        for strategy in ["agsp", "mbp", "rand"]:
            if strategy not in results[ratio]:
                continue
            runs = results[ratio][strategy]
            epoch1_losses = [r["epoch1_loss"] for r in runs if r.get("epoch1_loss") is not None]
            final_losses  = [r["final_loss"]  for r in runs if r.get("final_loss")  is not None]

            summary[ratio][strategy] = {
                "epoch1_loss": compute_stats(epoch1_losses),
                "final_loss":  compute_stats(final_losses),
                "seeds_completed": [r["seed"] for r in runs],
            }

    # Print updated Table 3
    print("\nUpdated Table 3: Pareto Sweep with Statistical Validation")
    print("─" * 80)
    print(f"{'Ratio':<8} {'Strategy':<8} {'Epoch-1 Loss':<28} {'Final Loss':<28} {'Seeds'}")
    print("─" * 80)

    for ratio in sorted(summary.keys()):
        ratio_pct = f"{int(ratio * 100)}%"
        for strategy in ["agsp", "mbp"]:
            if strategy not in summary[ratio]:
                continue
            s = summary[ratio][strategy]
            epoch1_str = format_cell(s["epoch1_loss"])
            final_str  = format_cell(s["final_loss"])
            seeds_str  = str(s["seeds_completed"])
            print(f"{ratio_pct:<8} {strategy.upper():<8} {epoch1_str:<28} {final_str:<28} {seeds_str}")

        # Print Δ (AGSP advantage)
        if "agsp" in summary[ratio] and "mbp" in summary[ratio]:
            agsp_final = summary[ratio]["agsp"]["final_loss"]["mean"]
            mbp_final  = summary[ratio]["mbp"]["final_loss"]["mean"]
            if agsp_final is not None and mbp_final is not None:
                delta = agsp_final - mbp_final
                sign  = "+" if delta > 0 else ""
                print(f"{'':>8} {'Δ':<8} {'':<28} {sign}{delta:.4f} (AGSP {'wins' if delta > 0 else 'LOSES'})")
        print()

    print("─" * 80)
    print("Positive Δ = AGSP achieves lower final loss (AGSP wins).")
    print("Gaps within ±0.010 are within single-run noise and should not be claimed as wins.")

    # Save JSON summary
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pareto_table_multiseed.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_path}")
    print("Use this to update Table 3 in the paper with mean ± std.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed Pareto results for AGSP Table 3"
    )
    parser.add_argument(
        "--ocean_dir", default=OCEAN_OUT,
        help=f"Ocean storage directory (default: {OCEAN_OUT})",
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=RATIOS,
        help="Pruning ratios to aggregate (default: 0.20 0.60)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to aggregate (default: 42 123 7)",
    )
    args = parser.parse_args()
    main(args)
