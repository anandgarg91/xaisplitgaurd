"""
experiments/collect_results.py
────────────────────────────────
Aggregates per-seed JSON results into mean ± std summary tables
matching Tables 2 and 3 in the paper.

Usage:
    python experiments/collect_results.py --log_dir ./results
"""

from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def collect(log_dir: str) -> Dict[str, List[dict]]:
    """Walk log_dir and collect all summary.json files."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for summary_path in Path(log_dir).rglob("summary.json"):
        with open(summary_path) as f:
            data = json.load(f)
        key = data.get("run_name", str(summary_path.parent.name))
        # Strip seed suffix: "cifar10_badnets_xai_splitshield_split2_s42" → "..._s*"
        base_key = "_s".join(key.split("_s")[:-1]) if "_s" in key else key
        groups[base_key].append(data)
    return groups


def summarize(groups: Dict[str, List[dict]]) -> None:
    """Print mean ± std for ASR and CA per group."""
    print(f"\n{'='*70}")
    print(f"{'Experiment':<45} {'ASR (%)':>10}  {'CA (%)':>10}  {'N':>4}")
    print(f"{'='*70}")

    for key in sorted(groups.keys()):
        runs = groups[key]
        asrs = [r["final_asr"] for r in runs if "final_asr" in r]
        cas  = [r["final_ca"]  for r in runs if "final_ca"  in r]
        if not asrs:
            continue
        asr_str = f"{np.mean(asrs):.2f} ± {np.std(asrs):.2f}"
        ca_str  = f"{np.mean(cas):.2f} ± {np.std(cas):.2f}"
        print(f"{key:<45} {asr_str:>14}  {ca_str:>14}  {len(asrs):>4}")

    print(f"{'='*70}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str, default="./results")
    args = p.parse_args()

    groups = collect(args.log_dir)
    if not groups:
        print(f"No summary.json files found in {args.log_dir}")
        return
    summarize(groups)


if __name__ == "__main__":
    main()
