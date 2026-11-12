"""
Final Step 13 (Global path): Global Saliency Index and behaviour, movieDM Segment 2.

Group map: mean |IG| of the Global Temporal-Pooling path over the autistic
participants within each fold, median across folds, mean over time; the top 5%
regions (12 of 246) are retained. Global Saliency Index (GSI): per participant mean
|IG| over the segment, averaged over these regions. Partial Pearson correlations
with behavioural scores in the autistic group, adjusting for age, sex, site and
mean framewise displacement. All joins are on subject ID.

Inputs: results/seg2/saved_models (supply-step2_timefiltered_train_and_ig.py);
TEST_MODE uses $TEST_OUTPUT_DIR/seg2/saved_models.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ig_io import (correlate_with_behaviour, covariates, global_group_map,  # noqa: E402
                   global_saliency_index, load_fold_array, load_sample, top_regions)

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
SEG2_ROOT = os.path.join(TEST_OUTPUT_DIR, "seg2") if TEST_OUTPUT_DIR else os.path.join("results", "seg2")

OUT_DIR = "correlation_results_global_wholetime_final"
CONFIGS = [{
    "name": "MovieDM_Seg2",
    "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
    "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
    "ig_root": os.path.join(SEG2_ROOT, "saved_models"),
}]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for conf in CONFIGS:
        print(f"Processing {conf['name']}...")
        if not os.path.exists(os.path.join(conf["ig_root"], "fold0", "ig_roi_outmean.npy")):
            print("  fold outputs not found")
            continue
        regions = top_regions(global_group_map(conf["ig_root"], "ig_roi_outmean.npy", 0))
        ids, ig, lbl = load_fold_array(conf["ig_root"], "ig_roi_outmean.npy", conf["fmri_file"])
        gsi = pd.DataFrame({"subject_id": ids[lbl == 0], "GSI": global_saliency_index(ig[lbl == 0], regions)})
        behav = pd.read_csv(conf["behavior_file"], low_memory=False)
        cols = [c for c in behav.columns if c != "Identifiers"]
        res = correlate_with_behaviour(gsi, covariates(load_sample(conf["fmri_file"])), behav, cols,
                                       5 if TEST_MODE else 15)
        if res.empty:
            print("  no valid correlations")
            continue
        res.sort_values("partial_p").to_csv(os.path.join(OUT_DIR, f"{conf['name']}_Top5pct_results.csv"), index=False)
        print(f"  top regions (data columns) {regions.tolist()}")


if __name__ == "__main__":
    main()
