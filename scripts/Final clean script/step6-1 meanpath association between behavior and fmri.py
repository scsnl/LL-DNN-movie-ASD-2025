"""
Step 6-1: Global Saliency Index and behaviour (Figure S3).

Global Saliency Index (GSI): for each participant, mean |IG| of the Global
Temporal-Pooling path over the whole viewing period, averaged over the top 5%
regions of the autistic group map (step 3-3). Partial Pearson correlations with
behavioural scores in the autistic group, adjusting for age, sex, site and mean
framewise displacement. IG rows, covariates and scores are joined on subject ID.

Environment: DATA_FILE (pickle used for training), TEST_MODE, TEST_DATA_DIR,
TEST_OUTPUT_DIR.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ig_io import (correlate_with_behaviour, covariates, global_saliency_index,  # noqa: E402
                   load_fold_array, load_sample)

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
TARGET_BEHAVIORS = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]


def main():
    fmri_file = os.environ.get("DATA_FILE", os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
                               if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz")
    behavior_file = os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv"
    fold_root = os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models"
    roi_file = os.path.join("top_roi_summary", "outmean_asd_top_roi.csv")
    if not os.path.exists(roi_file):
        print(f"ROI file not found: {roi_file}")
        return

    regions = pd.read_csv(roi_file).sort_values("rank")["roi_index"].astype(int).to_numpy()
    ids, ig, lbl = load_fold_array(fold_root, "ig_roi_outmean.npy", fmri_file)
    gsi = pd.DataFrame({"subject_id": ids[lbl == 0], "GSI": global_saliency_index(ig[lbl == 0], regions)})

    behav = pd.read_csv(behavior_file, low_memory=False)
    cols = [c for c in behav.columns if c != "Identifiers"]
    if TEST_MODE:
        cols = [c for c in cols if c in TARGET_BEHAVIORS]
    res = correlate_with_behaviour(gsi, covariates(load_sample(fmri_file)), behav, cols, 5 if TEST_MODE else 10)
    if res.empty:
        print("No valid correlations found.")
        return
    res.sort_values("partial_p").to_csv("meanpath_gsi_partial_correlations.csv", index=False)
    print("Saved meanpath_gsi_partial_correlations.csv")


if __name__ == "__main__":
    main()
