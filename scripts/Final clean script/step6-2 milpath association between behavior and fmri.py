"""
Step 6-2: event-level Neural Saliency Index and behaviour.

Events: contiguous runs of TRs in which attention weights differ between groups
(figures/attn_significance.csv, uncorrected p < 0.05). Neural Saliency Index
(NSI) per participant and event: median |IG| of the MIL path over the event's
TRs, averaged over the event's top 5% regions (top 12 of the autistic group mean
of these maps). Partial Pearson correlations with behavioural scores in the
autistic group, adjusting for age, sex, site and mean framewise displacement.
IG rows, covariates and scores are joined on subject ID.

Environment: DATA_FILE (pickle used for training), TEST_MODE, TEST_DATA_DIR,
TEST_OUTPUT_DIR.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ig_io import (correlate_with_behaviour, covariates, event_windows,  # noqa: E402
                   load_fold_array, load_sample, neural_saliency_index)

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
TARGET_BEHAVIORS = ["CBCL,CBCL_Total_T", "RBS,RBS_Total", "SRS,SRS_Total_T"]


def main():
    fmri_file = os.environ.get("DATA_FILE", os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz")
                               if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz")
    behavior_file = os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv"
    fold_root = os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models"
    attn_file = os.path.join("figures", "attn_significance.csv")
    if not os.path.exists(attn_file):
        print(f"Attention table not found: {attn_file}")
        return

    events = event_windows(attn_file)
    if not events:
        print("No event windows found.")
        return
    ids, ig, lbl = load_fold_array(fold_root, "ig_roi_milout.npy", fmri_file)
    ig_asd = ig[lbl == 0]
    nsi = pd.DataFrame({"subject_id": ids[lbl == 0]})
    for e, trs in enumerate(events, 1):
        nsi[f"E{e}"], _ = neural_saliency_index(ig_asd, trs)

    behav = pd.read_csv(behavior_file, low_memory=False)
    cols = [c for c in behav.columns if c != "Identifiers"]
    if TEST_MODE:
        cols = [c for c in cols if c in TARGET_BEHAVIORS]
    res = correlate_with_behaviour(nsi, covariates(load_sample(fmri_file)), behav, cols, 5 if TEST_MODE else 10)
    if res.empty:
        print("No valid correlations found.")
        return
    res.sort_values("partial_p").to_csv("milpath_nsi_partial_correlations.csv", index=False)
    print(f"{len(events)} events; saved milpath_nsi_partial_correlations.csv")


if __name__ == "__main__":
    main()
