"""
Final Step 13 (MIL path): event-level Neural Saliency Index and behaviour,
movieDM Segment 2.

Events: contiguous runs of TRs in which Segment 2 attention weights differ between
groups (uncorrected p < 0.05), indexed from segment onset. Neural Saliency Index
(NSI) per participant and event: median |IG| of the MIL path over the event's TRs,
averaged over the event's top 5% regions (top 12 of the autistic group mean of
these maps). Partial Pearson correlations with behavioural scores in the autistic
group, adjusting for age, sex, site and mean framewise displacement. All joins are
on subject ID.

Inputs: results/seg2/saved_models and results/seg2/figures/attn_significance.csv;
TEST_MODE uses $TEST_OUTPUT_DIR/seg2/.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ig_io import (correlate_with_behaviour, covariates, event_windows,  # noqa: E402
                   load_fold_array, load_sample, neural_saliency_index)

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "")
TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
SEG2_ROOT = os.path.join(TEST_OUTPUT_DIR, "seg2") if TEST_OUTPUT_DIR else os.path.join("results", "seg2")

OUT_DIR = "cluster_split_correlation_auto"
CONFIGS = [{
    "name": "MovieDM_Seg2",
    "fmri_file": os.path.join(TEST_DATA_DIR, "combined_asd_td_movieDM_data.pklz") if TEST_DATA_DIR else "combined_asd_td_movieDM_data.pklz",
    "behavior_file": os.path.join(TEST_DATA_DIR, "all subject behavior data.csv") if TEST_DATA_DIR else "all subject behavior data.csv",
    "ig_root": os.path.join(SEG2_ROOT, "saved_models"),
    "attn_csv": os.path.join(SEG2_ROOT, "figures", "attn_significance.csv"),
}]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for conf in CONFIGS:
        print(f"Processing {conf['name']}...")
        if not (os.path.exists(conf["attn_csv"]) and os.path.exists(os.path.join(conf["ig_root"], "fold0", "ig_roi_milout.npy"))):
            print("  attention table or fold outputs not found")
            continue
        events = event_windows(conf["attn_csv"])
        ids, ig, lbl = load_fold_array(conf["ig_root"], "ig_roi_milout.npy", conf["fmri_file"])
        ig_asd = ig[lbl == 0]
        nsi = pd.DataFrame({"subject_id": ids[lbl == 0]})
        for e, trs in enumerate(events, 1):
            if any(t < ig_asd.shape[1] for t in trs):
                nsi[f"E{e}"], _ = neural_saliency_index(ig_asd, trs)
        behav = pd.read_csv(conf["behavior_file"], low_memory=False)
        cols = [c for c in behav.columns if c != "Identifiers"]
        res = correlate_with_behaviour(nsi, covariates(load_sample(conf["fmri_file"])), behav, cols,
                                       5 if TEST_MODE else 15)
        if res.empty:
            print("  no valid correlations")
            continue
        res.sort_values("partial_p").to_csv(os.path.join(OUT_DIR, f"{conf['name']}_event_nsi_results.csv"), index=False)
        print(f"  {nsi.shape[1] - 1} events")


if __name__ == "__main__":
    main()
