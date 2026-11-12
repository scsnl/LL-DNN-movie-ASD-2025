"""
Step 3-3: top 5% regions of the group attribution maps.

Group map (per path and group): mean |IG| over the group's participants within each
fold, median across folds, mean over time. The 12 regions (5% of 246) with the
largest value are retained (Tables S4, S5, S8, S9; Figures S2, S9).
Data column c is Brainnetome region ID c.

Writes top_roi_summary/<path>_<group>_top_roi.csv with columns
rank, roi_index (data column), brainnetome_id, mean_abs_ig.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ig_io import column_to_bn_id, global_group_map, top_regions  # noqa: E402

TEST_OUTPUT_DIR = os.environ.get("TEST_OUTPUT_DIR", "")
FOLD_ROOT = sys.argv[1] if len(sys.argv) > 1 else (os.path.join(TEST_OUTPUT_DIR, "saved_models") if TEST_OUTPUT_DIR else "saved_models")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "top_roi_summary"

GROUPS = {"outmean_asd": ("ig_roi_outmean.npy", 0), "outmean_tdc": ("ig_roi_outmean.npy", 1),
          "mil_asd": ("ig_roi_milout.npy", 0), "mil_tdc": ("ig_roi_milout.npy", 1)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (fname, label) in GROUPS.items():
        gmap = global_group_map(FOLD_ROOT, fname, label)
        top = top_regions(gmap)
        df = pd.DataFrame({"rank": range(1, len(top) + 1), "roi_index": top,
                           "brainnetome_id": [column_to_bn_id(c) for c in top],
                           "mean_abs_ig": gmap[top]})
        df.to_csv(os.path.join(OUT_DIR, f"{name}_top_roi.csv"), index=False)
        print(f"{name}: top regions (data columns) {top.tolist()}")


if __name__ == "__main__":
    main()
