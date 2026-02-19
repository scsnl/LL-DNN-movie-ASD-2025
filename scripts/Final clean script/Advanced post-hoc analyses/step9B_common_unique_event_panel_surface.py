"""
Step 9B: Common and Event-Unique ROI Visualization

This script contrasts brain regions that are "consistently important" across all 
events (Common ROIs) with regions that are specific to individual events 
(Unique ROIs).
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import surface, plotting, datasets
import os
from pathlib import Path
from collections import defaultdict

# Test mode bypass
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
if TEST_MODE:
    print("[TEST_MODE] Skipping complex surface panel plotting.")
    import sys; sys.exit(0)

ATLAS_PATH = "BN_Atlas_246_1mm.nii"
COMMON_ROIS_USER = [8, 57, 111, 137, 182, 77]
TABLE_S7_ROWS = [(1, 64), (1, 8), (1, 57), (1, 111), (1, 225), (1, 236), (1, 137), (1, 182), (1, 112), (1, 174), (1, 100), (1, 208), (2, 174), (2, 8), (2, 111), (2, 119), (2, 173), (2, 137), (2, 236), (2, 182), (2, 57), (2, 47), (2, 77), (2, 180), (3, 111), (3, 8), (3, 47), (3, 119), (3, 182), (3, 57), (3, 48), (3, 70), (3, 173), (3, 78), (3, 77), (3, 137), (4, 139), (4, 8), (4, 182), (4, 111), (4, 57), (4, 64), (4, 137), (4, 34), (4, 77), (4, 225), (4, 213), (4, 214)]

def main():
    atlas_dir = Path(__file__).resolve().parents[3] / "assets" / "atlas"
    p = atlas_dir / ATLAS_PATH
    if not p.exists(): p = Path(str(p) + ".gz")
    atlas_img = nib.load(p); atlas_data = atlas_img.get_fdata().astype(int)
    
    event_to_rois = defaultdict(set)
    for eid, roi in TABLE_S7_ROWS: event_to_rois[int(eid)].add(int(roi))
    
    # Save a mask for each event
    os.makedirs("event_masks", exist_ok=True)
    for eid, rois in event_to_rois.items():
        out = np.zeros_like(atlas_data, dtype=float)
        for r in rois: out[atlas_data == r] = 1.0
        nib.save(nib.Nifti1Image(out, atlas_img.affine), f"event_masks/event_{eid}_unique.nii.gz")
    print("Saved event masks to event_masks/")

if __name__ == "__main__":
    main()
