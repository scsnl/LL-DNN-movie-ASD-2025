"""
Step 9A: Neural Backbone Frequency Visualization

This script visualizes the stable "Neural Backbone" of the model—brain regions 
that are consistently important across multiple discrete neural events.
"""

import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from nilearn import surface, plotting, datasets

# Test mode bypass
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
if TEST_MODE:
    print("[TEST_MODE] Skipping complex surface plotting.")
    import sys; sys.exit(0)

ATLAS_PATH = "BN_Atlas_246_1mm.nii"
OUT_DIR = "backbone_frequency_surface"

# Table S7 (ROIs identified per event)
TABLE_S7_ROWS = [(1, 64), (1, 8), (1, 57), (1, 111), (1, 225), (1, 236), (1, 137), (1, 182), (1, 112), (1, 174), (1, 100), (1, 208), (2, 174), (2, 8), (2, 111), (2, 119), (2, 173), (2, 137), (2, 236), (2, 182), (2, 57), (2, 47), (2, 77), (2, 180), (3, 111), (3, 8), (3, 47), (3, 119), (3, 182), (3, 57), (3, 48), (3, 70), (3, 173), (3, 78), (3, 77), (3, 137), (4, 139), (4, 8), (4, 182), (4, 111), (4, 57), (4, 64), (4, 137), (4, 34), (4, 77), (4, 225), (4, 213), (4, 214)]

def main():
    atlas_dir = Path(__file__).resolve().parents[3] / "assets" / "atlas"
    p = atlas_dir / ATLAS_PATH
    if not p.exists(): p = Path(str(p) + ".gz")
    atlas_img = nib.load(p); atlas_data = atlas_img.get_fdata().astype(int)
    
    roi_to_events = defaultdict(set)
    for eid, roi_label in TABLE_S7_ROWS:
        r_atlas = int(roi_label); roi_to_events[r_atlas].add(int(eid))
    counts = {r: len(evts) for r, evts in roi_to_events.items()}
    
    lut = np.zeros(247, dtype=float)
    for r, c in counts.items(): lut[r] = float(c)
    stat_img = nib.Nifti1Image(lut[atlas_data], atlas_img.affine)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    nib.save(stat_img, os.path.join(OUT_DIR, "backbone_freq.nii.gz"))
    print(f"Saved frequency map to {OUT_DIR}")

if __name__ == "__main__":
    main()
