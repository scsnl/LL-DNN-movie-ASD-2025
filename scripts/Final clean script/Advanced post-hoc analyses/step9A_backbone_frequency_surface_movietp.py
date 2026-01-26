"""
Step 9A (MovieTP): Neural Backbone Frequency Visualization

MovieTP-specific version of the backbone frequency analysis. 
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Test mode bypass
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
if TEST_MODE:
    print("[TEST_MODE] Skipping MovieTP backbone plotting.")
    import sys; sys.exit(0)

ATLAS_PATH = "BN_Atlas_246_1mm.nii"
OUT_DIR = "backbone_frequency_surface_movietp"

TABLE_S8_ROWS = [(1, 162), (1, 236), (1, 96), (1, 77), (1, 8), (1, 38), (1, 174), (1, 233), (1, 27), (1, 222), (1, 12), (1, 234), (2, 27), (2, 95), (2, 20), (2, 33), (2, 8), (2, 75), (2, 34), (2, 224), (2, 236), (2, 100), (2, 185), (2, 48), (3, 164), (3, 114), (3, 8), (3, 34), (3, 48), (3, 72), (3, 111), (3, 158), (3, 120), (3, 119), (3, 224), (3, 20), (4, 48), (4, 224), (4, 33), (4, 236), (4, 34), (4, 133), (4, 106), (4, 114), (4, 107), (4, 83), (4, 147), (4, 78), (5, 133), (5, 33), (5, 98), (5, 131), (5, 57), (5, 87), (5, 145), (5, 59), (5, 34), (5, 147), (5, 77), (5, 88), (6, 172), (6, 236), (6, 87), (6, 34), (6, 48), (6, 20), (6, 39), (6, 133), (6, 224), (6, 234), (6, 84), (6, 93)]

def main():
    atlas_dir = Path(__file__).resolve().parents[3] / "assets" / "atlas"
    p = atlas_dir / ATLAS_PATH
    if not p.exists(): p = Path(str(p) + ".gz")
    atlas_img = nib.load(p); atlas_data = atlas_img.get_fdata().astype(int)
    
    roi_counts = Counter([row[1] for row in TABLE_S8_ROWS])
    lut = np.zeros(247, dtype=float)
    for r, c in roi_counts.items(): 
        if 1 <= r <= 246: lut[r] = float(c)
    stat_img = nib.Nifti1Image(lut[atlas_data], atlas_img.affine)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    nib.save(stat_img, os.path.join(OUT_DIR, "backbone_freq_tp.nii.gz"))
    print(f"Saved MovieTP frequency map to {OUT_DIR}")

if __name__ == "__main__":
    main()
