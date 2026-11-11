"""
Paths, film settings, region table and colours shared by the cr*.py scripts.

Every path can be set with an environment variable:
  DATA_DIR                        folder with the input pickles (TEST_DATA_DIR in TEST_MODE)
  RUN_DIR                         folder holding saved_models/ and figures/ (TEST_OUTPUT_DIR in TEST_MODE)
  CR_OUT_DIR                      output folder (default RUN_DIR/common_regions)
  CR_<FILM>_FOLDS / _ATTN / _PICKLE / _WEIGHT_FOLDS   per-film overrides, FILM = MOVIETP or MOVIEDM_SEG2
  BN_ATLAS                        Brainnetome 246 volume (BN_Atlas_246_1mm.nii), used by cr3_surfaces.py
"""
import os
import sys
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
DATA_DIR = Path(os.environ.get("TEST_DATA_DIR") or os.environ.get("DATA_DIR") or SCRIPTS)
RUN_DIR = Path(os.environ.get("TEST_OUTPUT_DIR") or os.environ.get("RUN_DIR") or SCRIPTS)
OUT = Path(os.environ.get("CR_OUT_DIR") or RUN_DIR / "common_regions")
FIG = OUT / "figures"
MAPPING = REPO / "assets" / "mapping" / "brainnetome_devatlas.csv"
ATLAS = Path(os.environ.get("BN_ATLAS") or REPO / "assets" / "atlas" / "BN_Atlas_246_1mm.nii")
SEG2 = SCRIPTS / "Advanced post-hoc analyses" / "results" / "seg2"

# Candidate locations, first existing one is used. IG time index = TR index of the trained
# input (movieDM Segment 2: TR 0 = first TR of the segment, as in the Seg2 attention table).
FILMS = {
    "movieTP": dict(
        title="movieTP", pickle="combined_asd_td_movieTP_data.pklz",
        folds=[RUN_DIR / "saved_models"],
        attn=[RUN_DIR / "figures" / "attn_significance.csv", SCRIPTS / "figures" / "attn_significance.csv"],
        weight_folds=[RUN_DIR / "saved_models"]),
    "movieDM_Seg2": dict(
        title="movieDM Segment 2", pickle="combined_asd_td_movieDM_data.pklz",
        folds=[RUN_DIR / "seg2" / "saved_models", SEG2 / "saved_models"],
        attn=[RUN_DIR / "seg2" / "figures" / "attn_significance.csv", SEG2 / "figures" / "attn_significance.csv"],
        # surface weights for movieDM come from the whole-movie movieDM folds
        weight_folds=[RUN_DIR / "saved_models"]),
}
K = 12                      # top 5% of 246 regions
N_ROI = 246

SYSTEM_ORDER = ["DMN", "CON", "SAL", "ATN", "SMN", "VIS"]
SYSTEM_LONG = {"DMN": "Default Mode", "CON": "Control", "SAL": "Salience", "ATN": "Attention",
               "SMN": "Somatomotor", "VIS": "Visual"}
SYS_COLOR = {"DMN": "#2a78d6", "CON": "#eb6834", "SAL": "#1baf7a",
             "ATN": "#eda100", "SMN": "#e87ba4", "VIS": "#008300"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"


def _env(film, key):
    return os.environ.get(f"CR_{film.upper()}_{key}")


def film_path(film, key):
    """Resolved path for 'folds', 'attn', 'weight_folds' or 'pickle'; None if absent."""
    v = _env(film, key.upper())
    if v:
        return Path(v)
    if key == "pickle":
        return DATA_DIR / FILMS[film]["pickle"]
    for p in FILMS[film][key]:
        if Path(p).exists():
            return Path(p)
    return None


def regions():
    """Region table indexed by Brainnetome ID with circle_order (system, subnetwork, hemisphere, ID)."""
    m = pd.read_csv(MAPPING, keep_default_na=False).set_index("brainnetome_id")
    m["system_rank"] = m.system.map(SYSTEM_ORDER.index)
    order = m.sort_values(["system_rank", "subnetwork_index", "hemisphere"], kind="stable").index
    m.loc[order, "circle_order"] = np.arange(len(m))
    m["circle_order"] = m.circle_order.astype(int)
    return m


# data column c <-> Brainnetome ID c; column 0 has no label, ID 246 has no column
DATA_IDS = list(range(1, N_ROI))


def event_windows(attn_csv, n_tr=None):
    """Contiguous runs of TRs flagged sig_raw; TRs beyond the IG length are dropped."""
    d = pd.read_csv(attn_csv)
    trs = sorted(d.loc[d["sig_raw"].astype(str).str.upper().isin(["TRUE", "1"]), "timepoint"].astype(int))
    runs = [list(map(itemgetter(1), g)) for _, g in groupby(enumerate(trs), lambda ix: ix[0] - ix[1])]
    if n_tr is not None:
        runs = [[t for t in r if t < n_tr] for r in runs]
    return [r for r in runs if r]


def films_with(fname):
    """Films for which OUT/<fname with {film}> exists; prints a line for each skipped film."""
    out = []
    for f in FILMS:
        if (OUT / fname.format(film=f)).exists():
            out.append(f)
        else:
            print(f"{f}: {fname.format(film=f)} not found, skipped")
    return out


def save_jpg(fig, path, dpi=300, **kw):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, pil_kwargs={"quality": 95}, **kw)
