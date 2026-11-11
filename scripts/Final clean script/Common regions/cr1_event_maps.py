"""
Per-participant event maps of the MIL-path attribution.

Events are the contiguous runs of TRs with sig_raw in the attention table. For each
participant and event: median |IG| over the event's TRs, per region. Fold rows are
matched to participants by subject ID (ig_io.load_fold_array).

Output: <CR_OUT_DIR>/event_maps_<film>.npz with subject_id, group (ASD / TDC),
event_trs and maps (participants x events x 246).
"""
import numpy as np

from cr_common import FILMS, OUT, TEST_MODE, event_windows, film_path
from ig_io import load_fold_array

OUT.mkdir(parents=True, exist_ok=True)
for film in FILMS:
    folds, attn, pkl = film_path(film, "folds"), film_path(film, "attn"), film_path(film, "pickle")
    if folds is None or attn is None:
        if TEST_MODE:
            print(f"{film}: fold outputs or attention table not found, skipped")
            continue
        raise FileNotFoundError(f"{film}: fold outputs or attention table not found")
    ids, ig, lbl = load_fold_array(str(folds), "ig_roi_milout.npy", str(pkl) if pkl.exists() else None)
    events = event_windows(attn, n_tr=ig.shape[1])
    if not events:
        print(f"{film}: no event windows, skipped")
        continue
    mag = np.abs(ig)
    maps = np.stack([np.median(mag[:, e, :], axis=1) for e in events], axis=1)
    np.savez_compressed(OUT / f"event_maps_{film}.npz", subject_id=ids.astype(str),
                        group=np.where(lbl == 0, "ASD", "TDC"),
                        event_trs=np.array([",".join(map(str, e)) for e in events]), maps=maps)
    print(f"{film}: {len(ids)} participants ({int((lbl == 0).sum())} ASD), {len(events)} events")
