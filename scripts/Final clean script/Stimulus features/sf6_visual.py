"""
Visual channels from the extracted frames, and cut rate from the step-5 shots.

  motion_absdiff  mean |luminance(k) - luminance(k-1)|, 0-255; frame 0 copies frame 1
  lum_mean        mean luminance (covariate, not in the salience composite)
  rms_contrast    SD of luminance (covariate, not in the salience composite)
  cutrate_5s/10s  shot boundaries of visual_scene_emotions_<movie>.csv within a
                  5 s / 10 s window centred on each TR

Outputs: channels/<movie>__<channel>.csv
"""
import cv2
import numpy as np
import pandas as pd

from sf_common import CACHE, MOVIES, OUT, TR, ensure_dirs, frames_to_tr, save_channel, scene_csv

CUT_WINDOWS = {"5s": 5.0, "10s": 10.0}


def pixel_channels(movie):
    man = pd.read_csv(OUT / f"frames_{movie}.csv")
    fdir = CACHE / f"frames_{movie}"
    n = len(man)
    absdiff, lum, contrast = np.zeros(n), np.zeros(n), np.zeros(n)
    prev = None
    for i, fn in enumerate(man["file"]):
        gray = cv2.cvtColor(cv2.imread(str(fdir / fn), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum[i], contrast[i] = gray.mean(), gray.std()
        if prev is not None:
            absdiff[i] = np.abs(gray - prev).mean()
        prev = gray
    absdiff[0] = absdiff[1]
    times = man["target_time_sec"].to_numpy()
    for name, v in {"motion_absdiff": absdiff, "lum_mean": lum, "rms_contrast": contrast}.items():
        save_channel(movie, name, frames_to_tr(v), v, times)


def cut_counts(bounds, n_tr, window):
    centres = (np.arange(n_tr) + 0.5) * TR
    b = np.sort(bounds)
    return (np.searchsorted(b, centres + window / 2, "right")
            - np.searchsorted(b, centres - window / 2, "left")).astype(float)


def main():
    ensure_dirs()
    for m in MOVIES:
        pixel_channels(m)
        bounds = pd.read_csv(scene_csv(m)).sort_values("start_sec")["start_sec"].to_numpy()[1:].astype(float)
        for tag, w in CUT_WINDOWS.items():
            save_channel(m, f"cutrate_{tag}", cut_counts(bounds, MOVIES[m]["n_tr"], w))
        print(f"{m}: visual channels, {len(bounds)} shot boundaries")


if __name__ == "__main__":
    main()
