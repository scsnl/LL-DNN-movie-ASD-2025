"""
Frames and audio on the TR grid.

Frame k is sampled at t = (k + 0.5) / 2.5 + time_shift_sec (2 frames per TR).
Frames are decoded sequentially and the decoded frame nearest each target time
is kept. Borders that are black in every probed frame are cropped.
movieTP: 500 frames (200 s). movieDM: 1500 frames (600 s).

Outputs: cache/frames_<movie>/f######.jpg, cache/audio_<movie>.wav,
         frames_<movie>.csv, frame_extraction_log.json
"""
import json

import cv2
import numpy as np
import pandas as pd

from sf_common import CACHE, FRAME_RATE, FRAMES_PER_TR, MOVIES, OUT, ensure_dirs, ffmpeg_exe, media, run

JPEG_QUALITY = 95
BLACK_LUMA = 16
N_BAR_PROBE = 25


def measure_bars(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    top = bottom = left = right = h = w = None
    for i in np.linspace(total * 0.02, total * 0.98, N_BAR_PROBE).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        rm, cm = g.max(axis=1), g.max(axis=0)
        v = (int(np.argmax(rm > BLACK_LUMA)), int(np.argmax(rm[::-1] > BLACK_LUMA)),
             int(np.argmax(cm > BLACK_LUMA)), int(np.argmax(cm[::-1] > BLACK_LUMA)))
        top = v[0] if top is None else min(top, v[0])
        bottom = v[1] if bottom is None else min(bottom, v[1])
        left = v[2] if left is None else min(left, v[2])
        right = v[3] if right is None else min(right, v[3])
    cap.release()
    return {"height": h, "width": w, "top": int(top), "bottom": int(bottom),
            "left": int(left), "right": int(right)}


def extract_movie(name):
    cfg = MOVIES[name]
    video = media(cfg["video"])
    shift = cfg["time_shift_sec"]
    n_frames = cfg["n_tr"] * FRAMES_PER_TR
    targets = (np.arange(n_frames) + 0.5) / FRAME_RATE + shift
    bars = measure_bars(video)
    y0, y1 = bars["top"], bars["height"] - bars["bottom"]
    x0, x1 = bars["left"], bars["width"] - bars["right"]
    out_dir = CACHE / f"frames_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    recs, k, idx, prev = [], 0, 0, None
    while k < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        while k < n_frames and (t >= targets[k] or idx == src_n - 1):
            if prev is not None and abs(prev[1] - targets[k]) < abs(t - targets[k]):
                ci, ct, img = prev
            else:
                ci, ct, img = idx, t, frame
            fp = out_dir / f"f{k:06d}.jpg"
            cv2.imwrite(str(fp), img[y0:y1, x0:x1], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            recs.append({"frame_index": k, "tr_index": k // FRAMES_PER_TR,
                         "target_time_sec": round(float(targets[k]), 6),
                         "source_frame_index": int(ci), "source_time_sec": round(float(ct), 6),
                         "abs_time_error_sec": round(abs(float(ct) - float(targets[k])), 6),
                         "file": fp.name})
            k += 1
        prev = (idx, t, frame)
        idx += 1
    cap.release()
    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"frames_{name}.csv", index=False)

    wav = CACHE / f"audio_{name}.wav"
    run([ffmpeg_exe(), "-hide_banner", "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le", str(wav)])
    return {"source_fps": src_fps, "time_shift_sec": shift, "n_frames_requested": n_frames,
            "n_frames_extracted": int(len(df)),
            "max_abs_time_error_sec": float(df.abs_time_error_sec.max()),
            "letterbox": bars, "crop": [x0, x1, y0, y1], "audio_extracted": wav.exists()}


def main():
    ensure_dirs()
    log = {}
    for m in MOVIES:
        log[m] = extract_movie(m)
        print(f"{m}: {log[m]['n_frames_extracted']}/{log[m]['n_frames_requested']} frames, "
              f"max time error {log[m]['max_abs_time_error_sec']:.3f} s")
    (OUT / "frame_extraction_log.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
