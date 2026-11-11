"""
Synthetic inputs for a test run of the stimulus-feature scripts: two small
videos with audio at the real durations (200 s and 600 s), shot and semantic
tables, attention-significance tables and human valence ratings.

Run: TEST_MODE=1 TEST_OUTPUT_DIR=<dir> python make_test_media.py
"""
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import savemat

from sf_common import CACHE, MEDIA, MOVIES, SEED, ensure_dirs, ffmpeg_exe, load_prompts, run, scene_csv

FPS, W, H, SR = 10, 96, 64, 16000


def write_movie(movie, rng, prompts):
    import cv2
    dur = MOVIES[movie]["duration_sec"] + 1.0
    n = int(dur * FPS)
    cuts = np.cumsum(rng.uniform(2.0, 6.0, size=400))
    cuts = np.concatenate([[0.0], cuts[cuts < dur]])
    tmp = CACHE / f"test_{movie}_video.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    shot_of = np.searchsorted(cuts, np.arange(n) / FPS, side="right") - 1
    colours = rng.integers(40, 255, size=(len(cuts), 3))
    for i in range(n):
        img = np.full((H, W, 3), colours[shot_of[i]], dtype=np.uint8)
        x = int((i * 2) % (W - 16))
        img[20:36, x:x + 16] = 255 - colours[shot_of[i]]
        vw.write(img)
    vw.release()

    t = np.arange(int(dur * SR)) / SR
    level = rng.uniform(0.02, 0.3, size=len(cuts))[np.searchsorted(cuts, t, side="right") - 1]
    audio = level * (0.5 * rng.normal(size=len(t)) + np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.1 * t) > 0))
    wav = CACHE / f"test_{movie}_audio.wav"
    sf.write(wav, np.clip(audio, -1, 1).astype(np.float32), SR)
    run([ffmpeg_exe(), "-hide_banner", "-y", "-i", str(tmp), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
         "-shortest", str(MEDIA / MOVIES[movie]["video"])])

    ends = np.append(cuts[1:], dur)
    shots = pd.DataFrame({"scene_type": "visual", "scene_index": np.arange(1, len(cuts) + 1),
                          "start_sec": cuts.round(2), "end_sec": ends.round(2),
                          "emotion": rng.choice(prompts, size=len(cuts))})
    shots.to_csv(scene_csv(movie, "visual"), index=False)
    grp = np.arange(len(cuts)) // 3
    sem = shots.groupby(grp).agg(start_sec=("start_sec", "first"), end_sec=("end_sec", "last"),
                                 emotion=("emotion", "first")).reset_index(drop=True)
    sem.insert(0, "scene_index", np.arange(1, len(sem) + 1))
    sem.insert(0, "scene_type", "semantic")
    sem.to_csv(scene_csv(movie, "semantic"), index=False)


def write_attention(name, n_tr, starts):
    sig = np.zeros(n_tr, dtype=bool)
    for s in starts:
        sig[s:s + 6] = True
    pd.DataFrame({"timepoint": np.arange(n_tr), "sig_raw": sig}).to_csv(MEDIA / name, index=False)


def main():
    ensure_dirs()
    MEDIA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    prompts = load_prompts()
    for m in MOVIES:
        write_movie(m, rng, prompts)
    write_attention(MOVIES["movieTP"]["attn_csv"], 250, [20, 60, 100, 140, 180, 220])
    write_attention(MOVIES["movieDM"]["attn_csv"], 171, [15, 55, 95, 135])
    walk = np.cumsum(rng.normal(scale=0.2, size=(750, 60)), axis=0)
    savemat(str(MEDIA / MOVIES["movieDM"]["ratings_mat"]), {"ratings": np.clip(walk, -4, 4)})
    print(f"test media written to {MEDIA}")


if __name__ == "__main__":
    main()
