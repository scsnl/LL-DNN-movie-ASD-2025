"""
Shared paths, constants and helpers for the stimulus-feature scripts.

Input folder (STIM_MEDIA_DIR, default ./stimuli next to this file):
  TP.mp4, DM.mp4                          the two movies (not distributed)
  visual_scene_emotions_<movie>.csv       shot table from step 5
  semantic_event_emotions_<movie>.csv     semantic-event table from step 5
  attn_significance_movieTP.csv           event windows, movieTP model
  attn_significance_movieDM_seg2.csv      event windows, movieDM Segment 2 model
  emotion_final_ratings.mat               human valence ratings, movieDM
  lexicons/NRC-VAD-Lexicon.txt, lexicons/Ratings_Warriner_et_al.csv

Outputs go to STIM_OUT_DIR (default ./results next to this file).
With TEST_MODE=1 both default to folders under TEST_OUTPUT_DIR.
"""
import ast
import os
import subprocess
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
_TEST_ROOT = Path(os.environ.get("TEST_OUTPUT_DIR", HERE / "test_outputs"))

MEDIA = Path(os.environ.get("STIM_MEDIA_DIR", _TEST_ROOT / "stimulus_media" if TEST_MODE else HERE / "stimuli"))
OUT = Path(os.environ.get("STIM_OUT_DIR", _TEST_ROOT / "stimulus_features" if TEST_MODE else HERE / "results"))
LEXICON_DIR = Path(os.environ.get("STIM_LEXICON_DIR", MEDIA / "lexicons"))
CACHE = OUT / "cache"
CHANNELS = OUT / "channels"
TABLES = OUT / "tables"
STEP5_SCRIPT = HERE.parent / "step5 movies segment and emonational label.py"

TR = 0.8
FRAMES_PER_TR = 2
FRAME_RATE = FRAMES_PER_TR / TR          # 2.5 Hz
SEED = 20250803
N_SURROGATE = int(os.environ.get("N_SURROGATE", "5000"))
CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"

# movieTP: the scanned 200 s only. movieDM: the copy with audio runs two video
# frames (83.3 ms) ahead of the copy the step-5 annotation was built on, so every
# sampling time is shifted by that amount.
MOVIES = {
    "movieTP": {"video": "TP.mp4", "time_shift_sec": 0.0, "n_tr": 250, "duration_sec": 200.0,
                "attn_csv": "attn_significance_movieTP.csv", "attn_offset_tr": 0},
    "movieDM": {"video": "DM.mp4", "time_shift_sec": -0.08333333333333333, "n_tr": 750,
                "duration_sec": 600.0, "attn_csv": "attn_significance_movieDM_seg2.csv",
                "attn_offset_tr": 372, "ratings_mat": "emotion_final_ratings.mat"},
}

# movieDM segments, as in the Segment-2 training script
DM_SEGMENTS = {"Segment1": (0, 237), "Segment2": (372, 543), "Segment3": (547, 750)}

# prompt -> lexicon lookup term (27 GoEmotions categories, neutral, and two calm
# variants). Used only to look up published norms.
PROMPT_TO_TERM = {
    "a moment of admiration": "admiration", "an amusing scene": "amusement",
    "a moment of anger": "anger", "an annoying scene": "annoyance",
    "a moment of approval": "approval", "a caring and gentle moment": "caring",
    "a moment of confusion": "confusion", "a curious and exploring scene": "curiosity",
    "a scene of strong desire": "desire", "a disappointing moment": "disappointment",
    "a disapproving expression": "disapproval", "a disgusting moment": "disgust",
    "an embarrassing situation": "embarrassment", "an exciting and energetic scene": "excitement",
    "a fearful or scary moment": "fear", "a moment of gratitude": "gratitude",
    "a grieving and sorrowful moment": "grief", "a joyful and happy moment": "joy",
    "a loving and affectionate moment": "love", "a nervous or anxious scene": "nervousness",
    "an optimistic and uplifting moment": "optimism", "a proud and confident moment": "pride",
    "a moment of realization or discovery": "realization", "a relieving and calming moment": "relief",
    "a remorseful or regretful moment": "remorse", "a sad and emotional scene": "sadness",
    "a surprising and unexpected moment": "surprise", "a neutral and uneventful moment": "neutral",
    "a calm and quiet scene": "calm", "a transition with little emotion": "calm",
}
CATEGORY_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def ensure_dirs():
    for d in (OUT, CACHE, CHANNELS, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def media(name):
    return MEDIA / name


def scene_csv(movie, kind="visual"):
    stem = "visual_scene_emotions" if kind == "visual" else "semantic_event_emotions"
    return MEDIA / f"{stem}_{movie}.csv"


def _step5_literal(*names):
    tree = ast.parse(STEP5_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{names} not found in {STEP5_SCRIPT.name}")


def load_prompts():
    """The 30 emotion prompts, read from the step-5 script."""
    p = list(_step5_literal("EMOTION_PROMPTS", "goemotions_prompts"))
    assert len(p) == 30 and set(p) == set(PROMPT_TO_TERM), "prompt list differs from PROMPT_TO_TERM"
    return p


def load_category_map():
    """prompt -> positive / neutral / negative, read from the step-5 script."""
    return dict(_step5_literal("emotion_category_map"))


def ffmpeg_exe():
    from shutil import which
    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        exe = which("ffmpeg")
        if exe:
            return exe
    except Exception:  # noqa: BLE001
        pass
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout or "", p.stderr or "", p.returncode


def frames_to_tr(x):
    a = np.asarray(x, dtype=float)
    n = a.shape[0] // FRAMES_PER_TR
    return a[: n * FRAMES_PER_TR].reshape(n, FRAMES_PER_TR).mean(axis=1)


def save_channel(movie, channel, tr_values, frame_values=None, frame_times=None):
    CHANNELS.mkdir(parents=True, exist_ok=True)
    v = np.asarray(tr_values, dtype=float)
    tr = np.arange(len(v))
    pd.DataFrame({"tr_index": tr, "time_sec": (tr + 0.5) * TR, "value": v}).to_csv(
        CHANNELS / f"{movie}__{channel}.csv", index=False)
    if frame_values is not None:
        f = np.asarray(frame_values, dtype=float)
        fi = np.arange(len(f))
        ft = frame_times if frame_times is not None else (fi + 0.5) / FRAME_RATE
        pd.DataFrame({"frame_index": fi, "time_sec": ft, "value": f}).to_csv(
            CHANNELS / f"{movie}__{channel}__frames.csv", index=False)


def load_channel(movie, channel):
    p = CHANNELS / f"{movie}__{channel}.csv"
    return pd.read_csv(p)["value"].to_numpy() if p.exists() else None


def sig_clusters(attn_csv, col="sig_raw"):
    """(first, last) TR of each run of consecutive significant TRs."""
    df = pd.read_csv(attn_csv)
    sig = sorted(int(t) for t in df.loc[df[col].astype(bool), "timepoint"])
    out = []
    for _, g in groupby(enumerate(sig), lambda ix: ix[0] - ix[1]):
        c = [i for _, i in g]
        out.append((c[0], c[-1]))
    return out
