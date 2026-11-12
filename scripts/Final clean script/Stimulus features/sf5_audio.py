"""
Auditory channels per half-TR bin (0.4 s), then per TR.

  loudness_mom    ITU-R BS.1770 K-weighted loudness over the 0.4 s bin, LUFS
  loudness_short  the same over a 3 s window centred on the bin
  spectral_flux   half-wave rectified change of the STFT magnitude spectrum, bin mean
  speech_prob     proportion of the bin covered by Silero VAD speech segments

pyloudnorm's K-weighting filters are applied and per-block loudness
L = -0.691 + 10 log10(sum_ch G_ch mean(z_ch^2)) is taken without gating.
Bin edges carry the movie's time shift so audio and frames share one grid.

Outputs: channels/<movie>__<channel>.csv
"""
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal as sps

from sf_common import CACHE, FRAME_RATE, MOVIES, OUT, ensure_dirs, ffmpeg_exe, frames_to_tr, save_channel

BIN_SEC = 1.0 / FRAME_RATE
SHORT_TERM_SEC = 3.0
STFT_N, STFT_HOP = 2048, 512
EPS = 1e-12
CHANNEL_WEIGHTS = {1: [1.0], 2: [1.0, 1.0]}
VAD_RATE = 16000


def k_weight(x, rate):
    import pyloudnorm
    y = x.copy()
    for f in pyloudnorm.Meter(rate)._filters.values():
        y = f.apply_filter(y)
    return y


def block_loudness(zk, rate, centres, win_sec, g):
    half = int(round(win_sec * rate / 2))
    n = zk.shape[0]
    cs = np.concatenate([np.zeros((1, zk.shape[1])), np.cumsum(zk ** 2, axis=0)], axis=0)
    out = np.empty(len(centres))
    for i, c in enumerate(centres):
        s = int(round(c * rate)) - half
        s_c, e_c = max(0, s), min(n, s + 2 * half)
        if e_c <= s_c:
            out[i] = -np.inf
            continue
        out[i] = -0.691 + 10.0 * np.log10(float(np.dot(g, (cs[e_c] - cs[s_c]) / (e_c - s_c))) + EPS)
    return out


def bin_mean(values, times, edges):
    idx = np.digitize(times, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    ok = (idx >= 0) & (idx < len(out))
    s = np.bincount(idx[ok], weights=values[ok], minlength=len(out))
    c = np.bincount(idx[ok], minlength=len(out))
    out[c > 0] = s[c > 0] / c[c > 0]
    return out


def speech_proportion(wav, edges):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    pcm = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(wav), "-ac", "1", "-ar", str(VAD_RATE),
                          "-f", "s16le", "-acodec", "pcm_s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    ts = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(), sampling_rate=VAD_RATE,
                               return_seconds=True)
    prop = np.zeros(len(edges) - 1)
    for seg in ts:
        lo = np.maximum(edges[:-1], float(seg["start"]))
        hi = np.minimum(edges[1:], float(seg["end"]))
        prop += np.clip(hi - lo, 0, None)
    return np.clip(prop / np.diff(edges), 0.0, 1.0)


def run_movie(movie):
    wav = CACHE / f"audio_{movie}.wav"
    shift = MOVIES[movie]["time_shift_sec"]
    n_frames = len(pd.read_csv(OUT / f"frames_{movie}.csv"))
    centres = (np.arange(n_frames) + 0.5) / FRAME_RATE + shift
    edges = np.arange(n_frames + 1) / FRAME_RATE + shift

    x, rate = sf.read(str(wav), always_2d=True, dtype="float64")
    g = CHANNEL_WEIGHTS.get(x.shape[1], [1.0] * x.shape[1])
    zk = k_weight(x, rate)
    mono = x.mean(axis=1)
    f, t, Z = sps.stft(mono, fs=rate, nperseg=STFT_N, noverlap=STFT_N - STFT_HOP, window="hann",
                       boundary=None, padded=False)
    M = np.abs(Z)
    flux = np.zeros(M.shape[1])
    flux[1:] = np.sqrt((np.maximum(np.diff(M, axis=1), 0.0) ** 2).sum(axis=0))

    series = {"loudness_mom": block_loudness(zk, rate, centres, BIN_SEC, g),
              "loudness_short": block_loudness(zk, rate, centres, SHORT_TERM_SEC, g),
              "spectral_flux": bin_mean(flux, t, edges),
              "speech_prob": speech_proportion(wav, edges)}
    for name, v in series.items():
        v = np.asarray(v, dtype=float)
        if np.isnan(v).any():
            v = pd.Series(v).interpolate(limit_direction="both").to_numpy()
        if np.isneginf(v).any():
            fin = v[np.isfinite(v)]
            v = np.where(np.isneginf(v), fin.min() - 10.0 if fin.size else -120.0, v)
        save_channel(movie, name, frames_to_tr(v), v, centres)
    print(f"{movie}: audio {rate} Hz, {x.shape[1]} ch, speech in {np.nanmean(series['speech_prob']) * 100:.1f}% of run time")


def main():
    ensure_dirs()
    for m in MOVIES:
        run_movie(m)


if __name__ == "__main__":
    main()
