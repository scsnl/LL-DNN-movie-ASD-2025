# Stimulus features

Low-level audiovisual salience and semantic valence of the two movies, and the
comparison of movieDM Segment 2 with Segments 1 and 3 (Table S7). These
measures describe the stimuli only; none of them indexes the participants'
arousal.

| Script | Output | Manuscript |
|---|---|---|
| `sf1_extract.py` | frames at 2 per TR (2.5 Hz) and audio | Methods, stimulus salience |
| `sf2_clip.py` | CLIP probabilities over the 30 step-5 prompts | Methods, semantic valence and arousal |
| `sf3_lexicon.py` | NRC and Warriner norms, ranks, median split, lexicon agreement | Methods, semantic valence and arousal |
| `sf4_semantic.py` | valence rank series, three-way annotation series, Segment 2 category composition (24 variants) | Table S7, SI stimulus characterization |
| `sf5_audio.py` | loudness (BS.1770), spectral flux, speech presence (Silero VAD) | Table S7 |
| `sf6_visual.py` | motion energy, luminance, RMS contrast, cut rate | Table S7 |
| `sf7_salience.py` | PC1 of the z-scored channels (composite audiovisual salience) | Table S7 |
| `sf8_validation.py` | automatic vs human valence ratings, movieDM | Results, Figures S7-S8 |
| `sf9_segments.py` | Segment 2 and event-window comparisons, `tables/table_s7.csv` | Table S7 |
| `sf_common.py`, `sf_stats.py` | paths, constants, surrogate tests | |
| `make_test_media.py` | synthetic inputs for a test run | |

Run the scripts in numerical order from this folder.

## Inputs

The movies and the human valence ratings are not distributed. Place them in
`stimuli/` next to these scripts or set `STIM_MEDIA_DIR`:

- `TP.mp4`, `DM.mp4`
- `visual_scene_emotions_<movie>.csv`, `semantic_event_emotions_<movie>.csv` (step 5 output)
- `attn_significance_movieTP.csv`, `attn_significance_movieDM_seg2.csv` (step 3 output)
- `emotion_final_ratings.mat` (TR x rating series)

The NRC VAD lexicon and the Warriner norms are downloaded into
`STIM_LEXICON_DIR` (default `stimuli/lexicons/`) on first use, subject to their
own licences. Outputs go to `results/` or `STIM_OUT_DIR`.

## Test run

```bash
export TEST_MODE=1 TEST_OUTPUT_DIR=/path/to/test_outputs N_SURROGATE=50
python make_test_media.py
for s in sf1_extract sf2_clip sf3_lexicon sf4_semantic sf5_audio sf6_visual sf7_salience sf8_validation sf9_segments; do python $s.py; done
```

With `TEST_MODE=1` the CLIP model is not loaded (seeded placeholder
probabilities) and seeded synthetic lexicon tables are used.
