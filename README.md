# Deep learning localizes autism-related neural differences to emotionally demanding moments in naturalistic movies

Analysis code for the paper of the same title. DualPathNet classifies autistic and typically developing children from region-wise fMRI time series (Brainnetome, 246 regions) recorded during movie viewing (The Present, movieTP; Despicable Me, movieDM). It has two branches: a Global Temporal-Pooling path and an attention-based MIL path. Integrated Gradients (IG) attributions from the model are used to:

- define event-level brain features;
- identify cross-event common regions and event-specific regions;
- relate the features to behaviour with PLSC.

No participant data are included.

## Environment

Python 3.13. The code was tested with numpy 2.4, pandas 2.2, scikit-learn 1.7, scipy 1.15, torch 2.9, pytorch-lightning 2.6, captum 0.8, nilearn 0.12, transformers 5.0 and opencv 4.11.

```bash
pip install -r requirements.txt
```

## Smoke test

`make_test_data.py` writes a synthetic data set (10 participants per film, random time series, random behavioural scores) to `data/`. `run_all_test.sh` regenerates it and runs every script in order with `TEST_MODE=1`. In this mode the scripts:

- train for one epoch;
- use reduced permutation and bootstrap counts;
- replace the CLIP and BLIP-2 models with placeholders;
- use synthetic stimulus media.

```bash
./run_all_test.sh
```

Outputs go to `test_outputs/`. The numbers are meaningless; the test only checks that the code runs end to end. Set `DUMMY_MODE=1` to skip training and use random fold outputs.

## Layout

- `scripts/Final clean script/`: main pipeline (steps 1-6).
- `scripts/Final clean script/Advanced post-hoc analyses/`: segment-wise training, PLSC and other post-hoc analyses.
- `scripts/Final clean script/Common regions/`: cross-event common and event-specific regions.
- `scripts/Final clean script/Stimulus features/`: low-level audiovisual salience and semantic valence of the stimuli.
- `assets/mapping/brainnetome_devatlas.csv`: Brainnetome region to Dev-Atlas system (DMN, CON, SAL, ATN, SMN, VIS).
- `assets/mapping/subregion_func_network_Yeo.csv`: Brainnetome region to Yeo networks.
- `assets/behavior/`: lists of behavioural measures.

Fold outputs (`saved_models/fold*/`) hold one row per validation participant, together with `subject_ids.npy`. Downstream scripts read them through `scripts/Final clean script/ig_io.py`, which matches rows to participants by subject ID.

## Scripts and figures

| Script | Figure / table |
|---|---|
| `step 1 generate_cv_indices.py` | QC and cross-validation indices |
| `step 2-1 nested cv runner.py`, `step 2-1 5-fold training and ig.py`, `step 2-2 acc et summry and plot.py` | Figure 2B |
| `step 2-3 100 iteration of fivefold cv.py` | Figure S1 |
| `step3 analysis and plot attention weight.py`, `step3-1 plot mil attention weight.py` | Figures 3B, S4B, S8B |
| `step3-2 plot ig whole brainmap.py`, `step3-3 plot ig topk brain.py` | Figures S2, S9; Tables S4, S5, S8, S9 |
| `step3-4 plot ig difference and topk timepoints curve.py` | Figures S4A, S8A |
| `step5 movies segment and emonational label.py` | Figures 3A, 4B, S8C |
| `step5-2 add description of events.py` | event captions in Figures 3D, 5C |
| `step6-1 meanpath association between behavior and fmri.py` | Figure S3 |
| `Advanced post-hoc analyses/supply-step2_timefiltered_train_and_ig.py`, `supply-step2-1_timefiltered_eval.py` | Figure 4C |
| `Advanced post-hoc analyses/step17_v8_final_colors.py` | Figure 5A |
| `Advanced post-hoc analyses/final_step19_ig_only_plsc_totals.py` | Figures 3E-F, 5D-E, S6 |
| `Common regions/cr1-cr5` | Figures 3C-D, 5B-C, S5, S10; Tables S6, S10 |
| `Stimulus features/sf1-sf9` | Table S7 |

Each subfolder has a README with details.

## Data access

The fMRI and phenotypic data come from the Child Mind Institute Healthy Brain Network (CMI-HBN). They are available through the International Neuroimaging Data-sharing Initiative (INDI) portal after signing the HBN data usage agreement. The movie files and the movieDM valence ratings are not distributed here. Expected input files and formats are described in `make_test_data.py` and in the README of each subfolder.

## Citation

[citation to be added upon publication]
