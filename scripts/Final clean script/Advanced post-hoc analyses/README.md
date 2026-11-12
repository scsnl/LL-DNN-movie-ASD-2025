## Post-hoc analyses

Run these scripts from this folder. Fold outputs are read through `../ig_io.py`, which attaches each IG row to its subject ID.

### Segment-wise training (movieDM)
- `supply-step2_timefiltered_train_and_ig.py`: 5-fold training and IG for Segment 1 (TR 0-237), Segment 2 (TR 372-543) and Segment 3 (TR 547-end). Writes `results/<seg>/saved_models/fold*/`.
- `supply-step2-1_timefiltered_eval.py`: accuracy, precision, recall and F1 for each fold and segment.

### Segment 2 events
- `supply-step3-6_seg2_compute_events_and_combine.py`: overlap between the IG top time points and the attention-significant TRs.
- `step17_v8_final_colors.py`: attention time courses with event bands.

### Brain-behaviour association
- `final_step19_ig_only_plsc_totals.py`: PLSC between event-level Neural Saliency Indices and SRS, RBS-R and CBCL total scores. Writes weights and bootstrap ratios.
- `final_step13_global_wholetime_corr.py`: Global Saliency Index (mean |IG| over the Global-path top-5% regions) and behavioural scores, Segment 2.
- `final_step13_mil_cluster_corr_auto.py`: event-level Neural Saliency Index and behavioural scores, Segment 2.
- `final_step18_ig_weighted_cca.py`: CCA between event features and total scores.

Partial correlations adjust for age, sex, site and mean framewise displacement. The Segment 2 scripts read `results/seg2/saved_models` and `results/seg2/figures/attn_significance.csv`; the latter is written by `python "../step3 analysis and plot attention weight.py" results/seg2/saved_models results/seg2/figures`.

### Yeo network summaries
- `network_diff_plot.py`, `step7_mil_20net_diff_baseline.py` and `step16_network_diff_top10.py`: group comparisons of ROIs mapped to Yeo networks, using `assets/mapping/subregion_func_network_Yeo.csv`.

Cross-event common regions are in `../Common regions/`. Stimulus measures (Table S7) are in `../Stimulus features/`.
