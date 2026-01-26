## Advanced Post-hoc Analyses (MovieDM & MovieTP)

This directory contains consolidated, non-redundant scripts for advanced post-hoc analysis, focusing on MovieDM and MovieTP datasets. These scripts are organized for researchers to explore event-specific patterns, functional network mappings, and multivariate brain-behavior associations.

### 1) Time-filtered Training and Evaluation
- `supply-step2_timefiltered_train_and_ig.py`: Localized training on specific movie segments (seg1/seg2/seg3) and attribution extraction.
- `supply-step2-1_timefiltered_eval.py`: Performance evaluation and metric summary across segments and folds.

### 2) Event Integration (Seg2 Focus)
- `supply-step3-6_seg2_compute_events_and_combine.py`: Integrated analysis of accuracy bias, IG top timepoints, and attention significance to define neural events.

### 3) Visualizations
- `step17_v8_final_colors.py`: Stylized plotting of attention curves with standardized group colors and event-based hatching.

### 4) Surface-Based Analysis (Backbone & Common/Unique)
- `step8_threshold_stability_surface.py`: Identifies ROIs consistently appearing in top attribution ranks across folds (Stability Backbone).
- `step9A_backbone_frequency_surface.py`: Frequency of ROI appearance in Top-K across multiple neural events (MovieDM).
- `step9A_backbone_frequency_surface_movietp.py`: Frequency of ROI appearance in Top-K across multiple neural events (MovieTP).
- `step9B_common_unique_event_panel_surface.py`: Contrast between shared "common" regions and event-specific "unique" regions (MovieDM).
- `step9B_common_unique_event_panel_surface_movietp.py`: Contrast between shared "common" regions and event-specific "unique" regions (MovieTP).

### 5) Functional Network Analysis
- `network_diff_plot.py`: Maps ROIs to Yeo 7 networks and performs statistical group comparisons (ASD vs TDC).
- `step7_mil_20net_diff_baseline.py`: Baseline group comparison using 20 functional sub-networks without temporal constraints.
- `step16_network_diff_top10.py`: Specialized network analysis of Top 10% ROIs within defined neural events.

### 6) Brain-Behavior Correlation (Univariate)
- `final_step13_global_wholetime_corr.py`: Partial correlation between Global Mean path saliency and clinical traits (Age, Sex, Site controlled).
- `final_step13_mil_cluster_corr_auto.py`: Correlation analysis focusing on High/Low attention events identified by the MIL path.

### 7) Brain-Behavior Association (Multivariate)
- `final_step19_ig_only_plsc_totals.py`: Partial Least Squares Correlation (PLSC) between event saliency and behavioral total scores.
- `final_step18_ig_weighted_cca.py`: Canonical Correlation Analysis (CCA) with Freedman-Lane permutation and bootstrap resampling for stability.
