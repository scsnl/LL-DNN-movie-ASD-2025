# Project Pipeline Overview

This repository contains scripts for training and analyzing the DualPathNet (stDNN) model for ASD vs TDC classification using movie-viewing fMRI paradigms.

## Directory Structure

```
├── cv_generation/              # Cross-validation index generation
│   └── generate_cv_indices.py  # QC and stratified 5-fold split (10 random seeds)
│
├── training/                   # Model training and evaluation
│   ├── train_and_ig.py         # 5-fold training + Integrated Gradients
│   ├── summarize_metrics.py    # Aggregate accuracy/F1 and plot summaries
│   └── stability_cv.py         # 100 iterations of 5-fold CV for robustness analysis
│
├── interpretation/             # Model interpretation & visualization
│   ├── plot_attention_analysis.py    # Global attention weight analysis plots
│   ├── plot_mil_attention.py        # Frame-wise MIL attention visualizations
│   ├── plot_ig_full_brainmap.py     # Whole-brain IG heatmaps
│   ├── plot_ig_topk_rois.py         # Top-k ROI attribution plots
│   ├── plot_ig_timepoints.py        # IG curve comparison and top timepoint highlights
│   └── plot_frame_performance.py    # Frame-level classification performance plots
│
├── events_annotation/         # Stimulus event segmentation and labeling
│   ├── segment_and_label_emotion.py  # CLIP-based zero-shot emotion labels per segment
│   ├── describe_events.py           # BLIP-2-generated descriptions for each movie event
│   └── combine_event_annotations.py # Merge emotion, semantics, and attention event labels
│
├── dimensionality_reduction/   # UMAP & t-SNE visualization
│   └── umap_tsne_visualization.py   # Low-dimensional embedding plots colored by metadata
│
├── brain_behavior/             # Brain–behavior correlation analyses
│   ├── correlate_meanpath.py    # Correlate GlobalMean branch activations with behavior
│   └── correlate_milpath.py     # Correlate MIL path activations with behavioral scores
│
└── README.md                   # This file
```

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate CV indices**

   ```bash
   python cv_generation/generate_cv_indices.py --input harmonized_data.pklz --output cv_splits/
   ```

3. **Train and evaluate model**

   ```bash
   python training/train_and_ig.py --data harmonized_data.pklz --cv_dir cv_splits/ --out_dir models/
   python training/summarize_metrics.py --results_dir models/ --save_path figures/metrics_summary.png
   python training/stability_cv.py --data harmonized_data.pklz --out_dir stability_results/
   ```

4. **Interpretation & Visualization**

   ```bash
   python interpretation/plot_attention_analysis.py --input models/attention_weights/
   python interpretation/plot_mil_attention.py --input models/mil_attention/
   python interpretation/plot_ig_full_brainmap.py --input models/ig_scores/
   python interpretation/plot_ig_topk_rois.py --input models/ig_scores/
   python interpretation/plot_ig_timepoints.py --input models/ig_time_series/
   python interpretation/plot_frame_performance.py --input models/frame_logits/
   ```

5. **Dimensionality Reduction**

   ```bash
   python dimensionality_reduction/umap_tsne_visualization.py --embeddings embeddings.npy --meta metadata.csv
   ```

6. **Stimulus Annotation**

   ```bash
   python events_annotation/segment_and_label_emotion.py --video_dir frames/ --out emotion_labels.csv
   python events_annotation/describe_events.py --segments emotion_labels.csv --out descriptions.csv
   python events_annotation/combine_event_annotations.py --inputs emotion_labels.csv descriptions.csv attention_events.csv
   ```

7. **Brain–Behavior Correlation**

   ```bash
   python brain_behavior/correlate_meanpath.py --features meanpath_activations.npy --behavior behavior.csv
   python brain_behavior/correlate_milpath.py --features milpath_activations.npy --behavior behavior.csv
   ```

## Configuration

- All script-specific options (e.g., paths, parameters) can be set via `argparse` flags. Use `-h` to view usage.

## Contributions

Please open an issue or submit a pull request for bug fixes or new features.

## License

[Specify your license here]

