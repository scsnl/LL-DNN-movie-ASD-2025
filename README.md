# Neural Event Fingerprinting: DualPathNet Analysis Workflow

This repository contains the complete analytical pipeline for the paper "Neural Event Fingerprinting: Interpretable Deep Learning Reveals Stable and Context-Specific Brain Patterns in Autism".

## Quick Start

### 1. Environment Setup
Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Generate Test Dataset
Before running the full pipeline, generate a small 10-subject test dataset to verify the environment:
```bash
python make_test_data.py
```

### 3. Run Pipeline Smoke Test
Execute the automated test script to run all analysis steps in sequence using the test data:
```bash
./run_all_test.sh
```

---

## DualPathNet Architecture

The model has two branches: (1) **Global mean pooling path** (trait-level features) and (2) **MIL attention path** (event-level features). Each branch has its own fully connected layer producing 2-class logits. The outputs are fused using a weighted sum `α·logits_mean + (1−α)·logits_mil`. Final classification is obtained by argmax over the fused logits (α is a hyperparameter, default 0.6).

---

## Detailed Script Descriptions

### Module 1: Data Preparation
- `step 1 generate_cv_indices.py`: Performs data cleaning (QC filtering, NaN removal) and generates nested CV indices using StratifiedGroupKFold (by subject). Outputs `cv_indices/nested_indices_seed42.npz` (outer 5-fold + inner K-fold). Set `TEST_DATA_DIR` / `TEST_OUTPUT_DIR` for smoke test paths.

### Module 2: Model Training & Performance
- `step 2-1 5-fold training and ig.py`: Core training script for DualPathNet (Global temporal pooling + MIL paths). Saves best model weights and computes Integrated Gradients (IG). Uses indices from step 1.
- `step 2-1 nested cv runner.py`: Nested CV (outer 5-fold evaluation, inner K-fold for hyperparameter selection). Reads `nested_indices_seed42.npz`, performs 500-config random search, outputs `nested_results/nested_cv_summary.json`.
- `step 2-2 acc et summry and plot.py`: Reads `nested_results/nested_cv_summary.json` and plots 5-fold accuracy and F1 bar charts. Skips gracefully in smoke test when nested summary is absent.
- `step 2-3 100 iteration of fivefold cv.py`: 100x5-fold stratified CV for stability analysis. Skipped in TEST_MODE for fast smoke tests.

### Module 3: Model Interpretation & Feature Attribution
- `step3 analysis and plot attention weight.py`: Identifies statistically significant differences in temporal attention scores between ASD and TDC groups.
- `step3-1 plot mil attention weight.py`: Specialization of attention plotting for the MIL path component.
- `step3-2 plot ig whole brainmap.py`: Projects median IG attribution maps onto the 3D cortical surface using the Brainnetome atlas.
- `step3-3 plot ig topk brain.py`: Identifies and ranks the Top 5% most influential brain regions based on attribution magnitude.
- `step3-4 plot ig difference and topk timepoints curve.py`: Analyzes the group-wise difference in temporal attribution curves.
- `step3-6 combine all events.py`: Integrates accuracy bias, attention significance, and IG highlights into a unified temporal event plot.

### Module 4: Post-Hoc Network Analysis (Located in `Advanced post-hoc analyses/`)
- `network_diff_plot.py`: Aggregates ROI-level results into the Yeo 7-Network scheme and performs group comparisons.
- `step7_mil_20net_diff_baseline.py`: Baseline group comparison across 20 functional sub-networks without temporal windowing.
- `step16_network_diff_top10.py`: Network-level analysis focused on the Top 10% ROIs during specific identified events.
- `step17_v8_final_colors.py`: Stylized visualization of attention score curves with event category shading and hatching.

### Module 5: Brain-Behavior Association (Located in `Advanced post-hoc analyses/`)
- `final_step13_global_wholetime_corr.py`: Partial correlation analysis between Global Saliency Indices and clinical traits (SRS, RBS, CBCL).
- `final_step13_mil_cluster_corr_auto.py`: Event-specific brain-behavior association focusing on High/Low attention clusters.
- `final_step18_ig_weighted_cca.py`: Multivariate Canonical Correlation Analysis (CCA) between neural events and behavioral profiles.
- `final_step19_ig_only_plsc_totals.py`: Multivariate Partial Least Squares Correlation (PLSC) between brain events and clinical total scores.

### Module 6: Video Annotation
- `step 5 movies segment and emonational label.py`: Automated video interpretation using PySceneDetect (shots) and CLIP (emotions).
- `step 5-2 add description of events.py`: Generates natural language captions for model-identified neural events using BLIP-2.

---

## Assets and Data
- `assets/atlas/`: Contains the Brainnetome 246 atlas files used for mapping.
- `assets/mapping/`: CSV files defining ROI to functional network memberships.
- `assets/video/`: Video files (`DM.mp4`, `TP.mp4`) used for stimuli annotation.
- `data/`: Placeholder directory for fMRI `.pklz` and behavioral `.csv` files.
