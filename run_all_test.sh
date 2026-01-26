#!/usr/bin/env bash
set -euo pipefail

# Run the whole pipeline on 10-subject test data.
# This is intended to "smoke test" the GitHub package.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts/Final clean script"
DATA_DIR="$ROOT_DIR/data"
OUT_DIR="$ROOT_DIR/test_outputs"

export TEST_MODE=1
export TEST_DATA_DIR="$DATA_DIR"
export TEST_OUTPUT_DIR="$OUT_DIR"
export DUMMY_MODE="${DUMMY_MODE:-0}"
export N_PERM="${N_PERM:-50}"
export N_BOOT="${N_BOOT:-50}"

mkdir -p "$OUT_DIR"

echo "[1/2] Generate test data..."
python "$ROOT_DIR/make_test_data.py"

echo "[2/2] Run scripts in order..."
cd "$SCRIPTS_DIR"

# Link test data into the working directory (many legacy scripts expect local filenames)
ln -sfn "$DATA_DIR/combined_asd_td_movieDM_data.pklz" "combined_asd_td_movieDM_data.pklz"
ln -sfn "$DATA_DIR/combined_asd_td_movieTP_data.pklz" "combined_asd_td_movieTP_data.pklz"
ln -sfn "$DATA_DIR/all subject behavior data.csv" "all subject behavior data.csv"

# Link behavior-list CSVs for scripts that search relative paths
ln -sfn "$ROOT_DIR/assets/behavior/behavior-list.csv" "behavior-list.csv"
ln -sfn "$ROOT_DIR/assets/behavior/behavior-list cbcl.csv" "behavior-list cbcl.csv"
ln -sfn "$ROOT_DIR/assets/behavior/behavior-list rbs.csv" "behavior-list rbs.csv"
ln -sfn "$ROOT_DIR/assets/behavior/behavior-list srs.csv" "behavior-list srs.csv"
ln -sfn "$ROOT_DIR/assets/behavior/behavior-list WISC.csv" "behavior-list WISC.csv"

# Route heavy outputs to test_outputs but keep legacy paths working via symlinks
mkdir -p "$OUT_DIR/saved_models" "$OUT_DIR/cv_indices" "$OUT_DIR/crossval_results"
ln -sfn "$OUT_DIR/saved_models" "saved_models"
ln -sfn "$OUT_DIR/cv_indices" "cv_indices"
ln -sfn "$OUT_DIR/crossval_results" "crossval_results"

# Step 1: CV indices
python "step 1 generate_cv_indices.py"

# Step 2: Train + IG (set small epochs inside script when TEST_MODE=1)
python "step 2-1 5-fold training and ig.py"

# Step 2: Metrics summary
python "step 2-2 acc et summry and plot.py"

# Step 2: Repeated CV (TEST_MODE reduces to 1 run)
python "step 2-3 100 iteration of fivefold cv.py"

# Step 2: quick check script
python "step2-1 test and print.py"

# Step 3: Attention stats + significance CSV
python "step3 analysis and plot attention weight.py"
python "step3-1 plot mil attention weight.py"

# Step 3: Plots (may require outputs from step 2)
python "step3-2 plot ig whole brainmap.py"
python "step3-3 plot ig topk brain.py"
python "step3-4 plot ig difference and topk timepoints curve.py"
python "step3-5 train mil frame logits and plot frame-level performance.py"
python "step3-6 combine all events.py"

# Step 4: Embedding visualization (optional, but included)
python "step4 umap and tsne.py"

# Step 5: Video scripts (skip heavy model load in TEST_MODE inside scripts)
python "step5 movies segment and emonational label.py"
python "step5-2 add description of events.py"

# Step 6: Brain-behavior association (uses test behavior CSV)
python "step6-1 meanpath association between behavior and fmri.py"
python "step6-2 milpath association between behavior and fmri.py"

# Selected MovieDM add-on analyses
cd "$SCRIPTS_DIR/Advanced post-hoc analyses"

# also link test data here (some scripts assume local names)
ln -sfn "$DATA_DIR/combined_asd_td_movieDM_data.pklz" "combined_asd_td_movieDM_data.pklz"
ln -sfn "$DATA_DIR/combined_asd_td_movieTP_data.pklz" "combined_asd_td_movieTP_data.pklz"
ln -sfn "$DATA_DIR/all subject behavior data.csv" "all subject behavior data.csv"

# prepare seg2 attention csv fallback for scripts expecting results/seg2/figures
mkdir -p "results/seg2/figures"
if [[ -f "../figures/attn_significance.csv" ]]; then
  cp -f "../figures/attn_significance.csv" "results/seg2/figures/attn_significance.csv"
fi

# time-filtered training (TEST_MODE runs seg2 only)
python "supply-step2_timefiltered_train_and_ig.py"
python "supply-step2-1_timefiltered_eval.py"

python "step8_threshold_stability_surface.py"
python "step9A_backbone_frequency_surface.py"
python "step9B_common_unique_event_panel_surface.py"
python "step7_mil_20net_diff_baseline.py"
python "step16_network_diff_top10.py"
python "step17_v8_final_colors.py"
python "supply-step3-6_seg2_compute_events_and_combine.py"
python "final_step13_global_wholetime_corr.py"
python "final_step13_mil_cluster_corr_auto.py"

# Multivariate (reduce heavy permutation/bootstrap inside scripts when TEST_MODE=1)
python "final_step19_ig_only_plsc_totals.py"
python "final_step18_ig_weighted_cca.py"

echo "All scripts completed. Outputs in: $OUT_DIR"

