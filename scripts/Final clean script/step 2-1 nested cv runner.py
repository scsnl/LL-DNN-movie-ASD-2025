# step 2-1 nested cv runner.py
# Nested CV: outer 5-fold for unbiased evaluation, inner K-fold for hyperparameter selection.
# Uses indices from step 1 (nested_indices_seed42.npz) and train utilities from step 2-1 5-fold training and ig.py.

import os
import hashlib
import json
import importlib.util
import numpy as np
from sklearn.model_selection import train_test_split, ParameterSampler

# Import fit_one_split, eval_on_idx, load_and_preprocess_data from step 2-1 5-fold training and ig.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_step21_path = os.path.join(_script_dir, "step 2-1 5-fold training and ig.py")
_spec = importlib.util.spec_from_file_location("step2_1_module", _step21_path)
_step21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step21)
fit_one_split = _step21.fit_one_split
eval_on_idx = _step21.eval_on_idx
load_and_preprocess_data = _step21.load_and_preprocess_data

# Hyperparameter search: 500 random configs
N_HP_CONFIGS = 500
HP_SEED = 42

PARAM_DISTRIBUTIONS = {
    "lr": [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "alpha": np.linspace(0.3, 0.8, 15).tolist(),
    "dropout_rate": np.linspace(0.3, 0.7, 13).tolist(),
    "batch_size": [16, 32, 64],
    "max_epochs": [30, 40, 50, 60, 80],
}
PATIENCE_FIXED = 8


def _build_hp_grid(n_configs=N_HP_CONFIGS, seed=HP_SEED):
    """Generate n_configs random hyperparameter combinations (reproducible)."""
    rng = np.random.RandomState(seed)
    sampler = ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=n_configs,
        random_state=rng,
    )
    return [
        {
            "lr": float(hp["lr"]),
            "alpha": float(hp["alpha"]),
            "dropout_rate": float(hp["dropout_rate"]),
            "batch_size": int(hp["batch_size"]),
            "max_epochs": int(hp["max_epochs"]),
            "patience": PATIENCE_FIXED,
        }
        for hp in sampler
    ]


def _load_nested_indices(npz_path):
    """Reconstruct nested dict from npz."""
    data = np.load(npz_path, allow_pickle=True)
    y = data["y"]
    n_outer = int(data["n_outer"])
    n_inner = int(data["n_inner"])
    nested = {"outer": []}
    for i in range(n_outer):
        inner_splits = []
        for j in range(n_inner):
            inner_splits.append({
                "train": data[f"inner_train_{i}_{j}"],
                "val": data[f"inner_val_{i}_{j}"],
            })
        nested["outer"].append({
            "train": data[f"outer_train_{i}"],
            "test": data[f"outer_test_{i}"],
            "inner": inner_splits,
        })
    return nested, y


def inner_score_for_hp(X, meta, y, inner_splits, hp, metadata_dim, seed, workdir):
    """For one hp config, train on each inner fold and return mean val_acc."""
    scores = []
    for j, sp in enumerate(inner_splits):
        train_idx = sp["train"]
        val_idx = sp["val"]
        score, _ = fit_one_split(
            X, meta, y, train_idx, val_idx, hp, metadata_dim, seed,
            save_dir=os.path.join(workdir, f"inner{j}"),
        )
        scores.append(score)
    return float(np.mean(scores))


def make_inner_holdout_split(outer_train_idx, y, seed, val_frac=0.1):
    """Split outer_train into train/val for early stopping (90/10 stratified)."""
    n = len(outer_train_idx)
    n_val = max(2, int(n * val_frac))
    if n_val >= n:
        n_val = max(1, n - 1)
    y_train = y[outer_train_idx]
    train_idx, val_idx = train_test_split(
        np.arange(n),
        test_size=n_val,
        stratify=y_train,
        random_state=seed,
    )
    return outer_train_idx[train_idx], outer_train_idx[val_idx]


def retrain_on_outer_train(X, meta, y, outer_train_idx, hp, metadata_dim, seed, workdir):
    """Retrain on full outer_train with best hp; use small holdout inside for early stopping."""
    train_idx, val_idx = make_inner_holdout_split(outer_train_idx, y, seed)
    _, ckpt = fit_one_split(
        X, meta, y, train_idx, val_idx, hp, metadata_dim, seed,
        save_dir=os.path.join(workdir, "finalfit"),
    )
    return ckpt


def run_nested(X, meta, y, nested, seed=42, workdir_base="./nested_results", n_hp_configs=N_HP_CONFIGS):
    """
    Main nested CV loop.
    Returns (outer_results, chosen_hps).
    """
    metadata_dim = meta.shape[1]
    outer_results = []
    chosen_hps = []
    hp_grid = _build_hp_grid(n_configs=n_hp_configs, seed=seed)
    print(f"Hyperparameter search: {len(hp_grid)} configs")

    for i, outer in enumerate(nested["outer"]):
        outer_train_idx = outer["train"]
        outer_test_idx = outer["test"]
        inner_splits = outer["inner"]
        workdir = os.path.join(workdir_base, f"outer{i}")
        os.makedirs(workdir, exist_ok=True)

        # Inner loop: hyperparameter selection
        best_hp, best_score = None, -1e9
        for idx, hp in enumerate(hp_grid):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  Outer {i}: hp {idx+1}/{len(hp_grid)}")
            hp_key = hashlib.md5(str(sorted(hp.items())).encode()).hexdigest()[:12]
            s = inner_score_for_hp(
                X, meta, y, inner_splits, hp, metadata_dim, seed,
                workdir=os.path.join(workdir, f"hp_{hp_key}"),
            )
            if s > best_score:
                best_score, best_hp = s, hp

        chosen_hps.append(best_hp)

        # Refit on outer_train with best hp
        final_ckpt = retrain_on_outer_train(
            X, meta, y, outer_train_idx, best_hp, metadata_dim, seed, workdir,
        )

        # Unbiased evaluation on outer_test
        acc, f1, preds, _ = eval_on_idx(
            final_ckpt, X, meta, y, outer_test_idx, best_hp, metadata_dim,
        )
        outer_results.append({
            "outer_fold": i,
            "acc": float(acc),
            "f1": float(f1),
            "hp": best_hp,
        })
        print(f"Outer fold {i}: acc={acc:.4f}, f1={f1:.4f}, hp={best_hp}")

    return outer_results, chosen_hps


def main():
    test_data_dir = os.environ.get("TEST_DATA_DIR", "")
    test_output_dir = os.environ.get("TEST_OUTPUT_DIR", "")
    test_mode = os.environ.get("TEST_MODE", "0") == "1"

    if test_data_dir and test_output_dir:
        data_path = os.path.join(test_data_dir, "combined_asd_td_movieDM_data.pklz")
        npz_path = os.path.join(test_output_dir, "cv_indices", "nested_indices_seed42.npz")
        workdir_base = os.path.join(test_output_dir, "nested_results")
        n_hp = 2 if test_mode else int(os.environ.get("N_HP_CONFIGS", N_HP_CONFIGS))
    else:
        data_path = "# TODO: specify your data path"
        npz_path = "# TODO: specify path to nested_indices_seed42.npz"
        workdir_base = "./nested_results"
        n_hp = int(os.environ.get("N_HP_CONFIGS", N_HP_CONFIGS))

    if data_path.startswith("#") or npz_path.startswith("#"):
        raise ValueError("Set TEST_DATA_DIR and TEST_OUTPUT_DIR, or edit data_path and npz_path in script.")

    fmri, meta, labels = load_and_preprocess_data(data_path)
    nested, y_expected = _load_nested_indices(npz_path)

    # Sanity check: labels in npz should match loaded labels
    if not np.array_equal(y_expected, labels):
        raise ValueError("Labels in npz do not match loaded data. Ensure step 1 used same data_path and preprocessing.")

    outer_results, chosen_hps = run_nested(
        fmri, meta, labels, nested, seed=42, workdir_base=workdir_base, n_hp_configs=n_hp,
    )

    accs = [r["acc"] for r in outer_results]
    f1s = [r["f1"] for r in outer_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    summary = {
        "outer_results": outer_results,
        "chosen_hps": chosen_hps,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
    }
    os.makedirs(workdir_base, exist_ok=True)
    with open(os.path.join(workdir_base, "nested_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nNested CV summary:")
    print(f"  Outer test Acc: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Outer test F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Saved to {workdir_base}/nested_cv_summary.json")


if __name__ == "__main__":
    main()
