import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# === Function: Pearson + Partial Correlation ===

def compute_pearson_and_partial_corr(x, y, covariates):
    assert x.shape == y.shape
    assert x.ndim == 1 and y.ndim == 1 and covariates.ndim == 2
    assert len(x) == covariates.shape[0]

    pearson_r, pearson_p = pearsonr(x, y)

    def get_residuals(target, X):
        model = LinearRegression().fit(X, target)
        return target - model.predict(X)

    x_resid = get_residuals(x, covariates)
    y_resid = get_residuals(y, covariates)
    partial_r, partial_p = pearsonr(x_resid, y_resid)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'partial_r': partial_r,
        'partial_p': partial_p
    }

# === Step 0:  ===

base_dir = "./"
roi_dir = "./top_roi_summary/"
os.chdir(base_dir)

roi_file = os.path.join(roi_dir, "outmean_asd_top_roi.csv")
fmri_file = os.path.join(base_dir, "combined_asd_td_movieDM_data.pklz")
behavior_file = os.path.join(base_dir, "all subject behavior data.csv")

# ===  ===

roi_df = pd.read_csv(roi_file)
datao = pd.read_pickle(fmri_file)
behavior_df = pd.read_csv(behavior_file, low_memory=False)

# === fMRI  ===

datao = datao[(datao['percentofvolsrepaired'] <= 10) & (datao['mean_fd'] <= 0.5)]
shapes = [np.asarray(d).shape for d in datao.data]
main_shape = max(set(shapes), key=shapes.count)
datao = datao.iloc[[i for i, s in enumerate(shapes) if s == main_shape]].reset_index(drop=True)
fmri = np.stack([np.asarray(d) for d in datao.data])
nan_subs = np.unique(np.argwhere(np.isnan(fmri))[:, 0])
datao = datao.drop(index=nan_subs).reset_index(drop=True)
fmri = np.stack([np.asarray(d) for d in datao.data])

# ===  & ID  ===

datao['label'] = datao['label'].str.lower().map({'asd': 0, 'td': 1})
datao['subject_id'] = datao['subject_id'].astype(str).str.strip().str.lower()
behavior_df['Identifiers'] = behavior_df['Identifiers'].astype(str).str.strip().str.lower()

# === ASD  ===

asd_mask = datao['label'] == 0
fmri_asd = fmri[asd_mask.values]
asd_ids = datao.loc[asd_mask, 'subject_id'].values

# === Top 10 ROI ===

top_roi_indices = roi_df.sort_values('count', ascending=False).head(10)['Unnamed: 0'].astype(int).values

# ===  ===

results = []

for col in behavior_df.columns:
    if col == 'Identifiers':
        continue
    try:
        values_all = pd.to_numeric(behavior_df[col], errors='coerce')
        if values_all.notna().sum() < 5:
            continue

        valid_behavior_mask = values_all.notna()
        valid_behavior_ids = behavior_df.loc[valid_behavior_mask, 'Identifiers'].values
        common_ids = np.intersect1d(asd_ids, valid_behavior_ids)
        if len(common_ids) < 10:
            continue

        fmri_idx = [np.where(asd_ids == sid)[0][0] for sid in common_ids]
        behav_idx = [behavior_df[behavior_df['Identifiers'] == sid].index[0] for sid in common_ids]

        fmri_subset = fmri_asd[fmri_idx][:, :, top_roi_indices]
        roi_avg_activation = fmri_subset.mean(axis=(1, 2))
        behavior_values = pd.to_numeric(behavior_df.loc[behav_idx, col], errors='coerce')

        # age, gender, site

        meta_df = datao[asd_mask].copy().reset_index(drop=True).loc[fmri_idx].reset_index(drop=True)
        meta_df['age'] = pd.to_numeric(meta_df['age'], errors='coerce')
        meta_df['gender'] = meta_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
        meta_df['site'] = meta_df['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})

        # 

        combined_df = pd.DataFrame({
            'roi': roi_avg_activation,
            'behavior': behavior_values.values,
            'age': meta_df['age'],
            'gender': meta_df['gender'],
            'site': meta_df['site']
        }).dropna()

        if len(combined_df) < 10:
            continue

        roi_valid = combined_df['roi'].values
        behavior_valid = combined_df['behavior'].values
        X_covariates = combined_df[['age', 'gender', 'site']].values

        result = compute_pearson_and_partial_corr(roi_valid, behavior_valid, X_covariates)
        results.append({
            'scale': col,
            'pearson_r': result['pearson_r'],
            'pearson_p': result['pearson_p'],
            'partial_r': result['partial_r'],
            'partial_p': result['partial_p'],
            'n': len(combined_df),
            'sig_partial': '*' if result['partial_p'] < 0.05 else ''
        })

    except Exception as e:
        print(f" Error on {col}: {e}")
        continue

# ===  ===

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values(by='partial_p').reset_index(drop=True)
    print(" Top 10 partial correlations:")
    print(results_df.head(10))
    results_df.to_csv("meanpath_correlation_results_top10roi.csv", index=False)
else:
    print(" No valid correlations found. Please check your data again.")





import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

# 

plot_dir = "meanpath_correlation_plots_partial_sig"
os.makedirs(plot_dir, exist_ok=True)

#  partial_p < 0.05 

sig_df = results_df[results_df['partial_p'] < 0.05]

for _, row in sig_df.iterrows():
    col = row['scale']

    # 

    values_all = pd.to_numeric(behavior_df[col], errors='coerce')
    valid_behavior_mask = values_all.notna()
    valid_behavior_ids = behavior_df.loc[valid_behavior_mask, 'Identifiers'].values
    common_ids = np.intersect1d(asd_ids, valid_behavior_ids)
    if len(common_ids) < 10:
        continue

    fmri_idx = [np.where(asd_ids == sid)[0][0] for sid in common_ids]
    behav_idx = [behavior_df[behavior_df['Identifiers'] == sid].index[0] for sid in common_ids]
    fmri_subset = fmri_asd[fmri_idx][:, :, top_roi_indices]
    roi_avg_activation = fmri_subset.mean(axis=(1, 2))
    behavior_values = pd.to_numeric(behavior_df.loc[behav_idx, col], errors='coerce')

    # 

    meta_df = datao[asd_mask].copy().reset_index(drop=True).loc[fmri_idx].reset_index(drop=True)
    meta_df['age'] = pd.to_numeric(meta_df['age'], errors='coerce')
    meta_df['gender'] = meta_df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    meta_df['site'] = meta_df['site'].astype(str).str.strip().str.lower().map({'ru': 0, 'cuny': 1, 'cbic': 2})

    combined_df = pd.DataFrame({
        'roi': roi_avg_activation,
        'behavior': behavior_values.values,
        'age': meta_df['age'],
        'gender': meta_df['gender'],
        'site': meta_df['site']
    }).dropna()

    if len(combined_df) < 10:
        continue

    # 

    x = combined_df['roi'].values
    y = combined_df['behavior'].values

    # 

    r = row['partial_r']
    p = row['partial_p']

    # ===  +  ===

    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    x_sorted = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_sorted + intercept

    # 

    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = len(x) - 2
    t_val = stats.t.ppf(0.975, dof)
    s_err = np.sqrt(np.sum(residuals**2) / dof)
    conf = t_val * s_err * np.sqrt(1/len(x) + (x_sorted - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    upper = y_fit + conf
    lower = y_fit - conf

    # ===  ===

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    #  + 

    ax.scatter(x, y, color='#1f77b4', s=40, alpha=0.75, edgecolor='k', linewidth=0.5)

    ax.plot(x_sorted, y_fit, color='#0e3e66', linewidth=2)

    ax.fill_between(x_sorted, lower, upper, color='#0e3e66', alpha=0.15, label='95% CI')


    # 

    ax.text(0.05, 0.95,
            f"Partial r = {r:.2f}\np = {p:.3g}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    # 

    ax.set_xlabel("Mean Activation (Top 10 ROI)", fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(col, fontsize=13, weight='bold')

    # 

    ax.tick_params(axis='both', which='both', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    #  tick label

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[:-1])

    plt.tight_layout()

    # 

    safe_name = col.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(plot_dir, f"{safe_name}.png"))
    plt.close()

print(f" Finished: saved {len(sig_df)} beautiful plots with CI to '{plot_dir}'")
