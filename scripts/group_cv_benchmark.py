import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore') # Suppress correlation warnings for flat predictions

print("Loading datasets (this will take a moment)...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')

# 1. Input Assumption Verification
print("\n--- Verifying Input Assumptions ---")
assert expr.index.is_unique, "Expression matrix has duplicate indices!"
assert prism.index.is_unique, "PRISM matrix has duplicate indices!"
assert meta.index.is_unique, "Metadata matrix has duplicate indices!"

overlap = expr.index.intersection(prism.index).intersection(meta.index)
expr = expr.loc[overlap]
prism = prism.loc[overlap]
meta = meta.loc[overlap]

missing_lineage_count = meta['OncotreeLineage'].isna().sum()
if missing_lineage_count > 0:
    print(f"Dropping {missing_lineage_count} cell lines with missing 'OncotreeLineage' metadata to prevent false groupings.")
    meta = meta.dropna(subset=['OncotreeLineage'])
    overlap = overlap.intersection(meta.index)
    expr = expr.loc[overlap]
    prism = prism.loc[overlap]
    meta = meta.loc[overlap]

# 2. Compound Selection Rule
print("\nApplying Compound Selection Rule...")
compound_counts = prism.count()
max_count = compound_counts.max()
candidate_compounds = compound_counts[compound_counts == max_count].index
variances = prism[candidate_compounds].var()
target_compound = variances.idxmax()

y = prism[target_compound].dropna()
X = expr.loc[y.index]
groups = meta.loc[y.index, 'OncotreeLineage']

print(f"Target Compound Selected: {target_compound}")
print(f"Total usable cell lines: {len(y)}")
print(f"Number of distinct tissue lineages: {groups.nunique()}")

# 3. Cross-Validation & Permutation Setup
gkf = GroupKFold(n_splits=5)
results = []

# Storage for Pooled Out-Of-Fold (OOF) Metrics
oof_y_true = []
oof_y_pred_base = []
oof_y_pred_ridge = []
oof_y_pred_rf = []
oof_y_pred_rf_perm = []

print("\nStarting GroupKFold Cross-Validation (Strict nested feature selection)...")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Nested feature selection
    top_genes = X_train.var().nlargest(1000).index
    X_train_sel = X_train[top_genes]
    X_test_sel = X_test[top_genes]
    
    # Baseline
    y_pred_base = [y_train.mean()] * len(y_test)
    
    # Ridge (Standardized + Tuned via CV)
    ridge_pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 10)))
    ridge_pipe.fit(X_train_sel, y_train)
    y_pred_ridge = ridge_pipe.predict(X_test_sel)
    
    # Random Forest (100 estimators to match initial benchmark)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sel, y_train)
    y_pred_rf = rf.predict(X_test_sel)
    
    # Permutation Test (Shuffle labels to break feature-response relationship)
    np.random.seed(42 + fold)
    y_train_permuted = np.random.permutation(y_train)
    rf_perm = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_perm.fit(X_train_sel, y_train_permuted)
    y_pred_rf_perm = rf_perm.predict(X_test_sel)

    # Store OOF predictions
    oof_y_true.extend(y_test)
    oof_y_pred_base.extend(y_pred_base)
    oof_y_pred_ridge.extend(y_pred_ridge)
    oof_y_pred_rf.extend(y_pred_rf)
    oof_y_pred_rf_perm.extend(y_pred_rf_perm)

    # Fold Metrics
    def calc_corrs(t, p):
        return (pearsonr(t, p)[0], spearmanr(t, p)[0]) if np.std(p) > 0 else (0.0, 0.0)
    
    ridge_p, _ = calc_corrs(y_test, y_pred_ridge)
    rf_p, rf_s = calc_corrs(y_test, y_pred_rf)

    results.append({
        'Fold': fold,
        'Test_Size': len(y_test),
        'Base_MAE': mean_absolute_error(y_test, y_pred_base),
        'Ridge_MAE': mean_absolute_error(y_test, y_pred_ridge),
        'RF_MAE': mean_absolute_error(y_test, y_pred_rf),
        'Ridge_Pearson': ridge_p,
        'RF_Pearson': rf_p,
        'RF_Spearman': rf_s
    })
    print(f"Fold {fold} complete (Test size: {len(y_test)}).")

# 4. Summarize Results
res_df = pd.DataFrame(results)
print("\n--- Fold-Level Results ---")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Pooled OOF Metrics calculation
def oof_metrics(y_t, y_p):
    mae = mean_absolute_error(y_t, y_p)
    p, s = (pearsonr(y_t, y_p)[0], spearmanr(y_t, y_p)[0]) if np.std(y_p) > 0 else (0.0, 0.0)
    return mae, p, s

b_mae, _, _ = oof_metrics(oof_y_true, oof_y_pred_base)
ri_mae, ri_p, ri_s = oof_metrics(oof_y_true, oof_y_pred_ridge)
rf_mae, rf_p, rf_s = oof_metrics(oof_y_true, oof_y_pred_rf)
pm_mae, pm_p, pm_s = oof_metrics(oof_y_true, oof_y_pred_rf_perm)

print("\n--- Pooled Out-Of-Fold (OOF) Overall Performance ---")
print(f"Total Evaluated Cell Lines: {len(oof_y_true)}")
print(f"Baseline MAE:         {b_mae:.4f}")
print(f"Ridge MAE:            {ri_mae:.4f} | Pearson: {ri_p:.4f} | Spearman: {ri_s:.4f}")
print(f"Random Forest MAE:    {rf_mae:.4f} | Pearson: {rf_p:.4f} | Spearman: {rf_s:.4f}")
print(f"Permuted RF MAE:      {pm_mae:.4f} | Pearson: {pm_p:.4f} | Spearman: {pm_s:.4f}")

print("\nPermutation Test Interpretation:")
if rf_mae < pm_mae:
    print("-> The real RF model outperformed the permuted (randomized) model. It detected some genuine biological signal, even if weak.")
else:
    print("-> The real RF model performed no better than the permuted model. It failed to learn any genuine biological signal.")