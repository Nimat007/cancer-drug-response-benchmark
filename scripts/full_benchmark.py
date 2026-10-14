import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# EXECUTION TOGGLE
# ---------------------------------------------------------
PILOT_MODE = False  # Keep True for the 3-compound retest
# ---------------------------------------------------------

class TopVarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, k=1000): self.k = k
    def fit(self, X, y=None):
        self.top_cols_ = X.var().nlargest(self.k).index
        return self
    def transform(self, X): return X[self.top_cols_]

print("Loading datasets for Benchmark Execution...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')
panel = pd.read_csv(r"results\compound_panel_eligibility.csv", index_col='broad_id')

print("\n--- Running Preflight Integrity Checks ---")
assert expr.index.is_unique, "FATAL: Expression matrix has duplicate indices."
assert prism.index.is_unique, "FATAL: PRISM matrix has duplicate indices."
assert meta.index.is_unique, "FATAL: Metadata matrix has duplicate indices."

eligible_targets = panel[panel['Eligible'] == True].index.tolist()
assert len(eligible_targets) == 1198, f"FATAL: Expected exactly 1,198 frozen compounds, found {len(eligible_targets)}."

output_name = r"results\pilot_nested_check.csv" if PILOT_MODE else r"results\full_benchmark_results.csv"

if PILOT_MODE:
    eligible_targets = eligible_targets[:3]
    print(f"PILOT MODE ENABLED: Running {len(eligible_targets)} compounds.")
else:
    print(f"FULL BENCHMARK ENABLED: Proceeding with all {len(eligible_targets)} compounds.")

# 1. Safe Resume Logic
completed_compounds = set()
os.makedirs("results", exist_ok=True)
if os.path.exists(output_name):
    try:
        existing_df = pd.read_csv(output_name)
        if 'compound_id' in existing_df.columns:
            completed_compounds = set(existing_df['compound_id'].unique())
            print(f"Resume Mode: Found {len(completed_compounds)} already evaluated compounds. Skipping these.")
    except Exception as e:
        print(f"Could not read existing results: {e}. Starting fresh.")

# 2. Master Alignment & Fixed Splits
overlap = expr.index.intersection(prism.index).intersection(meta.index)
meta = meta.loc[overlap].dropna(subset=['OncotreeLineage'])
master_indices = meta.index
master_groups = meta['OncotreeLineage']

gkf = GroupKFold(n_splits=5)
master_folds = list(gkf.split(master_indices, master_indices, groups=master_groups))

for i, (tr, te) in enumerate(master_folds):
    tr_lin = set(master_groups.iloc[tr])
    te_lin = set(master_groups.iloc[te])
    assert len(tr_lin.intersection(te_lin)) == 0, f"FATAL: Leakage detected in master fold {i}!"

start_time = time.time()
successful_runs = 0

# 3. Main Benchmark Loop with Exception Handling
for idx, broad_id in enumerate(eligible_targets, 1):
    if broad_id in completed_compounds:
        continue
        
    elapsed = (time.time() - start_time) / 60
    print(f"Evaluating [{idx}/{len(eligible_targets)}]: {broad_id}... (Elapsed: {elapsed:.2f} min)")
    
    try:
        y = prism.loc[master_indices, broad_id].dropna()
        valid_cells = y.index
        
        oof_y_true, oof_y_pred_base, oof_y_pred_ridge, oof_y_pred_rf = [], [], [], []
        fold_metrics = []
        evaluated_cells = Counter()  # Now tracks exact counts per cell line
        
        for fold, (master_tr_idx, master_te_idx) in enumerate(master_folds, 1):
            tr_cells = master_indices[master_tr_idx].intersection(valid_cells)
            te_cells = master_indices[master_te_idx].intersection(valid_cells)
            
            if len(tr_cells) == 0 or len(te_cells) == 0: continue
            evaluated_cells.update(te_cells)
            
            X_train, X_test = expr.loc[tr_cells], expr.loc[te_cells]
            y_train, y_test = prism.loc[tr_cells, broad_id], prism.loc[te_cells, broad_id]
            tr_groups = master_groups.loc[tr_cells]
            
            pred_base = np.full(len(y_test), y_train.mean())
            
            ridge_pipe = Pipeline([
                ('feat_sel', TopVarFeatures(k=1000)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge())
            ])
            # Strict nested lineage-grouped CV for Ridge hyperparameter tuning
            inner_cv = GroupKFold(n_splits=3)
            ridge_cv = GridSearchCV(ridge_pipe, param_grid={'ridge__alpha': np.logspace(-3, 3, 10)}, cv=inner_cv, n_jobs=-1)
            ridge_cv.fit(X_train, y_train, groups=tr_groups)
            pred_ridge = ridge_cv.predict(X_test)
            
            rf_pipe = Pipeline([
                ('feat_sel', TopVarFeatures(k=1000)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            rf_pipe.fit(X_train, y_train)
            pred_rf = rf_pipe.predict(X_test)
            
            oof_y_true.extend(y_test)
            oof_y_pred_base.extend(pred_base)
            oof_y_pred_ridge.extend(pred_ridge)
            oof_y_pred_rf.extend(pred_rf)
            
            fold_metrics.append({
                f"fold_{fold}_test_size": len(y_test),
                f"fold_{fold}_base_mae": mean_absolute_error(y_test, pred_base),
                f"fold_{fold}_ridge_mae": mean_absolute_error(y_test, pred_ridge),
                f"fold_{fold}_rf_mae": mean_absolute_error(y_test, pred_rf)
            })

        # Strict OOF Bookkeeping Asserts
        assert all(count == 1 for count in evaluated_cells.values()), f"FATAL: Duplicate OOF predictions detected on {broad_id}."
        assert set(evaluated_cells.keys()) == set(valid_cells), f"FATAL: OOF prediction length mismatch on {broad_id}."

        if len(oof_y_true) > 0:
            base_mae = mean_absolute_error(oof_y_true, oof_y_pred_base)
            ri_mae = mean_absolute_error(oof_y_true, oof_y_pred_ridge)
            rf_mae = mean_absolute_error(oof_y_true, oof_y_pred_rf)
            
            def calc_corrs(t, p): return (pearsonr(t, p)[0], spearmanr(t, p)[0]) if np.std(p) > 0 else (np.nan, np.nan)
            ri_p, ri_s = calc_corrs(oof_y_true, oof_y_pred_ridge)
            rf_p, rf_s = calc_corrs(oof_y_true, oof_y_pred_rf)
            
            comp_res = {
                'compound_id': broad_id,
                'n_samples_evaluated': len(oof_y_true),
                'n_lineages_evaluated': meta.loc[valid_cells, 'OncotreeLineage'].nunique(),
                'baseline_oof_mae': base_mae,
                'ridge_oof_mae': ri_mae,
                'rf_oof_mae': rf_mae,
                'ridge_mae_delta': ri_mae - base_mae,
                'rf_mae_delta': rf_mae - base_mae,
                'ridge_oof_pearson': ri_p,
                'ridge_oof_spearman': ri_s,
                'rf_oof_pearson': rf_p,
                'rf_oof_spearman': rf_s,
                'status': 'Completed'
            }
            for d in fold_metrics: comp_res.update(d)
            successful_runs += 1
        else:
            comp_res = {'compound_id': broad_id, 'status': 'Failed: No valid OOF predictions'}
            
    except Exception as e:
        comp_res = {'compound_id': broad_id, 'status': f'Error: {str(e)}'}
        print(f"  -> ERROR on {broad_id}: {str(e)}")

    # 4. Safe Incremental Saving
    pd.DataFrame([comp_res]).to_csv(output_name, mode='a', header=not os.path.exists(output_name), index=False)

end_time = time.time()
print(f"\n--- Execution Summary ---")
print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
print(f"Compounds successfully evaluated this session: {successful_runs}")
print(f"Results saved to: {output_name}")