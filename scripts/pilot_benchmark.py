import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr
import os
import warnings
import time

warnings.filterwarnings('ignore')

print("Loading datasets for Pilot Benchmark...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')
panel = pd.read_csv(r"results\compound_panel_eligibility.csv", index_col='broad_id')

# 1. Master Alignment & Fixed Splits
print("\n--- Establishing Master Splits ---")
overlap = expr.index.intersection(prism.index).intersection(meta.index)
meta = meta.loc[overlap].dropna(subset=['OncotreeLineage'])
master_indices = meta.index
master_groups = meta['OncotreeLineage']

gkf = GroupKFold(n_splits=5)
master_folds = list(gkf.split(master_indices, master_indices, groups=master_groups))

print(f"Master folds generated for {len(master_indices)} cell lines across {master_groups.nunique()} lineages.")
for i, (tr, te) in enumerate(master_folds):
    tr_lin = set(master_groups.iloc[tr])
    te_lin = set(master_groups.iloc[te])
    assert len(tr_lin.intersection(te_lin)) == 0, f"Leakage detected in fold {i}!"

# 2. Pilot Selection (10 highest variance, 10 lowest variance from eligible)
eligible = panel[panel['Eligible'] == True].sort_values('Variance_AUC', ascending=False)
pilot_targets = eligible.head(10).index.tolist() + eligible.tail(10).index.tolist()
print(f"\nSelected {len(pilot_targets)} compounds for the Pilot Benchmark.")

results = []
start_time = time.time()

# 3. Main Benchmark Loop
for idx, broad_id in enumerate(pilot_targets, 1):
    print(f"Evaluating [{idx}/{len(pilot_targets)}]: {broad_id}...")
    
    y = prism.loc[master_indices, broad_id].dropna()
    valid_cells = y.index
    X = expr.loc[valid_cells]
    y = y.values
    
    oof_y_true, oof_y_pred_base, oof_y_pred_ridge, oof_y_pred_rf = [], [], [], []
    fold_maes = []
    
    for fold, (master_tr_idx, master_te_idx) in enumerate(master_folds, 1):
        # Mask master indices to only include valid cells for this compound
        tr_cells = master_indices[master_tr_idx].intersection(valid_cells)
        te_cells = master_indices[master_te_idx].intersection(valid_cells)
        
        if len(tr_cells) == 0 or len(te_cells) == 0:
            continue # Skip fold if insufficient data
            
        X_train, X_test = expr.loc[tr_cells], expr.loc[te_cells]
        y_train, y_test = prism.loc[tr_cells, broad_id], prism.loc[te_cells, broad_id]
        
        # Outer-Fold Feature Selection
        top_genes = X_train.var().nlargest(1000).index
        X_train_sel, X_test_sel = X_train[top_genes], X_test[top_genes]
        
        # Baselines & Models
        pred_base = np.full(len(y_test), y_train.mean())
        
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 10)))
        ridge.fit(X_train_sel, y_train)
        pred_ridge = ridge.predict(X_test_sel)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_sel, y_train)
        pred_rf = rf.predict(X_test_sel)
        
        # OOF Storage
        oof_y_true.extend(y_test)
        oof_y_pred_base.extend(pred_base)
        oof_y_pred_ridge.extend(pred_ridge)
        oof_y_pred_rf.extend(pred_rf)
        
        fold_maes.append({f"fold_{fold}_test_size": len(y_test), f"fold_{fold}_rf_mae": mean_absolute_error(y_test, pred_rf)})

    # Calculate Pooled OOF Metrics
    if len(oof_y_true) > 0:
        base_mae = mean_absolute_error(oof_y_true, oof_y_pred_base)
        ri_mae = mean_absolute_error(oof_y_true, oof_y_pred_ridge)
        rf_mae = mean_absolute_error(oof_y_true, oof_y_pred_rf)
        
        ri_delta = ri_mae - base_mae # Negative means improvement over baseline
        rf_delta = rf_mae - base_mae
        
        def calc_corrs(t, p): return (pearsonr(t, p)[0], spearmanr(t, p)[0]) if np.std(p) > 0 else (np.nan, np.nan)
        ri_p, ri_s = calc_corrs(oof_y_true, oof_y_pred_ridge)
        rf_p, rf_s = calc_corrs(oof_y_true, oof_y_pred_rf)
        
        comp_res = {
            'compound_id': broad_id,
            'n_samples': len(oof_y_true),
            'n_lineages': meta.loc[valid_cells, 'OncotreeLineage'].nunique(),
            'baseline_oof_mae': base_mae,
            'ridge_oof_mae': ri_mae,
            'rf_oof_mae': rf_mae,
            'ridge_mae_delta': ri_delta,
            'rf_mae_delta': rf_delta,
            'ridge_oof_pearson': ri_p,
            'rf_oof_pearson': rf_p,
            'status': 'Completed'
        }
        # Flatten fold diagnostics into row
        for d in fold_maes: comp_res.update(d)
        results.append(comp_res)
    else:
        results.append({'compound_id': broad_id, 'status': 'Failed: No valid OOF predictions'})

# 4. Save and Summarize
os.makedirs("results", exist_ok=True)
res_df = pd.DataFrame(results)
res_df.to_csv(r"results\pilot_benchmark_results.csv", index=False)

end_time = time.time()
print(f"\n--- Pilot Execution Summary ---")
print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
print(f"Compounds successfully evaluated: {len(res_df[res_df['status'] == 'Completed'])}")
print(f"Ridge improved over baseline (delta < 0): {sum(res_df['ridge_mae_delta'] < 0)}")
print(f"RF improved over baseline (delta < 0): {sum(res_df['rf_mae_delta'] < 0)}")
print("Detailed results saved to: results/pilot_benchmark_results.csv")