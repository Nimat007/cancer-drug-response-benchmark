import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os

# Helper for Fisher Z-transformation 95% Confidence Intervals
def get_ci(r, n):
    if n < 4 or abs(r) == 1.0: return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_high = np.tanh(z + 1.96 * se)
    return (ci_low, ci_high)

# Helper for Benjamini-Hochberg FDR correction
def bh_fdr(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    fdr_pvals = np.zeros(n)
    running_min = 1.0
    for i in range(n-1, -1, -1):
        fdr = min(1.0, sorted_pvals[i] * n / (i + 1))
        running_min = min(running_min, fdr)
        fdr_pvals[sorted_indices[i]] = running_min
    return fdr_pvals

print("Loading datasets for Biological Sanity Check...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')

# Define targets
pde3a_col = [col for col in expr.columns if 'PDE3A' in col.split(' ')[0]][0]
drugs = {
    'Anagrelide': 'BRD-K62200014-003-10-5',
    'Zardaverine': 'BRD-K37561857-001-15-5'
}

print("\n--- Applying Intersection Filters ---")
overlap = expr.index.intersection(prism.index).intersection(meta.index)
meta = meta.loc[overlap].dropna(subset=['OncotreeLineage'])
overlap = meta.index

expr_clean = expr.loc[overlap, [pde3a_col]]
prism_clean = prism.loc[overlap, list(drugs.values())]
meta_clean = meta

pooled_results = []
lineage_results = []

for drug_name, broad_id in drugs.items():
    print(f"\n--- Analyzing {drug_name} vs PDE3A ---")
    
    # Isolate valid data
    df = pd.concat([expr_clean, prism_clean[[broad_id]], meta_clean['OncotreeLineage']], axis=1).dropna()
    n_total = len(df)
    
    pde3a = df[pde3a_col]
    auc = df[broad_id]
    
    # 1. Pooled Analysis
    s_r, s_p = spearmanr(pde3a, auc)
    s_ci = get_ci(s_r, n_total)
    p_r, p_p = pearsonr(pde3a, auc)
    p_ci = get_ci(p_r, n_total)
    
    # Expected: High PDE3A = Sensitivity (Low AUC), so correlation should be negative.
    expected_met = s_r < 0 and s_p < 0.05
    
    print(f"Pooled (N={n_total}): Spearman r={s_r:.4f} (p={s_p:.2e}) | Expected biology confirmed: {expected_met}")
    
    pooled_results.append({
        'Compound': drug_name,
        'Biomarker': 'PDE3A',
        'N': n_total,
        'Spearman_r': s_r,
        'Spearman_CI_95': f"[{s_ci[0]:.4f}, {s_ci[1]:.4f}]",
        'Spearman_p': s_p,
        'Pearson_r': p_r,
        'Pearson_CI_95': f"[{p_ci[0]:.4f}, {p_ci[1]:.4f}]",
        'Pearson_p': p_p,
        'Literature_Validation': expected_met
    })
    
    # 2. Lineage-Confounding Analysis
    lineage_groups = df.groupby('OncotreeLineage')
    drug_lin_res = []
    
    for lin, group in lineage_groups:
        if len(group) >= 15: # Minimum threshold for meaningful correlation
            lr, lp = spearmanr(group[pde3a_col], group[broad_id])
            drug_lin_res.append({
                'Compound': drug_name,
                'Lineage': lin,
                'N': len(group),
                'Spearman_r': lr,
                'Spearman_p': lp
            })
            
    # Apply FDR correction
    if drug_lin_res:
        pvals = [x['Spearman_p'] for x in drug_lin_res]
        fdr_pvals = bh_fdr(pvals)
        for i, res in enumerate(drug_lin_res):
            res['FDR_adj_p'] = fdr_pvals[i]
        lineage_results.extend(drug_lin_res)

# Save Outputs
os.makedirs("results", exist_ok=True)
pd.DataFrame(pooled_results).to_csv(r"results\sanity_check_pooled.csv", index=False)
pd.DataFrame(lineage_results).to_csv(r"results\sanity_check_lineage.csv", index=False)

print("\nBiological sanity check complete! Check the 'results' folder for the statistical tables.")