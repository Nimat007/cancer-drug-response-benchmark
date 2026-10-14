import pandas as pd
from sklearn.model_selection import GroupKFold

print("Loading data for leakage verification...")
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')

overlap = prism.index.intersection(meta.index)
prism = prism.loc[overlap]
meta = meta.loc[overlap]

target_compound = 'BRD-K13662825-001-07-5'
y = prism[target_compound].dropna()
groups = meta.loc[y.index, 'OncotreeLineage'].fillna('Unknown')

gkf = GroupKFold(n_splits=5)
fold = 1

print("\n--- Cross-Validation Leakage Audit ---")
for train_idx, test_idx in gkf.split(y, y, groups=groups):
    train_lineages = set(groups.iloc[train_idx])
    test_lineages = set(groups.iloc[test_idx])
    
    # The intersection finds items that exist in both sets
    leaked_lineages = train_lineages.intersection(test_lineages)
    
    print(f"\nFold {fold}:")
    print(f"  Training cell lines: {len(train_idx)} | Test cell lines: {len(test_idx)}")
    print(f"  Training lineages: {len(train_lineages)} | Test lineages: {len(test_lineages)}")
    
    if len(leaked_lineages) == 0:
        print("  Leakage Check: PASS (0 overlapping lineages)")
    else:
        print(f"  Leakage Check: FAIL! Overlapping lineages: {leaked_lineages}")
        
    fold += 1