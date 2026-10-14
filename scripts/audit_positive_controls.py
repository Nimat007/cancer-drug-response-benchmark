import pandas as pd
import os

print("Loading datasets...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')
raw_prism = pd.read_csv(r"data\raw\secondary-screen-dose-response-curve-parameters.csv", low_memory=False, usecols=['broad_id', 'name'])

compound_info = raw_prism.drop_duplicates(subset=['broad_id']).set_index('broad_id')

print("\n--- Applying Strict Intersection (Main Benchmark Rules) ---")
overlap = expr.index.intersection(prism.index).intersection(meta.index)
missing_lineage_mask = meta.loc[overlap, 'OncotreeLineage'].isna()
overlap = overlap[~missing_lineage_mask]

expr_clean = expr.loc[overlap]
prism_clean = prism.loc[overlap]
meta_clean = meta.loc[overlap]

print(f"Usable cell lines after strict intersection: {len(prism_clean)}")

print("\n--- Auditing Biomarker Availability in Expression Data ---")
# DepMap column names are formatted like "GENE (ID)". We use substring matching to find them.
expected_biomarkers = ['MT1E', 'PDE3A']
found_genes = {}
for biomarker in expected_biomarkers:
    matched_cols = [col for col in expr_clean.columns if biomarker in col.split(' ')[0]]
    found_genes[biomarker] = matched_cols[0] if matched_cols else "NOT FOUND"
    print(f"Biomarker '{biomarker}': {found_genes[biomarker]}")

print("\n--- Auditing Candidate Compounds ---")
candidates = [
    {
        "expected_name": "disulfiram",
        "biomarker": "MT1E",
        "association": "Sensitivity associated with low MT1E expression",
        "source": "Corsello et al., Nature Cancer 2020"
    },
    {
        "expected_name": "anagrelide",
        "biomarker": "PDE3A",
        "association": "Sensitivity associated with PDE3A expression",
        "source": "Corsello et al., Nature Cancer 2020"
    },
    {
        "expected_name": "zardaverine",
        "biomarker": "PDE3A",
        "association": "Sensitivity associated with PDE3A expression",
        "source": "Corsello et al., Nature Cancer 2020"
    }
]

audit_results = []
for cand in candidates:
    print(f"\nEvaluating: {cand['expected_name'].upper()}")
    matches = compound_info[compound_info['name'].str.lower() == cand['expected_name'].lower()]
    
    if matches.empty:
        print(f"  -> STATUS: UNAVAILABLE in 19Q4 matrix.")
        continue
        
    for broad_id, row in matches.iterrows():
        if broad_id not in prism_clean.columns:
            print(f"  -> STATUS: Excluded during replicate resolution (ID: {broad_id}).")
            continue
            
        target_data = prism_clean[broad_id]
        valid_responses = target_data.dropna()
        valid_count = len(valid_responses)
        missing_count = target_data.isna().sum()
        variance = valid_responses.var()
        
        lineages = meta_clean.loc[valid_responses.index, 'OncotreeLineage']
        unique_lineages = lineages.nunique()
        top_lineages = ", ".join([f"{k} ({v})" for k, v in lineages.value_counts().head(3).items()])
        
        biomarker_status = found_genes[cand['biomarker']]
        
        print(f"  -> ID: {broad_id}")
        print(f"  -> Valid Responses: {valid_count} (Missing: {missing_count})")
        print(f"  -> AUC Variance: {variance:.4f}")
        print(f"  -> Distinct Lineages: {unique_lineages} (Top 3: {top_lineages})")
        print(f"  -> Biomarker Presence: {biomarker_status}")
        
        audit_results.append({
            "Compound": cand['expected_name'],
            "Broad_ID": broad_id,
            "Valid_Responses": valid_count,
            "Missing_Responses": missing_count,
            "AUC_Variance": variance,
            "Distinct_Lineages": unique_lineages,
            "Target_Biomarker": cand['biomarker'],
            "Biomarker_Found": biomarker_status,
            "Association": cand['association'],
            "Source": cand['source']
        })

if audit_results:
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(audit_results)
    df_results.to_csv(r"results\alternative_positive_controls.csv", index=False)
    print(f"\nAudit complete. Detailed table saved to: results/alternative_positive_controls.csv")
else:
    print("\nNo candidates met the evaluation criteria.")