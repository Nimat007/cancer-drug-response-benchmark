import pandas as pd
import os

print("Loading datasets...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix_resolved.csv", index_col=0)
meta = pd.read_csv(r"data\raw\Model.csv", index_col='ModelID')

# Load compound annotations to map IDs to readable names
raw_prism = pd.read_csv(r"data\raw\secondary-screen-dose-response-curve-parameters.csv", low_memory=False, usecols=['broad_id', 'name', 'moa', 'target'])
compound_info = raw_prism.drop_duplicates(subset=['broad_id']).set_index('broad_id')

print("\n--- Applying Intersection & Quality Filters ---")
initial_cell_lines = len(prism)
overlap = expr.index.intersection(prism.index).intersection(meta.index)

# Check for missing lineage metadata
missing_lineage_mask = meta.loc[overlap, 'OncotreeLineage'].isna()
missing_lineage_count = missing_lineage_mask.sum()
if missing_lineage_count > 0:
    print(f"Dropping {missing_lineage_count} cell lines due to missing 'OncotreeLineage' metadata.")
    overlap = overlap[~missing_lineage_mask]

expr_clean = expr.loc[overlap]
prism_clean = prism.loc[overlap]
meta_clean = meta.loc[overlap]

print(f"Final usable cell lines after strict intersection: {len(prism_clean)}")

print("\n--- Auditing Compound Panel Eligibility ---")
# Calculate statistics per compound
valid_counts = prism_clean.count()
variances = prism_clean.var()
missing_counts = prism_clean.isna().sum()

panel_df = pd.DataFrame({
    'Valid_Responses': valid_counts,
    'Missing_Responses': missing_counts,
    'Variance_AUC': variances
})

# Merge with readable names and mechanisms
panel_df = panel_df.join(compound_info)

# Apply fixed feasibility threshold (>= 400 valid cell lines)
THRESHOLD = 400
panel_df['Eligible'] = panel_df['Valid_Responses'] >= THRESHOLD
panel_df['Exclusion_Reason'] = panel_df.apply(lambda row: "None" if row['Eligible'] else f"Fails {THRESHOLD} response threshold", axis=1)

eligible_count = panel_df['Eligible'].sum()
print(f"Total compounds evaluated: {len(panel_df)}")
print(f"Compounds meeting the >={THRESHOLD} feasibility threshold: {eligible_count}")

# Save the frozen panel eligibility table
os.makedirs("results", exist_ok=True)
panel_df.to_csv(r"results\compound_panel_eligibility.csv", index_label='broad_id')
print(f"Complete eligibility table saved to: results/compound_panel_eligibility.csv")

print("\n--- Literature Candidate Audit (Positive Controls) ---")
# Pre-registered literature candidates based on Corsello et al., Nature Cancer 2020
candidates = [
    {
        "expected_name": "tepoxalin",
        "biomarker": "ABCB1",
        "association": "Expression associated with response",
        "context": "General ABC-transporter efflux mechanism",
        "source": "Corsello et al., Nature Cancer 2020"
    }
]

audit_results = []
for cand in candidates:
    print(f"\nChecking availability for literature candidate: {cand['expected_name']}")
    
    # Search for the drug by name in the resolved matrix annotations
    matches = panel_df[panel_df['name'].str.lower() == cand['expected_name'].lower()]
    
    if matches.empty:
        print(f"  -> STATUS: UNAVAILABLE in resolved 19Q4 matrix.")
        audit_results.append({**cand, "Status": "Unavailable in 19Q4 matrix", "broad_id": "N/A", "Eligible": False})
    else:
        for broad_id, row_data in matches.iterrows():
            status = "AVAILABLE & ELIGIBLE" if row_data['Eligible'] else f"AVAILABLE BUT EXCLUDED ({row_data['Exclusion_Reason']})"
            print(f"  -> STATUS: {status} (ID: {broad_id} | Valid Responses: {row_data['Valid_Responses']})")
            audit_results.append({
                **cand,
                "broad_id": broad_id,
                "Status": status,
                "Eligible": row_data['Eligible'],
                "Valid_Responses": row_data['Valid_Responses']
            })

cand_df = pd.DataFrame(audit_results)
cand_df.to_csv(r"results\positive_control_candidates.csv", index=False)
print(f"\nLiterature candidate audit saved to: results/positive_control_candidates.csv")
print("\nAudit complete. No models were trained. Panel parameters frozen.")