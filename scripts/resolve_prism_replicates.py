import pandas as pd
import os

input_file = r"data\raw\secondary-screen-dose-response-curve-parameters.csv"
print("Loading PRISM data...")
df = pd.read_csv(input_file, low_memory=False)
valid_df = df.dropna(subset=['depmap_id', 'broad_id'])

# 1. Identify all duplicates
counts = valid_df.groupby(['depmap_id', 'broad_id']).size()
dupe_pairs = counts[counts > 1].index

# 2. Split into singletons (safe) and dupes (needs resolution)
is_dupe = valid_df.set_index(['depmap_id', 'broad_id']).index.isin(dupe_pairs)
singletons = valid_df[~is_dupe]
dupes = valid_df[is_dupe]

# 3. Resolve dupes using the PI's MTS010 priority rule
mts_only = dupes[dupes['screen_id'] == 'MTS010']
mts_counts = mts_only.groupby(['depmap_id', 'broad_id']).size()

# Successfully resolved: Exactly ONE MTS010 record exists for the pair
resolved_keys = mts_counts[mts_counts == 1].index
resolved_records = mts_only[mts_only.set_index(['depmap_id', 'broad_id']).index.isin(resolved_keys)]

# Unresolved: 0 MTS010 records (meaning both are older screens), or >1 MTS010 records
unresolved_keys = dupe_pairs.difference(resolved_keys)
unresolved_records = dupes[dupes.set_index(['depmap_id', 'broad_id']).index.isin(unresolved_keys)]

print("\n--- Resolution Audit ---")
print(f"Total duplicated pairs to resolve: {len(dupe_pairs):,}")
print(f"Pairs successfully resolved (MTS010 priority): {len(resolved_keys):,}")
print(f"Pairs remaining unresolved: {len(unresolved_keys):,}")

if len(unresolved_keys) > 0:
    os.makedirs("results", exist_ok=True)
    unresolved_records.to_csv(r"results\unresolved_prism_duplicates.csv", index=False)
    print(f"\nSaved {len(unresolved_records):,} unresolved records to results/unresolved_prism_duplicates.csv")
    print("WARNING: These ambiguous pairs are EXCLUDED from the final matrix to ensure absolute data integrity.")

# 4. Build the final clean dataset
final_clean = pd.concat([singletons, resolved_records], ignore_index=True)
print(f"\nFinal clean dataset contains {len(final_clean):,} unique records.")

# 5. Pivot to matrix (Using .pivot() acts as a safety test: it crashes if duplicates exist)
print("Pivoting to 2D matrix (validating uniqueness)...")
matrix = final_clean.pivot(index='depmap_id', columns='broad_id', values='auc')
matrix.to_csv(r"data\raw\prism_auc_matrix_resolved.csv")
print("Success! Clean matrix saved to data/raw/prism_auc_matrix_resolved.csv")