import pandas as pd
import os
import sys

input_file = r"data\raw\secondary-screen-dose-response-curve-parameters.csv"
output_file = r"results\prism_duplicate_pairs.csv"

# Ensure the results folder exists
os.makedirs("results", exist_ok=True)

print(f"Reading PRISM data in chunks from {input_file}...")
chunks = []
total_rows = 0
missing_depmap = 0
missing_broad = 0
missing_auc = 0

try:
    # Use chunking and usecols to strictly limit memory usage
    for chunk in pd.read_csv(input_file, usecols=['depmap_id', 'broad_id', 'auc'], chunksize=50000, low_memory=False):
        total_rows += len(chunk)
        missing_depmap += chunk['depmap_id'].isna().sum()
        missing_broad += chunk['broad_id'].isna().sum()
        missing_auc += chunk['auc'].isna().sum()
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Please check the path.")
    sys.exit(1)
except ValueError as e:
    print(f"Error: A required column is missing from the file. {e}")
    sys.exit(1)

print("Analyzing duplicates...")
# Isolate pairs for duplicate detection, ignoring rows missing core identifiers
valid_pairs = df.dropna(subset=['depmap_id', 'broad_id'])

# Count occurrences of each (cell_line, compound) pair
pair_counts = valid_pairs.groupby(['depmap_id', 'broad_id']).size().reset_index(name='count')
unique_pairs = len(pair_counts)

# Filter down to only the pairs that occur more than once
dupes = pair_counts[pair_counts['count'] > 1]
num_duplicated_pairs = len(dupes)
excess_rows = dupes['count'].sum() - num_duplicated_pairs

print("\n--- PRISM Data Audit Summary ---")
print(f"Total rows: {total_rows:,}")
print(f"Missing depmap_id: {missing_depmap:,}")
print(f"Missing broad_id: {missing_broad:,}")
print(f"Missing auc: {missing_auc:,}")
print(f"Unique (depmap_id, broad_id) pairs: {unique_pairs:,}")
print(f"Pairs occurring >1 time: {num_duplicated_pairs:,}")
print(f"Excess rows from duplicates: {excess_rows:,}")

if num_duplicated_pairs > 0:
    print("\nAnalyzing AUC variations in duplicates...")
    
    # Merge the duplicate keys back with the original data to examine the AUC values
    dup_details = valid_pairs.merge(dupes[['depmap_id', 'broad_id']], on=['depmap_id', 'broad_id'])
    
    # Calculate variation statistics per duplicated pair
    dup_stats = dup_details.groupby(['depmap_id', 'broad_id']).agg(
        occurrences=('auc', 'size'),
        missing_auc_count=('auc', lambda x: x.isna().sum()),
        distinct_auc_values=('auc', 'nunique'),
        auc_min=('auc', 'min'),
        auc_max=('auc', 'max')
    ).reset_index()

    dup_stats['auc_diff'] = dup_stats['auc_max'] - dup_stats['auc_min']
    
    diff_auc_count = len(dup_stats[dup_stats['distinct_auc_values'] > 1])
    same_auc_count = len(dup_stats[dup_stats['distinct_auc_values'] <= 1])

    print(f"Duplicated pairs with EXACT SAME AUC: {same_auc_count:,}")
    print(f"Duplicated pairs with DIFFERING AUC:  {diff_auc_count:,}")

    if diff_auc_count > 0:
        print("\nExamples of duplicated pairs with differing AUC values:")
        print(dup_stats[dup_stats['distinct_auc_values'] > 1].head().to_string(index=False))

    dup_stats.sort_values(by='auc_diff', ascending=False).to_csv(output_file, index=False)
    print(f"\nDetailed duplicate statistics saved to: {output_file}")
else:
    print("\nNo duplicated pairs found. Using `pivot_table` with `mean` aggregation did not alter your data.")
    with open(output_file, 'w') as f:
        f.write("depmap_id,broad_id,status\nNo duplicates found")