import pandas as pd

print("Loading quarantined records...")
try:
    df = pd.read_csv(r"results\unresolved_prism_duplicates.csv")
    
    print("\n--- Auditing Quarantined Pairs ---")
    # Group by the cell line and drug, then see what screen_ids they have
    screen_combos = df.groupby(['depmap_id', 'broad_id'])['screen_id'].apply(lambda x: tuple(sorted(x)))
    combo_counts = screen_combos.value_counts()
    
    print("Screen ID combinations found in the quarantined data:")
    for combo, count in combo_counts.items():
        print(f"{combo}: {count:,} pairs")
        
    print(f"\nTotal unique cell lines affected: {df['depmap_id'].nunique():,}")
    print(f"Total unique compounds affected: {df['broad_id'].nunique():,}")

except FileNotFoundError:
    print("Could not find the unresolved duplicates file.")