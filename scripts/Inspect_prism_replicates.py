import pandas as pd

print("Loading PRISM data...")
df = pd.read_csv(r"data\raw\secondary-screen-dose-response-curve-parameters.csv", low_memory=False)

print("\n--- Available Columns ---")
print(df.columns.tolist())

print("\n--- Comparing a Duplicated Pair ---")
# Using the pair that had a large AUC difference (0.43 vs 0.86)
dupes = df[(df['depmap_id'] == 'ACH-000007') & (df['broad_id'] == 'BRD-A74914197-001-02-9')]

# Transpose (.T) so the columns print downward, making it easy to read
print(dupes.T)