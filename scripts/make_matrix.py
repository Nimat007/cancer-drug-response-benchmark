import pandas as pd

print("Loading PRISM data...")
df = pd.read_csv("data/raw/secondary-screen-dose-response-curve-parameters.csv")

print("Pivoting to a matrix using AUC values...")
# depmap_id holds the ACH cell line codes; broad_id holds the compound codes
matrix = df.pivot_table(index='depmap_id', columns='broad_id', values='auc', aggfunc='mean')
matrix.to_csv("data/raw/prism_auc_matrix.csv")

print("Done! Matrix saved.")