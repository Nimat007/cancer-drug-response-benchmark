import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("Loading datasets (this will take a moment)...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix.csv", index_col=0)

# 1. Keep only the 477 overlapping cell lines
overlap = expr.index.intersection(prism.index)
expr = expr.loc[overlap]
prism = prism.loc[overlap]

# 2. Pick ONE compound to start testing our pipeline (the first column)
target_compound = prism.columns[0]
y = prism[target_compound].dropna()
X = expr.loc[y.index] # Match expression rows to valid response rows

# 3. Split into 80% Training and 20% Testing (Unseen cell lines)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the "Dummy" Mean-Response Baseline
# (Predicting that every unseen cell line just has the average response)
mean_prediction = y_train.mean()
y_pred_baseline = [mean_prediction] * len(y_test)

mae = mean_absolute_error(y_test, y_pred_baseline)

print("\n--- Baseline Benchmark ---")
print(f"Compound: {target_compound}")
print(f"Training cell lines: {len(X_train)} | Testing cell lines: {len(X_test)}")
print(f"Baseline Mean Absolute Error (MAE): {mae:.4f}")