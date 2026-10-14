import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

print("Loading datasets...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix.csv", index_col=0)

# 1. Match overlap
overlap = expr.index.intersection(prism.index)
target_compound = prism.columns[0]
y = prism.loc[overlap, target_compound].dropna()
X = expr.loc[y.index]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Selection: Top 1000 highly variable genes
print("Selecting top 1000 highly variable genes...")
top_genes = X_train.var().nlargest(1000).index
X_train_selected = X_train[top_genes]
X_test_selected = X_test[top_genes]

# 4. Train Models
print("Training Ridge and Random Forest models (this may take a minute)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_selected, y_train)

# n_jobs=-1 tells it to use all your computer's CPU cores to speed it up
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_selected, y_train)

# 5. Evaluate
y_pred_baseline = [y_train.mean()] * len(y_test)
y_pred_ridge = ridge.predict(X_test_selected)
y_pred_rf = rf.predict(X_test_selected)

print("\n--- Final Benchmark Comparison ---")
print(f"Compound: {target_compound}")
print(f"1. Baseline MAE:      {mean_absolute_error(y_test, y_pred_baseline):.4f}")
print(f"2. Ridge MAE:         {mean_absolute_error(y_test, y_pred_ridge):.4f}")
print(f"3. Random Forest MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"RF Pearson Correl.:   {np.corrcoef(y_test, y_pred_rf)[0, 1]:.4f}")