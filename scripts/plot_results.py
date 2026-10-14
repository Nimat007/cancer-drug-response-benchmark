import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Loading data and training model (this takes a minute)...")
expr = pd.read_csv(r"data\raw\OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
prism = pd.read_csv(r"data\raw\prism_auc_matrix.csv", index_col=0)

overlap = expr.index.intersection(prism.index)
target_compound = prism.columns[0]
y = prism.loc[overlap, target_compound].dropna()
X = expr.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
top_genes = X_train.var().nlargest(1000).index

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train[top_genes], y_train)
y_pred = rf.predict(X_test[top_genes])

print("Generating publication-quality plot...")
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor='k')

# Draw the red "perfect prediction" line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel("Actual Observed AUC")
plt.ylabel("Model Predicted AUC")
plt.title(f"Random Forest Performance\nCompound: {target_compound}")
plt.legend()

plt.savefig(r"figures\rf_actual_vs_predicted.png", bbox_inches='tight', dpi=300)
print("Done! Check your figures folder.")