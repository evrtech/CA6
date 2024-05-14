import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("data_processed.csv")

# Target variable preparation
y = df.pop("cons_general").to_numpy()
y = np.where(y < 4, 0, 1)  # Binary classification

# Features scaling and imputation
X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Model with RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_imputed, y)

best_clf = grid_search.best_estimator_
yhat = cross_val_predict(best_clf, X_imputed, y, cv=5)

# Calculate metrics
acc = accuracy_score(y, yhat)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Output metrics to JSON
metrics = {"accuracy": acc, "specificity": specificity, "sensitivity": sensitivity}
with open("metrics.json", 'w') as outfile:
    json.dump(metrics, outfile)

# Visualization by region
df['pred_accuracy'] = (yhat == y).astype(int)
sns.set_color_codes("muted")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette="Blues_d")
ax.set(xlabel="Region", ylabel="Model accuracy")
plt.savefig("by_region.png", dpi=80)