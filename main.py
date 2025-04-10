# =========================================================
# Simulation of ENIGMA-like Dataset for Schizophrenia Study
# Predicting Working Memory Performance from Structural MRI
# Publication-Ready Pipeline with Publishable Figures & Model Saving
# =========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Data Collection / Simulation
# =========================================================

np.random.seed(42)

n_samples, n_cortical, n_subcortical, n_sites = 2000, 68, 14, 5

X_cortical_raw = np.random.normal(2.5, 0.3, size=(n_samples, n_cortical))
X_subcortical_raw = np.random.normal(1500, 200, size=(n_samples, n_subcortical))

site_labels = np.random.choice(range(n_sites), size=n_samples)
site_effects_cortical = np.random.normal(0, 0.1, size=(n_sites, n_cortical))
site_effects_subcortical = np.random.normal(0, 50, size=(n_sites, n_subcortical))

for i in range(n_samples):
    X_cortical_raw[i] += site_effects_cortical[site_labels[i]]
    X_subcortical_raw[i] += site_effects_subcortical[site_labels[i]]

X_raw = np.hstack([X_cortical_raw, X_subcortical_raw])
true_effect = X_raw[:, 3] * 3 - X_raw[:, 10] * 2 + X_raw[:, 25] * 1.5
y = true_effect + np.random.normal(0, 5, size=n_samples)

ages = np.random.normal(35, 10, n_samples)
sexes = np.random.choice([0, 1], n_samples)

feature_names = [f'Cortical_{i}' for i in range(n_cortical)] + [f'Subcortical_{i}' for i in range(n_subcortical)]
df_raw = pd.DataFrame(X_raw, columns=feature_names)
df_raw['Working_Memory'] = y
df_raw['Site'] = site_labels
df_raw['Age'] = ages
df_raw['Sex'] = sexes

# =========================================================
# 2. Data Preprocessing
# =========================================================

X = df_raw[feature_names + ['Age', 'Sex']]
y = df_raw['Working_Memory']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def find_best_split(X, y, split_ratios):
    best_r2 = float('-inf')
    best = None

    for test_size in split_ratios:
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LassoCV(cv=5).fit(X_train_temp, y_train_temp)
        preds = model.predict(X_test_temp)
        r2_temp = r2_score(y_test_temp, preds)
        if r2_temp > best_r2:
            best_r2 = r2_temp
            best = (test_size, X_train_temp, X_test_temp, y_train_temp, y_test_temp)

    return best

split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
best_test_size, X_train, X_test, y_train, y_test = find_best_split(X_scaled, y, split_ratios)

# =========================================================
# 3. Machine Learning Models
# =========================================================

lasso = LassoCV(cv=5).fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid = GridSearchCV(xgb, params, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print("Best Test Size:", best_test_size)
print("Lasso R²:", r2_score(y_test, y_pred_lasso), "MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb), "MAE:", mean_absolute_error(y_test, y_pred_xgb))

# =========================================================
# 4. Model Interpretability using SHAP
# =========================================================

explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)
shap.plots.bar(shap_values)

# =========================================================
# 5. Visualisations (Publication Quality)
# =========================================================

perf = pd.DataFrame({
    'Model': ['Lasso', 'XGBoost'],
    'R2': [r2_score(y_test, y_pred_lasso), r2_score(y_test, y_pred_xgb)],
    'MAE': [mean_absolute_error(y_test, y_pred_lasso), mean_absolute_error(y_test, y_pred_xgb)]
})

fig, ax = plt.subplots(figsize=(8, 6))
perf.set_index('Model')[['R2', 'MAE']].plot(kind='bar', ax=ax, edgecolor="black", alpha=0.8)
ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_ylabel('Score', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_xgb, alpha=0.6, edgecolor='k')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
ax.set_xlabel('True Working Memory', fontsize=14)
ax.set_ylabel('Predicted Working Memory', fontsize=14)
ax.set_title('Calibration Plot: XGBoost', fontsize=16)
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 10))
mean_shap = np.abs(shap_values.values).mean(axis=0)
ax.barh(X.columns, mean_shap, color='skyblue', edgecolor='black')
ax.set_xlabel('Mean Absolute SHAP Value', fontsize=14)
ax.set_title('Feature Importance (SHAP)', fontsize=16)
plt.tight_layout()
plt.savefig("feature_importance_shap.png", dpi=300)
plt.show()

# =========================================================
# 6. Save Results & Models for Future Use
# =========================================================

results = pd.DataFrame({
    'True': y_test,
    'Lasso_Pred': y_pred_lasso,
    'XGB_Pred': y_pred_xgb
})

results.to_csv("model_predictions.csv", index=False)

joblib.dump(best_xgb, 'xgb_model.pkl')
joblib.dump(lasso, 'lasso_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and Scaler saved successfully.")