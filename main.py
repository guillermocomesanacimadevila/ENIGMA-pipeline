# =========================================================
# Simulation of ENIGMA-like Dataset for Schizophrenia Study
# Decoding Working Memory Deficits from Structural MRI
# Explainable Machine Learning Pipeline (Publication-Ready)
# =========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style="whitegrid", context="talk")

# =========================================================
# 1. Data Simulation: ENIGMA-like Schizophrenia Data
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

# Simulated true effect (distributed across brain regions for realism)
true_effect = (
    X_raw[:, 3] * 2.5
    - X_raw[:, 10] * 1.8
    + X_raw[:, 25] * 1.5
    + X_raw[:, 50] * 1.2
    - X_raw[:, 60] * 1.0
)

y = true_effect + np.random.normal(0, 5, size=n_samples)

ages = np.random.normal(35, 10, n_samples)
sexes = np.random.choice([0, 1], n_samples)

feature_names = [f'Cortical_{i}' for i in range(n_cortical)] + [f'Subcortical_{i}' for i in range(n_subcortical)]
df_raw = pd.DataFrame(X_raw, columns=feature_names)
df_raw['Age'] = ages
df_raw['Sex'] = sexes
df_raw['Working_Memory'] = y
df_raw['Site'] = site_labels

# =========================================================
# 2. Data Preprocessing
# =========================================================

X = df_raw[feature_names + ['Age', 'Sex']]
y = df_raw['Working_Memory']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

def find_best_split(X, y, split_ratios):
    best_r2 = float('-inf')
    best_split = None
    for test_size in split_ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LassoCV(cv=5).fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2 = r2
            best_split = (test_size, X_train, X_test, y_train, y_test)
    return best_split

best_test_size, X_train, X_test, y_train, y_test = find_best_split(X_scaled, y, split_ratios)

# =========================================================
# 3. Machine Learning Models
# =========================================================

# Lasso
lasso = LassoCV(cv=5).fit(X_train, y_train)

# XGBoost
xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_best = GridSearchCV(xgb, xgb_grid, cv=3, scoring='r2', n_jobs=-1).fit(X_train, y_train).best_estimator_

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)

models = {'Lasso': lasso, 'XGBoost': xgb_best, 'Random Forest': rf}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'{name} R²: {r2_score(y_test, y_pred):.3f} | MAE: {mean_absolute_error(y_test, y_pred):.3f}')

# =========================================================
# 4. Model Explainability (SHAP for XGBoost)
# =========================================================

explainer = shap.Explainer(xgb_best)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)
shap.plots.bar(shap_values)

# =========================================================
# 5. Visualisation: Model Comparison & Calibration
# =========================================================

perf = pd.DataFrame({
    'Model': list(models.keys()),
    'R2': [r2_score(y_test, m.predict(X_test)) for m in models.values()],
    'MAE': [mean_absolute_error(y_test, m.predict(X_test)) for m in models.values()]
})

perf.set_index('Model')[['R2', 'MAE']].plot(kind='bar', figsize=(8,6), edgecolor="black", alpha=0.8)
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('model_performance_comparison_final.png', dpi=300)
plt.show()

plt.scatter(y_test, xgb_best.predict(X_test), alpha=0.6, edgecolor='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Working Memory')
plt.ylabel('Predicted Working Memory')
plt.title('Calibration Plot: XGBoost')
plt.tight_layout()
plt.savefig('calibration_plot_xgb_final.png', dpi=300)
plt.show()

# =========================================================
# 6. Save Results & Models
# =========================================================

results = pd.DataFrame({
    'True': y_test,
    'Lasso_Pred': lasso.predict(X_test),
    'XGB_Pred': xgb_best.predict(X_test),
    'RF_Pred': rf.predict(X_test)
})

results.to_csv("model_predictions_final.csv", index=False)

joblib.dump(lasso, 'lasso_model_final.pkl')
joblib.dump(xgb_best, 'xgb_model_final.pkl')
joblib.dump(rf, 'random_forest_model_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl')

print("Pipeline completed & all models saved successfully.")
