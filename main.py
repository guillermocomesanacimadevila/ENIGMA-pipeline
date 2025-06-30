#!/usr/bin/env python3

import os
import random
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
from jinja2 import Template
from datetime import datetime

# sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# neuroHarmonize import (updated)
try:
    from neuroHarmonize import harmonizationLearn
except ImportError:
    harmonizationLearn = None

warnings.filterwarnings('ignore')
sns.set(style="whitegrid", context="talk", palette="Set2")

# ============================
# SET SEEDS FOR REPRODUCIBILITY
# ============================
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# ========== Model Classes ===========

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        evs = explained_variance_score(y, y_pred)
        return {'R2': r2, 'MAE': mae, 'Explained Variance': evs}

    def save(self, path):
        joblib.dump(self.model, path)

class LassoModel(BaseModel):
    def __init__(self, cv, random_state=42):
        super().__init__('Lasso')
        from sklearn.linear_model import LassoCV
        self.model = LassoCV(cv=cv, random_state=random_state)

class XGBModel(BaseModel):
    def __init__(self, param_grid, cv, random_state=42):
        super().__init__('XGBoost')
        from xgboost import XGBRegressor
        base = XGBRegressor(objective='reg:squarederror', random_state=random_state)
        self.grid_search = GridSearchCV(base, param_grid, cv=cv, scoring='r2', n_jobs=-1)
        self.model = None

    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.model = self.grid_search.best_estimator_

    def save(self, path):
        joblib.dump(self.model, path)

class RFModel(BaseModel):
    def __init__(self, n_estimators=300, max_depth=10, random_state=42):
        super().__init__('Random Forest')
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

# ========== Pipeline Class ===========

class ENIGMAPipeline:
    def __init__(self, data_dir='Data', out_dir='Outputs', random_state=42):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.random_state = random_state
        self.scaler = None
        self.feature_names = []
        self.meta = None
        self.models = {}
        self.report_sections = []
        self.plots = []
        self.stats = {}
        os.makedirs(self.out_dir, exist_ok=True)
        for sub in ['Visualisations', 'EDA', 'Models', 'Predictions']:
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

    def log_environment(self):
        import sklearn
        env = {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': sklearn.__version__,
            'shap': shap.__version__,
            'seed': self.random_state,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        with open(f'{self.out_dir}/pipeline_reproducibility_log.txt', 'w') as f:
            for k, v in env.items():
                f.write(f"{k}: {v}\n")
        self.stats['env'] = env
        self.report_sections.append(f"<h2>Environment & Reproducibility</h2><pre>{env}</pre>")
        return env

    def load_data(self):
        try:
            df_sMRI = pd.read_csv(f'{self.data_dir}/enigma_like_sMRI.csv')
            df_WM = pd.read_csv(f'{self.data_dir}/enigma_like_WM.csv')
            df_raw = pd.merge(df_sMRI, df_WM, on='Sample_ID')
        except Exception as e:
            raise FileNotFoundError(f"Could not load required CSVs: {e}")
        self.meta = df_raw[['Sample_ID', 'Site', 'Age', 'Sex']]
        exclude_cols = ['Sample_ID', 'Site', 'Working_Memory']
        self.feature_names = [col for col in df_raw.columns if col not in exclude_cols]
        self.X_raw = df_raw[self.feature_names]
        self.y = df_raw['Working_Memory'].values
        self.df_raw = df_raw
        missing = self.df_raw.isnull().sum().sum()
        self.stats['missing'] = missing
        self.report_sections.append(f"<h2>Data Loaded</h2><ul><li>Features: {len(self.feature_names)}</li><li>Samples: {len(self.df_raw)}</li><li>Missing values: {missing}</li></ul>")
        print("Data loaded and merged.")

    def run_eda(self):
        eda_dir = f"{self.out_dir}/EDA"
        eda_imgs = []
        eda_stats = {}
        for var in ['Age', 'Sex', 'Site']:
            plt.figure(figsize=(5,3))
            if var == 'Sex':
                sns.countplot(x=var, data=self.df_raw, edgecolor='k', palette='Set2')
                eda_stats['sex_counts'] = self.df_raw[var].value_counts().to_dict()
            else:
                sns.histplot(self.df_raw[var], kde=True, bins=30, color="#3498db", edgecolor='black')
                eda_stats[f'{var.lower()}_mean'] = self.df_raw[var].mean()
                eda_stats[f'{var.lower()}_std'] = self.df_raw[var].std()
            plt.title(f'{var} Distribution', fontsize=16)
            plt.xlabel(var, fontsize=13)
            plt.tight_layout()
            fname = f"{eda_dir}/{var}_hist.png"
            plt.savefig(fname, dpi=200); plt.close()
            eda_imgs.append(fname)
        plt.figure(figsize=(6,3))
        sns.histplot(self.df_raw['Working_Memory'], bins=40, kde=True, color="#e67e22", edgecolor='black')
        plt.title('Working Memory Distribution', fontsize=16)
        plt.xlabel("Working Memory Score", fontsize=13)
        plt.tight_layout()
        fname = f"{eda_dir}/Working_Memory_hist.png"
        plt.savefig(fname, dpi=200); plt.close()
        eda_imgs.append(fname)
        eda_stats['wm_mean'] = self.df_raw['Working_Memory'].mean()
        eda_stats['wm_std'] = self.df_raw['Working_Memory'].std()
        plt.figure(figsize=(12, 3))
        means = self.X_raw.mean()
        sns.histplot(means, bins=40, color="#16a085", edgecolor='black')
        plt.title('Mean of sMRI Features', fontsize=16)
        plt.xlabel('Mean Value')
        plt.tight_layout()
        fname = f"{eda_dir}/Feature_Mean_hist.png"
        plt.savefig(fname, dpi=200); plt.close()
        eda_imgs.append(fname)
        plt.figure(figsize=(12, 3))
        stds = self.X_raw.std()
        sns.histplot(stds, bins=40, color="#c0392b", edgecolor='black')
        plt.title('Std Dev of sMRI Features', fontsize=16)
        plt.xlabel('Std Value')
        plt.tight_layout()
        fname = f"{eda_dir}/Feature_Std_hist.png"
        plt.savefig(fname, dpi=200); plt.close()
        eda_imgs.append(fname)
        plt.figure(figsize=(14, 1))
        sns.heatmap(self.df_raw.isnull(), cbar=False)
        plt.title('Missing Values', fontsize=15)
        fname = f"{eda_dir}/missing_values_heatmap.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200); plt.close()
        eda_imgs.append(fname)
        self.plots += eda_imgs
        self.stats['eda'] = eda_stats
        imgs_html = "".join([f'<img src="{os.path.relpath(x, self.out_dir)}" width="400">' for x in eda_imgs])
        stats_html = "<pre>" + "\n".join([f"{k}: {v}" for k, v in eda_stats.items()]) + "</pre>"
        self.report_sections.append(f"<h2>Exploratory Data Analysis</h2>{imgs_html}{stats_html}")
        print("EDA completed and saved.")

    def harmonize(self):
        if harmonizationLearn is None:
            self.X_harmonized = self.X_raw
            self.report_sections.append("<h2>ComBat Harmonization</h2><p><b>Skipped</b> (neuroHarmonize not installed).</p>")
            print('ComBat harmonization skipped: neuroHarmonize not installed.')
        else:
            try:
                covars = self.df_raw[['Site', 'Age', 'Sex']].copy()
                covars['SITE'] = covars['Site'].astype(str)
                covars = covars.drop(columns='Site')
                covars['Sex'] = covars['Sex'].astype(str)
                model, harmonized = harmonizationLearn(self.X_raw.values, covars)
                self.X_harmonized = pd.DataFrame(harmonized, columns=self.feature_names)
                self.report_sections.append("<h2>ComBat Harmonization</h2><p>Applied successfully.</p>")
                print('ComBat harmonization applied.')
            except Exception as e:
                self.X_harmonized = self.X_raw
                self.report_sections.append(f"<h2>ComBat Harmonization</h2><p><b>Skipped</b> (failed: {e}).</p>")
                print(f'ComBat harmonization skipped: {e}')
        harmo_save = self.df_raw[['Sample_ID', 'Site', 'Age', 'Sex']].copy()
        harmo_save = pd.concat([harmo_save, self.X_harmonized], axis=1)
        harmo_save.to_csv(f"{self.data_dir}/enigma_like_sMRI_harmonized.csv", index=False)

    def split_and_scale(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_harmonized)
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X_scaled, self.y, self.meta, test_size=0.2, random_state=self.random_state
        )
        pd.DataFrame({'Sample_ID': meta_train['Sample_ID']}).to_csv(f"{self.data_dir}/train_samples.csv", index=False)
        pd.DataFrame({'Sample_ID': meta_test['Sample_ID']}).to_csv(f"{self.data_dir}/test_samples.csv", index=False)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.meta_train, self.meta_test = meta_train, meta_test
        self.scaler = scaler

    def nested_cv(self, inner_cv=3, outer_cv=5):
        kf_outer = KFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        kf_inner = KFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        results = {}
        lasso = LassoModel(cv=kf_inner, random_state=self.random_state)
        lasso_scores = cross_val_score(lasso.model, self.scaler.transform(self.X_harmonized), self.y,
                                       cv=kf_outer, scoring='r2')
        results['Lasso'] = lasso_scores
        xgb_model = XGBModel(param_grid_xgb, cv=kf_inner, random_state=self.random_state)
        from xgboost import XGBRegressor
        base_xgb = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        xgb_grid = GridSearchCV(base_xgb, param_grid_xgb, cv=kf_inner, scoring='r2', n_jobs=-1)
        xgb_scores = cross_val_score(xgb_grid, self.scaler.transform(self.X_harmonized), self.y,
                                     cv=kf_outer, scoring='r2')
        results['XGBoost'] = xgb_scores
        rf = RFModel(n_estimators=300, max_depth=10, random_state=self.random_state)
        rf_scores = cross_val_score(rf.model, self.scaler.transform(self.X_harmonized), self.y,
                                    cv=kf_outer, scoring='r2')
        results['Random Forest'] = rf_scores
        self.stats['nested_cv'] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}
        text = "<h2>Nested Cross-Validation Results (5 Outer Folds)</h2><ul>"
        for m, scores in results.items():
            text += f"<li>{m}: {np.mean(scores):.3f} ± {np.std(scores):.3f} (R²)</li>"
        text += "</ul>"
        self.report_sections.append(text)
        self.nested_cv_results = results

    def train_final_models(self):
        lasso = LassoModel(cv=5, random_state=self.random_state)
        lasso.fit(self.X_train, self.y_train)
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        xgb = XGBModel(xgb_param_grid, cv=3, random_state=self.random_state)
        xgb.fit(self.X_train, self.y_train)
        rf = RFModel(n_estimators=300, max_depth=10, random_state=self.random_state)
        rf.fit(self.X_train, self.y_train)
        self.models = {'Lasso': lasso, 'XGBoost': xgb, 'Random Forest': rf}

    def evaluate(self):
        text = "<h2>Test Set Performance</h2><ul>"
        perf_dict = {}
        for name, model in self.models.items():
            scores = model.evaluate(self.X_test, self.y_test)
            perf_dict[name] = scores
            text += f"<li>{name}: R²={scores['R2']:.3f} | MAE={scores['MAE']:.3f} | Explained Variance={scores['Explained Variance']:.3f}</li>"
        text += "</ul>"
        self.stats['final_test'] = perf_dict
        self.report_sections.append(text)

    def interpret(self):
        try:
            explainer = shap.Explainer(self.models['XGBoost'].model)
            shap_values = explainer(self.X_test)
            # SHAP summary plot (with actual feature names)
            shap_summary_path = f"{self.out_dir}/Visualisations/SHAP_summary_xgb.png"
            plt.figure()
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
            plt.tight_layout(); plt.savefig(shap_summary_path, dpi=220, bbox_inches='tight'); plt.close()
            self.plots.append(shap_summary_path)
            # SHAP bar plot (actual feature names)
            shap_bar_path = f"{self.out_dir}/Visualisations/SHAP_bar_xgb.png"
            plt.figure()
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout(); plt.savefig(shap_bar_path, dpi=220, bbox_inches='tight'); plt.close()
            self.plots.append(shap_bar_path)
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[::-1][:10]
            top_feats = [self.feature_names[i] for i in top_idx]
            top_vals = mean_abs_shap[top_idx]
            text = "<h2>Top 10 XGBoost Features</h2><ol>"
            for i, val in enumerate(top_vals):
                text += f"<li>{top_feats[i]}: {val:.2f}</li>"
            text += "</ol>"
            self.stats['top10_shap'] = dict(zip(top_feats, top_vals))
            self.report_sections.append(text)
            perm = permutation_importance(self.models['Random Forest'].model, self.X_test, self.y_test,
                                          n_repeats=10, random_state=self.random_state)
            sorted_idx = perm.importances_mean.argsort()[::-1][:10]
            perm_names = [self.feature_names[i] for i in sorted_idx]
            perm_path = f"{self.out_dir}/Visualisations/RF_permutation_importance.png"
            plt.figure(figsize=(8, 4))
            sns.barplot(y=perm_names, x=perm.importances_mean[sorted_idx], palette='flare')
            plt.title("Top 10 RF Permutation Importances", fontsize=16)
            plt.xlabel("Mean Importance", fontsize=12)
            plt.tight_layout(); plt.savefig(perm_path, dpi=220); plt.close()
            self.plots.append(perm_path)
            self.stats['top10_rf'] = dict(zip(perm_names, perm.importances_mean[sorted_idx]))
            # PDP for XGBoost top 3
            for j in range(3):
                idx = top_idx[j]
                PDP_path = f"{self.out_dir}/Visualisations/PDP_Feature{j+1}_xgb.png"
                PartialDependenceDisplay.from_estimator(
                    self.models['XGBoost'].model, self.X_test, [idx], feature_names=self.feature_names
                )
                plt.title(f"Partial Dependence: {self.feature_names[idx]}", fontsize=15)
                plt.tight_layout(); plt.savefig(PDP_path, dpi=150); plt.close()
                self.plots.append(PDP_path)
        except Exception as e:
            self.report_sections.append(
                f"<h2>Interpretability</h2><p style='color:red;'>Interpretability step failed: {e}</p>")
            print("Interpretation step failed:", e)

    def visualise(self):
        perf = pd.DataFrame({
            'Model': list(self.models.keys()),
            'R2': [r2_score(self.y_test, m.predict(self.X_test)) for m in self.models.values()],
            'MAE': [mean_absolute_error(self.y_test, m.predict(self.X_test)) for m in self.models.values()]
        })
        bar_path = f'{self.out_dir}/Visualisations/model_performance_comparison_final.png'
        plt.figure(figsize=(7,5))
        perf.set_index('Model')[['R2', 'MAE']].plot(kind='bar', edgecolor="black", alpha=0.9, rot=0, ax=plt.gca())
        plt.title('Model Performance Comparison (Test Set)', fontsize=17, weight='bold')
        plt.ylabel("Score", fontsize=13)
        plt.tight_layout(); plt.savefig(bar_path, dpi=240); plt.close()
        self.plots.append(bar_path)
        # Calibration plot (XGBoost)
        cal_path = f'{self.out_dir}/Visualisations/calibration_plot_xgb_final.png'
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, self.models['XGBoost'].predict(self.X_test), alpha=0.7, edgecolor='k', s=50, c="#16a085")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--', lw=2)
        plt.xlabel('True Working Memory', fontsize=13)
        plt.ylabel('Predicted Working Memory', fontsize=13)
        plt.title('Calibration Plot: XGBoost', fontsize=17, weight='bold')
        plt.tight_layout(); plt.savefig(cal_path, dpi=240); plt.close()
        self.plots.append(cal_path)

    def save_all(self):
        results = pd.DataFrame({
            'Sample_ID': self.meta_test['Sample_ID'].values,
            'True': self.y_test,
            'Lasso_Pred': self.models['Lasso'].predict(self.X_test),
            'XGB_Pred': self.models['XGBoost'].predict(self.X_test),
            'RF_Pred': self.models['Random Forest'].predict(self.X_test)
        })
        results.to_csv(f"{self.out_dir}/Predictions/model_predictions_final.csv", index=False)
        self.models['Lasso'].save(f'{self.out_dir}/Models/lasso_model_final.pkl')
        self.models['XGBoost'].save(f'{self.out_dir}/Models/xgb_model_final.pkl')
        self.models['Random Forest'].save(f'{self.out_dir}/Models/random_forest_model_final.pkl')
        joblib.dump(self.scaler, f'{self.out_dir}/Models/scaler_final.pkl')

    def html_report(self, filename="ENIGMA_pipeline_report.html"):
        imgs_html = "".join([f'<img src="{os.path.relpath(x, self.out_dir)}" class="vizimg">' for x in self.plots])
        stats_html = "<h2>Pipeline Statistics</h2><pre>" + "\n".join(
            [f"{k}: {v}" for k, v in self.stats.items()]) + "</pre>"

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>ENIGMA Pipeline Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin:0; background:#f7f8fa; color:#222; transition: background 0.2s, color 0.2s; }
        .container { max-width:950px; margin: auto; padding: 1.5em 2em 2em 2em; background:var(--bg); box-shadow:0 2px 24px #0002; border-radius: 14px;}
        h1, h2, h3 { color: #202e3b; margin-top: 0.7em;}
        pre { background:#f2f2f4; border-radius:7px; padding:8px; font-size:1em; overflow-x:auto; }
        img.vizimg { margin:20px 14px 14px 0; border: 1.5px solid #ddd; box-shadow: 2px 2px 14px #0001; border-radius:8px; width:430px; max-width:100%; vertical-align:middle;}
        .toggle { float:right; margin:-10px 0 0 0; }
        .stats-box { background: #fcfcff; border-radius: 8px; box-shadow: 0 1px 8px #0001; margin: 18px 0 24px 0; padding: 14px 22px;}
        .section { margin-bottom: 30px; }
        @media (max-width:600px) { .container {padding:0.7em;} img.vizimg {width:100%;} }
        body.dark { background: #181e2b; color:#eee;}
        body.dark h1,body.dark h2,body.dark h3 {color:#b5cfff;}
        body.dark pre {background:#262d3a; color:#cdd6ee;}
        body.dark .container {background:#232b3b;}
        body.dark .stats-box {background: #222a39;}
        body.dark img.vizimg {border-color:#3b4660;}
        .button-toggle { padding: 6px 18px; border-radius: 8px; border:none; background: #464fd3; color:white; font-weight:600; margin-bottom:1em; cursor:pointer; float:right;}
        .button-toggle:hover {background: #2c3878;}
        </style>
        </head>
        <body>
        <div class="container">
        <button onclick="document.body.classList.toggle('dark')" class="button-toggle">Toggle Light/Dark</button>
        <h1>ENIGMA-like Pipeline Results</h1>
        {{report_sections}}
        {{stats_html}}
        <h2>All Plots</h2>
        {{imgs_html}}
        </div>
        </body>
        </html>
        """
        html = Template(html_template).render(
            report_sections=''.join(self.report_sections),
            imgs_html=imgs_html,
            stats_html=stats_html,
        )
        path = os.path.join(self.out_dir, filename)
        with open(path, "w") as f:
            f.write(html)
        print(f"HTML report saved as: {path}")

    def run_full_pipeline(self):
        self.log_environment()
        self.load_data()
        self.run_eda()
        self.harmonize()
        self.split_and_scale()
        self.nested_cv()
        self.train_final_models()
        self.evaluate()
        self.interpret()
        self.visualise()
        self.save_all()
        self.html_report()
        print("\nPipeline completed. Report and outputs saved to Output directory.")

# ========== Run Pipeline ===========

if __name__ == '__main__':
    pipeline = ENIGMAPipeline(data_dir='Data', out_dir='Outputs', random_state=42)
    pipeline.run_full_pipeline()
