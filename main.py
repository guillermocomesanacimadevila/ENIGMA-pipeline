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
from datetime import datetime

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
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': '#fafbfc', 'axes.edgecolor': '#dddddd'})

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
        self.top_feats_table = ""
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
        self.env_table = "<table class='env-table'><tr>" + "".join([f"<th>{k}</th>" for k in env]) + "</tr>" + \
            "<tr>" + "".join([f"<td>{v}</td>" for v in env.values()]) + "</tr></table>"
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
        self.dataset_table = f"""
        <table class='info-table'>
        <tr><th>Samples</th><td>{len(self.df_raw)}</td></tr>
        <tr><th>Features</th><td>{len(self.feature_names)}</td></tr>
        <tr><th>Missing Values</th><td>{missing}</td></tr>
        </table>
        """
        print("Data loaded and merged.")

    def run_eda(self):
        eda_dir = f"{self.out_dir}/EDA"
        eda_imgs = []
        eda_stats = {}
        for var in ['Age', 'Sex', 'Site']:
            plt.figure(figsize=(6,3))
            if var == 'Sex':
                sns.countplot(x=var, data=self.df_raw, edgecolor='k', palette='colorblind')
                eda_stats['sex_counts'] = self.df_raw[var].value_counts().to_dict()
            else:
                sns.histplot(self.df_raw[var], kde=True, bins=30, color="#1f77b4", edgecolor='black')
                eda_stats[f'{var.lower()}_mean'] = self.df_raw[var].mean()
                eda_stats[f'{var.lower()}_std'] = self.df_raw[var].std()
            plt.title(f'{var} Distribution', fontsize=18, weight='bold')
            plt.xlabel(var, fontsize=15)
            plt.ylabel("Count", fontsize=15)
            plt.tight_layout()
            fname = f"{eda_dir}/{var}_hist.png"
            plt.savefig(fname, dpi=250); plt.close()
            eda_imgs.append(fname)
        plt.figure(figsize=(6,3))
        sns.histplot(self.df_raw['Working_Memory'], bins=40, kde=True, color="#f39c12", edgecolor='black')
        plt.title('Working Memory Distribution', fontsize=18, weight='bold')
        plt.xlabel("Working Memory Score", fontsize=15)
        plt.tight_layout()
        fname = f"{eda_dir}/Working_Memory_hist.png"
        plt.savefig(fname, dpi=250); plt.close()
        eda_imgs.append(fname)
        eda_stats['wm_mean'] = self.df_raw['Working_Memory'].mean()
        eda_stats['wm_std'] = self.df_raw['Working_Memory'].std()
        plt.figure(figsize=(12, 3))
        means = self.X_raw.mean()
        sns.histplot(means, bins=40, color="#16a085", edgecolor='black')
        plt.title('Mean of sMRI Features', fontsize=17)
        plt.xlabel('Mean Value', fontsize=13)
        plt.tight_layout()
        fname = f"{eda_dir}/Feature_Mean_hist.png"
        plt.savefig(fname, dpi=250); plt.close()
        eda_imgs.append(fname)
        plt.figure(figsize=(12, 3))
        stds = self.X_raw.std()
        sns.histplot(stds, bins=40, color="#c0392b", edgecolor='black')
        plt.title('Std Dev of sMRI Features', fontsize=17)
        plt.xlabel('Std Value', fontsize=13)
        plt.tight_layout()
        fname = f"{eda_dir}/Feature_Std_hist.png"
        plt.savefig(fname, dpi=250); plt.close()
        eda_imgs.append(fname)
        plt.figure(figsize=(14, 1))
        sns.heatmap(self.df_raw.isnull(), cbar=False, cmap='Reds')
        plt.title('Missing Values', fontsize=16, weight='bold')
        fname = f"{eda_dir}/missing_values_heatmap.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=250); plt.close()
        eda_imgs.append(fname)
        self.plots += eda_imgs
        eda_df = pd.DataFrame([
            ["Sex (counts)", str(eda_stats.get('sex_counts', '-'))],
            ["Age (mean ± std)", f"{eda_stats.get('age_mean', '-'):0.1f} ± {eda_stats.get('age_std', '-'):0.1f}"],
            ["Site (mean ± std)", f"{eda_stats.get('site_mean', '-'):0.1f} ± {eda_stats.get('site_std', '-'):0.1f}"],
            ["Working Memory (mean ± std)", f"{eda_stats.get('wm_mean', '-'):0.2f} ± {eda_stats.get('wm_std', '-'):0.2f}"]
        ], columns=["Variable", "Stats"])
        eda_table = eda_df.to_html(index=False, classes='info-table', border=0)
        imgs_html = "".join([f'<img src="{os.path.relpath(x, self.out_dir)}" class="vizimg">' for x in eda_imgs])
        self.eda_section = f"<h2 id='eda'>Exploratory Data Analysis</h2>{eda_table}<div class='viz-grid'>{imgs_html}</div>"
        print("EDA completed and saved.")

    def harmonize(self):
        if harmonizationLearn is None:
            self.X_harmonized = self.X_raw
            self.harmo_section = "<h2 id='harmonization'>ComBat Harmonization</h2><div class='card'><b>Skipped</b> (neuroHarmonize not installed).</div>"
            print('ComBat harmonization skipped: neuroHarmonize not installed.')
        else:
            try:
                covars = self.df_raw[['Site', 'Age', 'Sex']].copy()
                covars['SITE'] = covars['Site'].astype(str)
                covars = covars.drop(columns='Site')
                covars['Sex'] = covars['Sex'].astype(str)
                model, harmonized = harmonizationLearn(self.X_raw.values, covars)
                self.X_harmonized = pd.DataFrame(harmonized, columns=self.feature_names)
                self.harmo_section = "<h2 id='harmonization'>ComBat Harmonization</h2><div class='card'>Applied successfully.</div>"
                print('ComBat harmonization applied.')
            except Exception as e:
                self.X_harmonized = self.X_raw
                self.harmo_section = f"<h2 id='harmonization'>ComBat Harmonization</h2><div class='card error'><b>Skipped</b> (failed: {e}).</div>"
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
        rows = "".join([
            f"<tr><td>{model}</td><td>{np.mean(scores):.3f} ± {np.std(scores):.3f}</td></tr>"
            for model, scores in results.items()
        ])
        self.cv_section = f"""
        <h2 id='crossval'>Nested Cross-Validation Results (5 Outer Folds)</h2>
        <table class='perf-table'><tr><th>Model</th><th>Mean R² (±SD)</th></tr>{rows}</table>
        """
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
        perf_dict = {}
        for name, model in self.models.items():
            scores = model.evaluate(self.X_test, self.y_test)
            perf_dict[name] = scores
        self.stats['final_test'] = perf_dict
        test_table = pd.DataFrame(perf_dict).T.reset_index().rename(columns={'index': 'Model'})
        test_html = test_table.to_html(index=False, classes='perf-table', float_format="{:.3f}".format, border=0)
        self.eval_section = f"<h2 id='testset'>Test Set Performance</h2>{test_html}"

    def interpret(self):
        try:
            explainer = shap.Explainer(self.models['XGBoost'].model)
            shap_values = explainer(self.X_test)
            shap_summary_path = f"{self.out_dir}/Visualisations/SHAP_summary_xgb.png"
            plt.figure()
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
            plt.tight_layout(); plt.savefig(shap_summary_path, dpi=260, bbox_inches='tight'); plt.close()
            self.plots.append(shap_summary_path)
            shap_bar_path = f"{self.out_dir}/Visualisations/SHAP_bar_xgb.png"
            plt.figure()
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout(); plt.savefig(shap_bar_path, dpi=260, bbox_inches='tight'); plt.close()
            self.plots.append(shap_bar_path)
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[::-1][:10]
            top_feats = [self.feature_names[i] for i in top_idx]
            top_vals = mean_abs_shap[top_idx]
            self.top_feats_table = "<table class='feat-table'><tr><th>Feature</th><th>Mean |SHAP value|</th></tr>"
            self.top_feats_table += "".join([
                f"<tr><td>{top_feats[i]}</td><td>{top_vals[i]:.2f}</td></tr>"
                for i in range(len(top_feats))
            ])
            self.top_feats_table += "</table>"
            self.stats['top10_shap'] = dict(zip(top_feats, top_vals))
            perm = permutation_importance(self.models['Random Forest'].model, self.X_test, self.y_test,
                                          n_repeats=10, random_state=self.random_state)
            sorted_idx = perm.importances_mean.argsort()[::-1][:10]
            perm_names = [self.feature_names[i] for i in sorted_idx]
            perm_vals = perm.importances_mean[sorted_idx]
            perm_table = "<table class='feat-table'><tr><th>Feature</th><th>Mean Importance</th></tr>"
            perm_table += "".join([
                f"<tr><td>{perm_names[i]}</td><td>{perm_vals[i]:.3f}</td></tr>"
                for i in range(len(perm_names))
            ])
            perm_table += "</table>"
            for j in range(3):
                idx = top_idx[j]
                PDP_path = f"{self.out_dir}/Visualisations/PDP_{top_feats[j].replace(' ', '_')}_xgb.png"
                PartialDependenceDisplay.from_estimator(
                    self.models['XGBoost'].model, self.X_test, [idx], feature_names=self.feature_names
                )
                plt.title(f"Partial Dependence: {self.feature_names[idx]}", fontsize=16)
                plt.tight_layout(); plt.savefig(PDP_path, dpi=220); plt.close()
                self.plots.append(PDP_path)
            self.interp_section = f"""
            <h2 id='interpret'>Interpretability</h2>
            <div class='viz-flex'>
                <div class='card'>
                    <h3>Top 10 XGBoost Features</h3>
                    {self.top_feats_table}
                </div>
                <div class='card'>
                    <h3>Top 10 RF Permutation Importances</h3>
                    {perm_table}
                </div>
            </div>
            """
        except Exception as e:
            self.interp_section = (
                f"<h2 id='interpret'>Interpretability</h2><div class='card error'>Interpretability step failed: {e}</div>")
            print("Interpretation step failed:", e)

    def visualise(self):
        perf = pd.DataFrame({
            'Model': list(self.models.keys()),
            'R2': [r2_score(self.y_test, m.predict(self.X_test)) for m in self.models.values()],
            'MAE': [mean_absolute_error(self.y_test, m.predict(self.X_test)) for m in self.models.values()]
        })
        bar_path = f'{self.out_dir}/Visualisations/model_performance_comparison_final.png'
        plt.figure(figsize=(7,5))
        perf.set_index('Model')[['R2', 'MAE']].plot(kind='bar', edgecolor="black", alpha=0.96, rot=0, ax=plt.gca())
        plt.title('Model Performance Comparison (Test Set)', fontsize=17, weight='bold')
        plt.ylabel("Score", fontsize=15)
        plt.xlabel("")
        plt.tight_layout(); plt.savefig(bar_path, dpi=260); plt.close()
        self.plots.append(bar_path)
        cal_path = f'{self.out_dir}/Visualisations/calibration_plot_xgb_final.png'
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, self.models['XGBoost'].predict(self.X_test), alpha=0.7, edgecolor='k', s=70, c="#16a085")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--', lw=2)
        plt.xlabel('True Working Memory', fontsize=15)
        plt.ylabel('Predicted Working Memory', fontsize=15)
        plt.title('Calibration Plot: XGBoost', fontsize=17, weight='bold')
        plt.tight_layout(); plt.savefig(cal_path, dpi=260); plt.close()
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

    # ============= MODERN, BEAUTIFUL HTML REPORT =============

    def html_report(self, filename="ENIGMA_pipeline_report.html"):
        imgs_html = "".join([
            f'<div class="imgbox"><img src="{os.path.relpath(x, self.out_dir)}" class="vizimg"></div>' for x in self.plots
        ])

        toc = """
        <nav class='toc'>
            <a href='#summary'><span class="toc-ico">🏆</span> Executive Summary</a>
            <a href='#dataset'><span class="toc-ico">📊</span> Dataset</a>
            <a href='#env'><span class="toc-ico">🔗</span> Reproducibility</a>
            <a href='#eda'><span class="toc-ico">🔍</span> EDA</a>
            <a href='#harmonization'><span class="toc-ico">🧬</span> Harmonization</a>
            <a href='#crossval'><span class="toc-ico">🧪</span> Cross-Validation</a>
            <a href='#testset'><span class="toc-ico">📈</span> Test Performance</a>
            <a href='#interpret'><span class="toc-ico">💡</span> Interpretability</a>
            <a href='#plots'><span class="toc-ico">🖼️</span> Visualizations</a>
        </nav>
        """

        best_model = max(self.stats['final_test'], key=lambda k: self.stats['final_test'][k]['R2'])
        best_score = self.stats['final_test'][best_model]['R2']
        summary_card = f"""
        <section id='summary'>
            <div class='headline'>
                <img src='https://img.icons8.com/fluency/48/brain.png' class='brandicon' alt='ENIGMA'>
                <span>ENIGMA Pipeline Results Dashboard</span>
                <span class='badge'>2024</span>
            </div>
            <div class='summary-cards'>
                <div class='card summary'>
                    <h3>Best Model</h3>
                    <div class='score-icon'>🏆</div>
                    <div class='num'>{best_model}</div>
                    <div class='score'>R² = {best_score:.3f}</div>
                </div>
                <div class='card summary'>
                    <h3>Samples</h3>
                    <div class='score-icon'>🧑‍🤝‍🧑</div>
                    <div class='num'>{len(self.df_raw)}</div>
                </div>
                <div class='card summary'>
                    <h3>Features</h3>
                    <div class='score-icon'>🔢</div>
                    <div class='num'>{len(self.feature_names)}</div>
                </div>
            </div>
            <div class='section-desc'>
                <span>This report presents the full ENIGMA pipeline analysis, from harmonization and feature engineering to cross-validation, model performance, and interpretability. <b>Interactive, visual, and ready for stakeholders.</b></span>
            </div>
        </section>
        """

        timeline = """
        <div class='timeline'>
            <div class='tl-step'><div class='tl-dot'>1</div><div>Data <span>Loaded</span></div></div>
            <div class='tl-step'><div class='tl-dot'>2</div><div>EDA <span>Exploration</span></div></div>
            <div class='tl-step'><div class='tl-dot'>3</div><div>Harmonization <span>(ComBat)</span></div></div>
            <div class='tl-step'><div class='tl-dot'>4</div><div>Model <span>Training</span></div></div>
            <div class='tl-step'><div class='tl-dot'>5</div><div>Validation <span>and Test</span></div></div>
            <div class='tl-step'><div class='tl-dot'>6</div><div>Interpretability <span>and Results</span></div></div>
        </div>
        """

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>ENIGMA Pipeline Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
            <style>
            body {{
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                margin: 0;
                background: linear-gradient(120deg,#f6f9fc 0%,#e0e7ff 100%);
                color:#1b2232;
            }}
            .container {{
                max-width:1100px;
                margin: 3.5em auto 2.5em auto;
                padding: 2.8em 2.2em 2.2em 2.2em;
                background:rgba(255,255,255,0.98);
                box-shadow:0 9px 54px #00235415;
                border-radius:32px;
            }}
            .brandicon {{
                height:48px;
                margin-right: 12px;
                vertical-align:middle;
            }}
            h1,h2,h3 {{
                color: #1a237e;
                font-weight:900;
                margin-top: 0.7em;
            }}
            h2 {{
                border-bottom:3px solid #f3f4fb;
                padding-bottom:0.18em;
                margin-bottom: 1.4em;
                letter-spacing:-0.02em;
            }}
            .headline {{
                display:flex;
                align-items:center;
                gap: 18px;
                margin-bottom:20px;
                font-size:2.3em;
                font-weight:900;
                letter-spacing:-0.02em;
            }}
            .badge {{
                background:#3a6cf6;
                color:#fff;
                padding:4px 18px;
                border-radius:20px;
                font-size:1em;
                font-weight:700;
                letter-spacing:0.06em;
                box-shadow: 0 2px 10px #3a6cf617;
                margin-left:auto;
            }}
            nav.toc {{
                display:flex;
                flex-wrap:wrap;
                gap:1.25em;
                padding:1em 0 1.4em 0;
                font-size:1.13em;
                background:rgba(80,111,221,0.06);
                border-radius:18px;
                margin-bottom:2.6em;
                border:1.5px solid #e7eafe;
                box-shadow: 0 1px 7px #d6e2ff24;
                font-weight:700;
            }}
            .toc-ico {{font-size:1.13em;vertical-align:middle;margin-right:0.11em;}}
            nav.toc a {{
                color:#2a3fa6;
                text-decoration:none;
                border-bottom:2px solid transparent;
                padding-bottom:1px;
                transition:all 0.17s;
                border-radius:4px;
            }}
            nav.toc a:hover {{
                color:#253082;
                border-bottom:2px solid #3a6cf6;
                background:#e6ecff;
            }}
            section {{
                margin-bottom:3em;
            }}
            .section-desc {{
                margin-top:18px;
                margin-bottom:0.8em;
                font-size:1.18em;
                color:#263c74;
                background: #f3f7fe;
                padding:14px 19px;
                border-radius:13px;
                box-shadow:0 2px 14px #e2eaf8;
            }}
            .info-table,.perf-table,.feat-table,.env-table {{
                width:100%;
                border-collapse:collapse;
                margin-bottom:2.2em;
                font-size:1.11em;
                box-shadow:0 2px 16px #e3eafc21;
                background:#f8faff;
                border-radius:8px;
                overflow:hidden;
            }}
            .info-table th, .info-table td,
            .perf-table th, .perf-table td,
            .feat-table th, .feat-table td,
            .env-table th, .env-table td {{
                border-bottom:1.2px solid #e8eefd;
                padding:14px 17px;
                text-align:left;
            }}
            .info-table th, .perf-table th, .feat-table th, .env-table th {{
                background:#eef4ff;
                font-weight:700;
                color:#264182;
                letter-spacing:0.01em;
            }}
            .info-table td, .perf-table td, .feat-table td, .env-table td {{
                background:#f8faff;
                font-weight:500;
            }}
            .info-table tr:nth-child(even) td,
            .perf-table tr:nth-child(even) td,
            .feat-table tr:nth-child(even) td,
            .env-table tr:nth-child(even) td {{
                background: #e7eafd;
            }}
            .card {{
                background:#f7f9ff;
                border-radius:17px;
                box-shadow:0 3px 18px #b6c4ea17;
                padding:28px 30px;
                margin-bottom:24px;
                border-left: 7px solid #3a6cf6;
                font-size:1.09em;
            }}
            .card.summary {{
                border-left: none;
                box-shadow:0 2px 10px #3a6cf616;
                text-align:center;
                margin:0 22px 0 0;
                min-width:160px; flex:1 1 0;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
            }}
            .card.summary .num {{ font-size:2.45em; font-weight:900; color:#3a6cf6; }}
            .card.summary .score {{ font-size:2.2em; font-weight:900; color:#14b9b2; }}
            .card.summary .score-icon {{ font-size:2.2em;margin-bottom:0.18em; }}
            .card.error {{ border-left:7px solid #e6344b; color:#be2238; background:#fff0f2; }}
            .summary-cards {{
                display:flex;
                gap:29px;
                margin-bottom:2.7em;
            }}
            .timeline {{
                display:flex;
                gap:19px;
                justify-content:center;
                align-items:center;
                margin-bottom:2.7em;
                margin-top:0.7em;
            }}
            .tl-step {{
                display:flex;
                flex-direction:column;
                align-items:center;
                font-weight:700;
                color:#4252a2;
                font-size:1.04em;
                min-width:85px;
            }}
            .tl-dot {{
                width:34px; height:34px;
                background:#3a6cf6;
                color:#fff;
                font-size:1.24em;
                font-weight:900;
                border-radius:50%;
                display:flex; align-items:center; justify-content:center;
                margin-bottom:5px;
                box-shadow: 0 3px 15px #3a6cf66c;
            }}
            .tl-step span {{
                display:block; color:#8386a6; font-size:0.92em; font-weight:400;
            }}
            .vizimg {{
                margin:10px 11px 12px 0;
                border: 2.5px solid #ccd9f7;
                box-shadow: 2px 8px 27px #d6e4fa66;
                border-radius:12px;
                width:420px;
                max-width:98%;
                vertical-align:middle;
                transition:box-shadow 0.2s, border 0.2s;
            }}
            .vizimg:hover {{
                box-shadow: 0 0 0 6px #9bc6fa44;
                border-color:#3a6cf6;
            }}
            .imgbox {{display:inline-block; margin-bottom:18px;}}
            .viz-grid {{
                display:flex;
                flex-wrap:wrap;
                gap:21px 17px;
                margin-bottom:24px;
            }}
            .viz-flex {{
                display:flex;
                flex-wrap:wrap;
                gap:29px 22px;
                margin-bottom:2.6em;
            }}
            @media (max-width:1000px) {{
                .container {{padding:0.9em;}}
                .vizimg, .viz-grid, .summary-cards, .viz-flex {{ width:100%; flex-direction:column; }}
                .vizimg {{width:100%;}}
                .summary-cards {{flex-direction:column; gap:12px;}}
            }}
            .button-toggle {{
                padding: 8px 27px;
                border-radius: 11px;
                border:none;
                background: #3a6cf6;
                color:white;
                font-weight:700;
                margin-bottom:2.2em;
                cursor:pointer;
                float:right;
                font-size:1.08em;
                box-shadow:0 3px 13px #3a6cf611;
            }}
            .button-toggle:hover {{background: #1b2478;}}
            .section-bg {{
                background:#f2f6fe;
                padding:2.2em 1.8em 0.5em 1.8em;
                border-radius:16px;
                margin-bottom:2.2em;
                box-shadow:0 3px 15px #c3d4f532;
            }}
            body.dark {{
                background: linear-gradient(120deg,#14192b 0%,#283869 100%);
                color:#e4ebfa;
            }}
            body.dark .container {{
                background:#202a43;
                color:#e4ebfa;
            }}
            body.dark h1,body.dark h2,body.dark h3 {{color:#aed1ff;}}
            body.dark nav.toc {{
                background:rgba(50,54,90,0.22);
                border:1px solid #384883;
            }}
            body.dark .info-table th,.perf-table th,.feat-table th,.env-table th{{background:#28396a;color:#cbe7ff;}}
            body.dark .info-table td,.perf-table td,.feat-table td,.env-table td{{background:#1b233c;}}
            body.dark .info-table tr:nth-child(even) td,
            body.dark .perf-table tr:nth-child(even) td,
            body.dark .feat-table tr:nth-child(even) td,
            body.dark .env-table tr:nth-child(even) td {{
                background: #223162;
            }}
            body.dark .card {{background:#18213a; border-left-color:#51d1e2;}}
            body.dark .card.summary {{background:#253364;}}
            body.dark .card.error {{background:#322035; border-left-color:#db4b5b;}}
            body.dark .vizimg {{border-color:#225088; box-shadow: 2px 6px 22px #25397a99;}}
            .section-divider {{
                height:0; border:0; border-top:2.6px solid #e0eaff; margin:3.8em 0 2.1em 0;
            }}
            body.dark .section-divider {{border-top-color: #39477a;}}
            ::selection {{background:#b1d0fd; color:#fff;}}
            body.dark ::selection {{background:#2c7aff; color:#fff;}}
            </style>
        </head>
        <body>
        <div class="container">
        <button onclick="document.body.classList.toggle('dark')" class="button-toggle">Toggle Light/Dark</button>
        {toc}
        {summary_card}
        {timeline}
        <section id='dataset'><h2>Dataset Information</h2>
            <div class='section-bg'>{self.dataset_table}</div>
        </section>
        <section id='env'><h2>Reproducibility & Environment</h2>
            <div class='section-bg'>{self.env_table}</div>
        </section>
        <section id='eda'>
            {self.eda_section}
        </section>
        <section id='harmonization'>
            {self.harmo_section}
        </section>
        <section id='crossval'>
            {self.cv_section}
        </section>
        <section id='testset'>
            {self.eval_section}
        </section>
        <section id='interpret'>
            {self.interp_section}
        </section>
        <h2 id='plots'>All Visualizations</h2>
        <div class='viz-grid'>{imgs_html}</div>
        <footer style='font-size:1.13em;margin-top:3.3em;text-align:right;color:#97a5ca;'>
            ENIGMA Pipeline Dashboard &mdash; generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </footer>
        </div>
        </body>
        </html>
        """

        path = os.path.join(self.out_dir, filename)
        with open(path, "w") as f:
            f.write(html_template)
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
