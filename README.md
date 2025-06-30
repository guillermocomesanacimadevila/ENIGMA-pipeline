# 🧠 ENIGMA-Pipeline

A reproducible, scalable, and modular machine learning pipeline for ENIGMA-like neuroimaging data, focusing on brain structure and working memory prediction. The pipeline performs data loading, exploratory data analysis (EDA), ComBat harmonisation (via neuroHarmonize), nested cross-validation, model training (Lasso, XGBoost, Random Forest), interpretation (SHAP), visualisation, and automated HTML reporting.

---

## 🚀 Features

- ⚡ **Flexible input:** accepts `.csv` files for sMRI features and working memory data
- 🧬 **ComBat harmonization:** batch/site correction using [neuroHarmonize](https://pypi.org/project/neuroHarmonize/)
- 🤖 **Multiple regression models:** Lasso, XGBoost, Random Forest (with nested CV and grid search)
- 🔍 **SHAP interpretability:** feature importance and visualizations
- 📊 **Auto EDA and reporting:** generates a comprehensive, mobile-friendly HTML report with plots, stats, and reproducibility log

---

## ⚙️ Installation & Environment

1. **Install [Anaconda](https://www.anaconda.com/products/distribution) (recommended), or Miniconda.**

2. **Clone the repository and run pipeline (step-by-step)**  
    ```bash
    git clone https://github.com/guillermocomesanacimadevila/ENIGMA-pipeline.git
    ```

   ```bash
    cd ENIGMA-pipeline
    ```

   ```bash
    chmod +x run_enigma_pipeline.sh && run_enigma_pipeline.sh
    ```

3. **RUN ALL-IN-ONE**
   ```bash
    git clone https://github.com/guillermocomesanacimadevila/ENIGMA-pipeline.git && cd ENIGMA-pipeline && chmod +x run_enigma_pipeline.sh && run_enigma_pipeline.sh
    ```

    This script will:
    - 🛠️ Create a new environment (`enigma-pipeline<N>`)
    - 📦 Install dependencies from conda and pip: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap, jinja2, neuroHarmonize, etc.
    - 🚦 Run the main pipeline.
