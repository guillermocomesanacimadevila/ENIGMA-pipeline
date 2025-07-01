#!/bin/bash

# =========================================================
# ENIGMA Pipeline Runner
# Sets up Conda environment, installs dependencies, runs pipeline
# =========================================================

ENV_NAME="enigma-pipeline"
PYTHON_VERSION=3.10
SCRIPT_NAME="main.py"   # Change to your actual Python filename if needed
REPORT_PATH="Outputs/ENIGMA_pipeline_report.html"

set -e  # Stop if any error occurs

# 1. Check for conda
if ! command -v conda &> /dev/null
then
    echo -e "\033[0;31m[ERROR]\033[0m Conda could not be found! Please install Anaconda or Miniconda first."
    exit 1
fi

# 2. Create or update the environment if needed
if conda env list | grep -q "$ENV_NAME"; then
    echo -e "\033[0;34m[INFO]\033[0m Conda environment $ENV_NAME already exists. Skipping creation."
else
    echo -e "\033[0;34m[INFO]\033[0m Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# 3. Activate environment (compatible for scripts)
echo -e "\033[0;34m[INFO]\033[0m Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 4. Install dependencies (safe for rerun)
echo -e "\033[0;34m[INFO]\033[0m Installing Python packages via conda and pip..."
conda install -y numpy pandas matplotlib seaborn scikit-learn joblib jinja2 nibabel 
pip install --upgrade pip
pip install xgboost shap neuroHarmonize neuroCombat

# 5. Print environment details for reproducibility
echo "----------------------------------------------"
echo "Environment details:"
conda info
conda list
echo "----------------------------------------------"

# 6. Ensure Outputs and subdirectories exist
mkdir -p Outputs Outputs/EDA Outputs/Visualisations Outputs/Models Outputs/Predictions

# 7. Run the pipeline
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "\033[0;31m[ERROR]\033[0m Pipeline script $SCRIPT_NAME not found! Please check your script name/path."
    exit 2
fi

echo -e "\033[0;32m[RUNNING]\033[0m ENIGMA pipeline script: $SCRIPT_NAME"
python "$SCRIPT_NAME" | tee Outputs/pipeline_run.log

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "----------------------------------------------"
    echo -e "\033[0;32mPipeline completed successfully!\033[0m"
    echo "See Outputs/ENIGMA_pipeline_report.html for results."
    echo "----------------------------------------------"
    # --- OPEN REPORT IN DEFAULT BROWSER ---
    if [ -f "$REPORT_PATH" ]; then
        # For macOS:
        open "$REPORT_PATH"
        # For Linux (uncomment if needed):
        # xdg-open "$REPORT_PATH" >/dev/null 2>&1 &
        # For Windows Git Bash (uncomment if needed):
        # start "" "$REPORT_PATH"
    else
        echo -e "\033[0;31m[ERROR]\033[0m Report not found at $REPORT_PATH."
    fi
else
    echo -e "\033[0;31mPipeline failed.\033[0m See Outputs/pipeline_run.log for error details."
    exit 3
fi

# 8. Deactivate environment (optional for scripts)
conda deactivate
