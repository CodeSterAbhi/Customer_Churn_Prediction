# Customer Churn MLOps Project

A production-style, reproducible machine learning pipeline and web application for predicting customer churn in the telecommunications industry. Built as an end-to-end portfolio project for Data Scientist and Machine Learning Engineer roles.

---

## 1. Project Overview & Business Problem
Customer churn is a critical metric for subscription-based businesses. Identifying high-risk customers allows companies to deploy proactive retention strategies. This project builds an automated, production-grade MLOps pipeline covering data validation, preprocessing, model comparison, Optuna hyperparameter tuning, MLflow tracking, testing, Docker containerization, and a Gradio user interface.

## 2. Dataset Description & Acquisition Instructions
* **Dataset**: Telco Customer Churn (7,043 rows, 21 columns).
* **Target Variable**: `Churn` (Yes/No).
* **Acquisition**: Download the dataset from Kaggle (Telco Customer Churn) and place the CSV file at:
  `data/raw/Telco-Customer-Churn.csv`

## 3. Architecture & Folder Structure
```
customer-churn-mlops/
    data/
        raw/
        processed/
    notebooks/
        churnAnalysis.ipynb
    src/
        churn_pipeline/
            __init__.py
            config.py
            data_validation.py
            data_preprocessing.py
            features.py
            train.py
            tune.py
            evaluate.py
            predict.py
            tracking.py
    configs/
        config.yaml
    tests/
        test_data_validation.py
        test_preprocessing.py
        test_training.py
        test_prediction.py
    reports/
    artifacts/
    app.py
    run_pipeline.py
    requirements.txt
    pyproject.toml
    Dockerfile
    .dockerignore
    .gitignore
    README.md
    .github/workflows/ci.yml
```

## 4. Methodology & Corrected Flaws
* **No Data Leakage**: Preprocessing pipelines are fitted exclusively on training data and applied without modification to validation and test sets.
* **Hyperparameter Tuning**: Optuna optimizes XGBoost hyperparameters strictly on validation/cross-validation splits; the test set remains untouched until final model evaluation.
* **Threshold Tuning**: Classification threshold is optimized on validation data to maximize F1-score rather than assuming a rigid 0.5 cutoff.

## 5. Local Setup & Installation (Windows PowerShell)
```powershell
# Clone or extract repository and navigate to root directory
cd customer-churn-mlops

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 6. Pipeline Execution
```powershell
# Run full training and evaluation pipeline
python run_pipeline.py
```

## 7. MLflow Tracking UI
```powershell
mlflow ui
```
Open `http://localhost:5000` in your browser.

## 8. Gradio User Interface
```powershell
python app.py
```
Open the local URL provided in the console to interact with the model.

## 9. Docker Instructions
```powershell
# Build Docker image
docker build -t churn-mlops-app .

# Run container
docker run -p 7860:7860 churn-mlops-app
```

## 10. Testing & CI
```powershell
# Run unit tests with pytest
pytest
```
GitHub Actions workflow runs automatically on push/PR.