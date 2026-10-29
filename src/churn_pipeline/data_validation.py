import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    pass

def validate_raw_data(df: pd.DataFrame) -> bool:
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")
    
    if df.empty:
        raise DataValidationError("Dataset is empty.")
    
    if 'Churn' in df.columns:
        valid_churn_vals = {'Yes', 'No', 1, 0}
        unique_vals = set(df['Churn'].dropna().unique())
        if not unique_vals.issubset(valid_churn_vals):
            raise DataValidationError(f"Invalid target values found in 'Churn': {unique_vals}")
            
    logger.info("Data validation passed successfully.")
    return True