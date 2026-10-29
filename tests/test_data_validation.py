import pandas as pd
import pytest
from churn_pipeline.data_validation import validate_raw_data, DataValidationError

def test_validate_raw_data_success():
    df = pd.DataFrame({
        'customerID': ['1234'], 'gender': ['Male'], 'SeniorCitizen': [0],
        'Partner': ['Yes'], 'Dependents': ['No'], 'tenure': [12],
        'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'InternetService': ['DSL'],
        'OnlineSecurity': ['No'], 'OnlineBackup': ['Yes'], 'DeviceProtection': ['No'],
        'TechSupport': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'], 'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Mailed check'], 'MonthlyCharges': [50.0],
        'TotalCharges': [600.0], 'Churn': ['No']
    })
    assert validate_raw_data(df) is True

def test_validate_raw_data_missing_column():
    df = pd.DataFrame({'customerID': ['1234']})
    with pytest.raises(DataValidationError):
        validate_raw_data(df)