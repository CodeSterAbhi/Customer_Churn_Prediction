import pandas as pd
import numpy as np
from churn_pipeline.data_preprocessing import get_preprocessing_pipeline

def test_preprocessing_pipeline():
    df = pd.DataFrame({
        'customerID': ['1234'], 'gender': ['Male'], 'SeniorCitizen': [0],
        'Partner': ['Yes'], 'Dependents': ['No'], 'tenure': [12],
        'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'InternetService': ['DSL'],
        'OnlineSecurity': ['No'], 'OnlineBackup': ['Yes'], 'DeviceProtection': ['No'],
        'TechSupport': ['No'], 'StreamingTV': ['No'], 'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'], 'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Mailed check'], 'MonthlyCharges': [50.0],
        'TotalCharges': ['600.0'], 'Churn': ['No']
    })
    pipeline = get_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1