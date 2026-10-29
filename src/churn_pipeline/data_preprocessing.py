import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, id_col: str = 'customerID'):
        self.id_col = id_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.id_col in X.columns:
            X = X.drop(columns=[self.id_col])
            
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
                
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = X['SeniorCitizen'].astype(int)
            
        return X

def get_preprocessing_pipeline() -> Pipeline:
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']),
            ('cat', categorical_transformer, ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'Contract', 'PaymentMethod', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'])
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline