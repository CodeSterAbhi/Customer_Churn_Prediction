import pandas as pd

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['TotalChargesCalculated'] = df['tenure'] * df['MonthlyCharges']
    return df