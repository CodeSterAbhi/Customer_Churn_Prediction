import pandas as pd
import pytest
from churn_pipeline.predict import ChurnPredictor

def test_predictor_unloaded(tmp_path):
    predictor = ChurnPredictor(artifacts_dir=str(tmp_path))
    df = pd.DataFrame({'tenure': [10]})
    with pytest.raises(RuntimeError):
        predictor.predict(df)