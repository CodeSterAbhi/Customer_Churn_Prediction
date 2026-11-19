import gradio as gr
import pandas as pd
from churn_pipeline.predict import ChurnPredictor

predictor = ChurnPredictor(artifacts_dir="artifacts")

def predict_churn(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):
    input_data = pd.DataFrame([{
        'customerID': 'TEST-9999',
        'gender': gender,
        'SeniorCitizen': int(senior_citizen),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': str(total_charges)
    }])
    
    try:
        result = predictor.predict(input_data)
        status = "Churn (Likely to leave)" if result['prediction'] == 1 else "Not Churn (Likely to stay)"
        prob = result['probability']
        return f"Prediction: {status}\nProbability of Churn: {prob:.2%}\n(Note: This is a model prediction, not a guarantee.)"
    except Exception as e:
        return f"Error generating prediction: {str(e)}. Please ensure model artifacts are trained and placed in 'artifacts/'."

demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Checkbox(label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Slider(0, 72, step=1, label="Tenure (Months)", value=12),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check"),
        gr.Number(label="Monthly Charges", value=70.0),
        gr.Number(label="Total Charges", value=840.0)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Customer Churn Prediction MLOps App",
    description="Predict whether a telecom customer is likely to churn using an optimized XGBoost MLOps pipeline."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
