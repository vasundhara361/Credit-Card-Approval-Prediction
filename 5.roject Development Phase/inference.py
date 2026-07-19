import joblib
import pandas as pd

def predict_approval(raw_json_data: dict, model_path: str = "models/credit_model_pipeline.pkl") -> dict:
    """Predicts credit card approval for a new incoming raw customer data dictionary."""
    # Load packaged artifacts
    pipeline = joblib.load(model_path)
    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([raw_json_data])
    
    # Pass through data pipeline
    processed_features = preprocessor.transform(input_df)
    
    # Run predictions
    prediction = int(model.predict(processed_features)[0])
    probability = float(model.predict_proba(processed_features)[0][1])
    
    return {
        "approved": prediction,
        "approval_probability": round(probability, 4)
    }

if __name__ == "__main__":
    # Simulate a new application coming from a web form
    new_applicant = {
        'Age': 34,
        'Income': 75000,
        'Credit_Score': 710,
        'Debt': 2300,
        'Prior_Default': 'No',
        'Employment_Status': 'Employed'
    }
    
    try:
        result = predict_approval(new_applicant)
        print("🔮 Inference Result:")
        print(result)
    except FileNotFoundError:
        print("❌ Error: Train the model first using train.py to generate the artifact package.")