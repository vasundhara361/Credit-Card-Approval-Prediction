import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from src.data_preprocessing import load_and_clean_data, get_feature_lists, build_preprocessing_pipeline

def train_model(data_path: str, target_col: str, model_save_path: str):
    print("✨ Starting Model Training Pipeline...")
    
    # 1. Load Data
    df = load_and_clean_data(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Setup Preprocessing
    num_cols, cat_cols = get_feature_lists(df, target_col)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # 4. Fit Preprocessor and transform data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 5. Initialize & Train XGBoost Model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("🏋️ Training XGBoost Classifier...")
    model.fit(X_train_proc, y_train)
    
    # 6. Evaluation
    preds = model.predict(X_test_proc)
    probs = model.predict_proba(X_test_proc)[:, 1]
    
    print("\n📊 Model Evaluation Metrics:")
    print(classification_report(y_test, preds))
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}\n")
    
    # 7. Package and Serialize the whole pipeline together
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    full_pipeline = {
        'preprocessor': preprocessor,
        'model': model,
        'features': {'numeric': num_cols, 'categorical': cat_cols}
    }
    
    joblib.dump(full_pipeline, model_save_path)
    print(f"💾 Production artifact successfully saved to {model_save_path}")

if __name__ == "__main__":
    train_model(
        data_path="data/raw/credit_applications.csv",
        target_col="Approved",
        model_save_path="models/credit_model_pipeline.pkl"
    )