import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads dataset and performs initial clean up."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    return df

def get_feature_lists(df: pd.DataFrame, target_col: str):
    """Splits columns into numerical and categorical lists based on data types."""
    X = df.drop(columns=[target_col])
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_features, cat_features

def build_preprocessing_pipeline(num_features, cat_features) -> ColumnTransformer:
    """Creates a robust preprocessor for numerical scaling and categorical encoding."""
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # Generate mock raw data for demonstration if user runs this directly
    import os
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Simulating a typical credit application dataset
    np.random.seed(42)
    n_samples = 1000
    mock_data = pd.DataFrame({
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.normal(55000, 20000, n_samples).clip(10000, 200000),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Debt': np.random.normal(5000, 3000, n_samples).clip(0, 50000),
        'Prior_Default': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Employment_Status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples),
        'Approved': np.random.choice([1, 0], n_samples, p=[0.4, 0.6])
    })
    
    raw_path = "data/raw/credit_applications.csv"
    mock_data.to_csv(raw_path, index=False)
    print(f"Mock raw data generated at: {raw_path}")