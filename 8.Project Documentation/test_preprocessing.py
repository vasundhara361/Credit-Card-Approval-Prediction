import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_clean_data, build_preprocessing_pipeline

def test_data_cleaning():
    # Setup test dummy frame with duplicates
    test_df = pd.DataFrame({
        'Age': [25, 25, 30],
        'Income': [50000, 50000, 60000]
    })
    
    # Write to temporary file space
    test_df.to_csv("tests/temp_test.csv", index=False)
    
    cleaned_df = load_and_clean_data("tests/temp_test.csv")
    
    # Assert duplicates are dropped properly
    assert len(cleaned_df) == 2
    
    import os
    os.remove("tests/temp_test.csv")

def test_pipeline_transform_shapes():
    num_cols = ['Age', 'Income']
    cat_cols = ['Prior_Default']
    
    df = pd.DataFrame({
        'Age': [23, 45, 60],
        'Income': [40000, np.nan, 80000],  # test median imputation triggers
        'Prior_Default': ['No', 'Yes', 'No']
    })
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    transformed_matrix = preprocessor.fit_transform(df)
    
    # Age(1) + Income(1) + OneHotEncoded Prior_Default(2 categories) = 4 Columns
    assert transformed_matrix.shape == (3, 4)