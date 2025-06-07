import joblib
import pandas as pd

# Load column order with duplicates (same as training)
column_order = joblib.load('model/column_order.pkl')

def preprocess_data(df):
    # Drop label column if present
    df = df.drop(columns=['label'], errors='ignore')

    # Ensure all required columns are present (including duplicates)
    for col in column_order:
        if col not in df.columns:
            df[col] = 0  # Fill missing

    # Reorder to match exact column order with duplicates
    df = df.reindex(columns=column_order)

    return df
