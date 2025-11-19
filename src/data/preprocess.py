import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Paths to your data - handle both repo structure and parent directory
# First check parent directory (current setup), then check repo data/ directory
if os.path.exists("../data/Training_BOP.csv"):
    TRAIN_PATH = "../data/Training_BOP.csv"
    TEST_PATH = "../data/Testing_BOP.csv"
else:
    TRAIN_PATH = "data/Training_BOP.csv"
    TEST_PATH = "data/Testing_BOP.csv"

# Columns to encode
CATEGORICAL_COLS = [
    'sku', 'potential_issue', 'deck_risk', 'oe_constraint',
    'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder'
]

def load_data():
    """Load train and test CSVs"""
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    print(f"✅ Training Data: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    print(f"✅ Testing Data: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
    return train_df, test_df

def preprocess_data(train_df, test_df, use_rfe=True, n_features=10):
    """Preprocess data: handle missing values, encode categorical features, feature selection"""
    
    print("🔧 Starting preprocessing...")

    # --- 1. Handle missing values ---
    missing_before = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    missing_after = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
    print(f"⚠️ Dropping {missing_before - missing_after} missing values...")

    # --- 2. Separate features and target BEFORE encoding ---
    target_col = 'went_on_backorder'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # Encode target separately (Yes/No -> 1/0)
    y_train = y_train.map({'Yes': 1, 'No': 0})
    
    # Remove target from test if it exists
    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
    else:
        X_test = test_df.copy()
    
    # --- 3. Encode categorical columns (excluding target) ---
    categorical_features = [col for col in CATEGORICAL_COLS if col != target_col]
    print(f"🧩 Encoding categorical columns: {categorical_features}")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_features] = encoder.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = encoder.transform(X_test[categorical_features])
    
    # --- 4. Feature selection using RFE ---
    if use_rfe:
        print(f"⚙️ Performing RFE to select top {n_features} features...")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_train, y_train)
        X_train = X_train.loc[:, selector.support_]
        X_test = X_test.loc[:, selector.support_]
        print(f"✅ Selected top {n_features} features: {list(X_train.columns)}")
    
    print("✅ Preprocessing complete.")
    return X_train, y_train, X_test

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test = preprocess_data(train_df, test_df, use_rfe=True, n_features=10)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
