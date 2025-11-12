import pandas as pd

# Paths to your data
TRAIN_PATH = "data/Training_BOP.csv"
TEST_PATH = "data/Testing_BOP.csv"

def load_train_test():
    """
    Load training and testing CSV files with safe options:
    - low_memory=False to avoid DtypeWarning
    - handles mixed types
    Returns: train_df, test_df
    """
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    print(f"✅ Training Data: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    print(f"✅ Testing Data: {test_df.shape[0]} rows × {test_df.shape[1]} columns")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_train_test()
