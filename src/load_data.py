import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the Uber/Lyft dataset from CSV, checks for missing values and duplicates,
    and prints a brief summary.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded successfully with shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        print("\n[INFO] First 5 rows:")
        print(df.head())
        
        # Basic info
        print("\n[INFO] Data types and non-null counts:")
        print(df.info())
        
        # Summary statistics
        print("\n[INFO] Summary statistics for numeric columns:")
        print(df.describe())
        
        # Missing values
        missing = df.isna().sum()
        print("\n[INFO] Missing values per column:")
        print(missing)
        
        # Duplicates
        num_duplicates = df.duplicated().sum()
        print(f"\n[INFO] Number of duplicate rows: {num_duplicates}")
        
        return df
    
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None
