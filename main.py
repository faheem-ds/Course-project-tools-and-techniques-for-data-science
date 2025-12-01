
from src.load_data import load_dataset
from src.eda import run_eda, top_fare_features
from src.cleaning import wrangle_data
from src.constants import DATA_RAW_PATH

def main():

    # Step 1: Load dataset
    df = load_dataset(DATA_RAW_PATH)

    if df is None:
        print("[ERROR] Dataset could not be loaded. Exiting.")
        return
    
     # Step 2: Run full EDA (all steps inside eda.py)
    run_eda(df)

    top_features = top_fare_features(df, target='price', top_n=5)
    print(f"Top features selected for report: {top_features}")

    #Step 3: Data wrangling/cleansing
    df = wrangle_data(df)

if __name__ == "__main__":
    main()
