
from src.load_data import load_dataset
from src.constants import DATA_RAW_PATH

def main():

    df = load_dataset(DATA_RAW_PATH)

    if df is None:
        print("[ERROR] Dataset could not be loaded. Exiting.")
        return

if __name__ == "__main__":
    main()
