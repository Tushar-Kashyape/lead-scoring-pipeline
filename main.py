from config import INPUT_DATA_PATH, OUTPUT_DATA_PATH
from src.preprocess import run_preprocessing

def main():
    df_clean = run_preprocessing(INPUT_DATA_PATH, OUTPUT_DATA_PATH)

if __name__ == '__main__':
    main()