from config import *

from src.evaluate import run_evaluate
from src.model import run_model
from src.preprocess import run_preprocessing
from src.features import run_feature_engineering

def main():
    df_clean = run_preprocessing(INPUT_DATA_PATH, OUTPUT_DATA_PATH)

    df_model = run_feature_engineering(df_clean, FEATURES_OUTPUT_PATH)

    model, test, pred = run_model(df_model, MODEL_OUTPUT_PATH)

    run_evaluate(model, test, pred, EVAL_OUTPUT_PATH)

if __name__ == '__main__':
    main()