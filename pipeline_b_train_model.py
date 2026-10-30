# pipeline_b_train_model.py
import pandas as pd

from config import PARQUET_CLUSTERED
from src.classification import (
    apply_cluster_labels, split_data, fit_encoders, encode_features,
    encode_target, train_random_forest, evaluate_model, save_artifacts,
)


def run_pipeline():
    df = pd.read_parquet(PARQUET_CLUSTERED)
    df = apply_cluster_labels(df)

    train_df, test_df = split_data(df)
    encoders = fit_encoders(train_df)

    X_train, y_train = encode_features(train_df, encoders), encode_target(train_df, encoders)
    X_test, y_test = encode_features(test_df, encoders), encode_target(test_df, encoders)

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test, encoders)
    save_artifacts(model, encoders)
    print("Script B selesai. Model dan encoder tersimpan di folder models/.")


if __name__ == "__main__":
    run_pipeline()