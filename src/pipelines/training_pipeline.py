from src.data.preprocess import load_data, preprocess
from src.training.train import train_model
from src.data.split import train_test_split
import os


def run_training():
    print("Loading data...")
    df = load_data("data/raw/ratings.csv")

    print("Preprocessing...")
    df, n_users, n_items = preprocess(df)

    print("Splitting data...")
    train_df, test_df = train_test_split(df)

    # ✅ Save ONLY encoded columns (IMPORTANT FIX)
    os.makedirs("data/splits", exist_ok=True)

    train_df[['user', 'item', 'rating']].to_csv(
        "data/splits/train.csv", index=False
    )

    test_df[['user', 'item', 'rating']].to_csv(
        "data/splits/test.csv", index=False
    )

    print("Training model...")
    train_model(train_df, n_users, n_items)

    print("Training completed!")


if __name__ == "__main__":
    run_training()