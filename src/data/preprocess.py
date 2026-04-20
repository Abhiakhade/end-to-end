import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder


# =========================
# Load MovieLens data
# =========================
def load_data(path):
    """
    Loads MovieLens 100K dataset (u.data format)
    """
    df = pd.read_csv(
        path,
        sep="\t",  # tab-separated file
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    return df


# =========================
# Preprocess data
# =========================
def preprocess(df, artifacts_path="artifacts/encoders"):
    """
    - Cleans data
    - Encodes user_id and item_id
    - Saves encoders
    """

    print("Preprocessing data...")

    # Create artifacts folder
    os.makedirs(artifacts_path, exist_ok=True)

    # Keep only required columns
    df = df[['user_id', 'item_id', 'rating']]

    # Remove duplicates
    df = df.drop_duplicates()

    # Initialize encoders
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    # Encode IDs
    df['user'] = user_enc.fit_transform(df['user_id'])
    df['item'] = item_enc.fit_transform(df['item_id'])

    # Save encoders
    with open(os.path.join(artifacts_path, "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_enc, f)

    with open(os.path.join(artifacts_path, "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_enc, f)

    print(f"Total Users: {len(user_enc.classes_)}")
    print(f"Total Items: {len(item_enc.classes_)}")

    return df, len(user_enc.classes_), len(item_enc.classes_)