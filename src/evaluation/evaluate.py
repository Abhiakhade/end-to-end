import torch
import pandas as pd
import pickle
import numpy as np

from src.models.matrix_factorization import MatrixFactorization


# =========================
# Load trained artifacts
# =========================
def load_artifacts():
    with open("artifacts/encoders/user_encoder.pkl", "rb") as f:
        user_enc = pickle.load(f)

    with open("artifacts/encoders/item_encoder.pkl", "rb") as f:
        item_enc = pickle.load(f)

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    model = MatrixFactorization(n_users, n_items)
    model.load_state_dict(torch.load("artifacts/models/model.pt"))
    model.eval()

    return model, user_enc, item_enc


# =========================
# Metrics
# =========================
def precision_at_k(recommended, actual, k):
    return len(set(recommended[:k]) & set(actual)) / k


def recall_at_k(recommended, actual, k):
    return len(set(recommended[:k]) & set(actual)) / len(actual)


def ndcg_at_k(recommended, actual, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in actual:
            dcg += 1 / np.log2(i + 2)

    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0


# =========================
# Recommendation (exclude seen)
# =========================
def get_top_k(model, user_idx, n_items, k, seen_items):
    items = torch.arange(n_items)
    users = torch.tensor([user_idx] * n_items)

    with torch.no_grad():
        scores = model(users, items)

    # ❗ Remove already seen items
    scores[seen_items] = -1e9

    top_items = torch.topk(scores, k).indices.numpy()
    return top_items


# =========================
# Evaluation Pipeline
# =========================
def evaluate(k=5):
    print("Loading train/test splits...")

    # ✅ Use SAME split as training
    train_df = pd.read_csv("data/splits/train.csv")
    test_df = pd.read_csv("data/splits/test.csv")

    print("Loading model & encoders...")
    model, user_enc, item_enc = load_artifacts()

    # Filter known users/items
# Data is already encoded → no transformation needed
# Just ensure correct columns exist

    assert 'user' in train_df.columns
    assert 'item' in train_df.columns
    assert 'user' in test_df.columns
    assert 'item' in test_df.columns

    # Group test data
    test_groups = test_df.groupby('user')

    precisions = []
    recalls = []
    ndcgs = []

    print("Evaluating...")

    for user, group in test_groups:
        actual_items = group['item'].tolist()

        # Seen items from training
        seen_items = train_df[train_df['user'] == user]['item'].tolist()

        recommended_items = get_top_k(
            model,
            user_idx=user,
            n_items=len(item_enc.classes_),
            k=k,
            seen_items=seen_items
        )

        p = precision_at_k(recommended_items, actual_items, k)
        r = recall_at_k(recommended_items, actual_items, k)
        n = ndcg_at_k(recommended_items, actual_items, k)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    print("\n===== Final Evaluation (No Leakage) =====")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"NDCG@{k}:      {np.mean(ndcgs):.4f}")
    print("=========================================")

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# =========================
# Run
# =========================
if __name__ == "__main__":
    evaluate()