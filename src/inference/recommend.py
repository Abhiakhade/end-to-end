import torch
import pickle
from src.models.matrix_factorization import MatrixFactorization

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


def recommend(user_id, top_k=5):
    model, user_enc, item_enc = load_artifacts()

    if user_id not in user_enc.classes_:
        return ["Popular items"]

    user_idx = user_enc.transform([user_id])[0]

    items = torch.arange(len(item_enc.classes_))
    users = torch.tensor([user_idx] * len(items))

    with torch.no_grad():
        scores = model(users, items)

    top_items = torch.topk(scores, top_k).indices.numpy()

    return item_enc.inverse_transform(top_items)