import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, k=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)

        return (u * i).sum(1)  # logits