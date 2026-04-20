import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=50):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)

        x = torch.cat([u, i], dim=1)
        return self.fc(x).squeeze()