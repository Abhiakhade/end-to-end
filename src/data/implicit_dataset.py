import torch
from torch.utils.data import Dataset
import random


class ImplicitDataset(Dataset):
    def __init__(self, df, n_items, num_negatives=2):
        self.users = df['user'].values
        self.items = df['item'].values
        self.n_items = n_items
        self.num_negatives = num_negatives

        # Create user -> positive items map
        self.user_item_set = {}
        for u, i in zip(self.users, self.items):
            self.user_item_set.setdefault(u, set()).add(i)

        self.data = self._generate()

    def _generate(self):
        data = []

        for u, i in zip(self.users, self.items):
            # Positive sample
            data.append((u, i, 1))

            # Negative samples
            for _ in range(self.num_negatives):
                neg_item = random.randint(0, self.n_items - 1)

                while neg_item in self.user_item_set[u]:
                    neg_item = random.randint(0, self.n_items - 1)

                data.append((u, neg_item, 0))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, label = self.data[idx]
        return (
            torch.tensor(u),
            torch.tensor(i),
            torch.tensor(label, dtype=torch.float32)
        )