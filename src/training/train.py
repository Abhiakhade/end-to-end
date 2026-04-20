import torch
from torch.utils.data import DataLoader
import os

from src.models.matrix_factorization import MatrixFactorization
from src.data.implicit_dataset import ImplicitDataset


def train_model(df, n_users, n_items, epochs=5, lr=0.001):
    print("Preparing implicit dataset (negative sampling)...")

    # Create dataset with negative samples
    dataset = ImplicitDataset(df, n_items, num_negatives=2)

    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Initializing model...")
    model = MatrixFactorization(n_users, n_items)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 🔥 Binary classification loss (important)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("Starting training...")

    for epoch in range(epochs):
        total_loss = 0

        for user, item, label in loader:
            pred = model(user, item)

            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
    os.makedirs("artifacts/models", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/models/model.pt")

    print("Model saved successfully!")

    return model