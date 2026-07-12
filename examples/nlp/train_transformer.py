"""Transformer sequence classification example.

Task: classify whether a sequence's first token index is even or odd.
This is a simple, learnable task that validates Transformer training works correctly.
"""
import orca
import orca.nn as nn
from orca.optim import Adam
from orca.optim.clip import clip_grad_norm_
from orca.data import Dataset, DataLoader
import numpy as np
import time


class SequenceDataset(Dataset):
    """Synthetic dataset where the label depends on the first token."""

    def __init__(self, num_samples, seq_len, vocab_size, num_classes):
        self.num_samples = num_samples

        # Random integers representing word indices
        self.X = np.random.randint(0, vocab_size, size=(num_samples, seq_len)).astype(np.int32)

        # Simple task: label = first_token % num_classes
        # This is learnable — the model just needs to attend to position 0.
        y = (self.X[:, 0] % num_classes).astype(np.int32)

        self.y = np.zeros((num_samples, num_classes))
        self.y[np.arange(num_samples), y] = 1.0

        # One-hot encode X
        self.X_oh = np.zeros((num_samples, seq_len, vocab_size), dtype=np.float32)
        for i in range(num_samples):
            self.X_oh[i, np.arange(seq_len), self.X[i]] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X_oh[idx].tolist(), self.y[idx].tolist()


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 1 layer, 2 heads
        self.transformer_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads=2, dim_feedforward=embed_dim * 2, dropout=0.0
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_len * embed_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.transformer_layer(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def main():
    print("Generating synthetic sequence dataset...")
    vocab_size = 10
    seq_len = 8
    num_classes = 2
    embed_dim = 16

    train_dataset = SequenceDataset(1000, seq_len, vocab_size, num_classes)
    test_dataset = SequenceDataset(200, seq_len, vocab_size, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = "gpu"
    print(f"Using device: {device}")

    model = TransformerClassifier(vocab_size, embed_dim, seq_len, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Standard Transformer LR: 1e-3 with small model, no warmup needed
    optimizer = Adam(model.parameters(), lr=0.001)

    epochs = 20
    print("Starting training Transformer...")

    for epoch in range(epochs):
        t0 = time.time()

        total_loss = 0
        batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.to_list()[0]
            batches += 1

        t1 = time.time()
        avg_train_loss = total_loss / batches

        # Test accuracy
        correct = 0
        total = 0
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            test_logits = model(X_batch)

            pred_list = test_logits.to_list()
            target_list = Y_batch.to_list()

            batch_size = X_batch.shape[0]

            for i in range(batch_size):
                start = i * num_classes
                end = start + num_classes
                pred_row = pred_list[start:end]
                target_row = target_list[start:end]
                if pred_row.index(max(pred_row)) == target_row.index(max(target_row)):
                    correct += 1
                total += 1

        acc = (correct / total) * 100
        print(
            f"Epoch {epoch + 1:2d}/{epochs} - {t1 - t0:.2f}s"
            f" - Loss: {avg_train_loss:.4f} - Test Acc: {acc:.2f}%"
        )


if __name__ == "__main__":
    main()
