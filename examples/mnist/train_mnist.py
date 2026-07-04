import orca
import orca.nn as nn
import orca.optim as optim
from orca.data import Dataset, DataLoader
import time
import os
from sklearn.datasets import load_digits
import numpy as np

# A simple Dataset for sklearn digits
class DigitsDataset(Dataset):
    def __init__(self, X, y):
        # X is (N, 64)
        # y is (N,)
        self.X = X
        
        # One-hot encode y (10 classes)
        self.y = np.zeros((len(y), 10))
        self.y[np.arange(len(y)), y] = 1.0
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx].tolist(), self.y[idx].tolist()

def main():
    print("Loading digits dataset...")
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1)) / 16.0 # Normalize 0-1
    y = digits.target
    
    # Split into train and test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    train_dataset = DigitsDataset(X_train, y_train)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    
    device = "gpu" if os.environ.get("USE_CPU") != "1" else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 20
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print("Starting training...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training loop
        total_loss = 0
        batches = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.to_list()[0]
            batches += 1
            
        avg_train_loss = total_loss / batches
        
        # Evaluation loop (computing accuracy)
        correct = 0
        total = 0
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            
            # to_list() returns a flat list
            pred_list = y_pred.to_list()
            target_list = Y_batch.to_list()
            
            batch_size = X_batch.shape[0]
            classes = Y_batch.shape[1]
            
            for i in range(batch_size):
                start = i * classes
                end = start + classes
                pred_row = pred_list[start:end]
                target_row = target_list[start:end]
                
                pred_idx = pred_row.index(max(pred_row))
                target_idx = target_row.index(max(target_row))
                
                if pred_idx == target_idx:
                    correct += 1
                total += 1
                
        accuracy = correct / total * 100.0
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch + 1:2d}/{epochs} - {elapsed:.2f}s - Loss: {avg_train_loss:.4f} - Test Acc: {accuracy:.2f}%")

    print("Saving model weights...")
    model.save_weights("mnist_model.safetensors")
    print("Model weights saved to mnist_model.safetensors")
    
    # Test Loading
    model2 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    model2.to(device)
    model2.load_weights("mnist_model.safetensors")
    
    # Evaluate model2 on a batch to confirm it works
    print("Evaluating loaded model...")
    correct = 0
    total = 0
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model2(X_batch)
        pred_list = y_pred.to_list()
        target_list = Y_batch.to_list()
        batch_size = X_batch.shape[0]
        classes = Y_batch.shape[1]
        for i in range(batch_size):
            start = i * classes
            end = start + classes
            pred_row = pred_list[start:end]
            target_row = target_list[start:end]
            if pred_row.index(max(pred_row)) == target_row.index(max(target_row)):
                correct += 1
            total += 1
    
    print(f"Loaded Model Test Acc: {correct/total*100.0:.2f}%")


if __name__ == "__main__":
    main()
