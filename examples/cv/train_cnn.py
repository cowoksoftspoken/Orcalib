import orca
import orca.nn as nn
from orca.optim import SGD, Adam
from orca.data import Dataset, DataLoader
from sklearn.datasets import load_digits
import numpy as np
import time

class DigitsDataset(Dataset):
    def __init__(self, X, y):
        # X is (N, 1, 8, 8)
        # y is (N,)
        self.X = X
        # One-hot encode y (10 classes)
        self.y = np.zeros((len(y), 10))
        self.y[np.arange(len(y)), y] = 1.0
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx].tolist(), self.y[idx].tolist()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 1, 8, 8]
        # Output after Conv2d (3x3 kernel, pad 1): [B, 4, 8, 8]
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Output after Flatten: [B, 4 * 8 * 8] = [B, 256]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Loading digits dataset...")
    digits = load_digits()
    X = digits.images.astype(np.float32)
    # Reshape to [N, 1, 8, 8]
    X = X.reshape(X.shape[0], 1, 8, 8) / 16.0 
    y = digits.target
    
    # Split into train and test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    train_dataset = DigitsDataset(X_train, y_train)
    test_dataset = DigitsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = "gpu"
    print(f"Using device: {device}")
    
    model = SimpleCNN()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print("Starting training CNN...")
    
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
            optimizer.step()
            
            total_loss += loss.to_list()[0]
            batches += 1
            
        t1 = time.time()
        avg_train_loss = total_loss / batches
        
        # Calculate test accuracy every epoch
        correct = 0
        total = 0
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            test_logits = model(X_batch)
            
            pred_list = test_logits.to_list()
            target_list = Y_batch.to_list()
            
            batch_size = X_batch.shape[0]
            classes = 10
            
            for i in range(batch_size):
                start = i * classes
                end = start + classes
                pred_row = pred_list[start:end]
                target_row = target_list[start:end]
                if pred_row.index(max(pred_row)) == target_row.index(max(target_row)):
                    correct += 1
                total += 1
                
        acc = (correct / total) * 100
        
        print(f"Epoch {epoch+1:2d}/{epochs} - {t1-t0:.2f}s - Loss: {avg_train_loss:.4f} - Test Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
