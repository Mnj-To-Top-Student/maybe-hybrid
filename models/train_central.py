import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from dataset import ISIC2019Dataset
from hybrid_backbone import HybridModel

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset (CLIENT 0 ONLY)
# -------------------------
train_dataset = ISIC2019Dataset(
    client_id=0,
    train=True,
    data_path=r"D:\Capstone\CodeBase\PFLlib\dataset"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0  # Windows-safe
)

print("Train samples:", len(train_dataset))

val_dataset = ISIC2019Dataset(
    client_id=0,
    train=False,
    data_path=r"D:\Capstone\CodeBase\PFLlib\dataset"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Val samples:", len(val_dataset))


# -------------------------
# Model
# -------------------------
model = HybridModel(num_classes=8).to(device)

# -------------------------
# Class imbalance handling
# -------------------------
labels = train_dataset.labels
class_counts = np.bincount(labels, minlength=8)
class_counts[class_counts == 0] = 1

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# -------------------------
# Training loop
# -------------------------
epochs = 20
best_val_acc = 0.0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"           LR: {current_lr:.6f}")
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}]  Avg Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    print(f"           Val Acc: {val_acc:.2f}%")

    if epoch == 0 or val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "hybrid_client0_best.pth")

    scheduler.step()


