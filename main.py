import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ── 1. Sozlamalar ─────────────────────────────────────────
DATASET_DIR = "dataset/Garbage classification"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
NUM_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Qurilma: {device}")

# ── 2. Transform (augmentation) ───────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 3. Dataset yuklash ────────────────────────────────────
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
print(f"Sinflar: {full_dataset.classes}")
print(f"Jami rasmlar: {len(full_dataset)}")

val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── 4. EfficientNet-B0 modeli ─────────────────────────────
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Oxirgi qatlamni o'zgartirish (6 sinfga)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
model = model.to(device)

# ── 5. Loss va Optimizer ──────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ── 6. O'qitish ───────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    train_acc = 100 * correct / total

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss    += loss.item()
            _, preds     = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100 * val_correct / val_total
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}%")

    # Eng yaxshi modelni saqlash
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✅ Model saqlandi (val_acc: {val_acc:.1f}%)")

print(f"\n🏁 O'qitish tugadi! Eng yaxshi: {best_val_acc:.1f}%")

# Sinf nomlarini saqlash
import json
with open("classes.json", "w") as f:
    json.dump(full_dataset.classes, f)
print("✅ classes.json saqlandi")