# import os
# from pathlib import Path
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms

# BATCH_SIZE = 32
# EPOCHS = 10
# LR = 0.001
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #Data augmentation and preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# data_dir = Path("data/gtsrb/GTSRB/Training")

# if not data_dir.exists():
#   print(
#       'Please ensure that the "data/gtsrb/GTSRB/Training" directory contains the'
#       " training set images (stored in subfolders according to categories)"
#   )

# train_dataset = datasets.ImageFolder(data_dir, transform=transform)
# train_loader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
# )

# #Load the model and modify the classification head
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 43)  
# model = model.to(DEVICE)

# #Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# #Training epoch
# best_acc = 0.0
# print("Start training")
# for epoch in range(EPOCHS):
#   model.train()
#   running_loss = 0.0
#   correct = 0
#   total = 0

#   for inputs, labels in train_loader:
#     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     running_loss += loss.item() * inputs.size(0)
#     _, predicted = outputs.max(1)
#     total += labels.size(0)
#     correct += predicted.eq(labels).sum().item()

#   epoch_loss = running_loss / total
#   epoch_acc = correct / total
#   print(
#       f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy:"
#       f" {epoch_acc:.4f}"
#   )


# os.makedirs("models", exist_ok=True)
# torch.save(model.state_dict(), "models/best_model.pth")
# print("Training completed! Model weights have been saved to models/best_model.pth")

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ----------------- Hyperparameters & Config -----------------
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/gtsrb/GTSRB/Training")
MODEL_SAVE_PATH = Path("models/best_model.pth")


def get_data_loaders(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Please ensure the GTSRB dataset is present."
        )

    # Preprocessing with data augmentation for robust real-world detection
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(str(data_dir))
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size

    train_data, val_data = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Assign correct transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_data_loaders(DATA_DIR)

    # ResNet-18 modified for 43 GTSRB classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 43)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        # ----------------- Training -----------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save model only when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved new best model with Validation Accuracy: {best_val_acc:.4f}")

    print(f"\nTraining completed. Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()