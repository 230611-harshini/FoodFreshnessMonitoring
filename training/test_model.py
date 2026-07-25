import torch
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path

# -----------------------------
# Dataset Paths
# -----------------------------
TEST_DIR = Path("dataset/split/test")
MODEL_PATH = "models/food_freshness_model.pth"

# -----------------------------
# Image Transformations
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Test Dataset
# -----------------------------
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Test Images :", len(test_dataset))
print("Classes :", test_dataset.classes)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", device)

# -----------------------------
# Load Model
# -----------------------------
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(test_dataset.classes)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model = model.to(device)
model.eval()

print("\nModel Loaded Successfully!")

# -----------------------------
# Testing
# -----------------------------
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print("\n==============================")
print("Testing Completed")
print("==============================")
print(f"Test Accuracy : {test_accuracy:.2f}%")