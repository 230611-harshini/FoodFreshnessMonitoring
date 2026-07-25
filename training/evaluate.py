import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
# Evaluation
# -----------------------------
all_labels = []
all_predictions = []

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

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total

print("\n==============================")
print("Evaluation Completed")
print("==============================")
print(f"Test Accuracy : {accuracy:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes
))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

plt.title("Food Freshness Confusion Matrix")
plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")

print("\n✅ Confusion Matrix saved to outputs/confusion_matrix.png")