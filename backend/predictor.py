import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Class names (must match training order)
CLASS_NAMES = [
    "fresh_bread",
    "fresh_dairy",
    "fresh_fruits",
    "fresh_vegetables",
    "spoiled_bread",
    "spoiled_dairy",
    "spoiled_fruits",
    "spoiled_vegetables"
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load("models/food_freshness_model.pth", map_location=device)
)

model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()]