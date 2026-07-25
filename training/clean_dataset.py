import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Dataset paths
RAW_DATASET = Path("dataset/raw")
CLEANED_DATASET = Path("dataset/cleaned")

# Create cleaned dataset folder if it doesn't exist
CLEANED_DATASET.mkdir(parents=True, exist_ok=True)

# Statistics
total_images = 0
valid_images = 0
corrupted_images = 0

# Process each class folder
for class_folder in RAW_DATASET.iterdir():

    # Skip if it is not a folder
    if not class_folder.is_dir():
        continue

    print(f"\nProcessing: {class_folder.name}")

    # Create corresponding folder in cleaned dataset
    output_folder = CLEANED_DATASET / class_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in tqdm(list(class_folder.iterdir()), desc=class_folder.name):

        total_images += 1

        try:
            # Open image
            image = Image.open(image_path)

            # Check whether image is corrupted
            image.verify()

            # Reopen image and convert to RGB
            image = Image.open(image_path).convert("RGB")

            # Copy valid image to cleaned dataset
            destination = output_folder / image_path.name
            shutil.copy2(image_path, destination)

            valid_images += 1

        except Exception:
            corrupted_images += 1
            continue

# Final Report
print("\n========== Dataset Cleaning Report ==========")
print(f"Total Images      : {total_images}")
print(f"Valid Images      : {valid_images}")
print(f"Corrupted Images  : {corrupted_images}")

print("\n✅ Cleaned dataset saved successfully!")