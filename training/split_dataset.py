import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset paths
CLEANED_DATASET = Path("dataset/cleaned")
SPLIT_DATASET = Path("dataset/split")

TRAIN_DIR = SPLIT_DATASET / "train"
VAL_DIR = SPLIT_DATASET / "val"
TEST_DIR = SPLIT_DATASET / "test"

# Create folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("Starting dataset split...\n")

# Process each class
for class_folder in CLEANED_DATASET.iterdir():

    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*"))

    random.shuffle(images)

    # 70% train, 30% remaining
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Split remaining into 15% validation and 15% test
    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    # Create class folders
    train_class = TRAIN_DIR / class_folder.name
    val_class = VAL_DIR / class_folder.name
    test_class = TEST_DIR / class_folder.name

    train_class.mkdir(parents=True, exist_ok=True)
    val_class.mkdir(parents=True, exist_ok=True)
    test_class.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy2(img, train_class / img.name)

    for img in val_images:
        shutil.copy2(img, val_class / img.name)

    for img in test_images:
        shutil.copy2(img, test_class / img.name)

    print(f"{class_folder.name}")
    print(f"  Train : {len(train_images)}")
    print(f"  Val   : {len(val_images)}")
    print(f"  Test  : {len(test_images)}\n")

print("✅ Dataset splitting completed successfully!")