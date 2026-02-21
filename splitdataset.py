import os
import shutil
import random

# Change this path to where extracted folder is
source_base = "real_and_fake_face"

real_source = os.path.join(source_base, "real")
fake_source = os.path.join(source_base, "fake")

# Create required structure
folders = [
    "dataset/train/real",
    "dataset/train/fake",
    "dataset/val/real",
    "dataset/val/fake"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

def split_data(source, train_dest, val_dest, split_ratio=0.8):
    images = os.listdir(source)
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(source, img), train_dest)

    for img in val_images:
        shutil.copy(os.path.join(source, img), val_dest)

split_data(real_source, "dataset/train/real", "dataset/val/real")
split_data(fake_source, "dataset/train/fake", "dataset/val/fake")

print("Dataset Split Completed ✅")
