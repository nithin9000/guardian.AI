import os
import random
import shutil

source_dir = "/Users/nithin/documents/guardianai/dataset/heavy/train/ai_generated"
destination_dir = "/Users/nithin/documents/guardianai/dataset/data/train/fake"

os.makedirs(destination_dir,exist_ok=True)
all_images = os.listdir(source_dir)

selected_images = random.sample(all_images,5000)

for image in selected_images:
    source_path = os.path.join(source_dir, image)
    dest_path = os.path.join(destination_dir, image)
    shutil.move(source_path, dest_path)