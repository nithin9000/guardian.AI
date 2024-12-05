import os
import random
import shutil

source_dir = "/Users/nithin/documents/guardianai/dataset/resized"
destination_dir = "/Users/nithin/documents/guardianai/dataset/val"

os.makedirs(destination_dir,exist_ok=True)
all_images = os.listdir(source_dir)

selected_images = random.sample(all_images,20000)

for image in selected_images:
    source_path = os.path.join(source_dir, image)
    dest_path = os.path.join(destination_dir, image)
    shutil.move(source_path, dest_path)