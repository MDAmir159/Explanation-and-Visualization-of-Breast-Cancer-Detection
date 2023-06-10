import os
import random
import shutil

# Path to the main folder
main_folder = "E:\\thesis4000\\preprocessed_dataset_images"

# Path to the new smaller dataset folder
smaller_dataset_folder = "E:\\thesis4000\\preprocessed_random_5000_75px"

# Number of images to randomly select
num_images = 5000

if not os.path.exists(smaller_dataset_folder):
    os.makedirs(smaller_dataset_folder)

# Randomly select images from the "0" folder
source_folder_0 = os.path.join(main_folder, "0")
destination_folder_0 = os.path.join(smaller_dataset_folder, "0")
if not os.path.exists(destination_folder_0):
    os.makedirs(destination_folder_0)
image_files_0 = random.sample(os.listdir(source_folder_0), num_images)
for file_name in image_files_0:
    source_path = os.path.join(source_folder_0, file_name)
    destination_path = os.path.join(destination_folder_0, file_name)
    shutil.copyfile(source_path, destination_path)

# Randomly select images from the "1" folder
source_folder_1 = os.path.join(main_folder, "1")
destination_folder_1 = os.path.join(smaller_dataset_folder, "1")
if not os.path.exists(destination_folder_1):
    os.makedirs(destination_folder_1)
image_files_1 = random.sample(os.listdir(source_folder_1), num_images)
for file_name in image_files_1:
    source_path = os.path.join(source_folder_1, file_name)
    destination_path = os.path.join(destination_folder_1, file_name)
    shutil.copyfile(source_path, destination_path)

print("Smaller dataset created successfully.")