# This script checks the dataset directory and counts the number of images in each class folder.
import os

dataset_path = r"C:\Users\mrsri\Desktop\micro-image\datasets" # Update this path to your dataset directory


print("Classes found:\n")# List all folders in the dataset directory and count the number of images in each folder

for folder in os.listdir(dataset_path):#
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))# Count the number of files in the folder
        print(folder, ":", count, "images")