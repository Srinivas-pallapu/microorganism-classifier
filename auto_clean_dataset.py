import os
import shutil
from PIL import Image
import imagehash

dataset_path = r"C:\Users\mrsri\Desktop\micro-image\datasets"
review_path = r"C:\Users\mrsri\Desktop\micro-image\review_duplicates"

os.makedirs(review_path, exist_ok=True)

hash_map = {}

print("Starting duplicate cleaning...\n")

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            img = Image.open(file_path)
            h = str(imagehash.average_hash(img))

            if h in hash_map:
                old_path, old_folder = hash_map[h]

                # Same class duplicate -> delete current
                if old_folder == folder:
                    os.remove(file_path)
                    print(f"Deleted same-class duplicate: {file_path}")

                # Cross-class duplicate -> move current to review
                else:
                    target = os.path.join(review_path, folder + "_" + file)
                    shutil.move(file_path, target)
                    print(f"Moved cross-class duplicate to review: {file_path}")

            else:
                hash_map[h] = (file_path, folder)

        except Exception as e:
            print("Error:", file_path)

print("\nCleaning complete.")