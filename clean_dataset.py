import os
from PIL import Image
import imagehash

dataset_path = r"C:\Users\mrsri\Desktop\micro-image\datasets"

print("🔍 DATASET CLEANING REPORT\n")

duplicate_map = {}

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        print(f"\n📁 Class: {folder}")
        files = os.listdir(folder_path)

        total = 0
        bad = 0

        for file in files:
            file_path = os.path.join(folder_path, file)

            if not os.path.isfile(file_path):
                continue

            total += 1

            try:
                img = Image.open(file_path)
                width, height = img.size

                # Check tiny images
                if width < 100 or height < 100:
                    print("⚠ Small image:", file, f"({width}x{height})")

                # Check weird ratio
                ratio = width / height
                if ratio > 3 or ratio < 0.3:
                    print("⚠ Strange ratio:", file, f"({width}x{height})")

                # Duplicate check
                h = str(imagehash.average_hash(img))
                if h in duplicate_map:
                    print("⚠ Duplicate:", file, "==", duplicate_map[h])
                else:
                    duplicate_map[h] = file_path

            except:
                print("❌ Corrupted / Invalid:", file)
                bad += 1

        print("Total Files:", total)
        print("Problem Files:", bad)

print("\n✅ Scan Complete")