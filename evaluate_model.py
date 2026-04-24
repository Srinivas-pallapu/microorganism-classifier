import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("micro_model.h5")

# -------------------------
# Dataset Path
# -------------------------
DATASET_PATH = r"C:\Users\mrsri\Desktop\micro-image\datasets"
IMG_SIZE = (224, 224)

class_names = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Classes:", class_names)

# -------------------------
# Collect image paths and labels
# -------------------------
image_paths = []
labels = []

for label_index, class_name in enumerate(class_names):
    class_path = os.path.join(DATASET_PATH, class_name)

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_paths.append(os.path.join(class_path, file))
            labels.append(label_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

# -------------------------
# Stratified Validation Split
# -------------------------
_, val_paths, _, y_true = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=123,
    stratify=labels
)

# -------------------------
# Preprocess Images
# IMPORTANT: Do NOT normalize here
# because model already has Rescaling layer
# -------------------------
X_val = []

for path in val_paths:
    img = Image.open(path).convert("RGB")
    img = ImageOps.pad(img, IMG_SIZE)
    img_array = np.array(img)
    X_val.append(img_array)

X_val = np.array(X_val)

# -------------------------
# Predict
# -------------------------
y_pred_probs = model.predict(X_val, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------
# Accuracy
# -------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()