import tensorflow as tf # TensorFlow is the main library for building and training the model
from tensorflow.keras import layers # layers is used for building the model architecture
from tensorflow.keras.applications import MobileNetV2# MobileNetV2 is a pre-trained model that we will use as the base for our model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# EarlyStopping is used to stop training when the model stops improving, and ModelCheckpoint is used to save the best model during training
from tensorflow.keras import models# models is used to create the full model by stacking layers on top
import matplotlib.pyplot as plt# matplotlib is used for plotting the training history (accuracy and loss) after training the model

# Set dataset path
DATASET_PATH = r"C:\Users\mrsri\Desktop\micro-image\datasets"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Base model
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Full model
model = models.Sequential([
    data_aug,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(8, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("micro_model.h5", save_best_only=True)
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

# Save final model
model.save("micro_model_final.h5")

# Plot Accuracy
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.show()