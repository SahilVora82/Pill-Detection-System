import os
import pandas as pd
import numpy as np
import cv2
import shutil
import string
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIG ---
OCR_CSV = "ocr_results.csv"
IMPRINTS_DIR = "imprints"
MIN_SAMPLES_PER_CLASS = 5
IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "ocr_class_data"

# --- UTILS ---
def normalize_ocr(text):
    text = str(text).upper().strip()
    text = text.replace(" ", "")
    return text.translate(str.maketrans("", "", string.punctuation))

# --- STEP 1: Load and normalize OCRs ---
df = pd.read_csv(OCR_CSV)
df["norm_ocr"] = df["ocr_text"].apply(normalize_ocr)

# --- STEP 2: Filter OCR strings by frequency ---
value_counts = df["norm_ocr"].value_counts()
valid_classes = value_counts[value_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
df_filtered = df[df["norm_ocr"].isin(valid_classes)].copy()

# --- STEP 3: Assign class indices and save crops into folders ---
class_to_index = {cls: i for i, cls in enumerate(sorted(valid_classes))}
df_filtered["class_index"] = df_filtered["norm_ocr"].map(class_to_index)

# Clean up and recreate data directory
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR)

print(f"[INFO] Using {len(class_to_index)} OCR pseudo-classes with ≥ {MIN_SAMPLES_PER_CLASS} samples")

# Copy images into class folders
for _, row in df_filtered.iterrows():
    folder = row["folder"]
    filename = row["filename"]
    class_idx = row["class_index"]
    src = os.path.join(IMPRINTS_DIR, str(folder), filename)
    dst_folder = os.path.join(DATA_DIR, str(class_idx))
    os.makedirs(dst_folder, exist_ok=True)
    dst_path = os.path.join(dst_folder, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst_path)

# --- STEP 4: Set up image generators ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# --- STEP 5: Build model ---
base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    Rescaling(1./255),
    base_model,
    Dense(128, activation='relu'),
    Dense(len(class_to_index), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- STEP 6: Train model ---
print(f"[INFO] Starting training on {train_gen.samples} samples...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=2
)

# --- STEP 7: Save model ---
model.save("ocr_imprint_classifier.h5")
print("[INFO] Model saved as ocr_imprint_classifier.h5")
