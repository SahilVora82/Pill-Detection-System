#!/usr/bin/env python3
"""
pill_classifier_with_orb_hsv.py
-------------------------------
Frozen MobileNetV2 feature extractor + ORB bag-of-keypoints + HSV color stats + cosine kNN.

Pipeline:
1. Load pill images from table.csv + folders
2. Resize to IMG_WIDTH x IMG_HEIGHT
3. Extract:
   - MobileNetV2 global features
   - ORB keypoint histogram (local imprint features)
   - HSV color stats (mean/std for each channel)
4. Concatenate [MobileNet | ORB | HSV]
5. Normalize features
6. Train + evaluate cosine kNN classifier
7. Print misclassified samples
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import time  # ⬅️ added

# ---------------- CONFIG ----------------
IMG_WIDTH = 128
IMG_HEIGHT = 128
TEST_SIZE = 0.2
CSV_PATH = "table.csv"
IMAGE_FOLDERS = [
    "image/images/gallery/300",
    "image/images/gallery/600",
    "image/images/gallery/800",
    "image/images/gallery/1024"
]

# ORB settings
N_KEYPOINTS = 500
VOCAB_SIZE = 256

# ---------------- ORB HELPER ----------------
def build_orb_vocab(images):
    print("[INFO] Building ORB vocabulary...")
    orb = cv2.ORB_create(nfeatures=N_KEYPOINTS)
    all_desc = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None:
            all_desc.append(des)

    all_desc = np.vstack(all_desc)
    print(f"[INFO] ORB descriptors shape: {all_desc.shape}")

    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=10000, random_state=42)
    kmeans.fit(all_desc)
    return kmeans

def orb_histogram(img, orb, kmeans):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    hist = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if des is not None:
        idxs = kmeans.predict(des)
        for i in idxs:
            hist[i] += 1
        hist /= (np.linalg.norm(hist) + 1e-7)
    return hist

# ---------------- HSV FEATURES ----------------
def hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.array([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ])

# ---------------- DATA LOADER ----------------
def load_data(csv_path, image_dirs):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["nlmImageFileName", "name"])

    images, labels, paths = [], [], []
    label_counts = df["name"].value_counts()
    valid_names = label_counts[label_counts > 1].index
    label_mapping = {name: i for i, name in enumerate(valid_names)}

    print(f"[INFO] Number of classes: {len(label_mapping)}")

    for folder in image_dirs:
        df["filepath"] = df["nlmImageFileName"].apply(lambda x: os.path.join(folder, x))
        for _, row in df.iterrows():
            if row["name"] not in label_mapping:
                continue
            if not os.path.exists(row["filepath"]):
                continue

            image = cv2.imread(row["filepath"])
            if image is None:
                continue

            image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image_resized)
            labels.append(label_mapping[row["name"]])
            paths.append(row["filepath"])

    print(f"[INFO] Total samples: {len(images)}")
    return np.array(images), np.array(labels), label_mapping, paths

# ---------------- FEATURE EXTRACTOR ----------------
def build_feature_extractor(img_width, img_height):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_width, img_height, 3),
        include_top=False,
        weights="imagenet",
        pooling='max'
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        base_model
    ])
    return model

# ---------------- MAIN ----------------
def main():
    start = time.time()  # ⬅️ start timer

    print("[INFO] Loading data...")
    images, labels, label_mapping, paths = load_data(CSV_PATH, IMAGE_FOLDERS)

    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        images, labels, paths, test_size=TEST_SIZE, random_state=42, stratify=labels
    )

    # ORB Vocabulary
    orb = cv2.ORB_create(nfeatures=N_KEYPOINTS)
    orb_kmeans = build_orb_vocab(x_train)

    # MobileNet Features
    print("[INFO] Building frozen MobileNetV2 feature extractor...")
    extractor = build_feature_extractor(IMG_WIDTH, IMG_HEIGHT)

    print("[INFO] Extracting MobileNet features...")
    train_feats_cnn = extractor.predict(x_train, verbose=1)
    test_feats_cnn = extractor.predict(x_test, verbose=1)

    # ORB Histograms
    print("[INFO] Extracting ORB histograms...")
    train_feats_orb = np.array([orb_histogram(img, orb, orb_kmeans) for img in x_train])
    test_feats_orb = np.array([orb_histogram(img, orb, orb_kmeans) for img in x_test])

    # HSV Stats
    print("[INFO] Extracting HSV features...")
    train_feats_hsv = np.array([hsv_stats(img) for img in x_train])
    test_feats_hsv = np.array([hsv_stats(img) for img in x_test])

    # Fusion
    train_feats = np.hstack([train_feats_cnn, train_feats_orb, train_feats_hsv])
    test_feats = np.hstack([test_feats_cnn, test_feats_orb, test_feats_hsv])

    # Normalize
    train_feats = normalize(train_feats, axis=1)
    test_feats = normalize(test_feats, axis=1)

    # kNN Classifier
    print("[INFO] Training cosine kNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
    knn.fit(train_feats, y_train)

    print("[INFO] Evaluating...")
    preds = knn.predict(test_feats)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] MobileNetV2 + ORB + HSV fused kNN Accuracy: {acc:.4f}")

    # Misclassified samples
    print("\n[REPORT] Misclassified samples:")
    errors = 0
    inv_map = {v: k for k, v in label_mapping.items()}
    for true, pred, path in zip(y_test, preds, paths_test):
        if true != pred:
            print(f" - {os.path.basename(path)}: True={inv_map[true]}, Pred={inv_map[pred]}")
            errors += 1
    if errors == 0:
        print(" 🎉 No misclassifications!")

    end = time.time()  # ⬅️ end timer
    print(f"[INFO] Total runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
