#!/usr/bin/env python3
"""
pill_classifier_extracted_orb_hsv_alpha.py
------------------------------------------
Frozen MobileNetV2 + ORB bag-of-keypoints + HSV color stats (pill-only via alpha channel) + cosine kNN.

Pipeline:
1. Walk through extracted_images/<class_name>/*.png
  - Ignores "edge_masks" subfolder
  - Uses alpha channel as mask (transparent background)
2. Resize to target size
3. Extract:
  - MobileNetV2 global features
  - ORB keypoint histogram (local imprint features)
  - HSV color stats (mean/std inside pill alpha mask only)
4. Concatenate [MobileNet | ORB | HSV]
5. Normalize features
6. Train + evaluate cosine kNN classifier
7. Print misclassified samples
8. Print average color per class
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import time  # ⬅️ added for runtime measurement

# ---------------- CONFIG ----------------
IMG_WIDTH = 256
IMG_HEIGHT = 256
TEST_SIZE = 0.2
DATASET_DIR = "./extracted_images"

# ORB settings
N_KEYPOINTS = 500
VOCAB_SIZE = 256

# ---------------- ORB HELPERS ----------------
def build_orb_vocab(images):
    print("[INFO] Building ORB vocabulary...")
    orb = cv2.ORB_create(nfeatures=N_KEYPOINTS)
    all_desc = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None:
            all_desc.append(des)
    if len(all_desc) == 0:
        raise RuntimeError("No ORB descriptors found in training set.")
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

# ---------------- HSV FEATURES (via alpha mask) ----------------
def hsv_stats(img):
    """Compute HSV mean/std inside the pill (using alpha mask)."""
    if img.shape[2] == 4:  # RGBA
        bgr, alpha = img[:, :, :3], img[:, :, 3]
        mask = alpha > 0
    else:  # fallback: no alpha, whole image
        bgr, mask = img, None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if mask is not None:
        h = h[mask]
        s = s[mask]
        v = v[mask]

    return np.array([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ])

# ---------------- DATA LOADER ----------------
def load_extracted_images(dataset_dir):
    images, labels, class_names, paths = [], [], [], []
    label_map = {}

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path) or class_name.lower() == "edge_masks":
            continue

        label_idx = len(class_names)
        class_names.append(class_name)
        label_map[class_name] = label_idx

        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)  # keep alpha if present
            if img is None:
                continue

            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            images.append(img_resized)
            labels.append(label_idx)
            paths.append(fpath)

    print(f"[INFO] Loaded {len(images)} images across {len(class_names)} classes")
    return np.array(images), np.array(labels), class_names, paths

# ---------------- FEATURE EXTRACTOR ----------------
def build_feature_extractor(img_width, img_height):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_width, img_height, 3),  # ignores alpha
        include_top=False,
        weights="imagenet",
        pooling="max"
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

    print("[INFO] Loading dataset...")
    images, labels, class_names, paths = load_extracted_images(DATASET_DIR)

    # strip alpha for CNN + ORB (they expect 3 channels)
    images_bgr = np.array([img[:, :, :3] if img.shape[2] == 4 else img for img in images])

    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        images, labels, paths, test_size=TEST_SIZE, random_state=42, stratify=labels
    )

    # ORB Vocabulary
    orb = cv2.ORB_create(nfeatures=N_KEYPOINTS)
    orb_kmeans = build_orb_vocab([img[:, :, :3] for img in x_train])  # BGR only

    # MobileNet Features
    print("[INFO] Building frozen MobileNetV2 feature extractor...")
    extractor = build_feature_extractor(IMG_WIDTH, IMG_HEIGHT)

    print("[INFO] Extracting MobileNet features...")
    train_feats_cnn = extractor.predict(np.array([img[:, :, :3] for img in x_train]), verbose=1)
    test_feats_cnn = extractor.predict(np.array([img[:, :, :3] for img in x_test]), verbose=1)

    # ORB Histograms
    print("[INFO] Extracting ORB histograms...")
    train_feats_orb = np.array([orb_histogram(img[:, :, :3], orb, orb_kmeans) for img in x_train])
    test_feats_orb = np.array([orb_histogram(img[:, :, :3], orb, orb_kmeans) for img in x_test])

    # HSV Stats (via alpha)
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
    print(f"[RESULT] CNN + ORB + HSV(alpha) + Cosine kNN Accuracy: {acc:.4f}")

    # Misclassified samples
    print("\n[REPORT] Misclassified samples:")
    errors = 0
    for true, pred, path in zip(y_test, preds, paths_test):
        if true != pred:
            print(f" - {os.path.basename(path)}: True={class_names[true]}, Pred={class_names[pred]}")
            errors += 1
    if errors == 0:
        print(" 🎉 No misclassifications!")

    # Color summary per class
    print("\n[COLOR REPORT] Average HSV per class (alpha-masked):")
    for class_idx, cname in enumerate(class_names):
        class_hsv = [hsv_stats(img) for img, lab in zip(images, labels) if lab == class_idx]
        if len(class_hsv) == 0:
            continue
        mean_hsv = np.mean(class_hsv, axis=0)
        print(f" - {cname}: H={mean_hsv[0]:.1f}, S={mean_hsv[2]:.1f}, V={mean_hsv[4]:.1f}")

    end = time.time()  # ⬅️ end timer
    print(f"[INFO] Total runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
