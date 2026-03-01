import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize  # for cosine (optional but recommended)

# =========================
# Config
# =========================
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
SEED = 42
K = 3
USE_COSINE = True  # set False to use distance

# ===== Define a few candidate policies to try =====
# magnitudes are "safe" for pill imprints; tweak as you like
POLICIES = [
    {"name": "light_geom", "rot":0.08, "tx":0.04, "ty":0.04, "zoom":0.08, "contr":0.10, "bright":0.10},
    {"name": "med_geom",   "rot":0.12, "tx":0.06, "ty":0.06, "zoom":0.12, "contr":0.20, "bright":0.18},
    {"name": "heavy_geom", "rot":0.16, "tx":0.08, "ty":0.08, "zoom":0.15, "contr":0.25, "bright":0.22},
    {"name": "no_geom_lc", "rot":0.00, "tx":0.00, "ty":0.00, "zoom":0.00, "contr":0.20, "bright":0.18},
    {"name": "rot_only",   "rot":0.12, "tx":0.00, "ty":0.00, "zoom":0.00, "contr":0.00, "bright":0.00},
    {"name": "shift_zoom", "rot":0.00, "tx":0.06, "ty":0.06, "zoom":0.12, "contr":0.00, "bright":0.00},
]

# =========================
# Data loading (same as your script)
# =========================
def load_data(csv_path, image_dirs):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["nlmImageFileName", "name"])

    images, labels = [], []
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

    print(f"[INFO] Total samples: {len(images)}")
    return np.array(images), np.array(labels)

def build_feature_extractor(img_width, img_height):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_width, img_height, 3),
        include_top=False,
        weights="imagenet",
        pooling='max'
    )
    base_model.trainable = False
    model = tf.keras.Sequential([ base_model ])  # inputs already scaled to [0,1] below
    return model

# =========================
# Augmentation builder
# =========================
def make_aug_layer(rot, tx, ty, zoom, contr, bright):
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(rot),
        tf.keras.layers.RandomTranslation(tx, ty),
        tf.keras.layers.RandomZoom(zoom, zoom),
        tf.keras.layers.RandomContrast(contr),
        tf.keras.layers.RandomBrightness(bright),
    ])

def apply_aug_numpy(imgs01, aug_layer, batch=256):
    """imgs01: float32 in [0,1], shape (N,H,W,3)"""
    out = []
    for i in range(0, len(imgs01), batch):
        t = tf.convert_to_tensor(imgs01[i:i+batch])
        out.append(aug_layer(t, training=True).numpy())
    return np.concatenate(out, axis=0)

# =========================
# Train & evaluate one policy
# =========================
def evaluate_policy(name, aug_layer, extractor, x_train01, y_train, x_test01, y_test):
    # Augment once, then concat (2x data)
    x_aug = apply_aug_numpy(x_train01, aug_layer)
    x_mix = np.concatenate([x_train01, x_aug], axis=0)
    y_mix = np.concatenate([y_train,   y_train], axis=0)

    # Extract features
    train_feats = extractor.predict(x_mix,  verbose=0)
    test_feats  = extractor.predict(x_test01, verbose=0)

    # Optional: cosine distance works well for embeddings
    if USE_COSINE:
        train_feats = normalize(train_feats, axis=1)
        test_feats  = normalize(test_feats,  axis=1)
        knn = KNeighborsClassifier(n_neighbors=K, metric='cosine', weights='distance')
    else:
        knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')

    knn.fit(train_feats, y_mix)
    preds = knn.predict(test_feats)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] {name:12s}  acc={acc:.4f}")
    return acc

# =========================
# Main sweep
# =========================
def main():
    # 1) Load & split once (same split for all policies)
    print("[INFO] Loading data...")
    images, labels = load_data(CSV_PATH, IMAGE_FOLDERS)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    # scale to [0,1]
    x_train01 = x_train.astype("float32") / 255.0
    x_test01  = x_test.astype("float32") / 255.0

    # 2) Build extractor once
    print("[INFO] Building frozen MobileNetV2 feature extractor...")
    extractor = build_feature_extractor(IMG_WIDTH, IMG_HEIGHT)

    # 3) Sweep
    results = []
    for p in POLICIES:
        aug = make_aug_layer(
            rot=p["rot"], tx=p["tx"], ty=p["ty"], zoom=p["zoom"],
            contr=p["contr"], bright=p["bright"]
        )
        acc = evaluate_policy(p["name"], aug, extractor, x_train01, y_train, x_test01, y_test)
        results.append((p["name"], acc, p))

    # 4) Leaderboard
    results.sort(key=lambda t: t[1], reverse=True)
    print("\n==== Leaderboard ====")
    for name, acc, _ in results:
        print(f"{name:12s}  acc={acc:.4f}")

    best_name, best_acc, best_p = results[0]
    print(f"\n[BEST] {best_name}  acc={best_acc:.4f}  params={best_p}")

if __name__ == "__main__":
    main()
