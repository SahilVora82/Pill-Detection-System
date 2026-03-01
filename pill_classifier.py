import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Constants
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

def main():
    start = time.time()  # start timer

    print("[INFO] Loading data...")
    images, labels = load_data(CSV_PATH, IMAGE_FOLDERS)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels),
        test_size=TEST_SIZE, random_state=42
    )

    print("[INFO] Loading MobileNetV2 (frozen)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False,
        weights="imagenet",
        pooling='max'   # flatten features
    )
    base_model.trainable = False

    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        base_model
    ])

    print("[INFO] Extracting features...")
    train_features = feature_extractor.predict(x_train, verbose=1)
    test_features = feature_extractor.predict(x_test, verbose=1)

    print("[INFO] Training kNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')  # cosine distance
    knn.fit(train_features, y_train)

    print("[INFO] Evaluating...")
    preds = knn.predict(test_features)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] MobileNetV2 + kNN Accuracy: {acc:.4f}")

    # End timer
    end = time.time()
    print(f"[INFO] Total runtime: {end - start:.2f} seconds")

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
    return images, labels

if __name__ == "__main__":
    main()
