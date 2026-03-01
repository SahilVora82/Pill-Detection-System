import os
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------
# Config
# -------------------
CSV_PATH = "ocr_results.csv"          # must have columns: filename, label, ocr_text, confidence, folder
MIN_CONF = 0.15                        # drop super-low-confidence OCR reads
MIN_SAMPLES_PER_CLASS = 5              # filter rare classes to keep eval sane
NGRAM_RANGE = (2, 4)                   # char-level n-grams (2–4 works well for imprints like "L484", "IP204")
K = 3                                  # K for KNN
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------
# Helpers
# -------------------
def norm_ocr_text(s: str) -> str:
    """Normalize OCR strings: uppercase, remove spaces & punctuation. Keep only A–Z0–9 and '-'."""
    if not isinstance(s, str):
        s = ""
    s = s.upper().strip()
    # Keep alnum and dash
    allowed = set(string.ascii_uppercase + string.digits + "-")
    s = "".join(ch for ch in s if ch in allowed)
    # Avoid empty strings (vectorizer can't handle all-empty)
    return s if s else "UNK"

# -------------------
# Load & clean
# -------------------
df = pd.read_csv(CSV_PATH)

required_cols = {"filename", "label", "ocr_text", "confidence"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {CSV_PATH}: {missing}. Expected {required_cols}")

# Drop low-confidence or missing ocr_text rows
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
df = df[df["confidence"] >= MIN_CONF].copy()

# Normalize OCR text
df["ocr_text_norm"] = df["ocr_text"].apply(norm_ocr_text)

# Filter classes with enough samples
counts = df["label"].value_counts()
keep_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df["label"].isin(keep_labels)].copy()

if len(df) == 0 or df["label"].nunique() < 2:
    raise ValueError("Not enough data after filtering. Lower MIN_CONF or MIN_SAMPLES_PER_CLASS.")

print(f"[INFO] Samples after filtering: {len(df)}")
print(f"[INFO] Classes kept (>= {MIN_SAMPLES_PER_CLASS} samples): {df['label'].nunique()}")

# -------------------
# Train / Test split
# -------------------
X_text = df["ocr_text_norm"].values
y = df["label"].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------
# Vectorize (char TF-IDF)
# -------------------
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=NGRAM_RANGE,
    lowercase=False,
    min_df=1
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# -------------------
# Train KNN (cosine)
# -------------------
knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", weights="distance")
knn.fit(X_train, y_train)

# -------------------
# Evaluate Top-1
# -------------------
y_pred = knn.predict(X_test)
top1_acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Top-1 accuracy: {top1_acc * 100:.2f}%")

# -------------------
# Evaluate Top-3
# -------------------
# Get distances to K neighbors; we’ll expand to 3-neighbor check even if K<3 by refitting temporarily
k_for_top3 = max(3, K)
knn_top3 = KNeighborsClassifier(n_neighbors=k_for_top3, metric="cosine", weights="distance")
knn_top3.fit(X_train, y_train)
dists, idxs = knn_top3.kneighbors(X_test)  # idxs shape: (n_samples, k_for_top3)

y_top3 = []
for row_idxs in idxs:
    cand_labels = set(y_train[row_idxs])
    y_top3.append(cand_labels)

top3_hits = sum(1 for true_lbl, cand_set in zip(y_test, y_top3) if true_lbl in cand_set)
top3_acc = top3_hits / len(y_test)
print(f"[RESULT] Top-3 accuracy: {top3_acc * 100:.2f}%")

# -------------------
# Per-class report (optional)
# -------------------
print("\n[INFO] Classification report (Top-1):")
print(classification_report(y_test, y_pred, digits=3))

# -------------------
# Save artifacts
# -------------------
# Save predictions for later error analysis
out = pd.DataFrame({
    "ocr_text_norm": X_test_text,
    "true_label": y_test,
    "pred_label_top1": y_pred
})
out.to_csv("ocr_text_knn_predictions.csv", index=False)

# Save vectorizer vocabulary & class list (lightweight)
# (You can joblib.dump(vectorizer, ...) if you want a full artifact.)
pd.Series(vectorizer.get_feature_names_out()).to_csv("ocr_text_vectorizer_vocab.csv", index=False)
pd.Series(sorted(df["label"].unique())).to_csv("ocr_text_knn_labels.csv", index=False)

print("\n[INFO] Saved: ocr_text_knn_predictions.csv, ocr_text_vectorizer_vocab.csv, ocr_text_knn_labels.csv")
