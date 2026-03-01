import cv2
import os
import pandas as pd
import easyocr

CSV_PATH = "table.csv"
IMAGE_FOLDERS = [
    "image/images/gallery/300",
    "image/images/gallery/600",
    "image/images/gallery/800",
    "image/images/gallery/1024"
]
IMPRINTS_DIR = "imprints"
RESULT_CSV = "ocr_results.csv"

reader = easyocr.Reader(['en'], gpu=False)

def detect_imprint_mser(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    mser = cv2.MSER_create(5, 60, 10000)
    regions, _ = mser.detectRegions(blur)

    boxes = []
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        ar = w / float(h + 1e-6)
        if 1.5 <= ar <= 12 and 8 <= h <= 150:
            boxes.append((x, y, w, h))

    if not boxes:
        return None

    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return gray[y:y+h, x:x+w]

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["nlmImageFileName", "name"])

    total_images = 0
    successful_detections = 0
    results = []

    for folder in IMAGE_FOLDERS:
        res_folder = os.path.basename(folder)  # e.g., "300"
        save_dir = os.path.join(IMPRINTS_DIR, res_folder)
        os.makedirs(save_dir, exist_ok=True)

        df["filepath"] = df["nlmImageFileName"].apply(lambda x: os.path.join(folder, x))
        for _, row in df.iterrows():
            filename = row["nlmImageFileName"]
            label = row["name"]
            path = row["filepath"]

            if not os.path.exists(path):
                continue

            image = cv2.imread(path)
            if image is None:
                continue

            total_images += 1
            imprint_crop = detect_imprint_mser(image)

            if imprint_crop is not None:
                successful_detections += 1

                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, imprint_crop)

                # Run OCR
                ocr_result = reader.readtext(imprint_crop)
                if ocr_result:
                    best_text, confidence = ocr_result[0][1], ocr_result[0][2]
                else:
                    best_text, confidence = "", 0.0

                results.append({
                    "filename": filename,
                    "label": label,
                    "ocr_text": best_text,
                    "confidence": confidence,
                    "folder": res_folder
                })

    print("\n----- MSER + OCR Detection Results -----")
    print(f"Total images scanned:        {total_images}")
    print(f"Successful MSER detections:  {successful_detections}")
    print(f"Detection success rate:      {successful_detections / total_images * 100:.2f}%")
    print(f"OCR results saved to:        {RESULT_CSV}")

    pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

if __name__ == "__main__":
    main()
