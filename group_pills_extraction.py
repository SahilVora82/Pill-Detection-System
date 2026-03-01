#!/usr/bin/env python3
"""
group_pills_extraction.py
-------------------------
Extract multiple pills from a single photo with debug visualization.

Fix:
- Ignores inner imprint/text contours using hierarchy.
- Rejects small/noisy contours.
- Debug image shows why each contour was kept or rejected.
"""

import os, glob, csv
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
CONFIG = {
    "INPUT_DIR": "./group_photos",
    "OUT_DIR": "./group_extracted",
    "MIN_AREA": 500,
    "DEBUG": True
}

# ---------------- edge segmentation ----------------
def find_pills(img):
    h, w = img.shape[:2]
    chans = cv2.split(img)
    edges_total = np.zeros((h, w), np.uint8)

    for ch in chans:
        e = cv2.Canny(ch, 60, 150)
        edges_total = cv2.bitwise_or(edges_total, e)

    edges_total = cv2.morphologyEx(edges_total, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges_total = cv2.dilate(edges_total, np.ones((3, 3), np.uint8), iterations=2)

    cnts, hierarchy = cv2.findContours(edges_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pills, debug_info = [], []

    if hierarchy is None:
        return pills, debug_info
    hierarchy = hierarchy[0]

    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)

        # reject small stuff
        if area < CONFIG["MIN_AREA"]:
            debug_info.append(("reject_small", c))
            continue

        # reject if it's a child (imprint inside another contour)
        if hierarchy[i][3] != -1:
            debug_info.append(("reject_child", c))
            continue

        # keep valid outer contour
        hull = cv2.convexHull(c)
        x, y, bw, bh = cv2.boundingRect(hull)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        pills.append((mask, (x, y, bw, bh)))
        debug_info.append(("keep", hull))

    return pills, debug_info

# ---------------- main ----------------
def process_group_photos():
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    manifest_path = os.path.join(CONFIG["OUT_DIR"], "manifest.csv")
    write_header = not os.path.exists(manifest_path)

    with open(manifest_path, "a", newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["group_name", "source_file", "pill_index",
                         "saved_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

        for img_path in glob.glob(os.path.join(CONFIG["INPUT_DIR"], "*.png")):
            img = cv2.imread(img_path)
            if img is None:
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            group_dir = os.path.join(CONFIG["OUT_DIR"], base)
            os.makedirs(group_dir, exist_ok=True)

            pills, debug_info = find_pills(img)
            print(f"[INFO] Found {len(pills)} pills in {img_path}")

            for i, (mask, (x, y, bw, bh)) in enumerate(pills):
                pill_roi = img[y:y+bh, x:x+bw]
                mask_roi = mask[y:y+bh, x:x+bw]

                rgba = cv2.cvtColor(pill_roi, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask_roi

                out_name = f"{base}_pill{i+1}.png"
                out_path = os.path.join(group_dir, out_name)
                cv2.imwrite(out_path, rgba)

                wr.writerow([base, img_path, i+1, out_path, x, y, bw, bh])

            # Debug
            if CONFIG["DEBUG"]:
                debug_img = img.copy()
                for status, c in debug_info:
                    color = (0, 255, 0) if status == "keep" else (0, 0, 255)
                    cv2.drawContours(debug_img, [c], -1, color, 2)
                    if status != "keep":
                        x, y, w_, h_ = cv2.boundingRect(c)
                        cv2.putText(debug_img, status, (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                for j, (mask, (x, y, bw, bh)) in enumerate(pills):
                    cv2.putText(debug_img, str(j+1), (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                debug_path = os.path.join(group_dir, f"{base}_debug.png")
                cv2.imwrite(debug_path, debug_img)

    print(f"[DONE] Pills + debug saved to {CONFIG['OUT_DIR']}")

# ---------------- runner ----------------
if __name__ == "__main__":
    process_group_photos()
