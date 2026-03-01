#!/usr/bin/env python3
"""
video_to_pictures_edge_only.py
------------------------------
Frame picker for pill videos with edge-only segmentation (transparent background).


Pipeline:
1. Sample frames at target FPS
2. Rank by sharpness, glare, imprint, exposure
3. Deduplicate + keep top N per video
4. Edge-based segmentation (color-aware Canny + morphological closing)
  - Always fill pill region (no hollow outlines)
  - Keep largest contour as pill
5. Save segmented frames as PNG with transparent background (RGBA)
6. Save edge masks + manifest CSV


Requirements:
   pip install opencv-python numpy Pillow
"""


import os, re, glob, csv, sys, random
from typing import List, Optional
import cv2, numpy as np


# =========================
# CONFIG
# =========================
CONFIG = {
   "INPUTS": ["./videos/*.mp4", "./videos/*.MOV", "./videos/*.mov", "./videos/*.MP4"],
   "OUT_DIR": "./extracted_images",


   "FPS": 12.0,
   "TOP_K": 1500,
   "MAX_PER_VIDEO": 12,


   "DEDUP": True,
   "DUP_HAMMING_THR": 10,


   # scoring weights
   "W_IMPRINT": 0.50,
   "W_SHARP":   0.35,
   "W_EXPO":    0.10,
   "W_GLARE":   0.05,
}


# ---------------- helpers ----------------
def group_from_base(base: str) -> str:
   return re.sub(r'_\d+$', '', base)


def sanitize_dirname(name: str) -> str:
   return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_') or "video"


def find_input_videos(inputs: List[str]) -> List[str]:
   exts = (".mp4",".mov",".avi",".mkv",".m4v",".webm",".MP4",".MOV")
   vids = []
   for p in inputs:
       if os.path.isdir(p):
           vids += [os.path.join(p,f) for f in os.listdir(p) if f.lower().endswith(exts)]
       elif any(ch in p for ch in "*?[]"):
           vids += [g for g in glob.glob(p) if g.lower().endswith(exts)]
       elif os.path.isfile(p) and p.lower().endswith(exts):
           vids.append(p)
   return list(dict.fromkeys(vids))  # dedup, preserve order


def get_frame_step(cap: cv2.VideoCapture, target_fps: float) -> int:
   src = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
   return 1 if target_fps <= 0 else max(int(round(src/target_fps)), 1)


# ---------------- metrics ----------------
def varlap(img: np.ndarray) -> float:
   g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   return float(cv2.Laplacian(g, cv2.CV_64F).var())


def glare_ratio(img: np.ndarray, bright_val: int = 250) -> float:
   g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   return float((g >= bright_val).sum()) / g.size


def exposure_quality(img: np.ndarray) -> float:
   g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   m = float(g.mean())
   mid, span = 145.0, 90.0
   q = max(0.0, 1.0 - abs(m - mid) / span)
   return q*q


def imprint_score(img: np.ndarray) -> float:
   g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   g = cv2.createCLAHE(2.0,(8,8)).apply(g)
   edges = cv2.Canny(g, 60, 120)
   edge_d = float((edges>0).sum()) / edges.size
   h,w = g.shape[:2]
   min_a = 30
   max_a = int(0.02*h*w)
   mser = cv2.MSER_create(delta=5, min_area=min_a, max_area=max_a)
   _, boxes = mser.detectRegions(g)
   cnt = 0
   for (x,y,bw,bh) in boxes:
       a=bw*bh
       if a<40 or a>0.15*h*w: continue
       if bh==0: continue
       ar=bw/float(bh)
       if 0.25<=ar<=5.0: cnt+=1
   mser_n = min(cnt/25.0, 1.0)
   edge_n = min(edge_d/0.25, 1.0)
   return 0.6*mser_n + 0.4*edge_n


# ---------------- dedup ----------------
def phash63(img: np.ndarray) -> np.ndarray:
   g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
   d=cv2.dct(g); block=d[:8,:8].flatten()[1:]; med=np.median(block)
   return (block>med).astype(np.bool_)


def ham(a: np.ndarray, b: np.ndarray) -> int:
   return int(np.count_nonzero(a ^ b))


def rank01(vals: np.ndarray, larger=True) -> np.ndarray:
   n = len(vals)
   if n <= 1: return np.ones(n, dtype=float)
   order = np.argsort(vals)
   r = np.empty(n, float); r[order] = np.linspace(0,1,n)
   return r if larger else (1.0 - r)


# ---------------- edge segmentation ----------------
def edge_segmentation(img: np.ndarray) -> np.ndarray:
    """Color-aware edge segmentation, pill mask fully filled in (no hollow outlines)."""
    h, w = img.shape[:2]
    chans = cv2.split(img)
    edges_total = np.zeros((h, w), np.uint8)

    # Combine edges from all channels
    for ch in chans:
        e = cv2.Canny(ch, 60, 150)
        edges_total = cv2.bitwise_or(edges_total, e)

    # Close small gaps and smooth edges
    edges_total = cv2.morphologyEx(edges_total, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    edges_total = cv2.dilate(edges_total, np.ones((3,3), np.uint8), iterations=2)

    # Find contours
    cnts, _ = cv2.findContours(edges_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)

    if cnts:
        # Take the largest contour (pill) and fill it solid
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        cv2.drawContours(mask, [hull], -1, 255, -1)  # fill inside

        # Ensure fully solid by flood filling from the center
        x, y, wbox, hbox = cv2.boundingRect(hull)
        seed_pt = (x + wbox // 2, y + hbox // 2)
        flood_mask = mask.copy()
        cv2.floodFill(flood_mask, None, seed_pt, 255)
        mask = flood_mask

    return mask


# ---------------- per-video ----------------
def process_video(video_path: str, out_root: str, base_name: str, cfg: dict, group_name: Optional[str] = None):
   out_dir = os.path.join(cfg["OUT_DIR"], group_name or base_name)
   os.makedirs(out_dir, exist_ok=True)
   edge_dir = os.path.join(out_dir, "edge_masks")
   os.makedirs(edge_dir, exist_ok=True)


   man_path = os.path.join(out_dir, "manifest.csv")
   write_header = not os.path.exists(man_path)


   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"[WARN] Could not open video: {video_path}", file=sys.stderr)
       return 0
   step = get_frame_step(cap, cfg["FPS"])
   start_offset = random.randrange(step)
   total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


   cands, idx, considered = [], -1, 0
   while True:
       ok, frame = cap.read()
       if not ok: break
       idx += 1
       if ((idx - start_offset) % step) != 0: continue


       sharp = varlap(frame)
       glare = glare_ratio(frame)
       impr  = imprint_score(frame)
       expo  = exposure_quality(frame)


       tnorm = (idx/float(total)) if total>0 else 0.0
       cands.append({
           "idx": idx, "tnorm": tnorm, "frame": frame,
           "metrics": {"sharp":sharp, "glare":glare, "imprint":impr, "exposure":expo},
           "hash": phash63(frame)
       })


       considered += 1
       if considered >= cfg["TOP_K"]: break


   cap.release()


   if not cands:
       print(f"[DONE] 0 frames -> {out_dir}")
       return 0


   # rank
   rI = rank01(np.array([c["metrics"]["imprint"] for c in cands]), True)
   rS = rank01(np.array([c["metrics"]["sharp"]   for c in cands]), True)
   rE = rank01(np.array([c["metrics"]["exposure"]for c in cands]), True)
   rG = rank01(np.array([c["metrics"]["glare"]   for c in cands]), False)
   scores = (cfg["W_IMPRINT"]*rI + cfg["W_SHARP"]*rS + cfg["W_EXPO"]*rE + cfg["W_GLARE"]*rG)
   for c,s in zip(cands,scores): c["score"]=float(s)
   cands.sort(key=lambda d:d["score"], reverse=True)


   kept, hashes = [], []
   for c in cands:
       if not cfg["DEDUP"] or not hashes:
           kept.append(c); hashes.append(c["hash"])
       else:
           if min(ham(c["hash"],h) for h in hashes) <= cfg["DUP_HAMMING_THR"]: continue
           kept.append(c); hashes.append(c["hash"])
       if len(kept) >= cfg["MAX_PER_VIDEO"]: break


   saved=0
   with open(man_path,("a" if not write_header else "w"),newline="") as f:
       wr = csv.writer(f)
       if write_header:
           wr.writerow(["video_path","frame_index","tnorm","sharp","glare","imprint","exposure","score","saved_path","edge_mask_path"])
       for c in kept:
           frame = c["frame"]
           mask = edge_segmentation(frame)


           # save edge mask
           edge_name=f"{base_name}_f{c['idx']:06d}_edge.png"
           edge_path=os.path.join(edge_dir,edge_name)
           cv2.imwrite(edge_path,mask)


           # save transparent RGBA
           rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
           rgba[:, :, 3] = mask  # alpha channel = pill mask


           name=f"{base_name}_f{c['idx']:06d}.png"
           path=os.path.join(out_dir,name)
           cv2.imwrite(path,rgba); saved+=1


           m=c["metrics"]
           wr.writerow([video_path,c["idx"],round(c["tnorm"],4),
                        round(m["sharp"],2),round(m["glare"],4),round(m["imprint"],4),
                        round(m["exposure"],4),round(c["score"],4),path,edge_path])


   print(f"[DONE] {saved} frames -> {out_dir}")
   return saved


# ---------------- runner ----------------
def main():
   cfg=CONFIG
   os.makedirs(cfg["OUT_DIR"],exist_ok=True)
   vids=find_input_videos(cfg["INPUTS"])
   if not vids:
       print("[ERROR] No input videos found."); sys.exit(1)
   for v in vids:
       base=os.path.splitext(os.path.basename(v))[0]
       safe=sanitize_dirname(base)
       group=sanitize_dirname(group_from_base(safe))
       print(f"[INFO] Processing: {v} -> group {group}")
       process_video(v,cfg["OUT_DIR"],safe,cfg,group_name=group)


if __name__=="__main__":
   main()

