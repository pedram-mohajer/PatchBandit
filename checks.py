#!/usr/bin/env python3
"""
checks.py  –  Run either YOLO-v5 (Ultralytics) **or** the Faster-R-CNN model
              you trained, draw the top-confidence prediction, and save the
              result as <original>_boxed.png.

Call examples
-------------
# YOLO (default)         ─────────────
python checks.py  --img /home/tigersec/Downloads/10_degree.jpg

# Faster-R-CNN           ─────────────
python checks.py  --img /home/tigersec/Downloads/3x_distance.jpg  --det frcnn
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataloader import VOC_CLASSES                # class-name list

# ---------------------------------------------------------- --
# CLI
# ------------------------------------------------------------
# VOCYOLO/images/val/000005.jpg
ap = argparse.ArgumentParser()
ap.add_argument("--img",  required=True, help="Path to the RGB image")
ap.add_argument("--det",  choices=["yolo", "frcnn"], default="yolo",
                help="Detector to use (yolo | frcnn)")
ap.add_argument("--conf", type=float, default=0.20,
                help="Score threshold")
args = ap.parse_args()

IMG_PATH   = Path(args.img).expanduser()
CONF_TH    = args.conf

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------
YOLO_WEIGHTS   = "runs/detect/train5/weights/best.pt"
FRCNN_WEIGHTS  = "runs/frcnn_voc/best.pth"

# ------------------------------------------------------------
# Build the desired detector
# ------------------------------------------------------------
if args.det == "yolo":
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)

    # Ultralytics already normalises + resizes internally
    def infer(np_img):
        return model.predict(source=np_img, conf=CONF_TH, verbose=False)[0].boxes

else:  # Faster-R-CNN
    from util import FasterRCNNWrapper
    model = FasterRCNNWrapper(FRCNN_WEIGHTS)      # auto-GPU

    # our wrapper mimics Ultralytics signature
    def infer(np_img):
        return model.predict(np_img, conf=CONF_TH, verbose=False)[0].boxes

# ------------------------------------------------------------
# Load image & run detector
# ------------------------------------------------------------
img_pil = Image.open(IMG_PATH).convert("RGB")
img_np  = np.array(img_pil)              # H×W×3  uint8 RGB

boxes = infer(img_np)

if len(boxes) == 0:
    print("[✗] No objects detected.")
    exit(0)

# ------------------------------------------------------------
# Pick highest-confidence detection
# ------------------------------------------------------------
conf_arr = boxes.conf.detach().cpu().numpy()
best_id  = int(np.argmax(conf_arr))

# xyxy coordinates
xyxy = (boxes.xyxy[best_id] if hasattr(boxes, "xyxy")
        else boxes.boxes[best_id]).detach().cpu().numpy()
x1, y1, x2, y2 = xyxy
cls_id = int(boxes.cls[best_id].item())
conf   = float(conf_arr[best_id])

label  = f"{VOC_CLASSES[cls_id]} {conf:.2f}"

# ------------------------------------------------------------
# Draw & save
# ------------------------------------------------------------
draw = ImageDraw.Draw(img_pil)
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
draw.text((x1, max(0, y1 - 12)), label, fill="red")

out_path = IMG_PATH.with_suffix("").as_posix() + "_boxed.png"
img_pil.save(out_path)
print("out_path : ", out_path)
print(f"[✓] Saved image with top prediction → {out_path}")




# ------------------------------------------------------------


# #!/usr/bin/env python3
# """
# checks.py — run YOLO on a single image, draw the top-confidence prediction,
# and save the result as <original name>_boxed.png.
# """

# from ultralytics import YOLO
# from PIL import Image, ImageDraw
# import numpy as np
# from dataloader import VOC_CLASSES    # class-name list

# # -----------------------------------------------------------
# # Paths
# # -----------------------------------------------------------

# # alternative
# IMAGE_PATH = "./results_single/img000_succeeded_patch_150_step_95_0_patch.png" 
# IMAGE_PATH = "/home/tigersec/Downloads/tv (1).jpg" 
# MODEL_PATH = "runs/detect/train5/weights/best.pt"

# # -----------------------------------------------------------
# # Load model + image
# # -----------------------------------------------------------
# model = YOLO(MODEL_PATH)

# img_pil = Image.open(IMAGE_PATH).convert("RGB")   # keep PIL copy for drawing
# img_np  = np.array(img_pil)                       # NumPy copy for inference

# # -----------------------------------------------------------
# # Inference
# # -----------------------------------------------------------
# results = model.predict(source=img_np, conf=0.3, verbose=False)
# boxes   = results[0].boxes

# if boxes is None or len(boxes) == 0:
#     print("[✗] No objects detected in the image.")
#     exit(0)

# # -----------------------------------------------------------
# # Pick the single best detection (highest confidence)
# # -----------------------------------------------------------
# conf_arr = boxes.conf.cpu().numpy()
# idx      = int(np.argmax(conf_arr))

# x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()   # (float) pixel coords
# cls_id         = int(boxes.cls[idx].item())
# conf           = float(conf_arr[idx])

# label = f"{VOC_CLASSES[cls_id]} {conf:.2f}"

# # -----------------------------------------------------------
# # Draw box + label
# # -----------------------------------------------------------
# draw = ImageDraw.Draw(img_pil)

# draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
# # Put text just above the box (no fancy font needed)
# text_y = max(0, y1 - 12)                        # keep inside image
# draw.text((x1, text_y), label, fill="red")

# # -----------------------------------------------------------
# # Save and done
# # -----------------------------------------------------------
# out_path = IMAGE_PATH.rsplit(".", 1)[0] + "_boxed.png"
# img_pil.save(out_path)

# print(f"[✓] Saved image with top-prediction bounding box → {out_path}")
