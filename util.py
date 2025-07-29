# util.py

from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from patch_bandit import PatchAttackRunner, AttackConfig
from dataloader import PascalVOCLoader, VOC_CLASSES
from types import SimpleNamespace
from torchvision.ops import nms
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import torch
import json
import os

VOC_YAML_PATH = "voc.yaml" #training for yolo
MODEL_PATH = "runs/detect/train5/weights/best.pt"
FRCNN_MODEL_PATH = "runs/frcnn_voc/best.pth"

# ------------------------------------------------------------------
# Helper: convert NumPy scalars → native Python so json.dump works
# ------------------------------------------------------------------
def _py(obj):
    """Recursively cast NumPy / torch values to plain Python types."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_py(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_py(v) for v in obj)

    # ---------- NEW: handle full arrays ----------
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())          # recurse on the resulting list
    # ---------------------------------------------

    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)

    return obj


# ------------------------------------------------------------------
# Generate (possibly adaptive) patch for a single image.
# Uses AttackConfig for: patch sizes, weights, budgets, placement policy.
# ------------------------------------------------------------------
def generate_patch_for_image(
    img_tensor,
    boxes,
    labels,
    model,
    *,
    mode="digital",
    attack_cfg: AttackConfig | None = None,
    save_prefix="results/picture",
):
    """
    Attempt an adversarial patch attack on ONE image.

    If `attack_cfg` is provided, we run a *single* adaptive session
    (budgeted size growth inside the runner). If None, we fall back to
    the old "loop over fixed sizes" behaviour.

    Returns: dict with keys:
        success (bool)
        steps   (int | None)
        final_size (int)
        history (list of logs)
    """
    if len(labels) == 0:
        print("[⚠] No objects – skip image")
        return {"success": False, "steps": None, "final_size": None, "history": []}

    if attack_cfg is None:
        # ===== Legacy multi-size sweep =====
        success_overall = False
        hist = []
        for k, sz in enumerate((60, 80, 100, 120, 160), 1):
            print(f"\n=====  SIZE {sz} (attempt {k})  =====")
            runner = PatchAttackRunner(
                image_tensor=img_tensor,
                gt_boxes=boxes,
                gt_labels=labels,
                min_size=sz,
                max_size=sz,
                grid_size=8,
                attack_type=mode,
                lambda_norm=0.5,
                use_bandit=True,
                attack_cfg=None,              # disable adaptive
                save_prefix=f"{save_prefix}_attempt_{k}",
            )
            succeeded, steps, _ = runner.run(model=model, max_iters=250)
            success_overall |= succeeded
            hist.append((sz, succeeded, steps))
        return {
            "success": success_overall,
            "steps": None,
            "final_size": None,
            "history": hist,
        }

    # ===== New adaptive attack =====
    runner = PatchAttackRunner(
        image_tensor=img_tensor,
        gt_boxes=boxes,
        gt_labels=labels,
        attack_type=mode,
        attack_cfg=attack_cfg,
        save_prefix=save_prefix,
    )
    succ, steps, info = runner.run(model=model, max_iters=attack_cfg.max_queries)
    return {
        "success": succ,
        "steps": steps,
        "final_size": info.get("final_size"),
        "history": info.get("history", []),
    }


# ------------------------------------------------------------------
# Save per-run summary (now NumPy-safe)
# ------------------------------------------------------------------
def save_attack_summary(summary_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_py(summary_dict), f, indent=2)
    print(f"[✓] Attack summary saved → {out_path}")





# ------------------------------------------------------------------
# YOLOv5 training and evaluation
# yolov5s.pt is the base model.
# ------------------------------------------------------------------
def train_yolo(data_yaml=VOC_YAML_PATH, force_train=False):
    if os.path.exists(MODEL_PATH) and not force_train:
        print(f"[✓] YOLO model already trained at {MODEL_PATH}")
        return

    print("[🔧] Training YOLOv5 on Pascal VOC...")
    model = YOLO("yolov5s.pt")

    _ = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=416,
        batch=16,
        name="train5",  # === CHANGED: keep consistent with MODEL_PATH ===
        workers=4,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    print(f"[✓] Training complete. Model saved to {MODEL_PATH}")


# ------------------------------------------------------------------
def evaluate_yolo_model(model_path=MODEL_PATH, conf_thresh=0.3):
    """
    Evaluate trained YOLO detector on Pascal VOC val split.
    This is *detector* performance, not patch-robustness evaluation.
    """
    model = YOLO(model_path)
    loader = PascalVOCLoader(root=".", year="2007", image_set="val", inp_size=416)

    all_gt_labels, all_pred_labels, all_pred_probs = [], [], []

    print(f"[✓] Evaluating on {len(loader)} samples...\n")
    for _ in tqdm(range(len(loader))):
        img_tensor, boxes, labels, _ = loader.next()
        if labels.numel() == 0:
            continue

        # Unnormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_unnorm = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_np = (img_unnorm * 255).astype(np.uint8)

        # YOLO prediction
        results = model.predict(source=img_np, conf=conf_thresh, verbose=False)
        preds = results[0].boxes

        if preds is None or len(preds) == 0:
            all_gt_labels.extend(labels.tolist())
            all_pred_labels.extend([-1] * len(labels))
            all_pred_probs.extend([0.0] * len(labels))
            continue

        pred_classes = preds.cls.cpu().numpy().astype(int)
        pred_scores = preds.conf.cpu().numpy()

        for gt_cls in labels.tolist():
            matched = False
            for j, pred_cls in enumerate(pred_classes):
                if pred_cls == gt_cls:
                    matched = True
                    all_pred_labels.append(pred_cls)
                    all_pred_probs.append(pred_scores[j])
                    break
            if not matched:
                all_pred_labels.append(-1)
                all_pred_probs.append(0.0)
            all_gt_labels.append(gt_cls)

    # === Report ===
    gt_arr = np.array(all_gt_labels)
    pred_arr = np.array(all_pred_labels)
    valid_mask = pred_arr != -1

    print("\n=== Classification Report ===")
    print(
        classification_report(
            gt_arr[valid_mask],
            pred_arr[valid_mask],
            labels=list(range(len(VOC_CLASSES))),
            target_names=VOC_CLASSES,
            zero_division=0,
        )
    )

    print("\n=== Micro F1-Score ===")
    print(f1_score(gt_arr[valid_mask], pred_arr[valid_mask], average="micro"))

    # ROC-AUC for 'person' (example)
    if "person" in VOC_CLASSES:
        target_class = VOC_CLASSES.index("person")
        binary_gt = (gt_arr == target_class).astype(int)
        binary_pred = np.array(
            [prob if gt == target_class else 0.0 for gt, prob in zip(gt_arr, all_pred_probs)]
        )
        fpr, tpr, _ = roc_curve(binary_gt, binary_pred)
        auc = roc_auc_score(binary_gt, binary_pred)
        print(f"\nROC AUC for class 'person': {auc:.3f}")


# ------------------------------------------------------------------
# faster R-CNN training and evaluation
# frcnn_voc/best.pth is the base model.
# ------------------------------------------------------------------

def _scale_boxes_to_416(boxes, anno):
    """
    Rescale Pascal-VOC GT boxes (original image size) to 416×416.

    boxes : Tensor[N,4]  (xmin, ymin, xmax, ymax) in original pixels
    anno  : raw annotation dict from VOCDetection
    """
    try:
        w = int(float(anno["size"]["width"][0]  if isinstance(anno["size"]["width"],  list) else anno["size"]["width"]))
        h = int(float(anno["size"]["height"][0] if isinstance(anno["size"]["height"], list) else anno["size"]["height"]))
    except Exception:
        w, h = 416, 416

    sx, sy = 416.0 / w, 416.0 / h
    boxes = boxes.clone()
    if boxes.numel():
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return boxes


# ------------------------------------------------------------------
#  mAP@0.5 evaluator (Pascal-VOC style)
# ------------------------------------------------------------------
def evaluate_faster_rcnn_map(
    weights_path: str = FRCNN_MODEL_PATH,
    split: str = "val",          # "train", "val", or "trainval"
    conf_thresh: float = 0.05,   # low threshold – AP is threshold-free inside mAP
    iou_thresh_nms: float = 0.5,
    device: str | None = None,
):
    """
    Compute Pascal-VOC mAP@0.5 and per-class AP for a Faster-R-CNN model.
    Prints a table and returns the full metric dict.
    """
    # ── setup ──────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_faster_r_cnn(weights_path, device=device).eval()

    loader = PascalVOCLoader(
        root=".", year="2007", image_set=split,
        inp_size=416, num_workers=2, shuffle=False
    )

    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5],     # Pascal metric
        class_metrics=True        # <<<<<< crucial
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    print(f"[✓] Evaluating {len(loader)} images from split “{split}”…\n")

    # ── loop ───────────────────────────────────────────────────────
    for _ in tqdm(range(len(loader)),disable=True):
        img_b, boxes_gt, labels_gt, anno = loader.next()

        # scale GT boxes to 416×416 (match model input)
        boxes_gt = _scale_boxes_to_416(boxes_gt, anno)

        # add batch dim → pixel space
        img_norm = img_b.squeeze(0).to(device)
        img_px   = (img_norm * std + mean).clamp(0, 1)

        # detector forward
        with torch.no_grad():
            det = model([img_px])[0]

        # filter detections
        keep = det["scores"] >= conf_thresh
        if keep.sum():
            b  = det["boxes"][keep].cpu()
            s  = det["scores"][keep].cpu()
            l  = det["labels"][keep].cpu()

            # extra NMS (torchvision already does one, but a second pass tightens duplicates)
            if b.size(0) > 1:
                keep_idx = nms(b, s, iou_thresh_nms)
                b, s, l = b[keep_idx], s[keep_idx], l[keep_idx]

            preds = [{"boxes": b, "scores": s, "labels": l}]
        else:
            preds = [{"boxes": torch.zeros((0,4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)}]

        # ground truth (even if empty)
        gts = [{"boxes": boxes_gt.cpu(), "labels": labels_gt.cpu()}]
        metric.update(preds, gts)

    # ── report ─────────────────────────────────────────────────────
    res  = metric.compute()                     # dict
    mAP  = res["map"].item() * 100
    print("\n========== Pascal-VOC mAP@0.5 ==========")
    print(f"mAP@0.5: {mAP:.2f} %\n")

    per_cls = (res["map_per_class"] * 100).tolist()
    for cid, ap in enumerate(per_cls):
        print(f"{VOC_CLASSES[cid]:<12}: {ap:5.2f} %")

    mar = res["mar_100"].item() * 100
    print(f"\nMean Average Recall (AR@100): {mar:.2f} %\n")

    return res


def _get_frcnn_model(num_classes: int):
    """
    Create a Faster R-CNN (ResNet-50 FPN) and swap the predictor head
    to emit `num_classes` (incl. background).
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # torchvision >= 0.13
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_faster_r_cnn(
    data_root: str = ".",
    year: str = "2007",
    train_split: str = "trainval",
    val_split: str = "val",
    epochs: int = 12,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    momentum: float = 0.9,
    print_every: int = 50,
    force_train: bool = False,
    save_path: str = FRCNN_MODEL_PATH,
    device: str | None = None,
):
    """
    Train Faster R‑CNN (ResNet‑50 FPN) on Pascal VOC.

    * Uses your existing PascalVOCLoader: images are resized to 416 × 416 and
      normalised with ImageNet stats; we undo that before feeding the detector.
    * Rescales ground‑truth boxes from original XML coords to 416 × 416.

    Returns
    -------
    str
        Path to the saved best model weights (.pth)
    """
    if os.path.exists(save_path) and not force_train:
        print(f"[✓] Faster R‑CNN model already trained at {save_path}")
        return save_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(VOC_CLASSES) + 1            # +1 for background
    model = _get_frcnn_model(num_classes).to(device)

    # ─── Data loaders ──────────────────────────────────────────────
    train_loader = PascalVOCLoader(
        root=data_root,
        year=year,
        image_set=train_split,
        inp_size=416,
        num_workers=4,
        shuffle=True,
    )
    val_loader = PascalVOCLoader(
        root=data_root,
        year=year,
        image_set=val_split,
        inp_size=416,
        num_workers=2,
        shuffle=False,
    )

    # ─── Optimiser / scheduler ─────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(epochs // 3, 1), gamma=0.1
    )

    # Stats needed to *undo* PascalVOCLoader’s normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _scale_boxes_to_416(boxes, anno):
        """Rescale VOC boxes to match 416 × 416 image size."""
        try:
            orig_w = int(float(
                anno["size"]["width"][0] if isinstance(anno["size"]["width"], list)
                else anno["size"]["width"]
            ))
            orig_h = int(float(
                anno["size"]["height"][0] if isinstance(anno["size"]["height"], list)
                else anno["size"]["height"]
            ))
        except Exception:
            orig_w, orig_h = 416, 416       # fallback

        sx, sy = 416.0 / orig_w, 416.0 / orig_h
        boxes = boxes.clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return boxes

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ─── Training loop ─────────────────────────────────────────────
    for epoch in range(epochs):
        # ----- Train pass -----
        model.train()
        running = 0.0
        n_train = len(train_loader)

        for i in range(n_train):
            img_norm, boxes_raw, labels, anno = train_loader.next()

            # Undo normalisation (→ [0,1]) and move to device
            img = (img_norm * std + mean).clamp(0, 1).to(device)
            boxes = _scale_boxes_to_416(boxes_raw, anno).to(device)

            target = {
                "boxes": boxes,
                "labels": labels.to(device),
                "image_id": torch.tensor([i], device=device),
            }

            loss_dict = model([img], [target])
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            if (i + 1) % print_every == 0 or (i + 1) == n_train:
                print(f"[Train][Epoch {epoch+1}/{epochs}] "
                      f"step {i+1}/{n_train}  loss={running/(i+1):.4f}")

        lr_scheduler.step()

        # ----- Validation pass -----
        model.train()                         # ← NEW: get loss dict
        val_loss = 0.0
        with torch.no_grad():
            n_val = len(val_loader)
            for i in range(n_val):
                img_norm, boxes_raw, labels, anno = val_loader.next()
                img = (img_norm * std + mean).clamp(0, 1).to(device)
                boxes = _scale_boxes_to_416(boxes_raw, anno).to(device)
                target = {
                    "boxes": boxes,
                    "labels": labels.to(device),
                    "image_id": torch.tensor([i], device=device),
                }
                loss_dict = model([img], [target])         # ← still returns dict
                val_loss += sum(loss_dict.values()).item()
        val_loss /= max(1, n_val)
        model.eval()                          # ← CHG: restore eval mode

        print(f"[Val][Epoch {epoch+1}/{epochs}] mean loss = {val_loss:.4f}")

        # ----- Save best model -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[✓] Saved best Faster R‑CNN → {save_path} "
                  f"(val loss {best_val_loss:.4f})")

    print(f"[✓] Training complete. Best model at {save_path}")
    return save_path


def load_faster_r_cnn(weights_path: str = FRCNN_MODEL_PATH, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_frcnn_model(len(VOC_CLASSES) + 1).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model





# mini container mimicking Ultralytics' Boxes object
class _MiniBoxes:
    def __init__(self, boxes, conf, cls):
        self.boxes = boxes      # Tensor[N,4]
        self.conf  = conf       # Tensor[N]
        self.cls   = cls        # Tensor[N]

    def __len__(self):
        return self.cls.size(0)

class FasterRCNNWrapper:
    """
    Wrap torchvision Faster-R-CNN so that:

        result = model.predict(img_uint8)[0].boxes
        len(result)          # number of detections
        result.boxes         # Tensor[N,4]
        result.conf          # Tensor[N]  (scores)
        result.cls           # Tensor[N]  (class IDs)

    behaves exactly like the Ultralytics YOLO API used elsewhere in
    Patch-Bandit.
    """

    def __init__(self,
                 weight_path: str,
                 device: str | None = None,
                 iou_nms: float = 0.5,
                 inp_size: int = 416):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device   = device
        self.iou_nms  = iou_nms
        self.inp_size = inp_size

        self.model = load_faster_r_cnn(weight_path, device=device).eval()

        # ImageNet statistics (already on correct device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    # ───────────────────────── helpers ────────────────────────────
    def _preprocess(self, img_uint8: np.ndarray) -> torch.Tensor:
        """
        img_uint8 : H×W×3  uint8 (RGB)
        returns   : 1×3×inp_size×inp_size  float32, ImageNet-norm, on self.device
        """
        # ensure positive contiguous strides (OpenCV BGR→RGB slices often aren’t)
        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        t = torch.from_numpy(img_uint8).to(self.device).permute(2, 0, 1).float() / 255.0

        # resize to model’s input size
        if t.shape[1] != self.inp_size or t.shape[2] != self.inp_size:
            t = torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=(self.inp_size, self.inp_size),
                    mode="bilinear", align_corners=False
                ).squeeze(0)

        # ImageNet normalisation
        t = (t - self.mean) / self.std
        return t.unsqueeze(0)               # add batch dim

    # ───────────────────────── public API ─────────────────────────
    def predict(self, source: np.ndarray, conf: float = 0.25, verbose: bool = False):
        """
        source : H×W×3 uint8 RGB     (identical signature to YOLO)
        conf   : score threshold
        returns: list with ONE element whose .boxes imitates Ultralytics
        """
        with torch.no_grad():
            out = self.model(self._preprocess(source))[0]    # dict: boxes/scores/labels

        scores = out["scores"].detach()
        keep   = scores >= conf

        if keep.sum() == 0:
            # return an empty Boxes object to stay compatible
            z  = torch.zeros((0,), device="cpu")
            mm = _MiniBoxes(torch.zeros((0, 4)), z, z)
            return [SimpleNamespace(boxes=mm)]

        boxes = out["boxes"][keep].detach()
        confs = scores[keep]
        clses = out["labels"][keep].detach()

        # second-stage NMS (YOLO behaviour)
        if boxes.size(0) > 1:
            idx = nms(boxes, confs, self.iou_nms)
            boxes, confs, clses = boxes[idx], confs[idx], clses[idx]

        mm = _MiniBoxes(boxes.cpu(), confs.cpu(), clses.cpu())
        return [SimpleNamespace(boxes=mm)]


# ------------------------------------------------------------------
# Stand-alone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_faster_rcnn_map(
        weights_path="runs/frcnn_voc/best.pth",
        split="val",      # or "trainval"
        conf_thresh=0.05
    )




# python - <<'PY'
# from util import evaluate_faster_rcnn_map
# evaluate_faster_rcnn_map(
#     "runs/frcnn_voc/best.pth",
#     split="val",
#     conf_thresh=0.05
# )
# PY




# class _MiniBoxes:
#     """
#     Minimal imitation of Ultralytics' Boxes object.
#     Provides .boxes  .cls  .conf   and   len()
#     """
#     def __init__(self, boxes, conf, cls):
#         self.boxes = boxes          # Tensor[N,4]
#         self.conf  = conf           # Tensor[N]
#         self.cls   = cls            # Tensor[N]

#     def __len__(self):
#         return self.cls.shape[0]

# class FasterRCNNWrapper:
#     """
#     Wrap torchvision Faster-R-CNN so that
#         model.predict(img_uint8)[0].boxes
#     behaves like Ultralytics YOLO.
#     """
#     def __init__(self, weight_path, device=None, iou_nms=0.5, inp_size=416):
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device   = device
#         self.inp_size = inp_size
#         self.model    = load_faster_r_cnn(weight_path, device=device).eval()
#         self.iou_nms  = iou_nms

#         self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
#         self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)

#     # ───────────────────────────────────────────────────────────────
#     # ───────────────────────────────────────────────────────────────
#     def _preprocess(self, img_uint8: np.ndarray) -> torch.Tensor:
#         """
#         img_uint8 : H×W×3  uint8  (RGB)
#         returns    : 1×3×416×416  float32  (ImageNet-norm, on self.device)
#         """
#         # 1. to float32 in [0,1] and send **directly** to GPU / self.device
#         t = torch.as_tensor(img_uint8, device=self.device).permute(2, 0, 1).float() / 255.0

#         # 2. resize to 416×416 if necessary
#         if t.shape[1] != self.inp_size or t.shape[2] != self.inp_size:
#             t = torch.nn.functional.interpolate(
#                     t.unsqueeze(0),
#                     size=(self.inp_size, self.inp_size),
#                     mode="bilinear",
#                     align_corners=False
#                 ).squeeze(0)

#         # 3. ImageNet normalisation (mean/std are already on the same device)
#         t = (t - self.mean) / self.std

#         return t.unsqueeze(0)   # add batch dimension
#     # ───────────────────────────────────────────────────────────────
#     def predict(self, source, conf=0.25, verbose=False):
#         """
#         Mimics Ultralytics signature:
#             result_list = model.predict(img_uint8, conf=0.25)
#         """
#         with torch.no_grad():
#             pred = self.model(self._preprocess(source))[0]   # dict

#         scores = pred["scores"]
#         keep   = scores >= conf
#         if keep.sum() == 0:
#             empty = _MiniBoxes(torch.zeros((0,4)),
#                                torch.zeros((0,)),
#                                torch.zeros((0,), dtype=torch.int64))
#             return [SimpleNamespace(boxes=empty)]

#         boxes = pred["boxes"][keep]
#         confs = scores[keep]
#         clses = pred["labels"][keep]

#         # extra NMS (YOLO does it internally; torchvision already ran one
#         # but we tighten duplicates similarly to the YOLO wrapper)
#         if boxes.size(0) > 1:
#             idx = nms(boxes, confs, self.iou_nms)
#             boxes, confs, clses = boxes[idx], confs[idx], clses[idx]

#         wrapped = _MiniBoxes(boxes.cpu(), confs.cpu(), clses.cpu())
#         return [SimpleNamespace(boxes=wrapped)]


